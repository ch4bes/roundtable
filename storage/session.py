import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field, asdict
import aiofiles


@dataclass
class Response:
    model: str
    content: str
    round: int
    timestamp: str
    position: int
    response_time_s: float | None = None


@dataclass
class RoundSummary:
    round: int
    summary: str
    timestamp: str


@dataclass
class AttributedSummary:
    round: int
    individual_summaries: dict[str, list[str]]  # model -> [points]
    agreement_analysis: str
    consensus_assessment: str  # "REACHED" / "NOT REACHED"
    confidence: str  # "HIGH" / "MEDIUM" / "LOW"
    full_text: str  # Original full moderator output
    timestamp: str


@dataclass
class SessionData:
    id: str
    prompt: str
    config_snapshot: dict
    created_at: str
    updated_at: str
    status: Literal["running", "completed", "stopped"]
    responses: list[Response]
    human_responses: list[Response]
    summaries: list[RoundSummary]
    attributed_summaries: list[AttributedSummary]
    completed_rounds: int
    consensus_reached: bool
    consensus_round: int | None
    similarity_matrices: list[dict]
    final_review: str | None


class Session:
    def __init__(
        self,
        prompt: str,
        config: dict,
        session_id: str | None = None,
    ):
        self.id = session_id or str(uuid.uuid4())
        self.prompt = prompt
        self.config_snapshot = config
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.status: Literal["running", "completed", "stopped"] = "running"
        self.responses: list[Response] = []
        self.human_responses: list[Response] = []
        self.summaries: list[RoundSummary] = []
        self.attributed_summaries: list[AttributedSummary] = []
        self.completed_rounds = 0
        self.consensus_reached = False
        self.consensus_round: int | None = None
        self.similarity_matrices: list[dict] = []
        self.final_review: str | None = None

    def add_response(self, model: str, content: str, round_num: int, position: int, response_time_s: float | None = None) -> Response:
        response = Response(
            model=model,
            content=content,
            round=round_num,
            timestamp=datetime.now().isoformat(),
            position=position,
            response_time_s=response_time_s,
        )
        self.responses.append(response)
        self.updated_at = datetime.now().isoformat()
        return response

    def add_human_response(self, content: str, round_num: int, position: int) -> Response:
        response = Response(
            model="human",
            content=content,
            round=round_num,
            timestamp=datetime.now().isoformat(),
            position=position,
        )
        self.human_responses.append(response)
        self.updated_at = datetime.now().isoformat()
        return response

    def get_round_human_responses(self, round_num: int) -> list[Response]:
        return [r for r in self.human_responses if r.round == round_num]

    def add_summary(self, round_num: int, summary: str) -> RoundSummary:
        round_summary = RoundSummary(
            round=round_num,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )
        self.summaries.append(round_summary)
        self.completed_rounds = round_num
        self.updated_at = datetime.now().isoformat()
        return round_summary

    def add_attributed_summary(
        self,
        round_num: int,
        individual_summaries: dict[str, list[str]],
        agreement_analysis: str,
        consensus_assessment: str,
        confidence: str,
        full_text: str,
    ) -> AttributedSummary:
        attr_summary = AttributedSummary(
            round=round_num,
            individual_summaries=individual_summaries,
            agreement_analysis=agreement_analysis,
            consensus_assessment=consensus_assessment,
            confidence=confidence,
            full_text=full_text,
            timestamp=datetime.now().isoformat(),
        )
        self.attributed_summaries.append(attr_summary)
        self.completed_rounds = round_num
        self.updated_at = datetime.now().isoformat()
        return attr_summary

    def mark_completed(self, consensus_round: int | None = None):
        self.status = "completed"
        self.consensus_reached = consensus_round is not None
        self.consensus_round = consensus_round
        self.updated_at = datetime.now().isoformat()

    def mark_stopped(self):
        self.status = "stopped"
        self.updated_at = datetime.now().isoformat()

    def add_similarity_matrix(self, round_num: int, matrix: list[list[float]], model_names: list[str]):
        matrix_entry = {
            "round": round_num,
            "matrix": matrix,
            "model_names": model_names,
            "timestamp": datetime.now().isoformat(),
        }
        self.similarity_matrices.append(matrix_entry)
        self.updated_at = datetime.now().isoformat()

    def get_similarity_matrix(self, round_num: int) -> dict | None:
        for m in self.similarity_matrices:
            if m["round"] == round_num:
                return m
        return None

    def add_final_review(self, review_text: str):
        self.final_review = review_text
        self.updated_at = datetime.now().isoformat()

    def get_round_responses(self, round_num: int) -> list[Response]:
        return [r for r in self.responses if r.round == round_num]

    def get_current_round_responses(self) -> list[Response]:
        if self.completed_rounds == 0:
            return []
        return self.get_round_responses(self.completed_rounds)

    def get_latest_summary(self) -> str | None:
        if not self.summaries:
            return None
        return self.summaries[-1].summary

    def get_summary(self, round_num: int) -> str | None:
        for s in self.summaries:
            if s.round == round_num:
                return s.summary
        return None

    def get_attributed_summary(self, round_num: int) -> AttributedSummary | None:
        for s in self.attributed_summaries:
            if s.round == round_num:
                return s
        return None

    def get_latest_attributed_summary(self) -> AttributedSummary | None:
        if not self.attributed_summaries:
            return None
        return self.attributed_summaries[-1]

    def to_data(self) -> SessionData:
        return SessionData(
            id=self.id,
            prompt=self.prompt,
            config_snapshot=self.config_snapshot,
            created_at=self.created_at,
            updated_at=self.updated_at,
            status=self.status,
            responses=self.responses,
            human_responses=self.human_responses,
            summaries=self.summaries,
            attributed_summaries=self.attributed_summaries,
            completed_rounds=self.completed_rounds,
            consensus_reached=self.consensus_reached,
            consensus_round=self.consensus_round,
            similarity_matrices=self.similarity_matrices,
            final_review=self.final_review,
        )

    def to_dict(self) -> dict:
        data = self.to_data()
        result = {
            "id": data.id,
            "prompt": data.prompt,
            "config_snapshot": data.config_snapshot,
            "created_at": data.created_at,
            "updated_at": data.updated_at,
            "status": data.status,
            "responses": [asdict(r) for r in data.responses],
            "human_responses": [asdict(r) for r in data.human_responses],
            "summaries": [asdict(s) for s in data.summaries],
            "attributed_summaries": [asdict(a) for a in data.attributed_summaries],
            "completed_rounds": data.completed_rounds,
            "consensus_reached": data.consensus_reached,
            "consensus_round": data.consensus_round,
            "similarity_matrices": data.similarity_matrices,
            "final_review": self.final_review,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        session = cls(
            prompt=data["prompt"],
            config=data.get("config_snapshot", {}),
            session_id=data["id"],
        )
        session.created_at = data["created_at"]
        session.updated_at = data["updated_at"]
        session.status = data["status"]
        session.responses = [Response(**r) for r in data.get("responses", [])]
        session.human_responses = [Response(**r) for r in data.get("human_responses", [])]
        session.summaries = [RoundSummary(**s) for s in data.get("summaries", [])]
        session.attributed_summaries = [
            AttributedSummary(**a) for a in data.get("attributed_summaries", [])
        ]
        session.completed_rounds = data.get("completed_rounds", 0)
        session.consensus_reached = data.get("consensus_reached", False)
        session.consensus_round = data.get("consensus_round")
        session.similarity_matrices = data.get("similarity_matrices", [])
        session.final_review = data.get("final_review")
        return session


class SessionManager:
    def __init__(self, sessions_dir: str = "./sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    async def save(self, session: Session) -> Path:
        path = self._get_session_path(session.id)
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(session.to_dict(), indent=2))
        return path

    async def load(self, session_id: str) -> Session | None:
        path = self._get_session_path(session_id)
        if not path.exists():
            return None
        async with aiofiles.open(path, "r") as f:
            data = json.loads(await f.read())
        return Session.from_dict(data)

    async def list_sessions(self) -> list[dict]:
        sessions = []
        for path in self.sessions_dir.glob("*.json"):
            async with aiofiles.open(path, "r") as f:
                data = json.loads(await f.read())
            sessions.append(
                {
                    "id": data["id"],
                    "prompt": data["prompt"][:80] + "..."
                    if len(data["prompt"]) > 100
                    else data["prompt"],
                    "status": data["status"],
                    "created_at": data["created_at"],
                    "completed_rounds": data.get("completed_rounds", 0),
                }
            )
        return sorted(sessions, key=lambda s: s["created_at"], reverse=True)

    async def delete(self, session_id: str) -> bool:
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False
