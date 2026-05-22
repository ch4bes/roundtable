import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import aiofiles

from .utils import sanitize_session_id, validate_path_within

logger = logging.getLogger(__name__)


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
    images: list[str]  # List of image file paths or base64 encoded images
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
        images: list[str] | None = None,
    ):
        self.id = session_id or str(uuid.uuid4())
        self.prompt = prompt
        self.images = images or []
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
        # Tracks how many items of each list have been flushed to the JSONL log.
        # Items from [offset:] are "pending" and will be appended on the next save.
        self._flush_offsets: dict[str, int] = {
            "responses": 0,
            "human_responses": 0,
            "summaries": 0,
            "attributed_summaries": 0,
            "similarity_matrices": 0,
        }

    def add_response(
        self,
        model: str,
        content: str,
        round_num: int,
        position: int,
        response_time_s: float | None = None,
    ) -> Response:
        """Record a model response for a given round."""
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

    def add_human_response(
        self, content: str, round_num: int, position: int
    ) -> Response:
        """Record a human-in-the-loop response for a given round."""
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
        """Record the model's summary for a specific round."""
        round_summary = RoundSummary(
            round=round_num,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )
        self.summaries.append(round_summary)
        self.completed_rounds = max(self.completed_rounds, round_num)
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
        self.completed_rounds = max(self.completed_rounds, round_num)
        self.updated_at = datetime.now().isoformat()
        return attr_summary

    def mark_completed(self, consensus_round: int | None = None) -> None:
        self.status = "completed"
        self.consensus_reached = consensus_round is not None
        self.consensus_round = consensus_round
        self.updated_at = datetime.now().isoformat()

    def mark_stopped(self) -> None:
        self.status = "stopped"
        self.updated_at = datetime.now().isoformat()

    def add_similarity_matrix(
        self, round_num: int, matrix: list[list[float]], model_names: list[str]
    ) -> None:
        """Store the similarity matrix from the SimilarityEngine."""
        matrix_entry = {
            "round": round_num,
            "matrix": matrix,
            "model_names": model_names,
            "timestamp": datetime.now().isoformat(),
        }
        self.similarity_matrices.append(matrix_entry)
        self.updated_at = datetime.now().isoformat()

    def get_similarity_matrix(self, round_num: int) -> dict | None:
        """Retrieve the stored similarity matrix for a specific round."""
        for m in self.similarity_matrices:
            if m["round"] == round_num:
                return m
        return None

    def add_final_review(self, review_text: str) -> None:
        self.final_review = review_text
        self.updated_at = datetime.now().isoformat()

    def get_round_responses(self, round_num: int) -> list[Response]:
        """Get all responses for a specific round."""
        return [r for r in self.responses if r.round == round_num]

    def get_current_round_responses(self) -> list[Response]:
        """Get responses for the current round (latest completed)."""
        if self.completed_rounds == 0:
            return []
        return self.get_round_responses(self.completed_rounds)

    def get_latest_summary(self) -> str | None:
        """Get the summary for the latest round."""
        if not self.summaries:
            return None
        return self.summaries[-1].summary

    def get_summary(self, round_num: int) -> str | None:
        """Get the summary for a specific round."""
        for s in self.summaries:
            if s.round == round_num:
                return s.summary
        return None

    def get_attributed_summary(self, round_num: int) -> AttributedSummary | None:
        """Get the attributed summary for a specific round."""
        for s in self.attributed_summaries:
            if s.round == round_num:
                return s
        return None

    def get_latest_attributed_summary(self) -> AttributedSummary | None:
        """Get the attributed summary for the latest round."""
        if not self.attributed_summaries:
            return None
        return self.attributed_summaries[-1]

    # ------------------------------------------------------------------
    # Incremental-save helpers (used by SessionManager)
    # ------------------------------------------------------------------

    def _get_pending_events(self) -> list[dict]:
        """Return list events not yet written to the JSONL append-log."""
        events: list[dict] = []
        for resp in self.responses[self._flush_offsets["responses"] :]:
            events.append({"type": "response", "data": asdict(resp)})
        for resp in self.human_responses[self._flush_offsets["human_responses"] :]:
            events.append({"type": "human_response", "data": asdict(resp)})
        for s in self.summaries[self._flush_offsets["summaries"] :]:
            events.append({"type": "summary", "data": asdict(s)})
        for a in self.attributed_summaries[
            self._flush_offsets["attributed_summaries"] :
        ]:
            events.append({"type": "attributed_summary", "data": asdict(a)})
        for m in self.similarity_matrices[self._flush_offsets["similarity_matrices"] :]:
            events.append({"type": "similarity_matrix", "data": m})
        return events

    def _mark_flushed(self) -> None:
        """Advance flush offsets to the current end of each list."""
        self._flush_offsets["responses"] = len(self.responses)
        self._flush_offsets["human_responses"] = len(self.human_responses)
        self._flush_offsets["summaries"] = len(self.summaries)
        self._flush_offsets["attributed_summaries"] = len(self.attributed_summaries)
        self._flush_offsets["similarity_matrices"] = len(self.similarity_matrices)

    def _to_header_dict(self) -> dict:
        """Serialize only session metadata (no list data) for the v2 storage format."""
        return {
            "format_version": 2,
            "id": self.id,
            "prompt": self.prompt,
            "images": self.images,
            "config_snapshot": self.config_snapshot,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "completed_rounds": self.completed_rounds,
            "consensus_reached": self.consensus_reached,
            "consensus_round": self.consensus_round,
            "final_review": self.final_review,
        }

    def to_data(self) -> SessionData:
        return SessionData(
            id=self.id,
            prompt=self.prompt,
            images=self.images,
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
        """Deserialize a Session from a dictionary."""
        session = cls(
            prompt=data["prompt"],
            config=data.get("config_snapshot", {}),
            session_id=data["id"],
            images=data.get("images", []),
        )
        session.created_at = data["created_at"]
        session.updated_at = data["updated_at"]
        session.status = data["status"]
        session.responses = [Response(**r) for r in data.get("responses", [])]
        session.human_responses = [
            Response(**r) for r in data.get("human_responses", [])
        ]
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
        """Get the JSON header file path for a session, with path traversal protection."""
        safe_name = sanitize_session_id(session_id)
        path = self.sessions_dir / f"{safe_name}.json"
        return validate_path_within(self.sessions_dir.resolve(), path.resolve())

    def _get_log_path(self, session_id: str) -> Path:
        """Get the JSONL append-log path for a session, with path traversal protection."""
        safe_name = sanitize_session_id(session_id)
        path = self.sessions_dir / f"{safe_name}.jsonl"
        return validate_path_within(self.sessions_dir.resolve(), path.resolve())

    async def save(self, session: Session) -> Path:
        """Incrementally persist a session.

        New list items (responses, summaries, etc.) are appended to a
        ``{session_id}.jsonl`` append-log so each call is O(1) regardless
        of how many items have accumulated.  A small ``{session_id}.json``
        header (metadata + scalar state) is written atomically via
        write-then-rename so readers always see a consistent snapshot.

        Old sessions stored in the v1 full-JSON format are automatically
        migrated: their entire contents are appended to the JSONL log on
        the first save, and subsequent calls are incremental.
        """
        header_path = self._get_session_path(session.id)
        log_path = self._get_log_path(session.id)

        # Append only the events that have not been persisted yet.
        pending = session._get_pending_events()
        if pending:
            lines = "\n".join(json.dumps(event) for event in pending) + "\n"
            async with aiofiles.open(log_path, "a") as f:
                await f.write(lines)
        session._mark_flushed()

        # Write the small header atomically (O(1) regardless of session size).
        tmp_path = header_path.with_suffix(".tmp")
        async with aiofiles.open(tmp_path, "w") as f:
            await f.write(json.dumps(session._to_header_dict(), indent=2))
        tmp_path.replace(header_path)  # atomic on POSIX; best-effort on Windows

        return header_path

    async def load(self, session_id: str) -> Session | None:
        """Load a session from disk.

        Supports both the legacy v1 format (single full-JSON file) and the
        current v2 format (header JSON + JSONL append-log).  Corrupt lines
        at the end of the JSONL log (e.g. from a mid-write crash) are
        skipped with a warning rather than aborting the load.
        """
        header_path = self._get_session_path(session_id)
        if not header_path.exists():
            return None

        async with aiofiles.open(header_path, "r") as f:
            header_data = json.loads(await f.read())

        # v1 legacy format: full session data in a single JSON file.
        if header_data.get("format_version", 1) != 2:
            session = Session.from_dict(header_data)
            # _flush_offsets default to 0, so the first save migrates all
            # existing items to the JSONL log automatically.
            return session

        # v2 format: reconstruct metadata from the header file.
        session = Session(
            prompt=header_data["prompt"],
            config=header_data.get("config_snapshot", {}),
            session_id=header_data["id"],
            images=header_data.get("images", []),
        )
        session.created_at = header_data["created_at"]
        session.updated_at = header_data["updated_at"]
        session.status = header_data["status"]
        session.completed_rounds = header_data.get("completed_rounds", 0)
        session.consensus_reached = header_data.get("consensus_reached", False)
        session.consensus_round = header_data.get("consensus_round")
        session.final_review = header_data.get("final_review")

        # Replay all events from the JSONL append-log.
        log_path = self._get_log_path(session_id)
        if log_path.exists():
            async with aiofiles.open(log_path, "r") as f:
                content = await f.read()
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping corrupt JSONL line %d in session %s",
                        line_num,
                        session_id,
                    )
                    continue
                etype = event.get("type")
                data = event.get("data", {})
                if etype == "response":
                    session.responses.append(Response(**data))
                elif etype == "human_response":
                    session.human_responses.append(Response(**data))
                elif etype == "summary":
                    session.summaries.append(RoundSummary(**data))
                elif etype == "attributed_summary":
                    session.attributed_summaries.append(AttributedSummary(**data))
                elif etype == "similarity_matrix":
                    session.similarity_matrices.append(data)
                else:
                    logger.debug(
                        "Unknown event type %r in session %s line %d",
                        etype,
                        session_id,
                        line_num,
                    )

        # Set flush offsets so only items added *after* this load are appended.
        session._mark_flushed()
        return session

    async def list_sessions(self) -> list[dict]:
        """List all saved sessions sorted by created_at (newest first).

        Both v1 and v2 formats are supported: the header JSON contains all
        the fields needed for listing regardless of format version.
        """
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
        """Delete a session from disk (both header JSON and JSONL log)."""
        header_path = self._get_session_path(session_id)
        if not header_path.exists():
            return False
        header_path.unlink()
        log_path = self._get_log_path(session_id)
        if log_path.exists():
            log_path.unlink()
        return True
