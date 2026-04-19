import pytest
import uuid
from storage.session import Session, Response, RoundSummary, AttributedSummary


class TestSession:
    def test_session_creation(self):
        session = Session(
            prompt="What is the meaning of life?",
            config={"models": []},
            session_id="test-123",
        )

        assert session.id == "test-123"
        assert session.prompt == "What is the meaning of life?"
        assert session.status == "running"
        assert session.completed_rounds == 0
        assert session.consensus_reached is False

    def test_add_response(self):
        session = Session(prompt="test", config={})

        response = session.add_response(
            model="llama3.2",
            content="42",
            round_num=1,
            position=0,
        )

        assert response.model == "llama3.2"
        assert response.content == "42"
        assert response.round == 1
        assert response.position == 0
        assert len(session.responses) == 1

    def test_add_human_response(self):
        session = Session(prompt="test", config={})

        response = session.add_human_response(
            content="I think it's 42",
            round_num=1,
            position=0,
        )

        assert response.model == "human"
        assert response.content == "I think it's 42"
        assert len(session.human_responses) == 1

    def test_add_summary(self):
        session = Session(prompt="test", config={})

        summary = session.add_summary(
            round_num=1,
            summary="All models agreed: 42",
        )

        assert summary.round == 1
        assert summary.summary == "All models agreed: 42"
        assert len(session.summaries) == 1

    def test_add_attributed_summary(self):
        session = Session(prompt="test", config={})

        attributed = session.add_attributed_summary(
            round_num=1,
            individual_summaries={
                "llama3.2": ["Point 1", "Point 2"],
                "mistral": ["Point 1", "Point 3"],
            },
            agreement_analysis="Models agreed on X, disagreed on Y",
            consensus_assessment="REACHED",
            confidence="HIGH",
            full_text="Full moderator output",
        )

        assert attributed.round == 1
        assert len(attributed.individual_summaries) == 2
        assert attributed.consensus_assessment == "REACHED"
        assert attributed.confidence == "HIGH"

    def test_get_round_responses(self):
        session = Session(prompt="test", config={})
        session.add_response("model1", "response1", 1, 0)
        session.add_response("model2", "response2", 1, 1)
        session.add_response("model1", "response3", 2, 0)

        round_1_responses = session.get_round_responses(1)
        round_2_responses = session.get_round_responses(2)

        assert len(round_1_responses) == 2
        assert len(round_2_responses) == 1

    def test_get_round_human_responses(self):
        session = Session(prompt="test", config={})
        session.add_human_response("human input", 1, 0)
        session.add_response("model", "model input", 1, 0)

        human_responses = session.get_round_human_responses(1)

        assert len(human_responses) == 1
        assert human_responses[0].content == "human input"

    def test_get_summary(self):
        session = Session(prompt="test", config={})
        session.add_summary(1, "Summary for round 1")
        session.add_summary(2, "Summary for round 2")

        assert session.get_summary(1) == "Summary for round 1"
        assert session.get_summary(2) == "Summary for round 2"
        assert session.get_summary(3) is None

    def test_get_attributed_summary(self):
        session = Session(prompt="test", config={})
        session.add_attributed_summary(
            1,
            individual_summaries={},
            agreement_analysis="agree",
            consensus_assessment="REACHED",
            confidence="HIGH",
            full_text="full",
        )

        attributed = session.get_attributed_summary(1)
        assert attributed is not None
        assert attributed.consensus_assessment == "REACHED"

    def test_get_latest_attributed_summary(self):
        session = Session(prompt="test", config={})
        session.add_attributed_summary(
            1,
            individual_summaries={},
            agreement_analysis="first",
            consensus_assessment="NOT REACHED",
            confidence="LOW",
            full_text="first",
        )
        session.add_attributed_summary(
            2,
            individual_summaries={},
            agreement_analysis="second",
            consensus_assessment="REACHED",
            confidence="HIGH",
            full_text="second",
        )

        latest = session.get_latest_attributed_summary()
        assert latest.consensus_assessment == "REACHED"

    def test_mark_completed_with_consensus(self):
        session = Session(prompt="test", config={})
        session.mark_completed(consensus_round=3)

        assert session.status == "completed"
        assert session.consensus_reached is True
        assert session.consensus_round == 3

    def test_mark_completed_without_consensus(self):
        session = Session(prompt="test", config={})
        session.mark_completed()

        assert session.status == "completed"
        assert session.consensus_reached is False

    def test_mark_stopped(self):
        session = Session(prompt="test", config={})
        session.mark_stopped()

        assert session.status == "stopped"

    def test_to_dict(self):
        session = Session(prompt="test", config={"key": "value"}, session_id="abc-123")
        session.add_response("model", "content", 1, 0)
        session.add_summary(1, "summary")

        data = session.to_dict()

        assert data["id"] == "abc-123"
        assert data["prompt"] == "test"
        assert data["status"] == "running"
        assert len(data["responses"]) == 1
        assert len(data["summaries"]) == 1

    def test_to_from_dict_roundtrip(self):
        session = Session(prompt="test", config={}, session_id="xyz")
        session.add_response("m1", "r1", 1, 0)
        session.mark_completed(1)

        data = session.to_dict()
        restored = Session(
            prompt=data["prompt"],
            config=data["config_snapshot"],
            session_id=data["id"],
        )
        restored.created_at = data["created_at"]
        restored.updated_at = data["updated_at"]
        restored.status = data["status"]
        restored.completed_rounds = data["completed_rounds"]
        restored.consensus_reached = data["consensus_reached"]
        restored.consensus_round = data.get("consensus_round")

        assert restored.id == session.id
        assert restored.prompt == session.prompt
        assert restored.status == session.status

    def test_round_trip_serialization(self):
        session = Session(prompt="round trip test", config={"testing": True})
        session.add_response("model1", "response1", 1, 0)
        session.add_response("model2", "response2", 1, 1)
        session.add_summary(1, "Round 1 summary")
        session.mark_completed(consensus_round=1)

        data = session.to_dict()
        restored = Session(
            prompt=data["prompt"],
            config=data["config_snapshot"],
            session_id=data["id"],
        )
        restored.responses = [Response(**r) for r in data.get("responses", [])]
        restored.completed_rounds = data["completed_rounds"]
        restored.consensus_reached = data["consensus_reached"]

        assert restored.id == session.id
        assert restored.prompt == session.prompt
        assert len(restored.responses) == len(session.responses)
        assert restored.consensus_reached == session.consensus_reached


class TestResponse:
    def test_response_creation(self):
        response = Response(
            model="test-model",
            content="test content",
            round=1,
            timestamp="2024-01-01T00:00:00",
            position=0,
        )

        assert response.model == "test-model"
        assert response.content == "test content"
        assert response.round == 1
        assert response.position == 0


class TestRoundSummary:
    def test_summary_creation(self):
        summary = RoundSummary(
            round=1,
            summary="Test summary",
            timestamp="2024-01-01T00:00:00",
        )

        assert summary.round == 1
        assert summary.summary == "Test summary"


class TestAttributedSummary:
    def test_attributed_summary_creation(self):
        attributed = AttributedSummary(
            round=1,
            individual_summaries={"model1": ["point1"]},
            agreement_analysis="agree",
            consensus_assessment="REACHED",
            confidence="HIGH",
            full_text="full text",
            timestamp="2024-01-01T00:00:00",
        )

        assert attributed.round == 1
        assert len(attributed.individual_summaries) == 1
        assert attributed.consensus_assessment == "REACHED"
        assert attributed.confidence == "HIGH"