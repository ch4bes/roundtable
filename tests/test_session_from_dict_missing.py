"""Tests for Session.from_dict() with missing optional fields - Issue #13 core logic coverage."""
import pytest
from storage.session import Session, Response, RoundSummary, AttributedSummary


class TestSessionFromDictMissingFields:
    """Test Session.from_dict() handles missing optional fields gracefully."""

    def test_from_dict_missing_config_snapshot(self):
        """Config snapshot missing → should not crash, use default."""
        data = {
            "id": "test-id-123",
            "prompt": "test prompt",
            # "config_snapshot" is missing
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
            "responses": [],
            "human_responses": [],
            "summaries": [],
            "attributed_summaries": [],
            "completed_rounds": 0,
            "consensus_reached": False,
            "consensus_round": None,
            "similarity_matrices": [],
            "final_review": None,
        }

        session = Session.from_dict(data)

        assert session.id == "test-id-123"
        assert session.prompt == "test prompt"
        # config should be None or empty dict, not crash
        assert session.config_snapshot is None or session.config_snapshot == {}

    def test_from_dict_missing_responses(self):
        """Responses list missing → should not crash."""
        data = {
            "id": "test-id-123",
            "prompt": "test prompt",
            "config_snapshot": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
            # "responses" key is missing entirely
            "human_responses": [],
            "summaries": [],
            "attributed_summaries": [],
            "completed_rounds": 0,
            "consensus_reached": False,
            "consensus_round": None,
            "similarity_matrices": [],
            "final_review": None,
        }

        session = Session.from_dict(data)

        assert session.responses == []
        assert isinstance(session.responses, list)

    def test_from_dict_missing_human_responses(self):
        """Human responses list missing → should not crash."""
        data = {
            "id": "test-id-123",
            "prompt": "test prompt",
            "config_snapshot": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
            "responses": [],
            # "human_responses" missing
            "summaries": [],
            "attributed_summaries": [],
            "completed_rounds": 0,
            "consensus_reached": False,
            "consensus_round": None,
            "similarity_matrices": [],
            "final_review": None,
        }

        session = Session.from_dict(data)

        assert session.human_responses == []
        assert isinstance(session.human_responses, list)

    def test_from_dict_missing_summaries(self):
        """Summaries list missing → should not crash."""
        data = {
            "id": "test-id-123",
            "prompt": "test prompt",
            "config_snapshot": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
            "responses": [],
            "human_responses": [],
            # "summaries" missing
            "attributed_summaries": [],
            "completed_rounds": 0,
            "consensus_reached": False,
            "consensus_round": None,
            "similarity_matrices": [],
            "final_review": None,
        }

        session = Session.from_dict(data)

        assert session.summaries == []
        assert isinstance(session.summaries, list)

    def test_from_dict_missing_attributed_summaries(self):
        """Attributed summaries list missing → should not crash."""
        data = {
            "id": "test-id-123",
            "prompt": "test prompt",
            "config_snapshot": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
            "responses": [],
            "human_responses": [],
            "summaries": [],
            # "attributed_summaries" missing
            "completed_rounds": 0,
            "consensus_reached": False,
            "consensus_round": None,
            "similarity_matrices": [],
            "final_review": None,
        }

        session = Session.from_dict(data)

        assert session.attributed_summaries == []
        assert isinstance(session.attributed_summaries, list)

    def test_from_dict_missing_similarity_matrices(self):
        """Similarity matrices list missing → should not crash."""
        data = {
            "id": "test-id-123",
            "prompt": "test prompt",
            "config_snapshot": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
            "responses": [],
            "human_responses": [],
            "summaries": [],
            "attributed_summaries": [],
            "completed_rounds": 0,
            "consensus_reached": False,
            "consensus_round": None,
            # "similarity_matrices" missing
            "final_review": None,
        }

        session = Session.from_dict(data)

        assert session.similarity_matrices == []
        assert isinstance(session.similarity_matrices, list)

    def test_from_dict_missing_final_review(self):
        """Final review missing → should not crash (already tested in test_session.py but good to verify)."""
        data = {
            "id": "test-id-123",
            "prompt": "test prompt",
            "config_snapshot": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "completed",
            "responses": [
                {
                    "model": "model1",
                    "content": "response content",
                    "round": 1,
                    "timestamp": "2024-01-01T00:00:00",
                    "position": 0,
                    "response_time_s": None,
                }
            ],
            "human_responses": [],
            "summaries": [
                {
                    "round": 1,
                    "summary": "summary text",
                    "timestamp": "2024-01-01T00:00:00",
                }
            ],
            "attributed_summaries": [],
            "completed_rounds": 1,
            "consensus_reached": True,
            "consensus_round": 1,
            "similarity_matrices": [],
            # "final_review" missing - older sessions may not have this
        }

        session = Session.from_dict(data)

        assert session.final_review is None
        assert session.status == "completed"
        assert session.consensus_reached is True

    def test_from_dict_all_optional_fields_missing(self):
        """All optional fields missing → should still create valid session."""
        data = {
            "id": "minimal-session-id",
            "prompt": "minimal prompt",
            # Only required fields
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
            # All optional lists missing
        }

        session = Session.from_dict(data)

        assert session.id == "minimal-session-id"
        assert session.prompt == "minimal prompt"
        assert session.status == "running"
        assert session.responses == []
        assert session.human_responses == []
        assert session.summaries == []
        assert session.attributed_summaries == []
        assert session.similarity_matrices == []
        assert session.completed_rounds == 0
        assert session.consensus_reached is False

    def test_from_dict_with_partial_responses(self):
        """Responses with some optional fields missing."""
        data = {
            "id": "test-id",
            "prompt": "test prompt",
            "config_snapshot": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
            "responses": [
                {
                    "model": "model1",
                    "content": "response content",
                    "round": 1,
                    "timestamp": "2024-01-01T00:00:00",
                    "position": 0,
                    # "response_time_s" is missing
                }
            ],
            "human_responses": [],
            "summaries": [],
            "attributed_summaries": [],
            "completed_rounds": 0,
            "consensus_reached": False,
            "consensus_round": None,
            "similarity_matrices": [],
            "final_review": None,
        }

        session = Session.from_dict(data)

        assert len(session.responses) == 1
        assert session.responses[0].response_time_s is None


class TestSessionFromDictRoundtripWithMissingFields:
    """Test that sessions with missing fields can still be used after restoration."""

    def test_restored_session_can_add_responses(self):
        """After restoring a session with missing fields, can still add responses."""
        data = {
            "id": "test-id",
            "prompt": "test prompt",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "running",
        }

        session = Session.from_dict(data)

        # Should be able to add responses
        session.add_response("model1", "response1", 1, 0)
        assert len(session.responses) == 1

        session.add_summary(1, "summary1")
        assert len(session.summaries) == 1

    def test_restored_session_can_export(self):
        """Session restored with missing fields can be exported."""
        data = {
            "id": "test-id",
            "prompt": "test prompt",
            "config_snapshot": {},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "completed",
            "responses": [],
            "human_responses": [],
            "summaries": [],
            "attributed_summaries": [],
            "completed_rounds": 0,
            "consensus_reached": True,
            "consensus_round": None,
            "similarity_matrices": [],
            "final_review": None,
        }

        session = Session.from_dict(data)

        # Should be able to convert to dict without error
        result = session.to_dict()

        assert result["id"] == "test-id"
        assert result["status"] == "completed"