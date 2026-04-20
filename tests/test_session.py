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


class TestSimilarityMatrices:
    def test_add_similarity_matrix(self):
        session = Session(prompt="test", config={})
        
        matrix = [[1.0, 0.9], [0.9, 1.0]]
        session.add_similarity_matrix(1, matrix, ["model1", "model2"])
        
        assert len(session.similarity_matrices) == 1
        assert session.similarity_matrices[0]["round"] == 1
        assert session.similarity_matrices[0]["model_names"] == ["model1", "model2"]
        assert session.similarity_matrices[0]["matrix"] == matrix

    def test_get_similarity_matrix(self):
        session = Session(prompt="test", config={})
        
        session.add_similarity_matrix(1, [[1.0, 0.85], [0.85, 1.0]], ["a", "b"])
        session.add_similarity_matrix(2, [[1.0, 0.7], [0.7, 1.0]], ["a", "b"])
        
        mat1 = session.get_similarity_matrix(1)
        mat2 = session.get_similarity_matrix(2)
        mat3 = session.get_similarity_matrix(3)
        
        assert mat1["round"] == 1
        assert mat2["round"] == 2
        assert mat3 is None

    def test_similarity_matrix_in_to_dict(self):
        session = Session(prompt="test", config={})
        session.add_similarity_matrix(1, [[1.0, 0.95], [0.95, 1.0]], ["m1", "m2"])
        
        data = session.to_dict()
        assert "similarity_matrices" in data
        assert len(data["similarity_matrices"]) == 1
        assert data["similarity_matrices"][0]["round"] == 1

    def test_similarity_matrix_roundtrip(self):
        session = Session(prompt="test", config={}, session_id="sim-rt")
        session.add_response("m1", "resp1", 1, 0)
        session.add_response("m2", "resp2", 1, 1)
        session.add_similarity_matrix(1, [[1.0, 0.72], [0.72, 1.0]], ["m1", "m2"])
        
        data = session.to_dict()
        restored = Session.from_dict(data)
        
        assert len(restored.similarity_matrices) == 1
        assert restored.similarity_matrices[0]["round"] == 1
        assert restored.similarity_matrices[0]["model_names"] == ["m1", "m2"]
        assert restored.similarity_matrices[0]["matrix"] == [[1.0, 0.72], [0.72, 1.0]]

    def test_similarity_matrices_in_session_data(self):
        session = Session(prompt="test", config={})
        session.add_similarity_matrix(1, [[1.0, 0.5]], ["a", "b"])
        
        data = session.to_data()
        assert data.similarity_matrices == session.similarity_matrices

    def test_sim_matrix_without_config_snapshot(self):
        session = Session(prompt="test", config={})
        data = session.to_dict()
        session.from_dict(data)
        assert hasattr(session, 'similarity_matrices')


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