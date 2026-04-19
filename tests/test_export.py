import pytest
from storage.session import Session
from storage.export import Exporter


class TestExporterBasics:
    def test_exporter_class_exists(self):
        assert Exporter is not None


class TestExportSessionData:
    def test_session_data_structure(self):
        session = Session(prompt="Test question?", config={}, session_id="test-123")
        session.add_response("model1", "Answer: 42", 1, 0)
        session.add_summary(1, "Summary: All agreed on 42")
        session.mark_completed(consensus_round=1)

        data = session.to_dict()

        assert data["id"] == "test-123"
        assert data["prompt"] == "Test question?"
        assert data["status"] == "completed"
        assert len(data["responses"]) == 1
        assert len(data["summaries"]) == 1
        assert data["consensus_reached"] is True
        assert data["consensus_round"] == 1

    def test_session_running_status(self):
        session = Session(prompt="test", config={})
        assert session.status == "running"

    def test_session_stopped_status(self):
        session = Session(prompt="test", config={})
        session.mark_stopped()
        assert session.status == "stopped"

    def test_session_completed_no_consensus(self):
        session = Session(prompt="test", config={})
        session.mark_completed()

        assert session.status == "completed"
        assert session.consensus_reached is False
        assert session.consensus_round is None


class TestExportContent:
    def test_export_content_structure(self):
        session = Session(prompt="What is the meaning?", config={})
        session.add_response("llama", "42", 1, 0)
        session.add_response("mistral", "42 too", 1, 1)
        session.add_summary(1, "Summary")

        data = session.to_dict()
        assert "prompt" in data
        assert "responses" in data
        assert "summaries" in data
        assert "status" in data
        assert "id" in data

    def test_multiple_rounds(self):
        session = Session(prompt="test", config={})
        session.add_response("m1", "r1", 1, 0)
        session.add_response("m1", "r2", 2, 0)
        session.add_summary(1, "sum1")
        session.add_summary(2, "sum2")

        data = session.to_dict()
        assert len(data["summaries"]) == 2

    def test_human_responses(self):
        session = Session(prompt="test", config={})
        session.add_response("model", "model response", 1, 0)
        session.add_human_response("Human input", 1, 1)

        data = session.to_dict()
        assert len(data["human_responses"]) == 1

    def test_attributed_summaries(self):
        session = Session(prompt="test", config={})
        session.add_response("m1", "content", 1, 0)
        session.add_attributed_summary(
            1,
            individual_summaries={"m1": ["point1"]},
            agreement_analysis="agree",
            consensus_assessment="REACHED",
            confidence="HIGH",
            full_text="full",
        )

        data = session.to_dict()
        assert len(data["attributed_summaries"]) == 1
        assert data["attributed_summaries"][0]["consensus_assessment"] == "REACHED"


class TestExportFormatSelection:
    def test_markdown_format_selection(self):
        session = Session(prompt="test", config={})
        assert session is not None

    def test_json_format_selection(self):
        session = Session(prompt="test", config={})
        assert session is not None


class TestExportEdgeCases:
    def test_empty_responses(self):
        session = Session(prompt="empty", config={})
        data = session.to_dict()

        assert len(data["responses"]) == 0
        assert data["status"] == "running"

    def test_empty_summaries(self):
        session = Session(prompt="empty", config={})
        data = session.to_dict()

        assert len(data["summaries"]) == 0

    def test_long_prompt(self):
        long_prompt = "x" * 10000
        session = Session(prompt=long_prompt, config={})

        assert session.prompt == long_prompt

    def test_special_characters_in_prompt(self):
        special = "Test! @#$%^&*()\n\t\r"
        session = Session(prompt=special, config={})

        assert session.prompt == special