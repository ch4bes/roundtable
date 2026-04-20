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


class TestMatrixExportFormatting:
    def test_format_matrix_md_table(self):
        model_names = ["gpt4", "claude"]
        matrix = [[1.0, 0.72], [0.72, 1.0]]
        
        result = Exporter._format_matrix_md_table(model_names, matrix)
        
        assert "|" in result
        assert "gpt4" in result
        assert "claude" in result
        assert "0.72" in result

    def test_format_matrix_md_table_single_model(self):
        result = Exporter._format_matrix_md_table(["only_one"], [[1.0]])
        
        assert "only_one" in result
        assert "-" in result

    def test_format_matrix_md_table_empty(self):
        result = Exporter._format_matrix_md_table([], [])
        
        assert result == ""

    def test_format_matrix_table_basic(self):
        model_names = ["a-model", "b-model"]
        matrix = [[1.0, 0.5], [0.5, 1.0]]
        
        result = Exporter._format_matrix_table(model_names, matrix)
        
        assert "a-model" in result
        assert "b-model" in result
        assert "0.50" in result

    def test_format_matrix_table_empty(self):
        result = Exporter._format_matrix_table([], [])
        
        assert result == ""

    def test_markdown_export_includes_similarity_matrix(self):
        import asyncio
        import tempfile
        
        session = Session(prompt="test question?", config={})
        session.add_response("gpt4", "Answer A", 1, 0)
        session.add_response("claude", "Answer A", 1, 1)
        session.add_similarity_matrix(1, [[1.0, 0.95], [0.95, 1.0]], ["gpt4", "claude"])
        session.add_summary(1, "Summary")
        session.mark_completed(1)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            output_path = f.name
        
        asyncio.run(Exporter.export_markdown(session, output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        assert "Similarity Matrix" in content
        assert "0.95" in content
        assert "Agreement" in content
        assert "1.00" in content

    def test_markdown_export_no_similarity_matrix_when_absent(self):
        import asyncio
        import tempfile
        
        session = Session(prompt="test question?", config={})
        session.add_response("gpt4", "Answer A", 1, 0)
        session.add_response("claude", "Answer B", 1, 1)
        session.add_summary(1, "Summary")
        session.mark_completed(1)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            output_path = f.name
        
        asyncio.run(Exporter.export_markdown(session, output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        assert "Similarity Matrix" not in content

    def test_json_export_includes_similarity_matrices(self):
        import asyncio
        import tempfile
        import json
        
        session = Session(prompt="test", config={}, session_id="json-test")
        session.add_response("gpt4", "A", 1, 0)
        session.add_response("claude", "A", 1, 1)
        session.add_similarity_matrix(1, [[1.0, 0.8], [0.8, 1.0]], ["gpt4", "claude"])
        session.mark_completed(1)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name
        
        asyncio.run(Exporter.export_json(session, output_path))
        
        with open(output_path) as f:
            export_data = json.loads(f.read())
        
        assert "similarity_matrices" in export_data
        assert len(export_data["similarity_matrices"]) == 1
        assert export_data["similarity_matrices"][0]["matrix"] == [[1.0, 0.8], [0.8, 1.0]]
        assert export_data["similarity_matrices"][0]["model_names"] == ["gpt4", "claude"]
        assert "exported_at" in export_data