import pytest
import numpy as np
from prompts.system_prompts import ModeratorPrompt, ParticipantPrompt
from storage.session import AttributedSummary


class TestModeratorPrompt:
    def test_system_prompt_exists(self):
        system = ModeratorPrompt.system()
        assert len(system) > 0
        assert "moderator" in system.lower() or "summarize" in system.lower()

    def test_template(self):
        responses = [
            {"model": "llama3.2", "content": "The answer is 42", "round": 1},
            {"model": "mistral", "content": "42 is the answer", "round": 1},
        ]
        prompt = ModeratorPrompt.template(responses, 1)

        assert "llama3.2" in prompt
        assert "mistral" in prompt
        assert "Round 1" in prompt

    def test_template_with_similarity_matrix(self):
        responses = [
            {"model": "model1", "content": "response1", "round": 1},
            {"model": "model2", "content": "response2", "round": 1},
        ]
        matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
        model_names = ["model1", "model2"]

        prompt = ModeratorPrompt.template_with_similarity_matrix(
            responses, 1, matrix, model_names
        )

        assert "model1" in prompt
        assert "model2" in prompt
        assert "SIMILARITY" in prompt

    def test_format_similarity_matrix(self):
        matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ])
        model_names = ["model-a", "model-b", "model-c"]

        result = ModeratorPrompt._format_similarity_matrix(matrix, model_names)

        assert "model-a" in result
        assert "model-b" in result
        assert "|" in result

    def test_format_similarity_matrix_empty(self):
        matrix = np.array([])
        result = ModeratorPrompt._format_similarity_matrix(matrix, [])

        assert "(No responses to compare)" in result


class TestParticipantPrompt:
    def test_system_prompt(self):
        system = ParticipantPrompt.system()
        assert len(system) > 0
        assert "roundtable" in system.lower() or "participant" in system.lower()

    def test_initial(self):
        prompt = ParticipantPrompt.initial(
            "What is AI?",
            model_position=1,
            total_models=3,
        )

        assert "What is AI?" in prompt
        assert "Model 1/3" in prompt

    def test_with_summary(self):
        prompt = ParticipantPrompt.with_summary(
            "What is AI?",
            "Summary: AI is intelligence demonstrated by machines.",
            model_position=2,
            total_models=3,
            round_num=2,
        )

        assert "What is AI?" in prompt
        assert "Summary" in prompt
        assert "Model 2/3" in prompt

    def test_with_attributed_summary(self):
        attributed = AttributedSummary(
            round=1,
            individual_summaries={
                "model1": ["Point A", "Point B"],
                "model2": ["Point A", "Point C"],
            },
            agreement_analysis="Both models agree on Point A",
            consensus_assessment="REACHED",
            confidence="HIGH",
            full_text="Full text",
            timestamp="2024-01-01T00:00:00",
        )

        prompt = ParticipantPrompt.with_attributed_summary(
            "Test prompt",
            attributed,
            model_position=2,
            total_models=3,
            round_num=2,
        )

        assert "Test prompt" in prompt
        assert "Point A" in prompt or "Point B" in prompt
        assert "REACHED" in prompt

    def test_with_context(self):
        context_responses = [
            {"model": "model1", "content": "Response 1"},
            {"model": "model2", "content": "Response 2"},
        ]

        prompt = ParticipantPrompt.with_context(
            "Original prompt?",
            context_responses,
            model_position=1,
            total_models=2,
            round_num=1,
        )

        assert "Original prompt?" in prompt
        assert "Response 1" in prompt
        assert "Response 2" in prompt


class TestPromptEdgeCases:
    def test_moderator_empty_responses(self):
        prompt = ModeratorPrompt.template([], 1)
        assert "Round 1" in prompt

    def test_moderator_single_response(self):
        responses = [{"model": "single", "content": "only one", "round": 1}]
        prompt = ModeratorPrompt.template(responses, 1)
        assert "single" in prompt

    def test_participant_first_position(self):
        prompt = ParticipantPrompt.initial("Question?", 1, 4)
        assert "Model 1/4" in prompt

    def test_participant_last_position(self):
        prompt = ParticipantPrompt.initial("Question?", 4, 4)
        assert "Model 4/4" in prompt

    def test_attributed_summary_empty(self):
        attributed = AttributedSummary(
            round=1,
            individual_summaries={},
            agreement_analysis="No analysis",
            consensus_assessment="NOT REACHED",
            confidence="LOW",
            full_text="",
            timestamp="",
        )

        prompt = ParticipantPrompt.with_attributed_summary(
            "prompt",
            attributed,
            1,
            1,
            1,
        )

        assert "prompt" in prompt