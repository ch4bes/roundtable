import pytest
from unittest.mock import AsyncMock, patch
from core.config import Config, ModelConfig, ContextConfig
from core.discussion import DiscussionOrchestrator
from storage.session import Session


class TestContextModesConfig:
    def test_valid_modes(self):
        for mode in ["full", "summary_only", "summary_plus_last_n"]:
            config = Config(context=ContextConfig(mode=mode))
            assert config.context.mode == mode

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            Config(context=ContextConfig(mode="invalid_mode"))

    def test_last_n_default(self):
        config = Config(context=ContextConfig(mode="summary_plus_last_n"))
        assert config.context.last_n_responses == 2


class TestContextModeBehavior:
    def test_first_round(self):
        session = Session(prompt="First round", config={})

        assert session.prompt == "First round"
        assert len(session.responses) == 0
        assert session.get_summary(1) is None

    def test_with_responses(self):
        session = Session(prompt="Test", config={})
        session.add_response("model1", "response 1", 1, 0)
        session.add_response("model1", "response 2", 2, 0)

        responses = session.get_round_responses(1)
        assert len(responses) == 1

    def test_with_summary(self):
        session = Session(prompt="Test", config={})
        session.add_response("model1", "response", 1, 0)
        session.add_summary(1, "Summary text")

        summary = session.get_summary(1)
        assert summary == "Summary text"

    def test_summary_plus_last_n_behavior(self):
        session = Session(prompt="Test", config={})
        session.add_response("model1", "old", 1, 0)
        session.add_response("model1", "recent", 2, 0)

        recent = session.responses[-2:]
        assert len(recent) == 2


class TestContextBuildEdgeCases:
    def test_empty_responses(self):
        session = Session(prompt="Empty test", config={})

        assert session.prompt == "Empty test"
        assert len(session.responses) == 0

    def test_multiple_models(self):
        session = Session(prompt="Multi model", config={})
        session.add_response("model1", "resp1", 1, 0)
        session.add_response("model2", "resp2", 1, 1)

        responses = session.get_round_responses(1)
        assert len(responses) == 2


class TestBuildContextFallback:
    """Tests for _build_context fallback path — ensures no TypeError on unknown modes or empty history."""

    def _make_orchestrator(self, context_mode: str):
        """Create a DiscussionOrchestrator with mocked Ollama dependencies."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            context=ContextConfig(mode=context_mode),
        )
        session = Session(prompt="What is the best programming language?", config={})

        with patch("core.discussion.OllamaClient") as MockOllama, \
             patch("core.discussion.SimilarityEngine") as MockSim, \
             patch("core.discussion.ConsensusDetector"):
            MockOllama.return_value = AsyncMock()
            MockSim.return_value = AsyncMock()
            orchestrator = DiscussionOrchestrator(config, session)
            return orchestrator, session

    @pytest.mark.asyncio
    async def test_build_context_unknown_mode_does_not_raise_type_error(self):
        """Unknown context_mode → falls back to ParticipantPrompt.initial(prompt) without TypeError.

        Regression test for issue #15: the old code called
        ParticipantPrompt.initial(prompt, model_position, total_models) which
        passes 3 args to a 1-arg method.
        """
        orchestrator, session = self._make_orchestrator("full")
         # Mutate mode to bypass Pydantic Literal validation
        orchestrator.config.context.mode = "unknown_mode"
        # Should not raise TypeError
        context = await orchestrator._build_context(
            model_position=1, total_models=2, round_num=1
        )
        assert session.prompt in context

    @pytest.mark.asyncio
    async def test_build_context_summary_plus_last_n_round_1_no_history(self):
        """summary_plus_last_n mode, round 1, no prior responses → uses ParticipantPrompt.initial."""
        orchestrator, session = self._make_orchestrator("summary_plus_last_n")
        context = await orchestrator._build_context(
            model_position=1, total_models=2, round_num=1
        )
        assert session.prompt in context
        assert "Provide your direct answer" in context

    @pytest.mark.asyncio
    async def test_build_context_full_mode_no_history(self):
        """full mode with no prior responses → falls back to ParticipantPrompt.initial."""
        orchestrator, session = self._make_orchestrator("full")
        context = await orchestrator._build_context(
            model_position=1, total_models=2, round_num=1
        )
        assert session.prompt in context
        assert "Provide your direct answer" in context

    @pytest.mark.asyncio
    async def test_build_context_summary_only_round_1_no_summary(self):
        """summary_only mode, round 1, no attributed summary → uses ParticipantPrompt.initial."""
        orchestrator, session = self._make_orchestrator("summary_only")
        context = await orchestrator._build_context(
            model_position=1, total_models=2, round_num=1
        )
        assert session.prompt in context
        assert "Provide your direct answer" in context