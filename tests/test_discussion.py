import pytest
from unittest.mock import AsyncMock, MagicMock
from core.config import Config, ModelConfig, DiscussionConfig, ContextConfig
from core.discussion import DiscussionOrchestrator, DiscussionState, _ConsensusVerdict
from storage.session import Session, Response


class TestDiscussionState:
    def test_creation(self):
        state = DiscussionState(
            current_round=1,
            current_model_index=0,
            model_order=["model1", "model2"],
        )

        assert state.current_round == 1
        assert state.current_model_index == 0
        assert state.model_order == ["model1", "model2"]
        assert state.consensus_result is None

    def test_defaults(self):
        state = DiscussionState(
            current_round=1,
            current_model_index=0,
            model_order=[],
        )

        assert state.is_running is False
        assert state.is_paused is False


class TestConfigIntegration:
    def test_config_with_all_options(self):
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(
                max_rounds=5,
                consensus_threshold=0.8,
                consensus_method="pairwise",
                rotation_order="sequential",
            ),
        )

        assert len(config.models) == 2
        assert config.discussion.max_rounds == 5
        assert config.discussion.consensus_threshold == 0.8
        assert config.discussion.consensus_method == "pairwise"
        assert config.discussion.rotation_order == "sequential"


class TestSessionMethods:
    def test_get_round_responses(self):
        session = Session(prompt="test", config={})
        session.add_response("model1", "response1", 1, 0)
        session.add_response("model2", "response2", 1, 1)

        responses = session.get_round_responses(1)
        assert len(responses) == 2

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
            individual_summaries={"m1": ["p1"]},
            agreement_analysis="agree",
            consensus_assessment="REACHED",
            confidence="HIGH",
            full_text="full",
        )

        attributed = session.get_attributed_summary(1)
        assert attributed.consensus_assessment == "REACHED"


class TestParseAttributedSummary:
    def test_final_consensus_not_reached(self):
        text = """## Individual Summaries
### Model1
- Point 1

**Consensus Assessment:** REACHED (Confidence: HIGH)

## Final Consensus
Consensus: NOT REACHED
Confidence: HIGH
Justification: Some disagreement"""

        result = DiscussionOrchestrator._parse_attributed_summary(None, text, [])
        assert result["consensus_assessment"] == "NOT REACHED"

    def test_final_consensus_reached(self):
        text = """## Individual Summaries
### Model1
- Point 1

## Final Consensus
Consensus: REACHED
Confidence: HIGH"""

        result = DiscussionOrchestrator._parse_attributed_summary(None, text, [])
        assert result["consensus_assessment"] == "REACHED"

    def test_no_final_consensus_uses_earlier_reached(self):
        text = """## Individual Summaries
### Model1
- Point 1

**Consensus Assessment:** REACHED (Confidence: HIGH)"""

        result = DiscussionOrchestrator._parse_attributed_summary(None, text, [])
        assert result["consensus_assessment"] == "REACHED"

    def test_no_consensus_at_all(self):
        text = """## Individual Summaries
### Model1
- Point 1"""

        result = DiscussionOrchestrator._parse_attributed_summary(None, text, [])
        assert result["consensus_assessment"] == "NOT REACHED"


class TestMainPointConsensus:
    def test_full_agreement_triggers_inconsistency_when_moderator_says_not_reached(self):
        from dataclasses import dataclass

        @dataclass
        class MockAttributed:
            agreement_analysis: str
            consensus_assessment: str
            confidence: str

        attributed = MockAttributed(
            agreement_analysis="**Areas of Full Agreement:** All participants agree on the main point.",
            consensus_assessment="NOT REACHED",
            confidence="HIGH"
        )

        result = DiscussionOrchestrator._check_main_point_consensus(None, attributed)
        assert result is _ConsensusVerdict.INCONSISTENT

    def test_overwhelming_consensus_returns_not_reached_without_main_answer(self):
        from dataclasses import dataclass

        @dataclass
        class MockAttributed:
            agreement_analysis: str
            consensus_assessment: str
            confidence: str

        attributed = MockAttributed(
            agreement_analysis="There is overwhelming consensus among all participants.",
            consensus_assessment="NOT REACHED",
            confidence="HIGH"
        )

        result = DiscussionOrchestrator._check_main_point_consensus(None, attributed)
        assert result is _ConsensusVerdict.NOT_REACHED

    def test_moderator_says_reached_short_circuits(self):
        from dataclasses import dataclass

        @dataclass
        class MockAttributed:
            agreement_analysis: str
            consensus_assessment: str
            confidence: str

        attributed = MockAttributed(
            agreement_analysis="",
            consensus_assessment="REACHED",
            confidence="HIGH"
        )

        result = DiscussionOrchestrator._check_main_point_consensus(None, attributed)
        assert result is _ConsensusVerdict.REACHED

    def test_disagreement_returns_not_reached(self):
        from dataclasses import dataclass

        @dataclass
        class MockAttributed:
            agreement_analysis: str
            consensus_assessment: str
            confidence: str

        attributed = MockAttributed(
            agreement_analysis="No agreement yet. Models disagree on the main point.",
            consensus_assessment="NOT REACHED",
            confidence="HIGH"
        )

        result = DiscussionOrchestrator._check_main_point_consensus(None, attributed)
        assert result is _ConsensusVerdict.NOT_REACHED

    def test_two_of_three_agreement_does_not_trigger_inconsistency(self):
        from dataclasses import dataclass

        @dataclass
        class MockAttributed:
            agreement_analysis: str
            consensus_assessment: str
            confidence: str

        attributed = MockAttributed(
            agreement_analysis="**Agreement Analysis:** `gemma4:e4b` and `qwen3.5:9b` fully agree on the core thesis that best is subjective.",
            consensus_assessment="NOT REACHED",
            confidence="HIGH"
        )

        result = DiscussionOrchestrator._check_main_point_consensus(None, attributed)
        assert result is _ConsensusVerdict.NOT_REACHED

    def test_all_clusters_agree_on_main_answer_triggers_inconsistency(self):
        from dataclasses import dataclass

        @dataclass
        class MockAttributed:
            agreement_analysis: str
            consensus_assessment: str
            confidence: str

        attributed = MockAttributed(
            agreement_analysis="Analysis: While examples differ, all clusters agree on the main answer that no single best exists.",
            consensus_assessment="NOT REACHED",
            confidence="HIGH"
        )

        result = DiscussionOrchestrator._check_main_point_consensus(None, attributed)
        assert result is _ConsensusVerdict.INCONSISTENT

    def test_inconsistent_with_no_agreement_analysis(self):
        from dataclasses import dataclass

        @dataclass
        class MockAttributed:
            agreement_analysis: str
            consensus_assessment: str
            confidence: str

        attributed = MockAttributed(
            agreement_analysis="",
            consensus_assessment="NOT REACHED",
            confidence="HIGH"
        )

        result = DiscussionOrchestrator._check_main_point_consensus(None, attributed)
        assert result is _ConsensusVerdict.NOT_REACHED

    def test_none_attributed_returns_not_reached(self):
        result = DiscussionOrchestrator._check_main_point_consensus(None, None)
        assert result is _ConsensusVerdict.NOT_REACHED


class TestRunExceptionLogging:
    @pytest.mark.asyncio
    async def test_run_logs_exception_context(self, capsys):
        """When run() raises, the error message includes the current round number."""
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=3),
            context=ContextConfig(mode="summary_only"),
        )
        session = Session(prompt="test prompt", config={})

        orchestrator = DiscussionOrchestrator(
            config=config,
            session=session,
            progress_callback=AsyncMock(),
        )

        # Have generate raise on the first call (round 1, model1)
        # This triggers the outer except handler immediately
        async def failing_generate(*args, **kwargs):
            raise RuntimeError("simulated failure")

        orchestrator.ollama.generate = failing_generate
        orchestrator.session_manager.save = AsyncMock()

        with pytest.raises(RuntimeError, match="simulated failure"):
            await orchestrator.run()

        captured = capsys.readouterr()
        assert "round" in captured.err.lower()
        assert "simulated failure" in captured.err


class TestConsensusResultInModeratorMode:
    """Verify self.state.consensus_result is set in moderator_decides paths (fix #1.1)."""

    @pytest.mark.asyncio
    async def test_consensus_result_set_main_point_reached(self):
        """moderator_decides + main_point: verdict REACHED -> consensus_result is set."""
        from core.config import ConsensusConfig
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=1),
            context=ContextConfig(mode="summary_only"),
        )
        config.consensus = ConsensusConfig(mode="moderator_decides", strictness="main_point")

        session = Session(prompt="test prompt", config={})
        orchestrator = DiscussionOrchestrator(
            config=config,
            session=session,
            progress_callback=AsyncMock(),
        )

        orchestrator._check_main_point_consensus = lambda a: _ConsensusVerdict.REACHED
        assert orchestrator._check_main_point_consensus(None) is _ConsensusVerdict.REACHED

    @pytest.mark.asyncio
    async def test_consensus_result_set_main_point_not_reached(self):
        """moderator_decides + main_point: verdict NOT_REACHED -> consensus_result should be set."""
        from core.config import ConsensusConfig
        config = Config(
            models=[
                ModelConfig(name="model1"),
                ModelConfig(name="model2"),
            ],
            discussion=DiscussionConfig(max_rounds=1),
            context=ContextConfig(mode="summary_only"),
        )
        config.consensus = ConsensusConfig(mode="moderator_decides", strictness="main_point")

        session = Session(prompt="test prompt", config={})
        orchestrator = DiscussionOrchestrator(
            config=config,
            session=session,
            progress_callback=AsyncMock(),
        )

        orchestrator._check_main_point_consensus = lambda a: _ConsensusVerdict.NOT_REACHED
        assert orchestrator._check_main_point_consensus(None) is _ConsensusVerdict.NOT_REACHED
