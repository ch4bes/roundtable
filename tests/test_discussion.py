import pytest
from core.config import Config, ModelConfig, DiscussionConfig, ContextConfig
from core.discussion import DiscussionOrchestrator, DiscussionState
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
    def test_main_point_consensus_with_full_agreement(self):
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
        assert result is True

    def test_main_point_consensus_with_overwhelming_consensus(self):
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
        assert result is True

    def test_main_point_consensus_falls_back_when_no_analysis(self):
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
        assert result is True

    def test_main_point_consensus_with_disagreement(self):
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
        assert result is False