import pytest
from core.config import Config, ModelConfig, DiscussionConfig, ContextConfig
from core.discussion import DiscussionState
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