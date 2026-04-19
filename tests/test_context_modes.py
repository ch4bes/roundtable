import pytest
from core.config import Config, ModelConfig, ContextConfig
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