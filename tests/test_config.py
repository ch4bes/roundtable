import pytest
import json
from pathlib import Path
from core.config import Config, ModelConfig


def test_default_config():
    config = Config()

    assert len(config.models) >= 2
    assert config.ollama.base_url == "http://localhost:11434"
    assert config.discussion.max_rounds == 10
    assert config.discussion.consensus_threshold == 0.75
    assert config.discussion.consensus_method == "clustering"


def test_config_from_dict():
    config_data = {
        "ollama": {
            "base_url": "http://localhost:11435",
            "timeout": 60,
        },
        "models": [
            {"name": "llama3.2", "temperature": 0.8},
            {"name": "mistral", "temperature": 0.7},
        ],
        "discussion": {
            "max_rounds": 5,
            "consensus_threshold": 0.9,
        },
    }

    config = Config(**config_data)

    assert config.ollama.base_url == "http://localhost:11435"
    assert config.ollama.timeout == 60
    assert config.models[0].name == "llama3.2"
    assert config.models[0].temperature == 0.8
    assert config.discussion.max_rounds == 5
    assert config.discussion.consensus_threshold == 0.9


def test_config_validation_insufficient_models():
    with pytest.raises(ValueError) as exc_info:
        Config(models=[ModelConfig(name="llama3.2")])

    assert "At least 2 models" in str(exc_info.value)


def test_config_temperature_bounds():
    with pytest.raises(ValueError):
        Config(
            models=[
                ModelConfig(name="llama3.2", temperature=-0.1),
                ModelConfig(name="mistral"),
            ]
        )

    with pytest.raises(ValueError):
        Config(
            models=[
                ModelConfig(name="llama3.2", temperature=2.5),
                ModelConfig(name="mistral"),
            ]
        )


def test_config_save_load(tmp_path):
    config_path = tmp_path / "test_config.json"

    original_config = Config(
        ollama={"base_url": "http://test:11434"},
        models=[
            ModelConfig(name="model1"),
            ModelConfig(name="model2"),
        ],
        discussion={"max_rounds": 3},
    )

    original_config.save(config_path)

    loaded_config = Config.load(config_path)

    assert loaded_config.ollama.base_url == "http://test:11434"
    assert len(loaded_config.models) == 2
    assert loaded_config.discussion.max_rounds == 3


def test_context_mode_validation():
    valid_modes = ["full", "summary_only", "summary_plus_last_n"]

    for mode in valid_modes:
        config = Config(context={"mode": mode})
        assert config.context.mode == mode

    with pytest.raises(ValueError):
        Config(context={"mode": "invalid_mode"})


def test_export_format_validation():
    config = Config(storage={"export_format": "md"})
    assert config.storage.export_format == "md"

    config = Config(storage={"export_format": "json"})
    assert config.storage.export_format == "json"

    with pytest.raises(ValueError):
        Config(storage={"export_format": "xml"})
