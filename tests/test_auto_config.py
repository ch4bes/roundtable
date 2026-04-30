"""Tests for scripts/auto_config.py — Phase 1 item 1.8 (bare except fixes)."""

import subprocess
from unittest.mock import patch

import pytest

from scripts.auto_config import (
    get_ollama_models,
    parse_size,
    select_diverse_models,
    has_embedding_model,
)


# ── parse_size ───────────────────────────────────────────────

class TestParseSize:

    def test_parse_size_valid_gb(self):
        assert parse_size("18.3 GB") == pytest.approx(18.3)
        assert parse_size("1 GB") == pytest.approx(1.0)

    def test_parse_size_valid_mb(self):
        assert parse_size("500 MB") == pytest.approx(500 / 1024)
        assert parse_size("1024 MB") == pytest.approx(1.0)

    def test_parse_size_dash(self):
        assert parse_size("-") == 0

    def test_parse_size_plain_number(self):
        assert parse_size("42") == pytest.approx(42.0)

    def test_parse_size_invalid(self):
        assert parse_size("not a number") == 0
        assert parse_size("") == 0
        assert parse_size("xyz GB") == 0

    def test_parse_size_does_not_catch_keyboard_interrupt(self):
        """Verify the except clause only catches ValueError/TypeError, not KeyboardInterrupt."""
        with patch("scripts.auto_config.float", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                parse_size("1 GB")

    def test_parse_size_does_not_catch_system_exit(self):
        """Verify SystemExit is not silently swallowed."""
        with patch("scripts.auto_config.float", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                parse_size("1 GB")


# ── get_ollama_models ──────────────────────────────────────────

class TestGetOllamaModels:

    def test_get_ollama_models_parses_output(self):
        fake_stdout = (
            "NAME                    ID              SIZE      MODIFIED\n"
            "gemma4:e4b              abc123          18.3 GB   2 days ago\n"
            "qwen3.5:9b              def456          5.8 GB    1 week ago\n"
        )
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(
                args=["ollama", "list"], stdout=fake_stdout, returncode=0
        )):
            models = get_ollama_models()
            assert len(models) == 2
            assert models[0]["name"] == "gemma4:e4b"
            assert models[0]["size_gb"] == pytest.approx(18.3)
            assert models[1]["name"] == "qwen3.5:9b"
            assert models[1]["size_gb"] == pytest.approx(5.8)

    def test_get_ollama_models_empty_output(self):
        fake_stdout = "NAME                    ID              SIZE      MODIFIED\n"
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(
                args=["ollama", "list"], stdout=fake_stdout, returncode=0
        )):
            models = get_ollama_models()
            assert models == []

    def test_get_ollama_models_subprocess_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("ollama not found")):
            models = get_ollama_models()
            assert models == []

    def test_get_ollama_models_partial_fields(self):
        """Lines with fewer than 3 parts are skipped (bounds check)."""
        fake_stdout = (
            "NAME                    ID              SIZE      MODIFIED\n"
            "gemma4:e4b              abc123          18.3 GB   2 days ago\n"
            "short line\n"
        )
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(
                args=["ollama", "list"], stdout=fake_stdout, returncode=0
        )):
            models = get_ollama_models()
            assert len(models) == 1

    def test_get_ollama_models_mishandled_modified_field(self):
        """Handles 'fewer seconds ago' or similar non-standard 'MODIFIED' fields gracefully."""
        fake_stdout = (
            "NAME                    ID              SIZE      MODIFIED\n"
            "gemma4:e4b              abc123          18.3 GB   2 minutes ago\n"
        )
        with patch("subprocess.run", return_value=subprocess.CompletedProcess(
                args=["ollama", "list"], stdout=fake_stdout, returncode=0
        )):
            models = get_ollama_models()
            assert len(models) == 1


# ── select_diverse_models ──────────────────────────────────

class TestSelectDiverseModels:

    def test_returns_all_when_fewer_than_count(self):
        models = [{"name": "m1", "size_gb": 1.0}, {"name": "m2", "size_gb": 2.0}]
        result = select_diverse_models(models, count=5)
        assert len(result) == 2

    def test_selects_diverse_range(self):
        models = [{"name": f"m{i}", "size_gb": float(i)} for i in range(1, 11)]
        result = select_diverse_models(models, count=3)
        assert len(result) == 3  # Should pick models spread across the list

    def test_does_not_select_duplicates(self):
        models = [{"name": f"m{i}", "size_gb": float(i)} for i in range(1, 11)]
        result = select_diverse_models(models, count=5)
        names = [m["name"] for m in result]
        assert len(names) == len(set(names))


# ── has_embedding_model ─────────────────────────────────────

class TestHasEmbeddingModel:

    def test_finds_embedding_model(self):
        models = [{"name": "nomic-embed-text"}, {"name": "gemma4:e4b"}]
        assert has_embedding_model(models) == "nomic-embed-text"

    def test_no_embedding_model(self):
        models = [{"name": "gemma4:e4b"}, {"name": "qwen3.5:9b"}]
        assert has_embedding_model(models) is None

    def test_case_insensitive(self):
        models = [{"name": "QWEN3-EMBEDDING:8b"}]
        assert has_embedding_model(models) == "QWEN3-EMBEDDING:8b"
