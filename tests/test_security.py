"""Tests for path traversal vulnerability fixes (Issue #31)."""

import asyncio
from pathlib import Path

import pytest

from storage import (
    sanitize_session_id,
    sanitize_export_path,
    validate_path_within,
    Exporter,
)
from storage.session import Session, SessionManager


# =============================================================================
# sanitize_session_id
# =============================================================================


class TestSanitizeSessionId:
    """Unit tests for path-traversal hardening of session identifiers."""

    def test_normal_uuid_returns_as_is(self):
        assert sanitize_session_id(
            "550e8400-e29b-41d4-a716-446655440000"
        ) == "550e8400-e29b-41d4-a716-446655440000"

    def test_alphanumeric_id_returns_as_is(self):
        assert sanitize_session_id("abc123") == "abc123"

    def test_id_with_hyphens_returns_as_is(self):
        assert sanitize_session_id("test-session-123") == "test-session-123"

    def test_id_with_underscores_returns_as_is(self):
        assert sanitize_session_id("sess_id_456") == "sess_id_456"

    def test_ollama_style_name_still_allowed(self):
        assert sanitize_session_id("qwen3:8b") == "qwen3:8b"

    def test_simple_dot_in_id_allowed(self):
        assert sanitize_session_id("my.session") == "my.session"

    def test_traversal_parent_ref_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("../../etc/passwd")

    def test_traversal_double_dot_anywhere_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("safe-dir/../../etc/passwd")

    def test_traversal_start_dotdot_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("./../etc/passwd")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("      ")

    def test_none_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id(None)

    def test_special_chars_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("id<evil>")

    def test_semicolon_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("id;rm -rf /")

    def test_backslash_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("id\\etc\\passwd")

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("id\x00evil")

    def test_url_encoded_traversal_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("..%2F..%2Fetc%2Fpasswd")

    def test_percent_encoded_slash_rejected(self):
        with pytest.raises(ValueError):
            sanitize_session_id("id%2F..%2F..%2Fetc%2Fpasswd")


# =============================================================================
# sanitize_export_path
# =============================================================================


class TestSanitizeExportPath:
    """Unit tests for hardening of export output paths."""

    @pytest.fixture
    def sessions_dir(self, tmp_path) -> Path:
        """Create a temporary sessions_dir for each test."""
        d = tmp_path / "sessions"
        d.mkdir()
        return d

    def test_normal_path_within_allowed(self, sessions_dir):
        p = sanitize_export_path(
            sessions_dir / "discussion_123.json", sessions_dir
        )
        assert p == sessions_dir / "discussion_123.json"

    def test_path_traversal_basename_stripped(self, sessions_dir):
        """Path traversal is neutralized by stripping to basename alone."""
        p = Path("/tmp/../../etc/passwd")
        result = sanitize_export_path(p, sessions_dir)
        assert result == sessions_dir / "passwd"

    def test_deep_traversal_basename_stripped(self, sessions_dir):
        p = Path("/../../../../../../etc/passwd")
        result = sanitize_export_path(p, sessions_dir)
        assert result == sessions_dir / "passwd"

    def test_subdir_traversal_basename_stripped(self, sessions_dir):
        p = Path("./sessions/../../etc/passwd")
        result = sanitize_export_path(p, sessions_dir)
        assert result == sessions_dir / "passwd"

    def test_special_chars_banned(self, sessions_dir):
        p = Path("discussion<>evil.json")
        with pytest.raises(ValueError):
            sanitize_export_path(p, sessions_dir)

    def test_sane_export_still_works(self, sessions_dir):
        p = sanitize_export_path(
            sessions_dir / "discussion_abc123.json", sessions_dir
        )
        assert p.name == "discussion_abc123.json"
        assert p.resolve().parent == sessions_dir.resolve()


# =============================================================================
# validate_path_within
# =============================================================================


class TestValidatePathWithin:
    """Unit tests for the cross-directory guard."""

    def test_child_within_parent_passes(self, tmp_path):
        child = tmp_path / "subdir" / "file.txt"
        child.parent.mkdir()
        child.touch()
        result = validate_path_within(tmp_path, child.resolve())
        assert result == child.resolve()

    def test_child_outside_parent_raises(self, tmp_path):
        """Test that a path resolving outside the parent raises."""
        outside = Path("/tmp/outside.txt")
        with pytest.raises(ValueError):
            validate_path_within(tmp_path, outside.resolve())

    def test_symlink_to_outside_banned(self, tmp_path):
        """Symlinks pointing outside the session dir should be blocked."""
        symlink = tmp_path / "link"
        real_file = Path("/tmp/symlink_test_target.txt")
        real_file.touch()
        symlink.symlink_to(real_file.resolve())
        with pytest.raises(ValueError):
            validate_path_within(tmp_path, symlink.resolve())


# =============================================================================
# SessionManager integration tests
# =============================================================================


class TestSessionManagerPathTraversal:
    """Integration tests for SessionManager with malicious session IDs."""

    @pytest.fixture
    def manager(self, tmp_path) -> SessionManager:
        return SessionManager(str(tmp_path / "sessions"))

    def test_load_safe_session_id_works(self, manager):
        session = Session(
            prompt="test",
            config={},
            session_id="safe-id-123",
        )
        asyncio.run(manager.save(session))
        loaded = asyncio.run(manager.load("safe-id-123"))
        assert loaded is not None
        assert loaded.id == "safe-id-123"

    def test_load_session_id_with_traversal_raises(self, manager):
        with pytest.raises(ValueError):
            asyncio.run(manager.load("../../etc/passwd"))

    def test_load_session_id_with_encoded_traversal_raises(self, manager):
        with pytest.raises(ValueError):
            asyncio.run(manager.load("./../etc/passwd"))

    def test_delete_safe_session_works(self, manager):
        session = Session(
            prompt="test",
            config={},
            session_id="del-me-456",
        )
        asyncio.run(manager.save(session))
        success = asyncio.run(manager.delete("del-me-456"))
        assert success is True
        assert asyncio.run(manager.load("del-me-456")) is None

    def test_delete_session_id_with_traversal_raises(self, manager):
        with pytest.raises(ValueError):
            asyncio.run(manager.delete("../../etc/passwd"))


# =============================================================================
# Exporter integration tests
# =============================================================================


class TestExporterPathTraversal:
    """Integration tests for Exporter with malicious paths."""

    @pytest.fixture
    def session(self, tmp_path) -> Session:
        s = Session(
            prompt="Test Q?", config={}, session_id="export-test-1"
        )
        s.add_response("m1", "Answer A", 1, 0)
        s.add_response("m2", "Answer B", 1, 1)
        s.add_summary(1, "Summary")
        s.mark_completed(1)
        return s

    @pytest.fixture
    def export_sessions_dir(self, tmp_path) -> Path:
        d = tmp_path / "sessions"
        d.mkdir()
        return d

    def test_export_md_normal_path_works(self, session, export_sessions_dir):
        p = sanitize_export_path(
            Path("output.md"), export_sessions_dir
        )
        result = asyncio.run(Exporter.export_markdown(session, str(p)))
        assert result.exists()
        assert result.name == "output.md"

    def test_export_json_normal_path_works(
        self, session, export_sessions_dir
    ):
        p = sanitize_export_path(
            Path("output.json"), export_sessions_dir
        )
        result = asyncio.run(Exporter.export_json(session, str(p)))
        assert result.exists()
        assert result.name == "output.json"

    def test_export_md_traversal_uses_safe_basename(
        self, session, export_sessions_dir
    ):
        """Path traversal is neutralized - basename 'evil.md' is safe."""
        output = Path("/tmp/../../etc/evil.md")
        p = sanitize_export_path(output, export_sessions_dir)
        result = asyncio.run(Exporter.export_markdown(session, str(p)))
        assert result.exists()
        assert result.name == "evil.md"

    def test_export_json_traversal_uses_safe_basename(
        self, session, export_sessions_dir
    ):
        output = Path("/tmp/../../etc/evil.json")
        p = sanitize_export_path(output, export_sessions_dir)
        result = asyncio.run(Exporter.export_json(session, str(p)))
        assert result.exists()
        assert result.name == "evil.json"
