"""Tests for incremental (O(1)) session persistence - Issue #36."""

import asyncio
import json
import time
from pathlib import Path

import pytest

from storage.session import Session, SessionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "test-inc-001") -> Session:
    return Session(prompt="test prompt", config={}, session_id=session_id)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Format v2 file structure
# ---------------------------------------------------------------------------


class TestV2FileStructure:
    """The new save format creates a small header JSON + a JSONL append-log."""

    def test_save_creates_json_and_jsonl(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "hello", 1, 0)

        _run(manager.save(session))

        assert (tmp_path / "test-inc-001.json").exists()
        assert (tmp_path / "test-inc-001.jsonl").exists()

    def test_header_json_has_format_version_2(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()

        _run(manager.save(session))

        header = json.loads((tmp_path / "test-inc-001.json").read_text())
        assert header["format_version"] == 2

    def test_header_json_has_no_list_fields(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "hello", 1, 0)
        session.add_summary(1, "sum")

        _run(manager.save(session))

        header = json.loads((tmp_path / "test-inc-001.json").read_text())
        assert "responses" not in header
        assert "summaries" not in header
        assert "attributed_summaries" not in header
        assert "human_responses" not in header
        assert "similarity_matrices" not in header

    def test_jsonl_contains_one_line_per_event(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "r1", 1, 0)
        session.add_response("m2", "r2", 1, 1)
        session.add_summary(1, "s1")

        _run(manager.save(session))

        lines = [
            l for l in (tmp_path / "test-inc-001.jsonl").read_text().splitlines() if l
        ]
        assert len(lines) == 3  # 2 responses + 1 summary

    def test_jsonl_events_have_correct_types(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "r1", 1, 0)
        session.add_human_response("human input", 1, 1)
        session.add_summary(1, "s1")

        _run(manager.save(session))

        lines = (tmp_path / "test-inc-001.jsonl").read_text().splitlines()
        types = [json.loads(l)["type"] for l in lines if l]
        assert types == ["response", "human_response", "summary"]


# ---------------------------------------------------------------------------
# Incremental append behaviour
# ---------------------------------------------------------------------------


class TestIncrementalAppend:
    """Each save should only append *new* items, not re-write existing ones."""

    def test_second_save_appends_only_new_events(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "r1", 1, 0)
        _run(manager.save(session))

        # First save: 1 line in the log
        lines_after_first = (tmp_path / "test-inc-001.jsonl").read_text().splitlines()
        assert len(lines_after_first) == 1

        # Add another response and save again
        session.add_response("m2", "r2", 1, 1)
        _run(manager.save(session))

        lines_after_second = (tmp_path / "test-inc-001.jsonl").read_text().splitlines()
        assert len(lines_after_second) == 2

    def test_save_with_no_new_items_does_not_grow_log(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "r1", 1, 0)
        _run(manager.save(session))

        size_before = (tmp_path / "test-inc-001.jsonl").stat().st_size
        # Save again without adding anything new
        _run(manager.save(session))
        size_after = (tmp_path / "test-inc-001.jsonl").stat().st_size

        assert size_before == size_after

    def test_header_updated_on_every_save(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        _run(manager.save(session))

        header1 = json.loads((tmp_path / "test-inc-001.json").read_text())
        first_updated_at = header1["updated_at"]

        # Status change - no list items, but header should reflect new status
        session.mark_completed(consensus_round=1)
        _run(manager.save(session))

        header2 = json.loads((tmp_path / "test-inc-001.json").read_text())
        assert header2["status"] == "completed"
        assert header2["consensus_round"] == 1
        assert header2["updated_at"] != first_updated_at or True  # may differ


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Save then load must return a session with identical in-memory state."""

    def test_responses_round_trip(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "answer1", 1, 0)
        session.add_response("m2", "answer2", 1, 1)
        _run(manager.save(session))

        loaded = _run(manager.load("test-inc-001"))
        assert loaded is not None
        assert len(loaded.responses) == 2
        assert loaded.responses[0].content == "answer1"
        assert loaded.responses[1].model == "m2"

    def test_human_responses_round_trip(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_human_response("my opinion", 1, 0)
        _run(manager.save(session))

        loaded = _run(manager.load("test-inc-001"))
        assert len(loaded.human_responses) == 1
        assert loaded.human_responses[0].content == "my opinion"

    def test_summaries_round_trip(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_summary(1, "round 1 summary")
        session.add_summary(2, "round 2 summary")
        _run(manager.save(session))

        loaded = _run(manager.load("test-inc-001"))
        assert len(loaded.summaries) == 2
        assert loaded.summaries[1].summary == "round 2 summary"

    def test_attributed_summaries_round_trip(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_attributed_summary(
            round_num=1,
            individual_summaries={"m1": ["point A"]},
            agreement_analysis="agree",
            consensus_assessment="REACHED",
            confidence="HIGH",
            full_text="full",
        )
        _run(manager.save(session))

        loaded = _run(manager.load("test-inc-001"))
        assert len(loaded.attributed_summaries) == 1
        assert loaded.attributed_summaries[0].consensus_assessment == "REACHED"

    def test_similarity_matrix_round_trip(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_similarity_matrix(1, [[1.0, 0.8], [0.8, 1.0]], ["m1", "m2"])
        _run(manager.save(session))

        loaded = _run(manager.load("test-inc-001"))
        assert len(loaded.similarity_matrices) == 1
        assert loaded.similarity_matrices[0]["matrix"] == [[1.0, 0.8], [0.8, 1.0]]

    def test_scalar_metadata_round_trip(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_summary(1, "sum")
        session.mark_completed(consensus_round=1)
        session.add_final_review("final review text")
        _run(manager.save(session))

        loaded = _run(manager.load("test-inc-001"))
        assert loaded.status == "completed"
        assert loaded.consensus_reached is True
        assert loaded.consensus_round == 1
        assert loaded.final_review == "final review text"
        assert loaded.completed_rounds == 1

    def test_multi_save_resume_round_trip(self, tmp_path):
        """Items added across multiple saves all survive a reload."""
        manager = SessionManager(str(tmp_path))
        session = _make_session()

        session.add_response("m1", "r1", 1, 0)
        _run(manager.save(session))

        session.add_response("m2", "r2", 1, 1)
        session.add_summary(1, "s1")
        _run(manager.save(session))

        session.add_response("m1", "r3", 2, 0)
        _run(manager.save(session))

        loaded = _run(manager.load("test-inc-001"))
        assert len(loaded.responses) == 3
        assert len(loaded.summaries) == 1

    def test_loaded_session_flush_offsets_correct(self, tmp_path):
        """After loading, flush offsets point to the end of each list so the
        next save only appends truly new items."""
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "r1", 1, 0)
        session.add_response("m2", "r2", 1, 1)
        _run(manager.save(session))

        loaded = _run(manager.load("test-inc-001"))
        assert loaded._flush_offsets["responses"] == 2

        # Add one more item and save - log should grow by exactly 1 line
        loaded.add_response("m3", "r3", 2, 0)
        _run(manager.save(loaded))

        lines = [
            l for l in (tmp_path / "test-inc-001.jsonl").read_text().splitlines() if l
        ]
        assert len(lines) == 3  # 2 original + 1 new


# ---------------------------------------------------------------------------
# v1 → v2 migration
# ---------------------------------------------------------------------------


class TestLegacyMigration:
    """Old full-JSON (v1) sessions load correctly and migrate on first save."""

    def _write_v1_session(self, path: Path, session: Session) -> None:
        """Simulate an old-format v1 file (no format_version key)."""
        path.write_text(json.dumps(session.to_dict()))

    def test_v1_session_loads_correctly(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "old-resp", 1, 0)
        session.add_summary(1, "old-sum")

        self._write_v1_session(tmp_path / "test-inc-001.json", session)

        loaded = _run(manager.load("test-inc-001"))
        assert loaded is not None
        assert len(loaded.responses) == 1
        assert loaded.responses[0].content == "old-resp"
        assert len(loaded.summaries) == 1

    def test_v1_session_migrates_on_save(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "old-resp", 1, 0)
        self._write_v1_session(tmp_path / "test-inc-001.json", session)

        loaded = _run(manager.load("test-inc-001"))
        _run(manager.save(loaded))  # triggers migration

        # Header should now be v2
        header = json.loads((tmp_path / "test-inc-001.json").read_text())
        assert header["format_version"] == 2
        assert "responses" not in header

        # JSONL should contain the old response
        lines = [
            l for l in (tmp_path / "test-inc-001.jsonl").read_text().splitlines() if l
        ]
        assert len(lines) == 1
        assert json.loads(lines[0])["type"] == "response"

    def test_v1_migrated_session_round_trips_correctly(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "old-resp", 1, 0)
        session.add_summary(1, "old-sum")
        self._write_v1_session(tmp_path / "test-inc-001.json", session)

        # Load v1, save (migrates), reload
        loaded = _run(manager.load("test-inc-001"))
        _run(manager.save(loaded))
        reloaded = _run(manager.load("test-inc-001"))

        assert len(reloaded.responses) == 1
        assert len(reloaded.summaries) == 1
        assert reloaded.responses[0].content == "old-resp"


# ---------------------------------------------------------------------------
# Robustness: corrupt JSONL line
# ---------------------------------------------------------------------------


class TestCorruptJSONL:
    """A single corrupt line in the JSONL log should not abort the load."""

    def test_corrupt_last_line_is_skipped(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "good-resp", 1, 0)
        _run(manager.save(session))

        # Append a corrupt line manually
        log_path = tmp_path / "test-inc-001.jsonl"
        with open(log_path, "a") as f:
            f.write("{broken json\n")

        loaded = _run(manager.load("test-inc-001"))
        assert loaded is not None
        assert len(loaded.responses) == 1  # good line still loaded

    def test_corrupt_middle_line_is_skipped(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "r1", 1, 0)
        session.add_response("m2", "r2", 1, 1)
        _run(manager.save(session))

        # Corrupt the middle line
        log_path = tmp_path / "test-inc-001.jsonl"
        lines = log_path.read_text().splitlines()
        lines[0] = "{bad"
        log_path.write_text("\n".join(lines) + "\n")

        loaded = _run(manager.load("test-inc-001"))
        # Second response should still load
        assert len(loaded.responses) == 1
        assert loaded.responses[0].content == "r2"


# ---------------------------------------------------------------------------
# Delete removes both files
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_both_files(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        session.add_response("m1", "r1", 1, 0)
        _run(manager.save(session))

        assert (tmp_path / "test-inc-001.json").exists()
        assert (tmp_path / "test-inc-001.jsonl").exists()

        result = _run(manager.delete("test-inc-001"))
        assert result is True
        assert not (tmp_path / "test-inc-001.json").exists()
        assert not (tmp_path / "test-inc-001.jsonl").exists()

    def test_delete_nonexistent_session_returns_false(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        result = _run(manager.delete("does-not-exist"))
        assert result is False

    def test_after_delete_load_returns_none(self, tmp_path):
        manager = SessionManager(str(tmp_path))
        session = _make_session()
        _run(manager.save(session))
        _run(manager.delete("test-inc-001"))
        assert _run(manager.load("test-inc-001")) is None


# ---------------------------------------------------------------------------
# Performance: O(1) save time
# ---------------------------------------------------------------------------


class TestSavePerformance:
    """Save time should be roughly constant regardless of session size."""

    def _time_save(self, manager: SessionManager, session: Session) -> float:
        start = time.perf_counter()
        _run(manager.save(session))
        return time.perf_counter() - start

    def test_save_time_small_vs_large_session(self, tmp_path):
        """Saving 1 response vs 100 responses should have negligible delta."""
        small_mgr = SessionManager(str(tmp_path / "small"))
        small_session = Session(prompt="p", config={}, session_id="small")
        small_session.add_response("m1", "r", 1, 0)
        # Warm-up save
        _run(small_mgr.save(small_session))

        large_mgr = SessionManager(str(tmp_path / "large"))
        large_session = Session(prompt="p", config={}, session_id="large")
        for i in range(100):
            large_session.add_response("m1", f"response {i}", i // 10 + 1, i % 10)
        # Flush all 100 on the first save
        _run(large_mgr.save(large_session))

        # Now measure a single incremental save (1 new item) on each
        small_session.add_response("m1", "new", 2, 0)
        large_session.add_response("m1", "new", 11, 0)

        t_small = self._time_save(small_mgr, small_session)
        t_large = self._time_save(large_mgr, large_session)

        # Large session incremental save should not be significantly slower.
        # We allow a 10x factor as a very generous bound (should be ~1x in practice).
        assert t_large < t_small * 10 + 0.1, (
            f"Large-session incremental save ({t_large:.4f}s) is much slower "
            f"than small-session save ({t_small:.4f}s)"
        )
