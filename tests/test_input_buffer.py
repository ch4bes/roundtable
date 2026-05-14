"""Tests for InputBuffer race condition fix - buffer is cleared after get()."""

import threading
from core.input_reader import InputBuffer


class TestInputBufferRaceCondition:
    """Verify that get() clears the buffer so subsequent calls return None."""

    def setup_method(self):
        self.buffer = InputBuffer()

    def test_get_clears_buffer_on_wait_true(self):
        """First get() returns data, second get() returns None."""
        self.buffer.put("hello")
        assert self.buffer.get(wait=True, timeout=1) == "hello"
        assert self.buffer.get(wait=True, timeout=0.1) is None

    def test_get_clears_buffer_on_wait_false(self):
        """First get() returns data, second non-blocking get() returns None."""
        self.buffer.put("world")
        assert self.buffer.get(wait=False) == "world"
        assert self.buffer.get(wait=False) is None

    def test_put_put_get_returns_only_latest(self):
        """Rapid puts followed by a single get returns only the last value."""
        self.buffer.put("first")
        import time
        time.sleep(0.01)  # Let first put settle
        self.buffer.put("second")
        assert self.buffer.get(wait=True, timeout=1) == "second"
        assert self.buffer.get(wait=True, timeout=0.1) is None

    def test_concurrent_puts_last_writer_wins(self):
        """Concurrent puts - last writer wins since buffer is single-value."""
        put_results = []

        def producer(text):
            import time
            time.sleep(0.01 * hash(text) % 10)
            self.buffer.put(text)
            put_results.append(text)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=producer, args=(f"msg{i}",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All 5 puts should have happened
        assert len(put_results) == 5

        # The buffer holds the last value written by any thread
        final_val = self.buffer.get(wait=True, timeout=1)
        assert final_val is not None
        assert final_val in put_results

    def test_clear_preserves_functionality(self):
        """After clear(), buffer is empty and subsequent puts work normally."""
        self.buffer.put("data")
        self.buffer.clear()
        assert self.buffer.get(wait=True, timeout=0.1) is None

        self.buffer.put("fresh")
        assert self.buffer.get(wait=True, timeout=1) == "fresh"
        assert self.buffer.get(wait=True, timeout=0.1) is None

    def test_is_ready_returns_false_after_get(self):
        """After consuming data with get(), is_ready() returns False."""
        self.buffer.put("test")
        assert self.buffer.is_ready() is True

        self.buffer.get(wait=True, timeout=1)
        assert self.buffer.is_ready() is False

    def test_is_ready_returns_true_after_put(self):
        """After putting data, is_ready() returns True."""
        assert self.buffer.is_ready() is False
        self.buffer.put("data")
        assert self.buffer.is_ready() is True


class TestInputBufferEdgeCases:
    """Edge case tests for InputBuffer."""

    def setup_method(self):
        self.buffer = InputBuffer()

    def test_get_empty_nonblocking_returns_none(self):
        assert self.buffer.get(wait=False) is None

    def test_get_empty_blocking_timeout_returns_none(self):
        assert self.buffer.get(wait=True, timeout=0.1) is None

    def test_put_empty_string_clears_buffer(self):
        self.buffer.put("data")
        self.buffer.put("")
        assert self.buffer.get(wait=True, timeout=1) == ""

    def test_get_after_clear_returns_none(self):
        self.buffer.put("data")
        self.buffer.clear()
        assert self.buffer.get(wait=True, timeout=1) is None
