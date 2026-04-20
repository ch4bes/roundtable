import pytest
import threading
import time
from unittest.mock import patch, MagicMock
from core.input_reader import InputBuffer, get_input_buffer


class TestInputBuffer:
    def test_init(self):
        buffer = InputBuffer()
        assert buffer._buffer == ""
        assert buffer.is_ready() is False

    def test_put(self):
        buffer = InputBuffer()
        buffer.put("test text")

        assert buffer._buffer == "test text"
        assert buffer.is_ready() is True

    def test_get_without_wait(self):
        buffer = InputBuffer()
        buffer.put("test")
        result = buffer.get(wait=False)

        assert result == "test"

    def test_get_without_wait_not_ready(self):
        buffer = InputBuffer()
        result = buffer.get(wait=False)

        assert result is None

    def test_get_with_wait_ready(self):
        buffer = InputBuffer()
        buffer.put("test")
        result = buffer.get(wait=True, timeout=1.0)

        assert result == "test"

    def test_get_with_wait_timeout(self):
        buffer = InputBuffer()
        result = buffer.get(wait=True, timeout=0.1)

        assert result is None

    def test_clear(self):
        buffer = InputBuffer()
        buffer.put("test")
        buffer.clear()

        assert buffer._buffer == ""
        assert buffer.is_ready() is False


class TestGlobalInstances:
    def test_get_input_buffer_singleton(self):
        buffer1 = get_input_buffer()
        buffer2 = get_input_buffer()

        assert buffer1 is buffer2


class TestInputBufferThread:
    def test_concurrent_put_get(self):
        buffer = InputBuffer()
        results = []

        def putter():
            time.sleep(0.05)
            buffer.put("concurrent data")

        thread = threading.Thread(target=putter)
        thread.start()

        result = buffer.get(wait=True, timeout=1.0)
        results.append(result)

        thread.join()

        assert "concurrent data" in results