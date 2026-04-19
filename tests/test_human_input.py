import pytest
import threading
import queue
import time
from unittest.mock import patch, MagicMock
from core.input_reader import AsyncInputReader, InputBuffer, get_input_buffer, get_async_input_reader


class TestAsyncInputReader:
    def test_init(self):
        reader = AsyncInputReader(prompt_text="Enter: ")
        assert reader.prompt_text == "Enter: "
        assert reader._started is False

    def test_start(self):
        reader = AsyncInputReader()
        reader.start()
        assert reader._started is True

    def test_start_already_started(self):
        reader = AsyncInputReader()
        reader.start()
        reader.start()
        assert reader._thread is not None

    def test_get_input_with_timeout(self):
        reader = AsyncInputReader()

        with patch('builtins.input', return_value="test input"):
            reader.start()
            time.sleep(0.1)
            result = reader.get_input(timeout=1.0)

        assert result == "test input"

    def test_get_input_empty(self):
        reader = AsyncInputReader()

        with patch('builtins.input', return_value=""):
            reader.start()
            time.sleep(0.1)
            result = reader.get_input(timeout=1.0)

        assert result == ""

    def test_get_input_none_on_timeout(self):
        reader = AsyncInputReader()

        result = reader.get_input(timeout=0.1)
        assert result is None

    def test_has_input(self):
        reader = AsyncInputReader()

        with patch('builtins.input', return_value="test"):
            reader.start()
            time.sleep(0.1)

        assert reader.has_input() is True

    def test_has_input_empty_queue(self):
        reader = AsyncInputReader()
        assert reader.has_input() is False

    def test_stop(self):
        reader = AsyncInputReader()
        reader.start()
        reader.stop()
        assert reader._started is False

    def test_stop_not_started(self):
        reader = AsyncInputReader()
        reader.stop()
        assert reader._started is False


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

    def test_get_async_input_reader_singleton(self):
        reader1 = get_async_input_reader()
        reader2 = get_async_input_reader()

        assert reader1 is reader2


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


class TestAsyncInputEdgeCases:
    def test_input_with_special_characters(self):
        reader = AsyncInputReader()

        with patch('builtins.input', return_value="Test! @#$%^&*()"):
            reader.start()
            time.sleep(0.1)
            result = reader.get_input(timeout=1.0)

        assert result == "Test! @#$%^&*()"

    def test_input_multiline(self):
        reader = AsyncInputReader()

        with patch('builtins.input', return_value="Line 1\nLine 2\nLine 3"):
            reader.start()
            time.sleep(0.1)
            result = reader.get_input(timeout=1.0)

        assert "Line 1" in result

    def test_input_unicode(self):
        reader = AsyncInputReader()

        with patch('builtins.input', return_value="Unicode: \u00e9\u00e8\u00ea"):
            reader.start()
            time.sleep(0.1)
            result = reader.get_input(timeout=1.0)

        assert "\u00e9" in result

    def test_input_very_long(self):
        long_text = "x" * 10000
        reader = AsyncInputReader()

        with patch('builtins.input', return_value=long_text):
            reader.start()
            time.sleep(0.1)
            result = reader.get_input(timeout=1.0)

        assert len(result) == 10000

    def test_multiple_readers_independent(self):
        reader1 = AsyncInputReader(prompt_text="First: ")
        reader2 = AsyncInputReader(prompt_text="Second: ")

        assert reader1.prompt_text != reader2.prompt_text
        assert reader1 is not reader2