import threading
import queue
from typing import Optional


class InputBuffer:
    def __init__(self):
        self._buffer: str = ""
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def put(self, text: str):
        with self._lock:
            self._buffer = text
        self._ready.set()

    def get(self, wait: bool = False, timeout: float = None) -> Optional[str]:
        if wait:
            if self._ready.wait(timeout=timeout):
                with self._lock:
                    return self._buffer
            return None
        else:
            if self._ready.is_set():
                with self._lock:
                    return self._buffer
            return None

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def clear(self):
        with self._lock:
            self._buffer = ""
            self._ready.clear()


input_buffer: Optional[InputBuffer] = None


def get_input_buffer() -> InputBuffer:
    global input_buffer
    if input_buffer is None:
        input_buffer = InputBuffer()
    return input_buffer
