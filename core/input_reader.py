import threading


class InputBuffer:
    def __init__(self) -> None:
        self._buffer: str = ""
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def put(self, text: str) -> None:
        with self._lock:
            self._buffer = text
        self._ready.set()

    def get(self, wait: bool = False, timeout: float = None) -> str | None:
        if wait:
            if self._ready.wait(timeout=timeout):
                with self._lock:
                    result = self._buffer
                    self._buffer = ""
                    self._ready.clear()
                return result
            return None
        else:
            if self._ready.is_set():
                with self._lock:
                    result = self._buffer
                    self._buffer = ""
                    self._ready.clear()
                return result
            return None

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def clear(self) -> None:
        with self._lock:
            self._buffer = ""
            self._ready.clear()


input_buffer: InputBuffer | None = None


def get_input_buffer() -> InputBuffer:
    global input_buffer
    if input_buffer is None:
        input_buffer = InputBuffer()
    return input_buffer
