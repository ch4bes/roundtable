import threading
import queue
from typing import Optional


class AsyncInputReader:
    def __init__(self, prompt_text: str = "Your response: "):
        self.prompt_text = prompt_text
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._read_input, daemon=True)
        self._thread.start()

    def _read_input(self):
        try:
            user_input = input(self.prompt_text)
            self._input_queue.put(user_input)
        except EOFError:
            self._input_queue.put("")
        except Exception:
            self._input_queue.put("")

    def get_input(self, timeout: float = None) -> Optional[str]:
        try:
            return self._input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_input(self) -> bool:
        return not self._input_queue.empty()

    def stop(self):
        self._started = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


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
        self._buffer = ""
        self._ready.clear()


async_input_reader: Optional[AsyncInputReader] = None
input_buffer: Optional[InputBuffer] = None


def get_input_buffer() -> InputBuffer:
    global input_buffer
    if input_buffer is None:
        input_buffer = InputBuffer()
    return input_buffer


def get_async_input_reader() -> AsyncInputReader:
    global async_input_reader
    if async_input_reader is None:
        async_input_reader = AsyncInputReader()
    return async_input_reader
