"""
Thread-safe stdout capture for background experiment execution.
Intercepts print() calls from experiment functions and stores them
in a shared buffer for UI polling.
"""

import io
import sys
import threading
from contextlib import contextmanager


class ThreadSafeLogBuffer:
    """
    Thread-safe buffer accumulating log lines from a background thread.
    Uses an internal lock to allow concurrent read from the UI thread.
    """

    def __init__(self, max_lines: int = 2000):
        self._lock = threading.Lock()
        self._lines: list = []
        self._max_lines = max_lines

    def write(self, message: str):
        """File-like write interface (called by stdout redirection)."""
        if not message or message.isspace():
            return
        with self._lock:
            for line in str(message).splitlines():
                stripped = line.rstrip()
                if stripped:
                    self._lines.append(stripped)
                    if len(self._lines) > self._max_lines:
                        self._lines = self._lines[-self._max_lines:]

    def flush(self):
        """File-like flush interface (no-op)."""
        pass

    def get_all(self) -> list:
        """Returns a snapshot copy of all accumulated lines."""
        with self._lock:
            return list(self._lines)

    def get_tail(self, n: int = 50) -> list:
        """Returns the last n lines."""
        with self._lock:
            return list(self._lines[-n:])

    def clear(self):
        """Empties the buffer."""
        with self._lock:
            self._lines.clear()

    def line_count(self) -> int:
        with self._lock:
            return len(self._lines)


class TeeStream:
    """
    File-like object that duplicates writes to two targets.
    Used to preserve real stdout while also capturing to buffer.
    """

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, message):
        try:
            self.primary.write(message)
        except Exception:
            pass
        try:
            self.secondary.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.primary.flush()
        except Exception:
            pass
        try:
            self.secondary.flush()
        except Exception:
            pass


@contextmanager
def capture_stdout(buffer: ThreadSafeLogBuffer, also_print: bool = False):
    """
    Context manager redirecting stdout to a ThreadSafeLogBuffer.

    Args:
        buffer: Target buffer to capture into.
        also_print: If True, keeps writing to the real stdout in parallel.
    """
    original_stdout = sys.stdout
    try:
        if also_print:
            sys.stdout = TeeStream(original_stdout, buffer)
        else:
            sys.stdout = buffer
        yield buffer
    finally:
        sys.stdout = original_stdout