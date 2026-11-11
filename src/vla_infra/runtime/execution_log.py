"""Non-blocking JSONL execution audit log (M3).

The executor thread appends immutable records to a bounded queue; a dedicated
writer thread serialises them to disk.  A slow/full disk can drop audit records
(the drop count is reported) but must never delay the control loop.  This is
intentionally separate from the episode recorder: M3 command audit must not be
coupled to the 100 Hz execution path.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Mapping, Optional


class ExecutionLog:
    def __init__(self, path: str | Path, *, queue_size: int = 4096) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: Queue = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = self.path.open("w", encoding="utf-8")
        self._dropped = 0
        self._written = 0
        self._lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("execution log already started")
        self._thread = threading.Thread(target=self._run, name="m3-execution-log", daemon=True)
        self._thread.start()

    def append(self, record: Mapping[str, Any]) -> bool:
        if self._closed:
            with self._lock:
                self._dropped += 1
            return False
        try:
            self._queue.put_nowait(dict(record))
            return True
        except Full:
            with self._lock:
                self._dropped += 1
            return False

    def _run(self) -> None:
        try:
            while True:
                try:
                    record = self._queue.get(timeout=0.05)
                except Empty:
                    if self._stop.is_set():
                        return
                    continue
                self._handle.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
                with self._lock:
                    self._written += 1
        finally:
            self._handle.flush()
            try:
                os.fsync(self._handle.fileno())
            except (OSError, ValueError):
                pass
            self._handle.close()

    def close(self, timeout_s: float = 5.0) -> bool:
        """Drain and close; return False if the writer thread is still alive."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                return False
        else:
            self._handle.flush()
            self._handle.close()
            self._thread = None
        self._closed = True
        return True

    @property
    def closed(self) -> bool:
        return self._closed

    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def written(self) -> int:
        with self._lock:
            return self._written

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped


def _default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


__all__ = ["ExecutionLog"]
