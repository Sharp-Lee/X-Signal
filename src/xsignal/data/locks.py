from __future__ import annotations

from pathlib import Path

from filelock import FileLock


class ExportLock:
    def __init__(self, lock_path: Path, timeout_seconds: int = 3600) -> None:
        self.lock_path = lock_path
        self.timeout_seconds = timeout_seconds
        self._lock = FileLock(str(lock_path), timeout=timeout_seconds)

    def __enter__(self) -> "ExportLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._lock.release()


def atomic_publish(temp_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.replace(target_path)
