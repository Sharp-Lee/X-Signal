from pathlib import Path

from xsignal.data.locks import ExportLock, atomic_publish


def test_export_lock_creates_parent_directory(tmp_path):
    lock_path = tmp_path / "canonical_bars" / "_locks" / "partition.lock"

    with ExportLock(lock_path):
        assert lock_path.parent.exists()


def test_atomic_publish_replaces_target(tmp_path):
    temp_path = tmp_path / ".bars.tmp.parquet"
    target_path = tmp_path / "bars.parquet"
    temp_path.write_bytes(b"new")
    target_path.write_bytes(b"old")

    atomic_publish(temp_path, target_path)

    assert target_path.read_bytes() == b"new"
    assert not temp_path.exists()
