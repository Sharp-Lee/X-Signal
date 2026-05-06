import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xsignal.data.canonical_bars import CanonicalRequest, Partition
from xsignal.data.canonical_export import ensure_canonical_bars
from xsignal.data.paths import CanonicalPaths


class FakeExporter:
    def __init__(self) -> None:
        self.calls = []

    def export(self, sql: str, path: Path) -> int:
        self.calls.append((sql, path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-parquet")
        return 3


class ZeroRowExporter:
    def export(self, sql: str, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"empty-parquet")
        return 0


class FailingExporter:
    def export(self, sql: str, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial-parquet")
        raise RuntimeError("export failed")


def test_ensure_exports_missing_partition(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    result = ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    assert len(exporter.calls) == 1
    assert paths.parquet_path(partition).exists()
    assert paths.manifest_path(partition).exists()
    assert result.partitions == [partition]


def test_ensure_reuses_complete_partition(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )
    ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    assert len(exporter.calls) == 1


def test_ensure_repairs_missing_catalog_for_complete_partition(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )
    catalog_path = paths.catalog_path("1h")
    catalog_path.unlink()

    ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    catalog = json.loads(catalog_path.read_text())
    assert len(exporter.calls) == 1
    assert partition.key in catalog["partitions"]


def test_ensure_rejects_zero_row_export_and_cleans_temp_parquet(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    with pytest.raises(ValueError, match="row_count"):
        ensure_canonical_bars(
            request=CanonicalRequest(timeframe="1h"),
            paths=paths,
            partitions=[partition],
            exporter=ZeroRowExporter(),
            now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
        )

    assert not paths.parquet_path(partition).exists()
    assert not paths.manifest_path(partition).exists()
    assert not paths.catalog_path("1h").exists()
    assert list(paths.partition_dir(partition).glob("*.tmp.parquet")) == []


def test_ensure_cleans_partial_temp_parquet_when_export_fails(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    with pytest.raises(RuntimeError, match="export failed"):
        ensure_canonical_bars(
            request=CanonicalRequest(timeframe="1h"),
            paths=paths,
            partitions=[partition],
            exporter=FailingExporter(),
            now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
        )

    assert not paths.parquet_path(partition).exists()
    assert not paths.manifest_path(partition).exists()
    assert not paths.catalog_path("1h").exists()
    assert list(paths.partition_dir(partition).glob("*.tmp.parquet")) == []


def test_ensure_cleans_temp_parquet_when_publish_fails(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths.parquet_path(partition).mkdir(parents=True)

    with pytest.raises(OSError):
        ensure_canonical_bars(
            request=CanonicalRequest(timeframe="1h"),
            paths=paths,
            partitions=[partition],
            exporter=exporter,
            now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
        )

    assert paths.parquet_path(partition).is_dir()
    assert not paths.manifest_path(partition).exists()
    assert not paths.catalog_path("1h").exists()
    assert list(paths.partition_dir(partition).glob("*.tmp.parquet")) == []


def test_ensure_cleans_temp_manifest_when_manifest_publish_fails(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths.manifest_path(partition).mkdir(parents=True)

    with pytest.raises(OSError):
        ensure_canonical_bars(
            request=CanonicalRequest(timeframe="1h"),
            paths=paths,
            partitions=[partition],
            exporter=exporter,
            now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
        )

    assert paths.parquet_path(partition).is_file()
    assert paths.manifest_path(partition).is_dir()
    assert not paths.catalog_path("1h").exists()
    assert list(paths.partition_dir(partition).glob(".manifest.*.tmp.json")) == []
