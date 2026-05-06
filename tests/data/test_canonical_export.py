from datetime import datetime, timezone
from pathlib import Path

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
