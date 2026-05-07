import json
from hashlib import sha256
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from xsignal.data.canonical_bars import CanonicalRequest, Partition
from xsignal.data import canonical_export as canonical_export_module
from xsignal.data.canonical_export import (
    discover_full_history_bounds,
    ensure_canonical_bars,
    main,
    partitions_for_full_history,
)
from xsignal.data.paths import CanonicalPaths


CANONICAL_COLUMNS = [
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "bar_count",
    "synthetic_1m_count",
    "expected_1m_count",
    "is_complete",
    "has_synthetic",
    "fill_policy",
]


def write_canonical_parquet(
    path: Path,
    row_count: int,
    *,
    bar_count: int = 60,
    fill_policy: str = "raw",
    synthetic_1m_count: int = 0,
    has_synthetic: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "symbol": [f"BTC{i}" for i in range(row_count)],
            "open_time": list(range(row_count)),
            "open": [1.0] * row_count,
            "high": [2.0] * row_count,
            "low": [0.5] * row_count,
            "close": [1.5] * row_count,
            "volume": [100.0] * row_count,
            "quote_volume": [150.0] * row_count,
            "trade_count": [10] * row_count,
            "taker_buy_volume": [50.0] * row_count,
            "taker_buy_quote_volume": [75.0] * row_count,
            "bar_count": [bar_count] * row_count,
            "synthetic_1m_count": [synthetic_1m_count] * row_count,
            "expected_1m_count": [60] * row_count,
            "is_complete": [True] * row_count,
            "has_synthetic": [has_synthetic] * row_count,
            "fill_policy": [fill_policy] * row_count,
        }
    )
    pq.write_table(table.select(CANONICAL_COLUMNS), path)


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class FakeExporter:
    def __init__(
        self,
        row_count: int = 3,
        *,
        bar_count: int = 60,
        fill_policy: str = "raw",
        synthetic_1m_count: int = 0,
        has_synthetic: bool = False,
    ) -> None:
        self.calls = []
        self.row_count = row_count
        self.bar_count = bar_count
        self.fill_policy = fill_policy
        self.synthetic_1m_count = synthetic_1m_count
        self.has_synthetic = has_synthetic

    def export(self, sql: str, path: Path) -> int:
        self.calls.append((sql, path))
        write_canonical_parquet(
            path,
            self.row_count,
            bar_count=self.bar_count,
            fill_policy=self.fill_policy,
            synthetic_1m_count=self.synthetic_1m_count,
            has_synthetic=self.has_synthetic,
        )
        return self.row_count


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


class BadSchemaExporter:
    def export(self, sql: str, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"symbol": ["BTCUSDT"]}), path)
        return 1


def manifest_parquet_path(paths: CanonicalPaths, partition: Partition) -> Path:
    manifest = json.loads(paths.manifest_path(partition).read_text())
    return Path(manifest["parquet_path"])


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
    assert paths.manifest_path(partition).exists()
    assert manifest_parquet_path(paths, partition).exists()
    assert result.partitions == [partition]


def test_ensure_rejects_paths_fill_policy_mismatch(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path, fill_policy="prev_close_zero_volume")
    partition = Partition(timeframe="1h", year=2026, month=5)

    with pytest.raises(ValueError, match="fill_policy"):
        ensure_canonical_bars(
            request=CanonicalRequest(timeframe="1h", fill_policy="raw"),
            paths=paths,
            partitions=[partition],
            exporter=exporter,
            now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
        )

    assert exporter.calls == []


def test_ensure_passes_filled_policy_to_query_builder(tmp_path):
    exporter = FakeExporter(
        bar_count=59,
        fill_policy="prev_close_zero_volume",
        synthetic_1m_count=1,
        has_synthetic=True,
    )
    paths = CanonicalPaths(root=tmp_path, fill_policy="prev_close_zero_volume")
    partition = Partition(timeframe="1h", year=2026, month=5)

    ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h", fill_policy="prev_close_zero_volume"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    assert len(exporter.calls) == 1
    assert "'prev_close_zero_volume' AS fill_policy" in exporter.calls[0][0]
    assert "fill_policy=prev_close_zero_volume" in str(paths.manifest_path(partition))
    manifest = json.loads(paths.manifest_path(partition).read_text())
    assert manifest["fill_policy"] == "prev_close_zero_volume"
    assert manifest["synthetic_generation_version"] == "prev-close-zero-volume-v1"
    assert manifest["synthetic_1m_count_total"] == 3
    assert manifest["incomplete_raw_bar_count"] == 3


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


def test_cli_ensure_reuses_complete_partition_without_opening_clickhouse(
    tmp_path,
    monkeypatch,
):
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

    def fail_if_opened(*_args, **_kwargs):
        raise AssertionError("ClickHouse client should not be opened for complete partitions")

    monkeypatch.setattr("xsignal.data.canonical_export.ClickHouseClient", fail_if_opened)

    exit_code = main(
        [
            "ensure",
            "--timeframe",
            "1h",
            "--year",
            "2026",
            "--month",
            "5",
            "--root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(exporter.calls) == 1


def test_cli_ensure_defaults_to_full_history_when_year_is_omitted(tmp_path, monkeypatch):
    class FakeClickHouseClient:
        instances = []

        def __init__(self, _config):
            self.queries = []
            self.exports = []
            FakeClickHouseClient.instances.append(self)

        def query_arrow(self, sql: str):
            self.queries.append(sql)
            return pa.table(
                {
                    "start": [datetime(2026, 5, 1, tzinfo=timezone.utc)],
                    "end": [datetime(2026, 7, 1, tzinfo=timezone.utc)],
                }
            )

        def write_parquet(self, sql: str, path: Path) -> int:
            self.exports.append((sql, path))
            write_canonical_parquet(path, row_count=3)
            return 3

    monkeypatch.setattr("xsignal.data.canonical_export.ClickHouseClient", FakeClickHouseClient)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["ensure", "--timeframe", "1h"])

    assert exit_code == 0
    assert len(FakeClickHouseClient.instances) == 1
    client = FakeClickHouseClient.instances[0]
    assert len(client.queries) == 1
    assert len(client.exports) == 2
    may_partition = Partition(timeframe="1h", year=2026, month=5)
    june_partition = Partition(timeframe="1h", year=2026, month=6)
    assert manifest_parquet_path(CanonicalPaths(root=Path("data")), may_partition).is_file()
    assert manifest_parquet_path(CanonicalPaths(root=Path("data")), june_partition).is_file()


def test_discover_full_history_bounds_accepts_clickhouse_epoch_seconds():
    class FakeClient:
        def query_arrow(self, _sql: str):
            return pa.table({"start": [1577836800], "end": [1577923200]})

    start, end = discover_full_history_bounds(FakeClient())

    assert start == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert end == datetime(2020, 1, 2, tzinfo=timezone.utc)


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


def test_ensure_cleans_temp_parquet_when_publish_fails(tmp_path, monkeypatch):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)
    original_atomic_publish = canonical_export_module.atomic_publish

    def fail_parquet_publish(temp_path: Path, target_path: Path) -> None:
        if target_path.suffix == ".parquet":
            raise OSError("parquet publish failed")
        original_atomic_publish(temp_path, target_path)

    monkeypatch.setattr(canonical_export_module, "atomic_publish", fail_parquet_publish)

    with pytest.raises(OSError, match="parquet publish failed"):
        ensure_canonical_bars(
            request=CanonicalRequest(timeframe="1h"),
            paths=paths,
            partitions=[partition],
            exporter=exporter,
            now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
        )

    assert not paths.manifest_path(partition).exists()
    assert not paths.catalog_path("1h").exists()
    assert list(paths.partition_dir(partition).glob("*.tmp.parquet")) == []
    assert list(paths.partition_dir(partition).glob("bars.*.parquet")) == []


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

    assert not paths.parquet_path(partition).exists()
    assert paths.manifest_path(partition).is_dir()
    assert not paths.catalog_path("1h").exists()
    assert list(paths.partition_dir(partition).glob(".manifest.*.tmp.json")) == []


def test_ensure_restores_existing_parquet_when_manifest_publish_fails(tmp_path, monkeypatch):
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=FakeExporter(row_count=2),
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )
    old_parquet_path = manifest_parquet_path(paths, partition)
    old_parquet_digest = file_digest(old_parquet_path)
    original_atomic_publish = canonical_export_module.atomic_publish

    def fail_manifest_publish(temp_path: Path, target_path: Path) -> None:
        if target_path.suffix == ".parquet":
            assert old_parquet_path.exists()
        if target_path == paths.manifest_path(partition):
            raise OSError("manifest publish failed")
        original_atomic_publish(temp_path, target_path)

    monkeypatch.setattr(canonical_export_module, "atomic_publish", fail_manifest_publish)

    with pytest.raises(OSError, match="manifest publish failed"):
        ensure_canonical_bars(
            request=CanonicalRequest(timeframe="1h", dataset_version="v2"),
            paths=paths,
            partitions=[partition],
            exporter=FakeExporter(row_count=3),
            now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
        )

    assert file_digest(old_parquet_path) == old_parquet_digest
    assert list(paths.partition_dir(partition).glob(".manifest.*.tmp.json")) == []


def test_ensure_rejects_invalid_published_parquet_before_catalog_update(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    with pytest.raises(ValueError, match="published parquet"):
        ensure_canonical_bars(
            request=CanonicalRequest(timeframe="1h"),
            paths=paths,
            partitions=[partition],
            exporter=BadSchemaExporter(),
            now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
        )

    assert not paths.catalog_path("1h").exists()
    assert not paths.manifest_path(partition).exists()


def test_partitions_for_full_history_uses_months_for_intraday_timeframes():
    partitions = partitions_for_full_history(
        timeframe="1h",
        start=datetime(2026, 4, 10, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1h", year=2026, month=4),
        Partition(timeframe="1h", year=2026, month=5),
    ]


def test_partitions_for_full_history_excludes_exact_month_start_end():
    partitions = partitions_for_full_history(
        timeframe="1h",
        start=datetime(2026, 4, 10, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, 0, 0, 0, 0, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1h", year=2026, month=4),
        Partition(timeframe="1h", year=2026, month=5),
    ]


def test_partitions_for_full_history_includes_month_for_end_with_seconds():
    partitions = partitions_for_full_history(
        timeframe="1h",
        start=datetime(2026, 4, 10, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, 0, 0, 30, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1h", year=2026, month=4),
        Partition(timeframe="1h", year=2026, month=5),
        Partition(timeframe="1h", year=2026, month=6),
    ]


def test_partitions_for_full_history_includes_month_for_end_with_microseconds():
    partitions = partitions_for_full_history(
        timeframe="1h",
        start=datetime(2026, 4, 10, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1h", year=2026, month=4),
        Partition(timeframe="1h", year=2026, month=5),
        Partition(timeframe="1h", year=2026, month=6),
    ]


def test_partitions_for_full_history_uses_years_for_daily_timeframe():
    partitions = partitions_for_full_history(
        timeframe="1d",
        start=datetime(2025, 12, 31, tzinfo=timezone.utc),
        end=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1d", year=2025),
        Partition(timeframe="1d", year=2026),
    ]


def test_partitions_for_full_history_excludes_exact_year_start_end_for_daily_timeframe():
    partitions = partitions_for_full_history(
        timeframe="1d",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1d", year=2025),
    ]


def test_partitions_for_full_history_includes_year_for_daily_end_with_microseconds():
    partitions = partitions_for_full_history(
        timeframe="1d",
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1d", year=2025),
        Partition(timeframe="1d", year=2026),
    ]


def test_partitions_for_full_history_normalizes_bounds_to_utc():
    utc_plus_8 = timezone(timedelta(hours=8))

    partitions = partitions_for_full_history(
        timeframe="1h",
        start=datetime(2026, 5, 1, 0, 30, tzinfo=utc_plus_8),
        end=datetime(2026, 6, 1, 8, 0, tzinfo=utc_plus_8),
    )

    assert partitions == [
        Partition(timeframe="1h", year=2026, month=4),
        Partition(timeframe="1h", year=2026, month=5),
    ]
