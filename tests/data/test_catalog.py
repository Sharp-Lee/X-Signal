from pathlib import Path

from xsignal.data.canonical_bars import Partition
from xsignal.data.catalog import Catalog, PartitionStatus
from xsignal.data.paths import CanonicalPaths
from xsignal.runs.manifest import ExportManifest


def make_manifest(partition: Partition, parquet_path: Path) -> ExportManifest:
    return ExportManifest(
        dataset_version="v1",
        source_table="xgate.klines_1m",
        timeframe=partition.timeframe,
        partition_key=partition.key,
        deduplication_mode="FINAL",
        aggregation_semantics_version="ohlcv-v1",
        query_hash="abc123",
        row_count=10,
        parquet_path=str(parquet_path),
        exported_at="2026-05-06T00:00:00Z",
    )


def test_catalog_marks_partition_complete(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = paths.parquet_path(partition)
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"fake-parquet")
    manifest = make_manifest(partition, parquet_path)
    paths.manifest_path(partition).write_text(manifest.model_dump_json(indent=2))

    catalog.mark_complete(manifest)

    status = catalog.status(partition, dataset_version="v1")
    assert status == PartitionStatus.COMPLETE


def test_catalog_treats_missing_manifest_as_missing(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths.parquet_path(partition).parent.mkdir(parents=True)
    paths.parquet_path(partition).write_bytes(b"fake-parquet")

    assert catalog.status(partition, dataset_version="v1") == PartitionStatus.MISSING
