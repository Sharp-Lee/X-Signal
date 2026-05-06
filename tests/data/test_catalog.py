import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from xsignal.data.canonical_bars import Partition
from xsignal.data.catalog import Catalog, PartitionStatus
from xsignal.data.paths import CanonicalPaths
from xsignal.runs.manifest import ExportManifest


def make_manifest(partition: Partition, parquet_path: Path, **overrides: object) -> ExportManifest:
    payload = {
        "dataset_version": "v1",
        "source_table": "xgate.klines_1m",
        "timeframe": partition.timeframe,
        "partition_key": partition.key,
        "deduplication_mode": "FINAL",
        "aggregation_semantics_version": "ohlcv-v1",
        "query_hash": "abc123",
        "row_count": 10,
        "parquet_path": str(parquet_path),
        "exported_at": "2026-05-06T00:00:00Z",
    }
    payload.update(overrides)
    return ExportManifest(**payload)


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


def test_catalog_mark_complete_preserves_concurrent_partition_updates(tmp_path, monkeypatch):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    first_partition = Partition(timeframe="1h", year=2026, month=5)
    second_partition = Partition(timeframe="1h", year=2026, month=6)
    catalog_path = paths.catalog_path("1h")
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(json.dumps({"timeframe": "1h", "partitions": {}}))
    first_manifest = make_manifest(first_partition, paths.parquet_path(first_partition))
    second_manifest = make_manifest(second_partition, paths.parquet_path(second_partition))
    original_loads = json.loads
    barrier = threading.Barrier(2)

    def loads_with_overlap(payload: str, *args, **kwargs):
        result = original_loads(payload, *args, **kwargs)
        if result == {"timeframe": "1h", "partitions": {}}:
            try:
                barrier.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass
        return result

    monkeypatch.setattr("xsignal.data.catalog.json.loads", loads_with_overlap)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(catalog.mark_complete, first_manifest),
            executor.submit(catalog.mark_complete, second_manifest),
        ]
        for future in futures:
            future.result()

    catalog_json = original_loads(catalog_path.read_text())
    assert set(catalog_json["partitions"]) == {
        first_partition.key,
        second_partition.key,
    }


def test_catalog_treats_missing_manifest_as_missing(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths.parquet_path(partition).parent.mkdir(parents=True)
    paths.parquet_path(partition).write_bytes(b"fake-parquet")

    assert catalog.status(partition, dataset_version="v1") == PartitionStatus.MISSING


def test_catalog_treats_corrupt_manifest_as_stale(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = paths.parquet_path(partition)
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"fake-parquet")
    paths.manifest_path(partition).write_text("{bad json")

    assert catalog.status(partition, dataset_version="v1") == PartitionStatus.STALE


def test_catalog_treats_missing_manifest_fields_as_stale(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = paths.parquet_path(partition)
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"fake-parquet")
    paths.manifest_path(partition).write_text("{}")

    assert catalog.status(partition, dataset_version="v1") == PartitionStatus.STALE


def test_catalog_treats_parquet_directory_as_stale(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = paths.parquet_path(partition)
    parquet_path.mkdir(parents=True)
    manifest = make_manifest(partition, parquet_path)
    paths.manifest_path(partition).write_text(manifest.model_dump_json(indent=2))

    assert catalog.status(partition, dataset_version="v1") == PartitionStatus.STALE


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_table", " "),
        ("query_hash", ""),
    ],
)
def test_export_manifest_rejects_empty_strings(tmp_path, field_name, value):
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = CanonicalPaths(root=tmp_path).parquet_path(partition)

    with pytest.raises(ValueError):
        make_manifest(partition, parquet_path, **{field_name: value})
