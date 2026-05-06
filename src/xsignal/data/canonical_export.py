from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from xsignal.data.canonical_bars import CanonicalRequest, Partition
from xsignal.data.catalog import Catalog, PartitionStatus
from xsignal.data.clickhouse import ClickHouseClient, ClickHouseConfig
from xsignal.data.locks import ExportLock, atomic_publish
from xsignal.data.paths import CanonicalPaths
from xsignal.data.query_templates import CLICKHOUSE_SOURCE_TABLE, build_aggregate_query, query_hash
from xsignal.runs.manifest import ExportManifest


class Exporter(Protocol):
    def export(self, sql: str, path: Path) -> int: ...


@dataclass(frozen=True)
class CanonicalDataset:
    request: CanonicalRequest
    root: Path
    partitions: Sequence[Partition]


def _partition_bounds(partition: Partition) -> tuple[datetime, datetime]:
    if partition.month is None:
        start = datetime(partition.year, 1, 1, tzinfo=timezone.utc)
        end = datetime(partition.year + 1, 1, 1, tzinfo=timezone.utc)
        return start, end

    start = datetime(partition.year, partition.month, 1, tzinfo=timezone.utc)
    if partition.month == 12:
        end = datetime(partition.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(partition.year, partition.month + 1, 1, tzinfo=timezone.utc)
    return start, end


def _repair_catalog(catalog: Catalog, partition: Partition) -> None:
    manifest = catalog.load_manifest(partition)
    if manifest is not None:
        catalog.mark_complete(manifest)


def ensure_canonical_bars(
    request: CanonicalRequest,
    paths: CanonicalPaths,
    partitions: Sequence[Partition],
    exporter: Exporter,
    now: Callable[[], datetime] | None = None,
) -> CanonicalDataset:
    clock = now or (lambda: datetime.now(timezone.utc))
    catalog = Catalog(paths)

    for partition in partitions:
        if partition.timeframe != request.timeframe:
            raise ValueError("Partition timeframe must match request timeframe")
        if catalog.status(partition, request.dataset_version) == PartitionStatus.COMPLETE:
            _repair_catalog(catalog, partition)
            continue

        with ExportLock(paths.lock_path(partition)):
            if catalog.status(partition, request.dataset_version) == PartitionStatus.COMPLETE:
                _repair_catalog(catalog, partition)
                continue

            run_id = uuid.uuid4().hex
            start, end = _partition_bounds(partition)
            sql = build_aggregate_query(request.timeframe, start, end)
            temp_parquet = paths.temp_parquet_path(partition, run_id)
            target_parquet = paths.parquet_path(partition)
            try:
                row_count = exporter.export(sql, temp_parquet)
            except Exception:
                temp_parquet.unlink(missing_ok=True)
                raise
            if row_count <= 0:
                temp_parquet.unlink(missing_ok=True)
                raise ValueError(f"Export returned non-positive row_count={row_count}")
            try:
                atomic_publish(temp_parquet, target_parquet)
            except Exception:
                temp_parquet.unlink(missing_ok=True)
                raise

            manifest = ExportManifest(
                dataset_version=request.dataset_version,
                source_table=CLICKHOUSE_SOURCE_TABLE,
                timeframe=request.timeframe,
                partition_key=partition.key,
                deduplication_mode="FINAL",
                aggregation_semantics_version="ohlcv-v1",
                query_hash=query_hash(sql),
                row_count=row_count,
                parquet_path=str(target_parquet),
                exported_at=clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            )
            temp_manifest = paths.temp_manifest_path(partition, run_id)
            temp_manifest.parent.mkdir(parents=True, exist_ok=True)
            temp_manifest.write_text(manifest.model_dump_json(indent=2))
            atomic_publish(temp_manifest, paths.manifest_path(partition))
            catalog.mark_complete(manifest)

    return CanonicalDataset(request=request, root=paths.base, partitions=partitions)


@dataclass(frozen=True)
class ClickHouseExporter:
    client: ClickHouseClient

    def export(self, sql: str, path: Path) -> int:
        return self.client.write_parquet(sql, path)


def _ensure_command(args: argparse.Namespace) -> CanonicalDataset:
    partition = Partition(timeframe=args.timeframe, year=args.year, month=args.month)
    request = CanonicalRequest(timeframe=args.timeframe)
    paths = CanonicalPaths(root=Path(args.root))
    exporter = ClickHouseExporter(ClickHouseClient(ClickHouseConfig()))
    return ensure_canonical_bars(
        request=request,
        paths=paths,
        partitions=[partition],
        exporter=exporter,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonical bar export orchestration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ensure_parser = subparsers.add_parser("ensure")
    ensure_parser.add_argument("--timeframe", required=True)
    ensure_parser.add_argument("--root", required=True)
    ensure_parser.add_argument("--year", required=True, type=int)
    ensure_parser.add_argument("--month", type=int)
    ensure_parser.set_defaults(func=_ensure_command)

    args = parser.parse_args(argv)
    dataset = args.func(args)
    print(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
