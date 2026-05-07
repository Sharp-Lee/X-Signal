from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from xsignal.data.canonical_bars import CanonicalRequest, Partition, partition_bounds
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


def partitions_for_full_history(
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list[Partition]:
    if (
        start.tzinfo is None
        or start.utcoffset() is None
        or end.tzinfo is None
        or end.utcoffset() is None
    ):
        raise ValueError("start and end must be timezone-aware")

    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("end must be after start")

    partitions: list[Partition] = []
    if timeframe == "1d":
        last_year = end.year
        if (
            end.month == 1
            and end.day == 1
            and end.hour == 0
            and end.minute == 0
            and end.second == 0
            and end.microsecond == 0
        ):
            last_year -= 1
        for year in range(start.year, last_year + 1):
            partitions.append(Partition(timeframe=timeframe, year=year))
        return partitions

    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        if (
            (year, month) == (end.year, end.month)
            and end.day == 1
            and end.hour == 0
            and end.minute == 0
            and end.second == 0
            and end.microsecond == 0
        ):
            break
        partitions.append(Partition(timeframe=timeframe, year=year, month=month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return partitions


def _repair_catalog(catalog: Catalog, partition: Partition) -> None:
    manifest = catalog.load_manifest(partition)
    if manifest is not None:
        catalog.mark_complete(manifest)


def _manifest_quality_counts(parquet_path: Path) -> tuple[int, int]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet_path).read(
        columns=["synthetic_1m_count", "bar_count", "expected_1m_count"],
    )
    synthetic_total = pc.sum(table["synthetic_1m_count"]).as_py() or 0
    incomplete_total = pc.sum(
        pc.cast(pc.less(table["bar_count"], table["expected_1m_count"]), "int64")
    ).as_py() or 0
    return int(synthetic_total), int(incomplete_total)


def ensure_canonical_bars(
    request: CanonicalRequest,
    paths: CanonicalPaths,
    partitions: Sequence[Partition],
    exporter: Exporter,
    now: Callable[[], datetime] | None = None,
) -> CanonicalDataset:
    if paths.fill_policy != request.fill_policy:
        raise ValueError("CanonicalPaths fill_policy must match request fill_policy")

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
            start, end = partition_bounds(partition)
            sql = build_aggregate_query(
                request.timeframe,
                start,
                end,
                fill_policy=request.fill_policy,
            )
            temp_parquet = paths.temp_parquet_path(partition, run_id)
            try:
                row_count = exporter.export(sql, temp_parquet)
            except Exception:
                temp_parquet.unlink(missing_ok=True)
                raise
            if row_count <= 0:
                temp_parquet.unlink(missing_ok=True)
                raise ValueError(f"Export returned non-positive row_count={row_count}")
            target_parquet = paths.published_parquet_path(partition, run_id)
            try:
                atomic_publish(temp_parquet, target_parquet)
            except Exception:
                temp_parquet.unlink(missing_ok=True)
                raise
            try:
                synthetic_1m_count_total, incomplete_raw_bar_count = _manifest_quality_counts(
                    target_parquet
                )
            except Exception as exc:
                target_parquet.unlink(missing_ok=True)
                raise ValueError("published parquet failed manifest validation") from exc

            synthetic_generation_version = (
                "none"
                if request.fill_policy == "raw"
                else "prev-close-zero-volume-v1"
            )
            manifest = ExportManifest(
                dataset_version=request.dataset_version,
                source_table=CLICKHOUSE_SOURCE_TABLE,
                timeframe=request.timeframe,
                fill_policy=request.fill_policy,
                partition_key=partition.key,
                deduplication_mode="FINAL",
                aggregation_semantics_version="ohlcv-v2",
                synthetic_generation_version=synthetic_generation_version,
                query_hash=query_hash(sql),
                row_count=row_count,
                parquet_path=str(target_parquet),
                exported_at=clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                synthetic_1m_count_total=synthetic_1m_count_total,
                incomplete_raw_bar_count=incomplete_raw_bar_count,
                symbol_bound_policy="observed_1m_bounds",
            )
            if (
                catalog.status_for_manifest(partition, request.dataset_version, manifest)
                != PartitionStatus.COMPLETE
            ):
                target_parquet.unlink(missing_ok=True)
                raise ValueError("published parquet failed manifest validation")
            temp_manifest = paths.temp_manifest_path(partition, run_id)
            temp_manifest.parent.mkdir(parents=True, exist_ok=True)
            temp_manifest.write_text(manifest.model_dump_json(indent=2))
            try:
                atomic_publish(temp_manifest, paths.manifest_path(partition))
            except Exception:
                temp_manifest.unlink(missing_ok=True)
                target_parquet.unlink(missing_ok=True)
                raise
            catalog.mark_complete(manifest)

    return CanonicalDataset(request=request, root=paths.base, partitions=partitions)


@dataclass(frozen=True)
class ClickHouseExporter:
    client: ClickHouseClient

    def export(self, sql: str, path: Path) -> int:
        return self.client.write_parquet(sql, path)


def _as_utc_aware(value: datetime | int | float) -> datetime:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def discover_full_history_bounds(client: ClickHouseClient) -> tuple[datetime, datetime]:
    table = client.query_arrow(
        f"""
SELECT
    min(open_time) AS start,
    max(open_time) + INTERVAL 1 minute AS end
FROM {CLICKHOUSE_SOURCE_TABLE}
""".strip()
    )
    rows = table.to_pylist()
    if not rows or rows[0]["start"] is None or rows[0]["end"] is None:
        raise ValueError("ClickHouse source table has no bars to export")
    start = _as_utc_aware(rows[0]["start"])
    end = _as_utc_aware(rows[0]["end"])
    if end <= start:
        raise ValueError("ClickHouse source range is empty")
    return start, end


@dataclass
class _LazyExporter:
    factory: Callable[[], Exporter]
    _exporter: Exporter | None = None

    def export(self, sql: str, path: Path) -> int:
        if self._exporter is None:
            self._exporter = self.factory()
        return self._exporter.export(sql, path)


def _ensure_command(args: argparse.Namespace) -> CanonicalDataset:
    request = CanonicalRequest(timeframe=args.timeframe, fill_policy=args.fill_policy)
    paths = CanonicalPaths(root=Path(args.root), fill_policy=args.fill_policy)
    if args.year is None:
        if args.month is not None:
            raise ValueError("--month requires --year")
        client = ClickHouseClient(ClickHouseConfig())
        start, end = discover_full_history_bounds(client)
        partitions = partitions_for_full_history(args.timeframe, start, end)
        exporter: Exporter = ClickHouseExporter(client)
    else:
        partitions = [Partition(timeframe=args.timeframe, year=args.year, month=args.month)]
        exporter = _LazyExporter(lambda: ClickHouseExporter(ClickHouseClient(ClickHouseConfig())))
    return ensure_canonical_bars(
        request=request,
        paths=paths,
        partitions=partitions,
        exporter=exporter,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonical bar export orchestration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ensure_parser = subparsers.add_parser("ensure")
    ensure_parser.add_argument("--timeframe", required=True)
    ensure_parser.add_argument("--root", default="data")
    ensure_parser.add_argument("--fill-policy", default="raw")
    ensure_parser.add_argument("--year", type=int)
    ensure_parser.add_argument("--month", type=int)
    ensure_parser.set_defaults(func=_ensure_command)

    args = parser.parse_args(argv)
    dataset = args.func(args)
    print(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
