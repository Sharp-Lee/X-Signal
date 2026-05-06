from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExportManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_version: str
    source_table: str
    timeframe: str
    partition_key: str
    deduplication_mode: str
    aggregation_semantics_version: str
    query_hash: str
    row_count: int
    parquet_path: str
    exported_at: str
