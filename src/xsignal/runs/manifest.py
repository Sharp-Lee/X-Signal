from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


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

    @field_validator(
        "dataset_version",
        "source_table",
        "timeframe",
        "partition_key",
        "deduplication_mode",
        "aggregation_semantics_version",
        "query_hash",
        "parquet_path",
        "exported_at",
    )
    @classmethod
    def _reject_empty_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value
