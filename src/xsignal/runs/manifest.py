from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class ExportManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_version: str
    source_table: str
    timeframe: str
    fill_policy: str = "raw"
    partition_key: str
    deduplication_mode: str
    aggregation_semantics_version: str
    synthetic_generation_version: str = "none"
    query_hash: str
    row_count: int
    parquet_path: str
    exported_at: str
    synthetic_1m_count_total: int = 0
    incomplete_raw_bar_count: int = 0
    symbol_bound_policy: str = "observed_1m_bounds"

    @field_validator(
        "dataset_version",
        "source_table",
        "timeframe",
        "fill_policy",
        "partition_key",
        "deduplication_mode",
        "aggregation_semantics_version",
        "synthetic_generation_version",
        "query_hash",
        "parquet_path",
        "exported_at",
        "symbol_bound_policy",
    )
    @classmethod
    def _reject_empty_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value

    @field_validator("synthetic_1m_count_total", "incomplete_raw_bar_count")
    @classmethod
    def _reject_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("field must be non-negative")
        return value
