from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import clickhouse_connect


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str = "127.0.0.1"
    port: int = 8123
    username: str = "default"
    password: str = ""
    database: str = "xgate"


class ClickHouseClient:
    def __init__(self, config: ClickHouseConfig) -> None:
        self.config = config
        self._client = clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
            database=config.database,
        )

    def query_arrow(self, sql: str):
        return self._client.query_arrow(sql)

    def write_parquet(self, sql: str, path: Path) -> int:
        table = self.query_arrow(sql)
        path.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq

        pq.write_table(table, path, compression="zstd")
        return table.num_rows
