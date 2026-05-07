from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


FIELD_MAP = {
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
    "quote_volume": 7,
    "trade_count": 8,
    "taker_buy_volume": 9,
    "taker_buy_quote_volume": 10,
}


def fetch_binance(symbol: str, interval: str, start_ms: int, limit: int) -> list[list[Any]]:
    query = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": limit,
        }
    )
    url = f"https://fapi.binance.com/fapi/v1/klines?{query}"
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _open_time_seconds(value: object) -> int:
    if hasattr(value, "timestamp"):
        return int(value.timestamp())
    return int(value)


def compare_rows(
    local_rows: list[dict[str, object]],
    remote_rows: list[list[Any]],
    tolerance: Decimal,
) -> list[dict[str, object]]:
    mismatches = []
    for local, remote in zip(local_rows, remote_rows, strict=False):
        local_open_time = _open_time_seconds(local["open_time"])
        remote_open_time = int(remote[0]) // 1000
        if local_open_time != remote_open_time:
            mismatches.append(
                {
                    "open_time": local_open_time,
                    "field": "open_time",
                    "local": local_open_time,
                    "remote": remote_open_time,
                }
            )
            continue

        for field, index in FIELD_MAP.items():
            if field == "trade_count":
                if int(local[field]) != int(remote[index]):
                    mismatches.append(
                        {
                            "open_time": local_open_time,
                            "field": field,
                            "local": local[field],
                            "remote": remote[index],
                        }
                    )
                continue
            diff = abs(Decimal(str(local[field])) - Decimal(str(remote[index])))
            if diff > tolerance:
                mismatches.append(
                    {
                        "open_time": local_open_time,
                        "field": field,
                        "local": local[field],
                        "remote": remote[index],
                        "diff": str(diff),
                    }
                )
    if len(remote_rows) != len(local_rows):
        mismatches.append(
            {
                "field": "row_count",
                "local": len(local_rows),
                "remote": len(remote_rows),
            }
        )
    return mismatches


def _load_local_rows(manifest: dict[str, object], symbol: str, limit: int) -> list[dict[str, object]]:
    table = pq.ParquetFile(str(manifest["parquet_path"])).read()
    filtered = table.filter(
        pc.and_(
            pc.equal(table["symbol"], symbol),
            pc.equal(pc.cast(table["is_complete"], pa.uint8()), 1),
        )
    ).slice(0, limit)
    return filtered.to_pylist()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate raw canonical bars against Binance USD-M Futures klines"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--tolerance", default="0.000001")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["fill_policy"] != "raw":
        raise SystemExit("Binance raw comparison requires fill_policy=raw")

    local_rows = _load_local_rows(manifest, args.symbol, args.limit)
    if not local_rows:
        raise SystemExit("No complete local rows found for requested symbol")

    remote_rows = fetch_binance(
        args.symbol,
        args.timeframe,
        _open_time_seconds(local_rows[0]["open_time"]) * 1000,
        len(local_rows),
    )
    mismatches = compare_rows(local_rows, remote_rows, Decimal(args.tolerance))
    result = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "checked": len(local_rows),
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:5],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
