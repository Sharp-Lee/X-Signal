from __future__ import annotations

import argparse
import subprocess
import time
import uuid
from pathlib import Path

from xsignal.strategies.volume_price_efficiency_v1.artifacts import write_run_artifacts
from xsignal.strategies.volume_price_efficiency_v1.baseline import build_baseline_events
from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import (
    load_offline_ohlcv_table,
    prepare_ohlcv_arrays,
)
from xsignal.strategies.volume_price_efficiency_v1.events import build_event_rows
from xsignal.strategies.volume_price_efficiency_v1.features import compute_features
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _run_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    config = VolumePriceEfficiencyConfig(
        atr_window=args.atr_window,
        volume_window=args.volume_window,
        efficiency_lookback=args.efficiency_lookback,
        efficiency_percentile=args.efficiency_percentile,
        volume_floor=args.volume_floor,
        min_move_unit=args.min_move_unit,
        min_volume_unit=args.min_volume_unit,
        min_close_position=args.min_close_position,
        min_body_ratio=args.min_body_ratio,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        baseline_seed=args.baseline_seed,
    )
    table, manifests = load_offline_ohlcv_table(Path(args.root), fill_policy=config.fill_policy)
    arrays = prepare_ohlcv_arrays(table)
    features = compute_features(arrays, config)
    events = build_event_rows(arrays, features, config)
    baseline_events = build_baseline_events(arrays, features, config)
    runtime_seconds = time.perf_counter() - started
    run_id = args.run_id or uuid.uuid4().hex
    output = write_run_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        run_id=run_id,
        config=config,
        events=events,
        baseline_events=baseline_events,
        symbols=arrays.symbols,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
    )
    print(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run volume_price_efficiency_v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", default="data")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--offline", action="store_true", required=True)
    run_parser.add_argument("--atr-window", type=int, default=14)
    run_parser.add_argument("--volume-window", type=int, default=60)
    run_parser.add_argument("--efficiency-lookback", type=int, default=120)
    run_parser.add_argument("--efficiency-percentile", type=float, default=0.90)
    run_parser.add_argument("--volume-floor", type=float, default=0.2)
    run_parser.add_argument("--min-move-unit", type=float, default=0.5)
    run_parser.add_argument("--min-volume-unit", type=float, default=0.3)
    run_parser.add_argument("--min-close-position", type=float, default=0.7)
    run_parser.add_argument("--min-body-ratio", type=float, default=0.4)
    run_parser.add_argument("--fee-bps", type=float, default=5.0)
    run_parser.add_argument("--slippage-bps", type=float, default=5.0)
    run_parser.add_argument("--baseline-seed", type=int, default=17)
    run_parser.set_defaults(func=_run_command)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
