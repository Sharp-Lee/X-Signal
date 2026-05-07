from __future__ import annotations

import argparse
import subprocess
import time
import uuid
from pathlib import Path

from xsignal.strategies.momentum_rotation_v1.artifacts import write_run_artifacts
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.kernel import run_backtest
from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import compute_momentum_signals


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def prepare_from_canonical(root: Path, config: MomentumRotationConfig) -> tuple[PreparedArrays, list[str]]:
    raise RuntimeError("canonical preparation is not connected")


def _run_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    config = MomentumRotationConfig(
        top_n=args.top_n,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        min_rolling_7d_quote_volume=args.min_rolling_7d_quote_volume,
    )
    arrays, canonical_manifests = prepare_from_canonical(Path(args.root), config)
    signals = compute_momentum_signals(arrays, config)
    result = run_backtest(arrays, signals, config)
    runtime_seconds = time.perf_counter() - started
    paths = MomentumRotationPaths(root=Path(args.root))
    run_id = args.run_id or uuid.uuid4().hex
    output = write_run_artifacts(
        paths=paths,
        run_id=run_id,
        config=config,
        symbols=arrays.symbols,
        rebalance_times=arrays.rebalance_times,
        result=result,
        canonical_manifests=canonical_manifests,
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
    )
    print(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run momentum_rotation_v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", default="data")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--top-n", type=int, default=10)
    run_parser.add_argument("--fee-bps", type=float, default=5.0)
    run_parser.add_argument("--slippage-bps", type=float, default=5.0)
    run_parser.add_argument("--min-rolling-7d-quote-volume", type=float, default=0.0)
    run_parser.set_defaults(func=_run_command)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
