from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

from xsignal.strategies.momentum_rotation_v1.artifacts import write_run_artifacts
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.data import (
    collect_offline_manifest_paths,
    collect_strategy_inputs,
)
from xsignal.strategies.momentum_rotation_v1.kernel import run_backtest
from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths
from xsignal.strategies.momentum_rotation_v1.prepare import (
    PreparedArrays,
    load_prepared_arrays,
    prepare_daily_arrays,
    save_prepared_arrays,
)
from xsignal.strategies.momentum_rotation_v1.signals import compute_momentum_signals


PREPARED_CACHE_VERSION = "momentum-rotation-prepared-v2"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _prepared_cache_key(config: MomentumRotationConfig, manifest_paths: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(PREPARED_CACHE_VERSION.encode())
    digest.update(b"\0")
    digest.update(config.strategy_name.encode())
    digest.update(b"\0")
    digest.update(config.fill_policy.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(config.timeframes), separators=(",", ":")).encode())
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        digest.update(b"\0manifest-path\0")
        digest.update(str(path).encode())
        digest.update(b"\0manifest-bytes\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "cache_manifest.json"


def _write_cache_manifest(
    cache_dir: Path,
    *,
    cache_key: str,
    config: MomentumRotationConfig,
    canonical_manifests: list[str],
) -> None:
    payload = {
        "cache_key": cache_key,
        "cache_version": PREPARED_CACHE_VERSION,
        "config_hash": config.config_hash(),
        "canonical_manifests": canonical_manifests,
    }
    _cache_manifest_path(cache_dir).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _cache_manifest_matches(
    cache_dir: Path,
    *,
    cache_key: str,
    canonical_manifests: list[str],
) -> bool:
    path = _cache_manifest_path(cache_dir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        payload.get("cache_key") == cache_key
        and payload.get("cache_version") == PREPARED_CACHE_VERSION
        and payload.get("canonical_manifests") == canonical_manifests
    )


def prepare_from_canonical(
    root: Path,
    config: MomentumRotationConfig,
    *,
    offline: bool = False,
    use_cache: bool = True,
) -> tuple[PreparedArrays, list[str]]:
    offline_manifest_paths = None
    if offline:
        offline_manifest_paths = collect_offline_manifest_paths(root=root, config=config)
        canonical_manifests = [str(path) for path in offline_manifest_paths]
    else:
        canonical_manifests = []
    paths = MomentumRotationPaths(root=root)
    cache_key = _prepared_cache_key(config, canonical_manifests)
    cache_dir = paths.cache / "prepared" / cache_key
    if use_cache and _cache_manifest_matches(
        cache_dir,
        cache_key=cache_key,
        canonical_manifests=canonical_manifests,
    ):
        return load_prepared_arrays(cache_dir), canonical_manifests

    inputs = collect_strategy_inputs(root=root, config=config, offline=offline)
    if not canonical_manifests:
        canonical_manifests = [str(path) for path in inputs.manifest_paths]
        cache_key = _prepared_cache_key(config, canonical_manifests)
        cache_dir = paths.cache / "prepared" / cache_key
        if use_cache and _cache_manifest_matches(
            cache_dir,
            cache_key=cache_key,
            canonical_manifests=canonical_manifests,
        ):
            return load_prepared_arrays(cache_dir), canonical_manifests

    arrays = prepare_daily_arrays(
        bars_1h=inputs.bars_1h,
        bars_4h=inputs.bars_4h,
        bars_1d=inputs.bars_1d,
    )
    if use_cache:
        save_prepared_arrays(cache_dir, arrays)
        _write_cache_manifest(
            cache_dir,
            cache_key=cache_key,
            config=config,
            canonical_manifests=canonical_manifests,
        )
    return arrays, canonical_manifests


def _run_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    config = MomentumRotationConfig(
        top_n=args.top_n,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        min_rolling_7d_quote_volume=args.min_rolling_7d_quote_volume,
    )
    arrays, canonical_manifests = prepare_from_canonical(
        Path(args.root),
        config,
        offline=args.offline,
        use_cache=not args.no_cache,
    )
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
    run_parser.add_argument(
        "--offline",
        action="store_true",
        help="Only read existing canonical Parquet manifests; never connect to ClickHouse or export",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild prepared arrays even if a prepared cache entry exists",
    )
    run_parser.set_defaults(func=_run_command)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
