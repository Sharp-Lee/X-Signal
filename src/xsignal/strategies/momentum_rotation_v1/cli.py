from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import time
import uuid
from datetime import date
from pathlib import Path

from xsignal.strategies.momentum_rotation_v1.artifacts import (
    build_backtest_summary,
    write_run_artifacts,
    write_scan_artifacts,
)
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
from xsignal.strategies.momentum_rotation_v1.splits import (
    filter_by_rebalance_date,
    split_research_and_holdout,
)


PREPARED_CACHE_VERSION = "momentum-rotation-prepared-v3"


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


def _parse_csv_values(value: str, caster, field_name: str) -> tuple:
    try:
        parsed = tuple(caster(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{field_name} must be a comma-separated list") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{field_name} must contain at least one value")
    return parsed


def _parse_int_grid(value: str) -> tuple[int, ...]:
    return _parse_csv_values(value, int, "grid")


def _parse_float_grid(value: str) -> tuple[float, ...]:
    return _parse_csv_values(value, float, "grid")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _run_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    config = MomentumRotationConfig(
        top_n=args.top_n,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        min_rolling_7d_quote_volume=args.min_rolling_7d_quote_volume,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    arrays, canonical_manifests = prepare_from_canonical(
        Path(args.root),
        config,
        offline=args.offline,
        use_cache=not args.no_cache,
    )
    arrays = filter_by_rebalance_date(
        arrays,
        start_date=config.start_date,
        end_date=config.end_date,
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


def _scan_configs(args: argparse.Namespace) -> list[MomentumRotationConfig]:
    configs = []
    for top_n, fee_bps, slippage_bps, min_volume in itertools.product(
        args.top_n,
        args.fee_bps,
        args.slippage_bps,
        args.min_rolling_7d_quote_volume,
    ):
        configs.append(
            MomentumRotationConfig(
                top_n=top_n,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                min_rolling_7d_quote_volume=min_volume,
            )
        )
    return configs


def _signal_cache_key(config: MomentumRotationConfig) -> tuple[float, float, float, float, int, int, int]:
    return (
        config.min_rolling_7d_quote_volume,
        config.short_return_weight,
        config.medium_return_weight,
        config.long_return_weight,
        config.short_window_hours,
        config.medium_window_days,
        config.long_window_days,
    )


def _scan_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    configs = _scan_configs(args)
    base_config = configs[0]
    arrays, canonical_manifests = prepare_from_canonical(
        Path(args.root),
        base_config,
        offline=args.offline,
        use_cache=not args.no_cache,
    )
    arrays, _holdout, data_split = split_research_and_holdout(
        arrays,
        holdout_days=args.holdout_days,
    )
    scan_id = args.scan_id or uuid.uuid4().hex
    rows = []
    signal_cache = {}
    for config in configs:
        signal_key = _signal_cache_key(config)
        signals = signal_cache.get(signal_key)
        if signals is None:
            signals = compute_momentum_signals(arrays, config)
            signal_cache[signal_key] = signals
        result = run_backtest(arrays, signals, config)
        row = {
            "scan_id": scan_id,
            "config_hash": config.config_hash(),
            "top_n": config.top_n,
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "min_rolling_7d_quote_volume": config.min_rolling_7d_quote_volume,
        }
        row.update(build_backtest_summary(result))
        rows.append(row)
    runtime_seconds = time.perf_counter() - started
    output = write_scan_artifacts(
        paths=MomentumRotationPaths(root=Path(args.root)),
        scan_id=scan_id,
        base_config=base_config,
        rows=rows,
        canonical_manifests=canonical_manifests,
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
        symbol_count=len(arrays.symbols),
        data_split=data_split,
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
    run_parser.add_argument("--start-date", type=_parse_date)
    run_parser.add_argument("--end-date", type=_parse_date)
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

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--root", default="data")
    scan_parser.add_argument("--scan-id")
    scan_parser.add_argument("--top-n", type=_parse_int_grid, default=(10,))
    scan_parser.add_argument("--fee-bps", type=_parse_float_grid, default=(5.0,))
    scan_parser.add_argument("--slippage-bps", type=_parse_float_grid, default=(5.0,))
    scan_parser.add_argument(
        "--min-rolling-7d-quote-volume",
        type=_parse_float_grid,
        default=(0.0,),
    )
    scan_parser.add_argument(
        "--holdout-days",
        type=int,
        default=180,
        help="Reserve the most recent N days from parameter scans for final production testing",
    )
    scan_parser.add_argument(
        "--offline",
        action="store_true",
        help="Only read existing canonical Parquet manifests; never connect to ClickHouse or export",
    )
    scan_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild prepared arrays even if a prepared cache entry exists",
    )
    scan_parser.set_defaults(func=_scan_command)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
