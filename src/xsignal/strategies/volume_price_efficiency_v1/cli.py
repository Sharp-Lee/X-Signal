from __future__ import annotations

import argparse
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

from xsignal.strategies.volume_price_efficiency_v1.artifacts import (
    write_run_artifacts,
    write_scan_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.baseline import build_baseline_events
from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import (
    load_offline_ohlcv_table,
    prepare_ohlcv_arrays,
)
from xsignal.strategies.volume_price_efficiency_v1.events import build_event_rows
from xsignal.strategies.volume_price_efficiency_v1.features import (
    FeatureArrays,
    build_signal_mask,
    compute_features,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.scan import (
    build_bucket_summary_rows,
    build_scan_configs,
    build_scan_row,
    select_top_configs,
)
from xsignal.strategies.volume_price_efficiency_v1.splits import (
    holdout_mask_for_open_times,
    split_research_and_holdout,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing import (
    simulate_trailing_stop,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_diagnostics import (
    build_trailing_bucket_summary_rows,
    build_trailing_time_summary_rows,
    enrich_trades_with_market_context,
    write_trailing_diagnostic_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_artifacts import (
    write_trailing_scan_artifacts,
    write_trailing_run_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_scan import (
    build_trailing_scan_row,
    select_top_trailing_configs,
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


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return parsed


def _scan_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    configs = build_scan_configs(
        efficiency_percentiles=args.efficiency_percentile,
        min_move_units=args.min_move_unit,
        min_volume_units=args.min_volume_unit,
        min_close_positions=args.min_close_position,
        min_body_ratios=args.min_body_ratio,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        baseline_seed=args.baseline_seed,
    )
    base_config = configs[0]
    if args.ranking_horizon not in base_config.horizons:
        raise ValueError("ranking_horizon must be one of the configured horizons")

    table, manifests = load_offline_ohlcv_table(Path(args.root), fill_policy=base_config.fill_policy)
    arrays = prepare_ohlcv_arrays(table)
    research_arrays, _holdout_arrays, data_split = split_research_and_holdout(
        arrays,
        holdout_days=args.holdout_days,
    )

    scan_id = args.scan_id or uuid.uuid4().hex
    rows = []
    bucket_rows = []
    feature_cache = {}
    for config in configs:
        feature_key = (
            config.atr_window,
            config.volume_window,
            config.efficiency_lookback,
            config.efficiency_percentile,
            config.volume_floor,
        )
        cached_features = feature_cache.get(feature_key)
        if cached_features is None:
            cached_features = compute_features(research_arrays, config)
            feature_cache[feature_key] = cached_features
        features = replace(
            cached_features,
            signal=build_signal_mask(research_arrays, cached_features, config),
        )
        events = build_event_rows(research_arrays, features, config)
        baseline_events = build_baseline_events(research_arrays, features, config)
        rows.append(
            build_scan_row(
                scan_id=scan_id,
                config=config,
                events=events,
                baseline_events=baseline_events,
                symbols=research_arrays.symbols,
                ranking_horizon=args.ranking_horizon,
            )
        )
        bucket_rows.extend(
            build_bucket_summary_rows(
                config=config,
                events=events,
                horizons=config.horizons,
                bucket_count=args.bucket_count,
            )
        )

    top_configs = select_top_configs(rows, top_k=args.top_k)
    runtime_seconds = time.perf_counter() - started
    output = write_scan_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        scan_id=scan_id,
        base_config=base_config,
        rows=rows,
        top_configs=top_configs,
        bucket_rows=bucket_rows,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
        symbol_count=len(research_arrays.symbols),
        data_split=data_split,
    )
    print(output)
    return output


def _trail_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    if args.atr_multiplier != 2.0:
        raise ValueError("atr_multiplier must stay fixed at 2.0 for the holdout test")
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
    _research_arrays, holdout_arrays, data_split = split_research_and_holdout(
        arrays,
        holdout_days=args.holdout_days,
    )
    if holdout_arrays is None:
        raise ValueError("trail requires a positive holdout window")
    full_features = compute_features(arrays, config)
    holdout_mask = holdout_mask_for_open_times(arrays.open_times, holdout_days=args.holdout_days)
    features = FeatureArrays(
        true_range=full_features.true_range[holdout_mask],
        atr=full_features.atr[holdout_mask],
        move_unit=full_features.move_unit[holdout_mask],
        volume_baseline=full_features.volume_baseline[holdout_mask],
        volume_unit=full_features.volume_unit[holdout_mask],
        efficiency=full_features.efficiency[holdout_mask],
        efficiency_threshold=full_features.efficiency_threshold[holdout_mask],
        close_position=full_features.close_position[holdout_mask],
        body_ratio=full_features.body_ratio[holdout_mask],
        signal=full_features.signal[holdout_mask],
    )
    result = simulate_trailing_stop(
        holdout_arrays,
        features,
        config,
        atr_multiplier=args.atr_multiplier,
    )
    runtime_seconds = time.perf_counter() - started
    run_id = args.run_id or uuid.uuid4().hex
    output = write_trailing_run_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        run_id=run_id,
        config=config,
        result=result,
        symbols=holdout_arrays.symbols,
        open_times=holdout_arrays.open_times,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
        data_split=data_split,
        atr_multiplier=args.atr_multiplier,
    )
    print(output)
    return output


def _trail_scan_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    if args.atr_multiplier != 2.0:
        raise ValueError("atr_multiplier must stay fixed at 2.0 for research trail-scan")
    configs = build_scan_configs(
        efficiency_percentiles=args.efficiency_percentile,
        min_move_units=args.min_move_unit,
        min_volume_units=args.min_volume_unit,
        min_close_positions=args.min_close_position,
        min_body_ratios=args.min_body_ratio,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        baseline_seed=args.baseline_seed,
    )
    base_config = configs[0]
    table, manifests = load_offline_ohlcv_table(Path(args.root), fill_policy=base_config.fill_policy)
    arrays = prepare_ohlcv_arrays(table)
    research_arrays, _holdout_arrays, data_split = split_research_and_holdout(
        arrays,
        holdout_days=args.holdout_days,
    )

    scan_id = args.scan_id or uuid.uuid4().hex
    rows = []
    feature_cache = {}
    for config in configs:
        feature_key = (
            config.atr_window,
            config.volume_window,
            config.efficiency_lookback,
            config.efficiency_percentile,
            config.volume_floor,
        )
        cached_features = feature_cache.get(feature_key)
        if cached_features is None:
            cached_features = compute_features(research_arrays, config)
            feature_cache[feature_key] = cached_features
        features = replace(
            cached_features,
            signal=build_signal_mask(research_arrays, cached_features, config),
        )
        result = simulate_trailing_stop(
            research_arrays,
            features,
            config,
            atr_multiplier=args.atr_multiplier,
        )
        rows.append(
            build_trailing_scan_row(
                scan_id=scan_id,
                config=config,
                result=result,
                symbols=research_arrays.symbols,
                atr_multiplier=args.atr_multiplier,
            )
        )

    top_configs = select_top_trailing_configs(
        rows,
        top_k=args.top_k,
        min_trades=args.min_trades,
    )
    runtime_seconds = time.perf_counter() - started
    output = write_trailing_scan_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        scan_id=scan_id,
        base_config=base_config,
        rows=rows,
        top_configs=top_configs,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
        symbol_count=len(research_arrays.symbols),
        data_split=data_split,
        atr_multiplier=args.atr_multiplier,
        min_trades=args.min_trades,
    )
    print(output)
    return output


def _trail_diagnose_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    if args.atr_multiplier != 2.0:
        raise ValueError("atr_multiplier must stay fixed at 2.0 for trail diagnostics")
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
    research_arrays, holdout_arrays, data_split = split_research_and_holdout(
        arrays,
        holdout_days=args.holdout_days,
    )
    if holdout_arrays is None:
        raise ValueError("trail-diagnose requires a positive holdout window")

    research_base = compute_features(research_arrays, config)
    research_features = replace(
        research_base,
        signal=build_signal_mask(research_arrays, research_base, config),
    )
    research_result = simulate_trailing_stop(
        research_arrays,
        research_features,
        config,
        atr_multiplier=args.atr_multiplier,
    )

    full_features = compute_features(arrays, config)
    holdout_mask = holdout_mask_for_open_times(arrays.open_times, holdout_days=args.holdout_days)
    holdout_features = FeatureArrays(
        true_range=full_features.true_range[holdout_mask],
        atr=full_features.atr[holdout_mask],
        move_unit=full_features.move_unit[holdout_mask],
        volume_baseline=full_features.volume_baseline[holdout_mask],
        volume_unit=full_features.volume_unit[holdout_mask],
        efficiency=full_features.efficiency[holdout_mask],
        efficiency_threshold=full_features.efficiency_threshold[holdout_mask],
        close_position=full_features.close_position[holdout_mask],
        body_ratio=full_features.body_ratio[holdout_mask],
        signal=full_features.signal[holdout_mask],
    )
    holdout_result = simulate_trailing_stop(
        holdout_arrays,
        holdout_features,
        config,
        atr_multiplier=args.atr_multiplier,
    )

    research_trades = enrich_trades_with_market_context(
        research_result.trades,
        research_arrays,
        lookback_bars=args.lookback_bars,
    )
    holdout_trades = enrich_trades_with_market_context(
        holdout_result.trades,
        holdout_arrays,
        lookback_bars=args.lookback_bars,
    )
    time_rows = [
        *build_trailing_time_summary_rows(research_trades, data_set="research"),
        *build_trailing_time_summary_rows(holdout_trades, data_set="holdout"),
    ]
    bucket_rows = [
        *build_trailing_bucket_summary_rows(research_trades, data_set="research"),
        *build_trailing_bucket_summary_rows(holdout_trades, data_set="holdout"),
    ]
    runtime_seconds = time.perf_counter() - started
    diagnostic_id = args.diagnostic_id or uuid.uuid4().hex
    output = write_trailing_diagnostic_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        diagnostic_id=diagnostic_id,
        config=config,
        time_rows=time_rows,
        bucket_rows=bucket_rows,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
        data_split=data_split,
        atr_multiplier=args.atr_multiplier,
        lookback_bars=args.lookback_bars,
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

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--root", default="data")
    scan_parser.add_argument("--scan-id")
    scan_parser.add_argument("--offline", action="store_true", required=True)
    scan_parser.add_argument("--efficiency-percentile", type=_parse_float_grid, default=(0.90, 0.95))
    scan_parser.add_argument("--min-move-unit", type=_parse_float_grid, default=(0.5, 0.8))
    scan_parser.add_argument("--min-volume-unit", type=_parse_float_grid, default=(0.3,))
    scan_parser.add_argument("--min-close-position", type=_parse_float_grid, default=(0.7, 0.8))
    scan_parser.add_argument("--min-body-ratio", type=_parse_float_grid, default=(0.4,))
    scan_parser.add_argument("--fee-bps", type=float, default=5.0)
    scan_parser.add_argument("--slippage-bps", type=float, default=5.0)
    scan_parser.add_argument("--baseline-seed", type=int, default=17)
    scan_parser.add_argument("--holdout-days", type=int, default=180)
    scan_parser.add_argument("--ranking-horizon", type=int, default=30)
    scan_parser.add_argument("--top-k", type=int, default=20)
    scan_parser.add_argument("--bucket-count", type=int, default=5)
    scan_parser.set_defaults(func=_scan_command)

    trail_parser = subparsers.add_parser("trail")
    trail_parser.add_argument("--root", default="data")
    trail_parser.add_argument("--run-id")
    trail_parser.add_argument("--offline", action="store_true", required=True)
    trail_parser.add_argument("--atr-window", type=int, default=14)
    trail_parser.add_argument("--volume-window", type=int, default=60)
    trail_parser.add_argument("--efficiency-lookback", type=int, default=120)
    trail_parser.add_argument("--efficiency-percentile", type=float, default=0.90)
    trail_parser.add_argument("--volume-floor", type=float, default=0.2)
    trail_parser.add_argument("--min-move-unit", type=float, default=0.5)
    trail_parser.add_argument("--min-volume-unit", type=float, default=0.3)
    trail_parser.add_argument("--min-close-position", type=float, default=0.7)
    trail_parser.add_argument("--min-body-ratio", type=float, default=0.4)
    trail_parser.add_argument("--fee-bps", type=float, default=5.0)
    trail_parser.add_argument("--slippage-bps", type=float, default=5.0)
    trail_parser.add_argument("--baseline-seed", type=int, default=17)
    trail_parser.add_argument("--holdout-days", type=int, default=180)
    trail_parser.add_argument("--atr-multiplier", type=float, default=2.0)
    trail_parser.set_defaults(func=_trail_command)

    trail_scan_parser = subparsers.add_parser("trail-scan")
    trail_scan_parser.add_argument("--root", default="data")
    trail_scan_parser.add_argument("--scan-id")
    trail_scan_parser.add_argument("--offline", action="store_true", required=True)
    trail_scan_parser.add_argument("--efficiency-percentile", type=_parse_float_grid, default=(0.90, 0.95))
    trail_scan_parser.add_argument("--min-move-unit", type=_parse_float_grid, default=(0.5, 0.8))
    trail_scan_parser.add_argument("--min-volume-unit", type=_parse_float_grid, default=(0.3,))
    trail_scan_parser.add_argument("--min-close-position", type=_parse_float_grid, default=(0.7, 0.8))
    trail_scan_parser.add_argument("--min-body-ratio", type=_parse_float_grid, default=(0.4,))
    trail_scan_parser.add_argument("--fee-bps", type=float, default=5.0)
    trail_scan_parser.add_argument("--slippage-bps", type=float, default=5.0)
    trail_scan_parser.add_argument("--baseline-seed", type=int, default=17)
    trail_scan_parser.add_argument("--holdout-days", type=int, default=180)
    trail_scan_parser.add_argument("--atr-multiplier", type=float, default=2.0)
    trail_scan_parser.add_argument("--top-k", type=int, default=20)
    trail_scan_parser.add_argument("--min-trades", type=int, default=200)
    trail_scan_parser.set_defaults(func=_trail_scan_command)

    trail_diagnose_parser = subparsers.add_parser("trail-diagnose")
    trail_diagnose_parser.add_argument("--root", default="data")
    trail_diagnose_parser.add_argument("--diagnostic-id")
    trail_diagnose_parser.add_argument("--offline", action="store_true", required=True)
    trail_diagnose_parser.add_argument("--atr-window", type=int, default=14)
    trail_diagnose_parser.add_argument("--volume-window", type=int, default=60)
    trail_diagnose_parser.add_argument("--efficiency-lookback", type=int, default=120)
    trail_diagnose_parser.add_argument("--efficiency-percentile", type=float, default=0.90)
    trail_diagnose_parser.add_argument("--volume-floor", type=float, default=0.2)
    trail_diagnose_parser.add_argument("--min-move-unit", type=float, default=0.5)
    trail_diagnose_parser.add_argument("--min-volume-unit", type=float, default=0.3)
    trail_diagnose_parser.add_argument("--min-close-position", type=float, default=0.7)
    trail_diagnose_parser.add_argument("--min-body-ratio", type=float, default=0.4)
    trail_diagnose_parser.add_argument("--fee-bps", type=float, default=5.0)
    trail_diagnose_parser.add_argument("--slippage-bps", type=float, default=5.0)
    trail_diagnose_parser.add_argument("--baseline-seed", type=int, default=17)
    trail_diagnose_parser.add_argument("--holdout-days", type=int, default=180)
    trail_diagnose_parser.add_argument("--atr-multiplier", type=float, default=2.0)
    trail_diagnose_parser.add_argument("--lookback-bars", type=int, default=30)
    trail_diagnose_parser.set_defaults(func=_trail_diagnose_command)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
