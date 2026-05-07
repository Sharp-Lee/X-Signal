from __future__ import annotations

import argparse
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

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
from xsignal.strategies.volume_price_efficiency_v1.trailing_walk_forward import (
    build_walk_forward_fold_row,
    build_walk_forward_folds,
    slice_feature_arrays,
    slice_ohlcv_arrays,
    write_trailing_walk_forward_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_scan import (
    DEFAULT_REGIME_FEATURES,
    apply_regime_filter_rule,
    build_regime_filter_rules,
    build_regime_scan_row,
    build_regime_value_arrays,
    select_top_regime_filters,
    write_trailing_regime_scan_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_walk_forward import (
    build_regime_walk_forward_fold_row,
    build_regime_stability_summary,
    select_stable_regime_filters,
    write_trailing_regime_walk_forward_artifacts,
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


def _parse_string_grid(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
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


def _feature_cache_key(config: VolumePriceEfficiencyConfig) -> tuple[int, int, int, float, float]:
    return (
        config.atr_window,
        config.volume_window,
        config.efficiency_lookback,
        config.efficiency_percentile,
        config.volume_floor,
    )


def _signal_features(
    arrays,
    config: VolumePriceEfficiencyConfig,
    feature_cache: dict[tuple[int, int, int, float, float], FeatureArrays],
) -> FeatureArrays:
    feature_key = _feature_cache_key(config)
    cached_features = feature_cache.get(feature_key)
    if cached_features is None:
        cached_features = compute_features(arrays, config)
        feature_cache[feature_key] = cached_features
    return replace(
        cached_features,
        signal=build_signal_mask(arrays, cached_features, config),
    )


def _trail_walk_forward_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    if args.atr_multiplier != 2.0:
        raise ValueError("atr_multiplier must stay fixed at 2.0 for research trail-walk-forward")
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
    config_by_hash = {config.config_hash(): config for config in configs}
    table, manifests = load_offline_ohlcv_table(Path(args.root), fill_policy=base_config.fill_policy)
    arrays = prepare_ohlcv_arrays(table)
    research_arrays, _holdout_arrays, data_split = split_research_and_holdout(
        arrays,
        holdout_days=args.holdout_days,
    )
    folds = build_walk_forward_folds(
        research_arrays.open_times,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )

    walk_forward_id = args.walk_forward_id or uuid.uuid4().hex
    fold_rows = []
    selection_rows = []
    selected_train_rows = []
    research_feature_cache: dict[tuple[int, int, int, float, float], FeatureArrays] = {}
    research_signal_cache: dict[str, FeatureArrays] = {}
    for fold_index, fold in enumerate(folds):
        train_arrays = slice_ohlcv_arrays(research_arrays, fold.train_indices)
        train_rows = []
        for config in configs:
            features = research_signal_cache.get(config.config_hash())
            if features is None:
                features = _signal_features(research_arrays, config, research_feature_cache)
                research_signal_cache[config.config_hash()] = features
            train_features = slice_feature_arrays(features, fold.train_indices)
            result = simulate_trailing_stop(
                train_arrays,
                train_features,
                config,
                atr_multiplier=args.atr_multiplier,
            )
            row = build_trailing_scan_row(
                scan_id=walk_forward_id,
                config=config,
                result=result,
                symbols=train_arrays.symbols,
                atr_multiplier=args.atr_multiplier,
            )
            row.update(
                {
                    "walk_forward_id": walk_forward_id,
                    "fold_index": fold_index,
                    "train_start": fold.train_start.isoformat(),
                    "train_end": fold.train_end.isoformat(),
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                    "is_selected": False,
                }
            )
            train_rows.append(row)

        selected = select_top_trailing_configs(
            train_rows,
            top_k=1,
            min_trades=args.min_trades,
        )
        selected_train_row = selected[0] if selected else None
        selected_config = None
        validation_row = None
        if selected_train_row is not None:
            selected_train_row["is_selected"] = True
            selected_train_rows.append(selected_train_row)
            selected_config = config_by_hash[str(selected_train_row["config_hash"])]
            selected_features = research_signal_cache.get(selected_config.config_hash())
            if selected_features is None:
                selected_features = _signal_features(
                    research_arrays,
                    selected_config,
                    research_feature_cache,
                )
                research_signal_cache[selected_config.config_hash()] = selected_features
            validation_arrays = slice_ohlcv_arrays(research_arrays, fold.test_indices)
            validation_features = slice_feature_arrays(selected_features, fold.test_indices)
            validation_result = simulate_trailing_stop(
                validation_arrays,
                validation_features,
                selected_config,
                atr_multiplier=args.atr_multiplier,
            )
            validation_row = build_trailing_scan_row(
                scan_id=walk_forward_id,
                config=selected_config,
                result=validation_result,
                symbols=validation_arrays.symbols,
                atr_multiplier=args.atr_multiplier,
            )

        selection_rows.extend(train_rows)
        fold_rows.append(
            build_walk_forward_fold_row(
                walk_forward_id=walk_forward_id,
                fold_index=fold_index,
                fold=fold,
                selected_train_row=selected_train_row,
                validation_row=validation_row,
                selected_config=selected_config,
            )
        )

    top_configs = select_top_trailing_configs(
        selected_train_rows,
        top_k=args.top_k,
        min_trades=args.min_trades,
    )
    runtime_seconds = time.perf_counter() - started
    output = write_trailing_walk_forward_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        walk_forward_id=walk_forward_id,
        base_config=base_config,
        fold_rows=fold_rows,
        selection_rows=selection_rows,
        top_configs=top_configs,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
        symbol_count=len(research_arrays.symbols),
        data_split=data_split,
        atr_multiplier=args.atr_multiplier,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        min_trades=args.min_trades,
    )
    print(output)
    return output


def _trail_regime_scan_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    if args.atr_multiplier != 2.0:
        raise ValueError("atr_multiplier must stay fixed at 2.0 for research trail-regime-scan")
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
    research_arrays, _holdout_arrays, data_split = split_research_and_holdout(
        arrays,
        holdout_days=args.holdout_days,
    )
    base_features = compute_features(research_arrays, config)
    features = replace(
        base_features,
        signal=build_signal_mask(research_arrays, base_features, config),
    )
    values_by_feature = build_regime_value_arrays(
        research_arrays,
        features,
        lookback_bars=args.lookback_bars,
    )
    rules = build_regime_filter_rules(
        features.signal,
        values_by_feature,
        feature_names=args.feature_name,
        quantiles=args.quantile,
    )
    regime_scan_id = args.regime_scan_id or uuid.uuid4().hex
    base_signal_count = int(features.signal.sum())
    base_result = simulate_trailing_stop(
        research_arrays,
        features,
        config,
        atr_multiplier=args.atr_multiplier,
    )
    rows = [
        build_regime_scan_row(
            regime_scan_id=regime_scan_id,
            config=config,
            result=base_result,
            symbols=research_arrays.symbols,
            rule=None,
            base_signal_count=base_signal_count,
            filtered_signal_count=base_signal_count,
            atr_multiplier=args.atr_multiplier,
        )
    ]
    for rule in rules:
        filtered_signal = apply_regime_filter_rule(features.signal, values_by_feature, rule)
        filtered_features = replace(features, signal=filtered_signal)
        result = simulate_trailing_stop(
            research_arrays,
            filtered_features,
            config,
            atr_multiplier=args.atr_multiplier,
        )
        rows.append(
            build_regime_scan_row(
                regime_scan_id=regime_scan_id,
                config=config,
                result=result,
                symbols=research_arrays.symbols,
                rule=rule,
                base_signal_count=base_signal_count,
                filtered_signal_count=int(filtered_signal.sum()),
                atr_multiplier=args.atr_multiplier,
            )
        )

    top_filters = select_top_regime_filters(
        rows,
        top_k=args.top_k,
        min_trades=args.min_trades,
    )
    runtime_seconds = time.perf_counter() - started
    output = write_trailing_regime_scan_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        regime_scan_id=regime_scan_id,
        config=config,
        rows=rows,
        top_filters=top_filters,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
        symbol_count=len(research_arrays.symbols),
        data_split=data_split,
        atr_multiplier=args.atr_multiplier,
        lookback_bars=args.lookback_bars,
        quantiles=args.quantile,
        feature_names=args.feature_name,
    )
    print(output)
    return output


def _slice_regime_values(
    values_by_feature: dict[str, object],
    indices,
) -> dict[str, object]:
    indexer = indices if isinstance(indices, np.ndarray) else np.array(indices, dtype=np.int64)
    return {name: values[indexer] for name, values in values_by_feature.items()}


def _stability_segments(row_count: int, split_count: int) -> list[tuple[int, ...]]:
    if split_count <= 1:
        return []
    indices = np.arange(row_count, dtype=np.int64)
    return [
        tuple(int(index) for index in segment)
        for segment in np.array_split(indices, split_count)
        if segment.size
    ]


def _stability_min_trades(args: argparse.Namespace) -> int:
    if args.stability_min_trades > 0:
        return args.stability_min_trades
    if args.stability_splits <= 1:
        return 0
    return max(1, args.min_trades // args.stability_splits)


def _stability_min_positive_splits(args: argparse.Namespace) -> int:
    if args.stability_min_positive_splits > 0:
        return args.stability_min_positive_splits
    if args.stability_splits <= 1:
        return 0
    return max(1, (args.stability_splits + 1) // 2)


def _trail_regime_walk_forward_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    if args.atr_multiplier != 2.0:
        raise ValueError(
            "atr_multiplier must stay fixed at 2.0 for research trail-regime-walk-forward"
        )
    if args.stability_splits <= 0:
        raise ValueError("stability_splits must be positive")
    if args.stability_min_trades < 0:
        raise ValueError("stability_min_trades must be non-negative")
    if args.stability_min_positive_splits < 0:
        raise ValueError("stability_min_positive_splits must be non-negative")
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
    research_arrays, _holdout_arrays, data_split = split_research_and_holdout(
        arrays,
        holdout_days=args.holdout_days,
    )
    base_features = compute_features(research_arrays, config)
    features = replace(
        base_features,
        signal=build_signal_mask(research_arrays, base_features, config),
    )
    values_by_feature = build_regime_value_arrays(
        research_arrays,
        features,
        lookback_bars=args.lookback_bars,
    )
    folds = build_walk_forward_folds(
        research_arrays.open_times,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )

    regime_walk_forward_id = args.regime_walk_forward_id or uuid.uuid4().hex
    fold_rows = []
    selection_rows = []
    selected_train_rows = []
    stability_min_trades = _stability_min_trades(args)
    stability_min_positive_splits = _stability_min_positive_splits(args)
    for fold_index, fold in enumerate(folds):
        train_arrays = slice_ohlcv_arrays(research_arrays, fold.train_indices)
        train_features = slice_feature_arrays(features, fold.train_indices)
        train_values = _slice_regime_values(values_by_feature, fold.train_indices)
        stability_segments = _stability_segments(train_arrays.open.shape[0], args.stability_splits)
        train_rules = build_regime_filter_rules(
            train_features.signal,
            train_values,
            feature_names=args.feature_name,
            quantiles=args.quantile,
        )
        rule_by_id = {rule.rule_id: rule for rule in train_rules}
        train_base_signal_count = int(train_features.signal.sum())
        train_rows = []
        for rule in train_rules:
            filtered_train_signal = apply_regime_filter_rule(
                train_features.signal,
                train_values,
                rule,
            )
            result = simulate_trailing_stop(
                train_arrays,
                replace(train_features, signal=filtered_train_signal),
                config,
                atr_multiplier=args.atr_multiplier,
            )
            row = build_regime_scan_row(
                regime_scan_id=regime_walk_forward_id,
                config=config,
                result=result,
                symbols=train_arrays.symbols,
                rule=rule,
                base_signal_count=train_base_signal_count,
                filtered_signal_count=int(filtered_train_signal.sum()),
                atr_multiplier=args.atr_multiplier,
            )
            row.update(
                {
                    "regime_walk_forward_id": regime_walk_forward_id,
                    "fold_index": fold_index,
                    "train_start": fold.train_start.isoformat(),
                    "train_end": fold.train_end.isoformat(),
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                    "threshold_source": "train_fold_signal_distribution",
                    "is_selected": False,
                }
            )
            if stability_segments:
                segment_rows = []
                for segment_index, segment_indices in enumerate(stability_segments):
                    segment_arrays = slice_ohlcv_arrays(train_arrays, segment_indices)
                    segment_features = slice_feature_arrays(train_features, segment_indices)
                    segment_values = _slice_regime_values(train_values, segment_indices)
                    filtered_segment_signal = apply_regime_filter_rule(
                        segment_features.signal,
                        segment_values,
                        rule,
                    )
                    segment_result = simulate_trailing_stop(
                        segment_arrays,
                        replace(segment_features, signal=filtered_segment_signal),
                        config,
                        atr_multiplier=args.atr_multiplier,
                    )
                    segment_row = build_regime_scan_row(
                        regime_scan_id=regime_walk_forward_id,
                        config=config,
                        result=segment_result,
                        symbols=segment_arrays.symbols,
                        rule=rule,
                        base_signal_count=int(segment_features.signal.sum()),
                        filtered_signal_count=int(filtered_segment_signal.sum()),
                        atr_multiplier=args.atr_multiplier,
                    )
                    segment_row.update(
                        {
                            "regime_walk_forward_id": regime_walk_forward_id,
                            "fold_index": fold_index,
                            "stability_segment_index": segment_index,
                        }
                    )
                    segment_rows.append(segment_row)
                row.update(
                    build_regime_stability_summary(
                        segment_rows,
                        split_count=args.stability_splits,
                        min_trades=stability_min_trades,
                    )
                )
            train_rows.append(row)

        if args.stability_splits > 1:
            selected = select_stable_regime_filters(
                train_rows,
                top_k=1,
                min_trades=args.min_trades,
                stability_min_positive_splits=stability_min_positive_splits,
            )
        else:
            selected = select_top_regime_filters(
                train_rows,
                top_k=1,
                min_trades=args.min_trades,
            )
        selected_train_row = selected[0] if selected else None
        selected_rule = None
        validation_row = None
        if selected_train_row is not None:
            selected_train_row["is_selected"] = True
            selected_train_rows.append(selected_train_row)
            selected_rule = rule_by_id[str(selected_train_row["rule_id"])]
            validation_arrays = slice_ohlcv_arrays(research_arrays, fold.test_indices)
            validation_features = slice_feature_arrays(features, fold.test_indices)
            validation_values = _slice_regime_values(values_by_feature, fold.test_indices)
            filtered_validation_signal = apply_regime_filter_rule(
                validation_features.signal,
                validation_values,
                selected_rule,
            )
            validation_result = simulate_trailing_stop(
                validation_arrays,
                replace(validation_features, signal=filtered_validation_signal),
                config,
                atr_multiplier=args.atr_multiplier,
            )
            validation_row = build_regime_scan_row(
                regime_scan_id=regime_walk_forward_id,
                config=config,
                result=validation_result,
                symbols=validation_arrays.symbols,
                rule=selected_rule,
                base_signal_count=int(validation_features.signal.sum()),
                filtered_signal_count=int(filtered_validation_signal.sum()),
                atr_multiplier=args.atr_multiplier,
            )

        selection_rows.extend(train_rows)
        fold_rows.append(
            build_regime_walk_forward_fold_row(
                regime_walk_forward_id=regime_walk_forward_id,
                fold_index=fold_index,
                fold=fold,
                selected_train_row=selected_train_row,
                validation_row=validation_row,
                selected_rule=selected_rule,
                selected_config=config,
            )
        )

    if args.stability_splits > 1:
        top_filters = select_stable_regime_filters(
            selected_train_rows,
            top_k=args.top_k,
            min_trades=args.min_trades,
            stability_min_positive_splits=stability_min_positive_splits,
        )
    else:
        top_filters = select_top_regime_filters(
            selected_train_rows,
            top_k=args.top_k,
            min_trades=args.min_trades,
        )
    runtime_seconds = time.perf_counter() - started
    output = write_trailing_regime_walk_forward_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        regime_walk_forward_id=regime_walk_forward_id,
        config=config,
        fold_rows=fold_rows,
        selection_rows=selection_rows,
        top_filters=top_filters,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
        symbol_count=len(research_arrays.symbols),
        data_split=data_split,
        atr_multiplier=args.atr_multiplier,
        lookback_bars=args.lookback_bars,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        min_trades=args.min_trades,
        stability_splits=args.stability_splits,
        stability_min_trades=stability_min_trades,
        stability_min_positive_splits=stability_min_positive_splits,
        quantiles=args.quantile,
        feature_names=args.feature_name,
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

    trail_walk_forward_parser = subparsers.add_parser("trail-walk-forward")
    trail_walk_forward_parser.add_argument("--root", default="data")
    trail_walk_forward_parser.add_argument("--walk-forward-id")
    trail_walk_forward_parser.add_argument("--offline", action="store_true", required=True)
    trail_walk_forward_parser.add_argument(
        "--efficiency-percentile",
        type=_parse_float_grid,
        default=(0.90, 0.95),
    )
    trail_walk_forward_parser.add_argument("--min-move-unit", type=_parse_float_grid, default=(0.5, 0.8))
    trail_walk_forward_parser.add_argument("--min-volume-unit", type=_parse_float_grid, default=(0.3,))
    trail_walk_forward_parser.add_argument(
        "--min-close-position",
        type=_parse_float_grid,
        default=(0.7, 0.8),
    )
    trail_walk_forward_parser.add_argument("--min-body-ratio", type=_parse_float_grid, default=(0.4,))
    trail_walk_forward_parser.add_argument("--fee-bps", type=float, default=5.0)
    trail_walk_forward_parser.add_argument("--slippage-bps", type=float, default=5.0)
    trail_walk_forward_parser.add_argument("--baseline-seed", type=int, default=17)
    trail_walk_forward_parser.add_argument("--holdout-days", type=int, default=180)
    trail_walk_forward_parser.add_argument("--atr-multiplier", type=float, default=2.0)
    trail_walk_forward_parser.add_argument("--train-days", type=int, default=720)
    trail_walk_forward_parser.add_argument("--test-days", type=int, default=90)
    trail_walk_forward_parser.add_argument("--step-days", type=int, default=90)
    trail_walk_forward_parser.add_argument("--top-k", type=int, default=20)
    trail_walk_forward_parser.add_argument("--min-trades", type=int, default=200)
    trail_walk_forward_parser.set_defaults(func=_trail_walk_forward_command)

    trail_regime_scan_parser = subparsers.add_parser("trail-regime-scan")
    trail_regime_scan_parser.add_argument("--root", default="data")
    trail_regime_scan_parser.add_argument("--regime-scan-id")
    trail_regime_scan_parser.add_argument("--offline", action="store_true", required=True)
    trail_regime_scan_parser.add_argument("--atr-window", type=int, default=14)
    trail_regime_scan_parser.add_argument("--volume-window", type=int, default=60)
    trail_regime_scan_parser.add_argument("--efficiency-lookback", type=int, default=120)
    trail_regime_scan_parser.add_argument("--efficiency-percentile", type=float, default=0.90)
    trail_regime_scan_parser.add_argument("--volume-floor", type=float, default=0.2)
    trail_regime_scan_parser.add_argument("--min-move-unit", type=float, default=0.5)
    trail_regime_scan_parser.add_argument("--min-volume-unit", type=float, default=0.3)
    trail_regime_scan_parser.add_argument("--min-close-position", type=float, default=0.7)
    trail_regime_scan_parser.add_argument("--min-body-ratio", type=float, default=0.4)
    trail_regime_scan_parser.add_argument("--fee-bps", type=float, default=5.0)
    trail_regime_scan_parser.add_argument("--slippage-bps", type=float, default=5.0)
    trail_regime_scan_parser.add_argument("--baseline-seed", type=int, default=17)
    trail_regime_scan_parser.add_argument("--holdout-days", type=int, default=180)
    trail_regime_scan_parser.add_argument("--atr-multiplier", type=float, default=2.0)
    trail_regime_scan_parser.add_argument("--lookback-bars", type=int, default=30)
    trail_regime_scan_parser.add_argument(
        "--feature-name",
        type=_parse_string_grid,
        default=DEFAULT_REGIME_FEATURES,
    )
    trail_regime_scan_parser.add_argument(
        "--quantile",
        type=_parse_float_grid,
        default=(0.2, 0.4, 0.6, 0.8),
    )
    trail_regime_scan_parser.add_argument("--top-k", type=int, default=20)
    trail_regime_scan_parser.add_argument("--min-trades", type=int, default=200)
    trail_regime_scan_parser.set_defaults(func=_trail_regime_scan_command)

    trail_regime_walk_forward_parser = subparsers.add_parser("trail-regime-walk-forward")
    trail_regime_walk_forward_parser.add_argument("--root", default="data")
    trail_regime_walk_forward_parser.add_argument("--regime-walk-forward-id")
    trail_regime_walk_forward_parser.add_argument("--offline", action="store_true", required=True)
    trail_regime_walk_forward_parser.add_argument("--atr-window", type=int, default=14)
    trail_regime_walk_forward_parser.add_argument("--volume-window", type=int, default=60)
    trail_regime_walk_forward_parser.add_argument("--efficiency-lookback", type=int, default=120)
    trail_regime_walk_forward_parser.add_argument("--efficiency-percentile", type=float, default=0.90)
    trail_regime_walk_forward_parser.add_argument("--volume-floor", type=float, default=0.2)
    trail_regime_walk_forward_parser.add_argument("--min-move-unit", type=float, default=0.5)
    trail_regime_walk_forward_parser.add_argument("--min-volume-unit", type=float, default=0.3)
    trail_regime_walk_forward_parser.add_argument("--min-close-position", type=float, default=0.7)
    trail_regime_walk_forward_parser.add_argument("--min-body-ratio", type=float, default=0.4)
    trail_regime_walk_forward_parser.add_argument("--fee-bps", type=float, default=5.0)
    trail_regime_walk_forward_parser.add_argument("--slippage-bps", type=float, default=5.0)
    trail_regime_walk_forward_parser.add_argument("--baseline-seed", type=int, default=17)
    trail_regime_walk_forward_parser.add_argument("--holdout-days", type=int, default=180)
    trail_regime_walk_forward_parser.add_argument("--atr-multiplier", type=float, default=2.0)
    trail_regime_walk_forward_parser.add_argument("--lookback-bars", type=int, default=30)
    trail_regime_walk_forward_parser.add_argument("--train-days", type=int, default=720)
    trail_regime_walk_forward_parser.add_argument("--test-days", type=int, default=90)
    trail_regime_walk_forward_parser.add_argument("--step-days", type=int, default=90)
    trail_regime_walk_forward_parser.add_argument(
        "--feature-name",
        type=_parse_string_grid,
        default=DEFAULT_REGIME_FEATURES,
    )
    trail_regime_walk_forward_parser.add_argument(
        "--quantile",
        type=_parse_float_grid,
        default=(0.2, 0.4, 0.6, 0.8),
    )
    trail_regime_walk_forward_parser.add_argument("--top-k", type=int, default=20)
    trail_regime_walk_forward_parser.add_argument("--min-trades", type=int, default=200)
    trail_regime_walk_forward_parser.add_argument("--stability-splits", type=int, default=1)
    trail_regime_walk_forward_parser.add_argument("--stability-min-trades", type=int, default=0)
    trail_regime_walk_forward_parser.add_argument(
        "--stability-min-positive-splits",
        type=int,
        default=0,
    )
    trail_regime_walk_forward_parser.set_defaults(func=_trail_regime_walk_forward_command)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
