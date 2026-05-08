from __future__ import annotations

from typing import Any

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing import (
    TrailingStopResult,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_artifacts import (
    build_trailing_summary,
)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def build_trailing_scan_row(
    *,
    scan_id: str,
    config: VolumePriceEfficiencyConfig,
    result: TrailingStopResult,
    symbols: tuple[str, ...],
    atr_multiplier: float = 2.0,
) -> dict[str, Any]:
    summary = build_trailing_summary(result)
    total_return = summary["total_return"]
    max_drawdown = summary["max_drawdown"]
    score = None
    if total_return is not None and max_drawdown is not None:
        score = float(total_return) - float(max_drawdown)
    return {
        "scan_id": scan_id,
        "config_hash": config.config_hash(),
        "efficiency_percentile": config.efficiency_percentile,
        "signal_mode": config.signal_mode,
        "min_move_unit": config.min_move_unit,
        "min_volume_unit": config.min_volume_unit,
        "min_close_position": config.min_close_position,
        "min_body_ratio": config.min_body_ratio,
        "seed_efficiency_lookback": config.seed_efficiency_lookback,
        "seed_min_efficiency_ratio_to_max": config.seed_min_efficiency_ratio_to_max,
        "seed_min_efficiency_ratio_to_mean": config.seed_min_efficiency_ratio_to_mean,
        "seed_max_volume_unit": config.seed_max_volume_unit,
        "seed_bottom_lookback": config.seed_bottom_lookback,
        "seed_max_close_position_in_range": config.seed_max_close_position_in_range,
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "baseline_seed": config.baseline_seed,
        "atr_multiplier": atr_multiplier,
        "symbol_count": len(symbols),
        "trade_count": summary["trade_count"],
        "win_rate": summary["win_rate"],
        "mean_net_realized_return": summary["mean_net_realized_return"],
        "median_net_realized_return": summary["median_net_realized_return"],
        "average_holding_bars": summary["average_holding_bars"],
        "total_ignored_signal_count": summary["total_ignored_signal_count"],
        "final_equity": summary["final_equity"],
        "total_return": summary["total_return"],
        "max_drawdown": summary["max_drawdown"],
        "score": _rounded(score),
    }


def select_top_trailing_configs(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    min_trades: int,
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if int(row.get("trade_count") or 0) >= min_trades]
    return sorted(
        eligible,
        key=lambda row: (
            -(float(row["score"]) if row.get("score") is not None else float("-inf")),
            str(row["config_hash"]),
        ),
    )[:top_k]
