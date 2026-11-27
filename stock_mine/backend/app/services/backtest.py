from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from app.services.flag_pattern import FLAG_DIRECTIONS
from app.services.market_calendar import calendar_info, session_dates
from app.services.market_regime import regime_on
from app.services.validation import VALIDATION_EXECUTION_MODEL, trade_on_calendar

from app.services.strategy import (
    SignalCategory,
    SignalResult,
    StrategyConfig,
    STRATEGY_VERSION,
    version_for_engine,
    _analyze_normalized_bars,
    _normalize_bars,
)


FORWARD_WINDOWS = (5, 10, 20)
EXECUTION_MODEL = VALIDATION_EXECUTION_MODEL
ROUNDTRIP_COST_BPS = 20.0


def _execution_inputs(bars: pd.DataFrame, calendar: list[str] | None = None,
                      as_of: str | None = None) -> tuple[list[str], dict[str, dict]]:
    if bars.empty:
        return [], {}
    stamps = pd.to_datetime(bars["trade_date"]).dt.strftime("%Y-%m-%d")
    end = as_of or stamps.iloc[-1]
    dates = session_dates(stamps.iloc[0], end) if calendar is None else [
        day for day in calendar if as_of is None or day <= as_of
    ]
    if dates != sorted(set(dates)):
        raise ValueError("execution calendar must be sorted and unique")
    rows = bars.assign(trade_date=stamps).to_dict("records")
    return dates, {row["trade_date"]: row for row in rows}


def compute_trade_metrics(
    bars: pd.DataFrame, signal_idx: int,
    windows: tuple[int, ...] = FORWARD_WINDOWS,
    roundtrip_cost_bps: float = ROUNDTRIP_COST_BPS,
    *, calendar: list[str] | None = None, as_of: str | None = None,
    _prices: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Adapt the shared calendar execution ledger to the existing API fields."""
    if not np.isfinite(roundtrip_cost_bps) or roundtrip_cost_bps < 0:
        raise ValueError("roundtrip_cost_bps must be finite and non-negative")
    if _prices is None:
        calendar, _prices = _execution_inputs(bars, calendar, as_of)
    if calendar is None:
        raise ValueError("execution prices require an explicit calendar")
    signal_date = pd.Timestamp(bars["trade_date"].iloc[signal_idx]).strftime("%Y-%m-%d")
    out: dict[str, Any] = {
        "execution_model": EXECUTION_MODEL,
        "roundtrip_cost_bps": roundtrip_cost_bps,
        "entry_status": "pending",
        "entry_date": None,
        "entry_price": None,
    }
    for window in windows:
        out.update({
            f"gross_return_{window}": None, f"net_return_{window}": None,
            f"exit_date_{window}": None, f"exit_delay_{window}": None,
        })
    for window in windows:
        trade = trade_on_calendar("", {"signal_date": signal_date}, _prices, calendar, window, roundtrip_cost_bps)
        out.update({key: trade.get(key) for key in ("entry_status", "entry_date", "entry_price", "entry_reason")})
        net = trade["net_return_pct"]
        out.update({
            f"gross_return_{window}": round(net + roundtrip_cost_bps / 100, 6) if net is not None else None,
            f"net_return_{window}": round(net, 6) if net is not None else None,
            f"exit_date_{window}": trade["exit_date"],
            f"exit_delay_{window}": trade.get("exit_delay_sessions"),
            f"planned_exit_date_{window}": trade.get("planned_exit_date"),
        })
    return out


def dedupe_signal_indices(indices: Iterable[int], min_gap: int = 20) -> list[int]:
    deduped: list[int] = []
    for index in sorted(indices):
        if not deduped or index - deduped[-1] >= min_gap:
            deduped.append(index)
    return deduped


def dedupe_event_signals(raw: dict[int, Any]) -> list[int]:
    """V2 counts independent frozen breakout events, not an arbitrary date cooldown."""
    seen = set()
    kept = []
    for index, result in sorted(raw.items()):
        flags = result.flag_pattern
        key = result.signal_event_id or (
            f"flag:{flags['id']}" if result.pattern_type == "flag_breakout" and flags else f"day:{index}"
        )
        if key not in seen:
            seen.add(key)
            kept.append(index)
    return kept


def compute_forward_metrics(
    bars: pd.DataFrame,
    signal_idx: int,
    windows: tuple[int, ...] = FORWARD_WINDOWS,
    target_pct: float = 0.05,
    stop_pct: float = 0.05,
    roundtrip_cost_bps: float = ROUNDTRIP_COST_BPS,
    *, calendar: list[str] | None = None, as_of: str | None = None,
    _prices: dict[str, dict] | None = None,
) -> dict[str, Any]:
    close = float(bars["close"].iloc[signal_idx])
    metrics = compute_trade_metrics(
        bars, signal_idx, windows, roundtrip_cost_bps, calendar=calendar, as_of=as_of, _prices=_prices,
    )
    metrics["legacy_path_clock"] = "observed_stock_bars"
    for window in windows:
        target_idx = signal_idx + window
        metrics[f"forward_return_{window}"] = (
            round((float(bars["close"].iloc[target_idx]) / close - 1) * 100, 2)
            if close > 0 and target_idx < len(bars)
            else None
        )
    next_open_idx = signal_idx + 1
    week_exit_idx = signal_idx + 5
    if next_open_idx < len(bars) and week_exit_idx < len(bars):
        next_open = float(bars["open"].iloc[next_open_idx])
        week_exit = float(bars["close"].iloc[week_exit_idx])
        metrics["forward_return_5_from_next_open"] = _pct_change(week_exit, next_open)
        metrics["next_open_gap_pct"] = _pct_change(next_open, close)
    else:
        metrics["forward_return_5_from_next_open"] = None
        metrics["next_open_gap_pct"] = None

    first_week = bars.iloc[signal_idx + 1 : signal_idx + 6]
    if len(first_week) == 5 and close > 0:
        week_highs = first_week["high"].astype(float)
        week_lows = first_week["low"].astype(float)
        metrics.update(
            {
                "mfe_5": round((float(week_highs.max()) / close - 1) * 100, 2),
                "mae_5": round((float(week_lows.min()) / close - 1) * 100, 2),
                "target_3pct_5": bool((week_highs >= close * 1.03).any()),
                "stop_3pct_5": bool((week_lows <= close * 0.97).any()),
            }
        )
    else:
        metrics.update({"mfe_5": None, "mae_5": None, "target_3pct_5": None, "stop_3pct_5": None})

    future = bars.iloc[signal_idx + 1 : signal_idx + 21]
    if len(future) < 20:
        metrics.update({"target_5pct_hit": None, "stop_5pct_hit": None, "outcome_20": "incomplete", "mfe_20": None, "mae_20": None})
        return metrics
    highs = future["high"].astype(float).to_numpy()
    lows = future["low"].astype(float).to_numpy()
    target_hits = np.flatnonzero(highs >= close * (1 + target_pct))
    stop_hits = np.flatnonzero(lows <= close * (1 - stop_pct))
    target_hit, stop_hit = len(target_hits) > 0, len(stop_hits) > 0
    if target_hit and (not stop_hit or target_hits[0] < stop_hits[0]):
        outcome = "target_first"
    elif stop_hit and (not target_hit or stop_hits[0] < target_hits[0]):
        outcome = "stop_first"
    elif target_hit and stop_hit:
        outcome = "same_day_unknown"
    else:
        outcome = "neither"
    metrics.update(
        {
            "target_5pct_hit": target_hit,
            "stop_5pct_hit": stop_hit,
            "outcome_20": outcome,
            "mfe_20": round((float(highs.max()) / close - 1) * 100, 2),
            "mae_20": round((float(lows.min()) / close - 1) * 100, 2),
        }
    )
    return metrics


def _pct_change(current: float, previous: float) -> float | None:
    if previous <= 0:
        return None
    return round((current / previous - 1) * 100, 3)


def _date_index(df: pd.DataFrame, value: str | None) -> int | None:
    if not value:
        return None
    dates = pd.to_datetime(df["trade_date"], errors="coerce")
    matches = df.index[dates.dt.strftime("%Y-%m-%d") == value]
    return int(matches[-1]) if len(matches) else None


def compute_signal_features(bars: pd.DataFrame, signal_idx: int, result: SignalResult) -> dict[str, Any]:
    """Extract only information observable at the close of the signal day."""
    visible = bars.iloc[: signal_idx + 1]
    close = float(visible["close"].iloc[-1])
    open_price = float(visible["open"].iloc[-1])
    high = float(visible["high"].iloc[-1])
    low = float(visible["low"].iloc[-1])
    previous_close = float(visible["close"].iloc[-2]) if len(visible) > 1 else close

    def trailing_return(days: int) -> float | None:
        if len(visible) <= days:
            return None
        return _pct_change(close, float(visible["close"].iloc[-1 - days]))

    closes = visible["close"].astype(float)
    highs = visible["high"].astype(float)
    lows = visible["low"].astype(float)
    volumes = visible["volume"].astype(float)
    ma5 = closes.rolling(5).mean()
    ma10 = closes.rolling(10).mean()
    ma20 = closes.rolling(20).mean()
    ma60 = closes.rolling(60).mean()
    prior_volumes = volumes.iloc[-21:-1]
    volume_ratio_20 = float(volumes.iloc[-1] / prior_volumes.mean()) if len(prior_volumes) == 20 and prior_volumes.mean() > 0 else None

    previous = closes.shift(1)
    true_range = pd.concat(
        [(highs - lows), (highs - previous).abs(), (lows - previous).abs()], axis=1
    ).max(axis=1)
    atr20 = float(true_range.iloc[-20:].mean()) if len(true_range) >= 20 else None
    volatility20 = float(closes.pct_change().iloc[-20:].std(ddof=1) * 100) if len(closes) >= 21 else None
    recent_high_60 = float(highs.iloc[-60:].max()) if len(highs) else high
    recent_close_high_20 = float(closes.iloc[-20:].max()) if len(closes) else close
    breakout_idx = _date_index(bars, result.breakout_date)
    formation_idx = _date_index(bars, result.line_formation_date)
    day_range = high - low
    dif = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    histogram = dif - dif.ewm(span=9, adjust=False).mean()
    pre_breakout = (
        volumes.iloc[max(0, breakout_idx - 20):breakout_idx]
        if breakout_idx is not None and 0 <= breakout_idx <= signal_idx else pd.Series(dtype=float)
    )

    features: dict[str, Any] = {
        "distance_to_line_pct": result.distance_to_line_pct,
        "breakout_volume_ratio": result.breakout_volume_ratio,
        "pullback_volume_ratio": result.pullback_volume_ratio,
        "breakout_date": result.breakout_date,
        "pullback_date": result.pullback_date,
        "line_formation_date": result.line_formation_date,
        "bars_since_breakout": signal_idx - breakout_idx if breakout_idx is not None else None,
        "line_age_bars": signal_idx - formation_idx if formation_idx is not None else None,
        "signal_day_return_pct": _pct_change(close, previous_close),
        "signal_body_pct": _pct_change(close, open_price),
        "signal_gap_pct": _pct_change(open_price, previous_close),
        "signal_range_pct": round(day_range / previous_close * 100, 3) if previous_close > 0 else None,
        "close_location_pct": round((close - low) / day_range * 100, 2) if day_range > 0 else 50.0,
        "prior_return_5_pct": trailing_return(5),
        "prior_return_10_pct": trailing_return(10),
        "prior_return_20_pct": trailing_return(20),
        "prior_return_60_pct": trailing_return(60),
        "volume_ratio_20": round(volume_ratio_20, 3) if volume_ratio_20 is not None else None,
        "pullback_vs_prebreakout_volume": (
            round(float(volumes.iloc[-1] / pre_breakout.mean()), 4)
            if len(pre_breakout) == 20 and pre_breakout.mean() > 0 else None
        ),
        "macd_hist_pct": round(float(histogram.iloc[-1] / close * 100), 6),
        "macd_hist_rising": bool(
            len(histogram) >= 3 and histogram.iloc[-1] > histogram.iloc[-2] > histogram.iloc[-3]
        ),
        "volatility_20_pct": round(volatility20, 3) if volatility20 is not None and not np.isnan(volatility20) else None,
        "atr_20_pct": round(atr20 / close * 100, 3) if atr20 is not None and close > 0 else None,
        "distance_to_ma5_pct": _pct_change(close, float(ma5.iloc[-1])) if not np.isnan(ma5.iloc[-1]) else None,
        "distance_to_ma10_pct": _pct_change(close, float(ma10.iloc[-1])) if not np.isnan(ma10.iloc[-1]) else None,
        "distance_to_ma20_pct": _pct_change(close, float(ma20.iloc[-1])) if not np.isnan(ma20.iloc[-1]) else None,
        "distance_to_ma60_pct": _pct_change(close, float(ma60.iloc[-1])) if not np.isnan(ma60.iloc[-1]) else None,
        "ma20_slope_5_pct": (
            _pct_change(float(ma20.iloc[-1]), float(ma20.iloc[-6]))
            if len(ma20) >= 25 and not np.isnan(ma20.iloc[-1]) and not np.isnan(ma20.iloc[-6])
            else None
        ),
        "ma5_above_ma20": bool(ma5.iloc[-1] >= ma20.iloc[-1]) if not np.isnan(ma5.iloc[-1]) and not np.isnan(ma20.iloc[-1]) else None,
        "drawdown_from_20d_close_high_pct": _pct_change(close, recent_close_high_20),
        "distance_to_60d_high_pct": _pct_change(close, recent_high_60),
    }
    return features


def evaluate_bars(
    stock_code: str,
    stock_name: str,
    bars: pd.DataFrame,
    lookback_days: int = 0,
    min_signal_gap: int = 20,
    config: StrategyConfig | None = None,
    *, calendar: list[str] | None = None, as_of: str | None = None,
    market_regimes: pd.DataFrame | None = None,
    dedupe: bool = True,
) -> dict[str, Any]:
    """逐日重放；market_regimes 为 market_regime.regime_table 的逐日沪深300环境。

    dedupe=False 返回全部原始预警日（研究脚本先套附加条件再自行去重时使用）。
    """
    df = _normalize_bars(bars)
    if as_of is not None:
        df = df[df.trade_date <= pd.Timestamp(as_of)].reset_index(drop=True)
    execution_dates, execution_prices = _execution_inputs(df, calendar, as_of)
    cfg = config or StrategyConfig(signal_offset=0)
    cfg = StrategyConfig(**{**cfg.__dict__, "signal_offset": 0, "include_chart": False})
    if cfg.market_filter == "downgrade" and market_regimes is None:
        raise ValueError("market_filter=downgrade 的回测需要沪深300历史行情，不能静默跳过大盘过滤")
    stamps = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d").tolist()
    regimes: dict[int, dict[str, Any] | None] = {}
    minimum = max(cfg.min_history_bars, cfg.ma_period + cfg.ma_slope_days)
    start_idx = minimum - 1 if lookback_days <= 0 else max(minimum - 1, len(df) - lookback_days)
    raw: dict[int, Any] = {}
    pattern_events: list[dict[str, Any]] = []
    seen_events: set[tuple[str, str]] = set()
    if cfg.precision_engine == "platform_events" and cfg.strategy_mode != "flag_breakout":
        from app.services.platform_events import iter_platform_results
        evaluated = iter_platform_results(stock_code, stock_name, df, cfg, start_idx)
    else:
        def replay():
            for index in range(start_idx, len(df)):
                beginning = 0 if cfg.lookback_days is None else max(0, index + 1 - max(1, cfg.lookback_days))
                yield index, _analyze_normalized_bars(stock_code, stock_name, df.iloc[beginning:index + 1], cfg)
        evaluated = replay()
    for idx, result in evaluated:
        if result is not None and (result.flag_observation or result.flag_pattern):
            pattern = result.flag_observation or result.flag_pattern
            key = (pattern["id"], pattern["status"])
            if key not in seen_events:
                pattern_events.append(pattern)
                seen_events.add(key)
        if result is not None and result.category == SignalCategory.BUY_ALERT:
            regime = regime_on(market_regimes, stamps[idx]) if market_regimes is not None else None
            # 与实盘同一口径：偏弱日降为观察，不计入预警；同一事件之后的日子仍可能触发。
            if cfg.market_filter == "downgrade" and regime is not None and regime["weak"]:
                continue
            raw[idx] = result
            regimes[idx] = regime

    signals: list[dict[str, Any]] = []
    if not dedupe:
        selected_indices = sorted(raw)
    elif cfg.precision_engine == "platform_events":
        selected_indices = dedupe_event_signals(raw)
    else:
        selected_indices = dedupe_signal_indices(raw, min_signal_gap)
    for idx in selected_indices:
        result = raw[idx]
        confirmed_flag = next((
            item for item in result.pattern_matches
            if item["pattern_type"] == "flag_breakout" and item["category"] == "buy_alert"
        ), None)
        flag_direction = confirmed_flag["flag_direction"] if confirmed_flag else (
            result.flag_pattern["direction"]
            if not result.pattern_matches and result.pattern_type == "flag_breakout" and result.flag_pattern else None
        )
        record = {
            "signal_index": idx,
            "signal_date": pd.to_datetime(df["trade_date"].iloc[idx]).strftime("%Y-%m-%d"),
            "signal_price": round(float(df["close"].iloc[idx]), 3),
            "key_line_price": result.key_line_price,
            "line_strength": result.line_strength,
            "structure_score": round(float(result.structure_score or result.score), 2),
            "pattern_type": result.pattern_type,
            "flag_direction": flag_direction,
            "flag_pattern": result.flag_pattern,
            "pattern_matches": result.pattern_matches,
            "prebreakout_check": result.prebreakout_check,
            "pullback_check": result.pullback_check,
            "resistance_anchor": result.resistance_anchor,
            "require_breakout_volume": result.require_breakout_volume,
            "breakout_volume_threshold": result.breakout_volume_threshold,
            "precision_engine": result.precision_engine,
            "platform_state": result.platform_state,
            "signal_event_id": result.signal_event_id,
            "line_snapshot": result.line_snapshot,
            "breakout_limit_up": result.breakout_limit_up,
            "pullback_normal_volume_ratio": result.pullback_normal_volume_ratio,
        }
        regime = regimes.get(idx)
        record.update({
            "market_weak": regime["weak"] if regime else None,
            "market_return_20_pct": regime["return_20_pct"] if regime else None,
            "market_ma20_slope_5_pct": regime["ma20_slope_5_pct"] if regime else None,
        })
        record.update(compute_signal_features(df, idx, result))
        record.update(compute_forward_metrics(df, idx, calendar=execution_dates, _prices=execution_prices))
        signals.append(record)

    summary = summarize_signals(signals)
    if cfg.strategy_mode in {"flag_breakout", "combined"}:
        summary["flag_directions"] = summarize_flag_directions(signals)
    return {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "lookback_days": lookback_days,
        "history_scope": "all" if lookback_days <= 0 else "limited",
        "strategy_version": version_for_engine(cfg.precision_engine),
        "precision_engine": cfg.precision_engine,
        "raw_signal_day_count": len(raw),
        "deduplication": "breakout_event" if cfg.precision_engine == "platform_events" else "trading_bar_gap",
        "strategy_mode": cfg.strategy_mode,
        "flag_direction": cfg.flag_direction,
        "max_prebreakout_extension_pct": cfg.max_prebreakout_extension_pct,
        "prebreakout_reject_above_days": cfg.prebreakout_reject_above_days,
        "prebreakout_above_buffer_pct": cfg.prebreakout_above_buffer_pct,
        "require_pullback": cfg.require_pullback,
        "min_pullback_days": cfg.min_pullback_days,
        "max_pullback_days": cfg.max_pullback_days,
        "require_breakout_volume": cfg.require_breakout_volume,
        "breakout_vol_mult": cfg.breakout_vol_mult,
        "pullback_max_normal_volume_ratio": cfg.pullback_max_normal_volume_ratio,
        "max_post_breakout_runup_pct": cfg.max_post_breakout_runup_pct,
        "max_event_age_bars": cfg.max_event_age_bars,
        "market_filter": cfg.market_filter,
        "market_regime_available": market_regimes is not None and not market_regimes.empty,
        "execution_model": EXECUTION_MODEL,
        "execution_calendar": calendar_info() if calendar is None else {"source": "explicit_frozen_sessions"},
        "evaluation_as_of": as_of or (str(df.trade_date.iloc[-1].date()) if len(df) else None),
        "methodology": (
            f"信号按当日收盘后可知信息重放；按共同交易日历次日开盘入场，T+5/10/20市场交易日收盘退出；"
            f"双边费用与滑点合计假设 {ROUNDTRIP_COST_BPS:g} bps。"
            "计划入场日缺失、一字或零成交量时不买入，不顺延到复牌日；卖出受阻则延后，买入当天不卖出。"
            + ("同一冻结突破事件只留首个预警；" if cfg.precision_engine == "platform_events" else "同股20日内只留首个预警；")
            + "当前行业快照不回填历史。"
            "联合模式任一形态确认即可，同股同日只计一笔；待确认标签不计入旗形突破胜率。"
            "仅为日线成交近似，不能确认开盘排队成交，缺失记录不能区分停牌与数据缺口；未模拟历史风险警示状态。"
            "旧路径字段仍按股票K线根数观察，不是共同交易日历的执行收益；历史信号不等于实际扫描时已经可见。"
        ),
        "signals": signals,
        "pattern_events": pattern_events,
        "summary": summary,
    }


def summarize_signals(signals: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"signal_count": len(signals)}
    trades = [item for item in signals if item.get("execution_model") == EXECUTION_MODEL]
    if trades or not signals:
        summary["execution_model"] = EXECUTION_MODEL
        summary["execution_sample_count"] = len(trades)
        summary["blocked_entry_count"] = sum(item.get("entry_status") == "blocked" for item in trades)
        summary["pending_entry_count"] = sum(item.get("entry_status") == "pending" for item in trades)
        summary["legacy_signal_count"] = len(signals) - len(trades)
        summary["filled_entry_count"] = sum(item.get("entry_status") == "filled" for item in trades)
        mature_entries = summary["filled_entry_count"] + summary["blocked_entry_count"]
        summary["entry_feasibility_sample_count"] = mature_entries
        summary["entry_feasibility_rate"] = round(100 * summary["filled_entry_count"] / mature_entries, 3) if mature_entries else None
        for window in FORWARD_WINDOWS:
            values = [float(item[f"net_return_{window}"]) for item in trades if item.get(f"net_return_{window}") is not None]
            summary[f"net_t{window}_sample_count"] = len(values)
            summary[f"net_t{window}_positive_rate"] = round(sum(v > 0 for v in values) / len(values) * 100, 1) if values else None
            summary[f"net_t{window}_average_return"] = round(float(np.mean(values)), 3) if values else None
            summary[f"net_t{window}_median_return"] = round(float(np.median(values)), 3) if values else None
            summary[f"net_t{window}_pending_count"] = sum(
                item.get("entry_status") == "filled" and item.get(f"net_return_{window}") is None for item in trades
            )
            gains = sum(value for value in values if value > 0)
            losses = -sum(value for value in values if value < 0)
            wins = [value for value in values if value > 0]
            losing = [-value for value in values if value < 0]
            summary[f"net_t{window}_profit_factor"] = round(gains / losses, 4) if losses else None
            summary[f"net_t{window}_profit_factor_status"] = "finite" if losses else "no_losses" if values else "no_closed_trades"
            summary[f"net_t{window}_payoff_ratio"] = round(float(np.mean(wins)) / float(np.mean(losing)), 4) if wins and losing else None
    for window in FORWARD_WINDOWS:
        values = [float(item[f"forward_return_{window}"]) for item in signals if item.get(f"forward_return_{window}") is not None]
        summary[f"t{window}_sample_count"] = len(values)
        summary[f"t{window}_positive_rate"] = round(sum(value > 0 for value in values) / len(values) * 100, 1) if values else None
        summary[f"t{window}_average_return"] = round(sum(values) / len(values), 2) if values else None
        summary[f"t{window}_median_return"] = round(float(np.median(values)), 2) if values else None
    complete = [item for item in signals if item.get("target_5pct_hit") is not None]
    summary["target_5pct_sample_count"] = len(complete)
    summary["target_5pct_hit_rate"] = round(sum(bool(item["target_5pct_hit"]) for item in complete) / len(complete) * 100, 1) if complete else None
    summary["stop_5pct_hit_rate"] = round(sum(bool(item["stop_5pct_hit"]) for item in complete) / len(complete) * 100, 1) if complete else None
    summary["target_before_stop_rate"] = round(sum(item.get("outcome_20") == "target_first" for item in complete) / len(complete) * 100, 1) if complete else None
    return summary


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    signals = [signal for report in reports for signal in report.get("signals", [])]
    summary = summarize_signals(signals)
    if any(report.get("strategy_mode") in {"flag_breakout", "combined"} for report in reports):
        summary["flag_directions"] = summarize_flag_directions(signals)
    summary.update(
        {
            "stock_count": len(reports),
            "stocks_with_signals": sum(bool(report.get("signals")) for report in reports),
        }
    )
    return summary


def summarize_flag_directions(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"direction": direction, "label": label, **summarize_signals([
            item for item in signals if item.get("flag_direction") == direction
        ])}
        for direction, label in FLAG_DIRECTIONS.items()
    ]
