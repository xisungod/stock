"""Point-in-time bull-flag prototype; orientation does not imply profitability."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from app.services.zigzag import confirmed_swings


FLAG_VERSION = "flag-channel-v1"
FLAG_DIRECTIONS = {"descending": "下倾旗形", "horizontal": "水平旗形", "ascending": "上倾旗形"}


@dataclass(frozen=True)
class FlagConfig:
    pole_min_bars: int = 3
    pole_max_bars: int = 15
    pole_min_gain: float = .10
    pole_min_atr: float = 4.0
    pole_min_efficiency: float = .65
    flag_min_bars: int = 4
    flag_max_bars: int = 20
    max_retracement: float = .50
    max_width_to_pole: float = .50
    max_slope_ratio: float = .50
    parallel_tolerance: float = .30
    flat_drift_atr: float = .50
    pivot_reversal_atr: float = .50
    touch_tolerance_atr: float = .25
    volume_contraction: float = .80
    range_expansion: float = 1.20
    breakout_buffer_atr: float = .20
    volume_lookback: int = 20


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    previous = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(abs(high - previous), abs(low - previous)))
    return pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()


def _fit_channel(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, opens: np.ndarray,
    atr: float, cfg: FlagConfig,
) -> dict[str, Any] | None:
    reversal = np.full(len(close), max(.2, cfg.pivot_reversal_atr * atr / float(np.median(close)) * 100))
    peaks, valleys = confirmed_swings(high, low, close, reversal, opens=opens)
    if len(peaks) < 2 or len(valleys) < 2:
        return None
    upper = Polynomial.fit(peaks, high[peaks], 1).convert().coef
    lower = Polynomial.fit(valleys, low[valleys], 1).convert().coef
    # Polynomial conversion trims an exactly zero slope in horizontal channels.
    upper = np.pad(upper, (0, max(0, 2 - len(upper))))
    lower = np.pad(lower, (0, max(0, 2 - len(lower))))
    x = np.arange(len(close), dtype=float)
    ceiling, floor = upper[0] + upper[1] * x, lower[0] + lower[1] * x
    widths = ceiling - floor
    width = float(np.mean(widths))
    if width <= 0 or float(widths.min()) <= 0:
        return None
    # Separate fits are necessary: fitting a shared slope would manufacture parallelism.
    divergence = abs(float(upper[1] - lower[1])) * (len(close) - 1) / width
    if divergence > cfg.parallel_tolerance:
        return None
    tolerance = cfg.touch_tolerance_atr * atr
    contained = (high <= ceiling + tolerance) & (low >= floor - tolerance)
    if float(contained.mean()) < .9:
        return None
    if np.any(close > ceiling + cfg.breakout_buffer_atr * atr) or np.any(close < floor - cfg.breakout_buffer_atr * atr):
        return None
    slope = float((upper[1] + lower[1]) / 2)
    drift = slope * (len(close) - 1)
    direction = "horizontal" if abs(drift) <= cfg.flat_drift_atr * atr else "ascending" if slope > 0 else "descending"
    return {
        "upper": upper, "lower": lower, "width": float(widths.max()),
        "slope": slope, "direction": direction, "peaks": peaks, "valleys": valleys,
        "parallel_error": divergence,
    }


def detect_flag(bars: pd.DataFrame, config: FlagConfig | None = None, breakout_vol_mult: float = 1.5) -> dict[str, Any] | None:
    cfg = config or FlagConfig()
    n = len(bars)
    if n < 20 + cfg.pole_min_bars + cfg.flag_min_bars + 2:
        return None
    values = bars[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    opens, high, low, close, volume = values.T
    if (values[:, :4] <= 0).any() or (volume < 0).any():
        return None
    atrs = _atr(high, low, close)
    # Exclude the entire signal bar, including its range and volume, from every fit.
    end = n - 2
    candidates = []
    for length in range(cfg.flag_min_bars, cfg.flag_max_bars + 1):
        start, pole_end = end - length + 1, end - length
        if pole_end - cfg.pole_min_bars < 20:
            continue
        pole_candidates = []
        for duration in range(cfg.pole_min_bars, cfg.pole_max_bars + 1):
            pole_start = pole_end - duration
            if pole_start < 20:
                continue
            base_atr = float(atrs[pole_start - 1])
            gain = float(close[pole_end] - close[pole_start])
            if base_atr <= 0 or gain < max(cfg.pole_min_gain * close[pole_start], cfg.pole_min_atr * base_atr):
                continue
            travel = float(np.abs(np.diff(close[pole_start:pole_end + 1])).sum())
            efficiency = gain / travel if travel > 0 else 0
            if efficiency < cfg.pole_min_efficiency or close[pole_end] < close[pole_start:pole_end + 1].max() - base_atr * .5:
                continue
            if (volume[pole_start:end + 1] <= 0).any():
                continue
            pole_candidates.append((gain / (base_atr * np.sqrt(duration)), pole_start, gain, efficiency))
        if not pole_candidates:
            continue
        half = length // 2
        early_vol = float(np.median(volume[start:start + half]))
        later_vol = float(np.median(volume[start + half:end + 1]))
        contraction = later_vol / early_vol if early_vol > 0 else np.inf
        if contraction > cfg.volume_contraction:
            continue
        early_range = float(np.median(high[start:start + half] - low[start:start + half]))
        later_range = float(np.median(high[start + half:end + 1] - low[start + half:end + 1]))
        if early_range <= 0 or later_range > early_range * cfg.range_expansion:
            continue
        flag_atr = float(atrs[end])
        if not np.isfinite(flag_atr) or flag_atr <= 0:
            continue
        channel = _fit_channel(high[start:end + 1], low[start:end + 1], close[start:end + 1], opens[start:end + 1], flag_atr, cfg)
        if channel is None:
            continue
        for pole_quality, pole_start, gain, efficiency in sorted(pole_candidates, reverse=True):
            pole_slope = gain / (pole_end - pole_start)
            slope_ratio = abs(channel["slope"]) / pole_slope
            retracement = max(0.0, float(close[pole_end] - low[start:end + 1].min()) / gain)
            width_ratio = channel["width"] / gain
            if slope_ratio > cfg.max_slope_ratio or retracement > cfg.max_retracement or width_ratio > cfg.max_width_to_pole:
                continue
            candidates.append({
                **channel, "start": start, "end": end, "pole_start": pole_start, "pole_end": pole_end,
                "pole_quality": pole_quality, "gain": gain, "efficiency": efficiency,
                "slope_ratio": slope_ratio, "retracement": retracement, "width_ratio": width_ratio,
                "contraction": contraction, "atr": flag_atr,
            })
            break
    if not candidates:
        return None
    # Select by yesterday's structural evidence, never by which boundary today's price crosses.
    best = max(candidates, key=lambda p: (len(p["peaks"]) + len(p["valleys"]), p["pole_quality"], -p["start"]))
    start, pole_start, pole_end = best["start"], best["pole_start"], best["pole_end"]
    upper, lower = best["upper"], best["lower"]
    offset = n - 1 - start
    upper_now = float(upper[0] + upper[1] * offset)
    lower_now = float(lower[0] + lower[1] * offset)
    mean_volume = float(np.mean(volume[max(0, n - 1 - cfg.volume_lookback):n - 1]))
    volume_ratio = float(volume[-1] / mean_volume) if mean_volume > 0 else 0
    buffer = best["atr"] * cfg.breakout_buffer_atr
    if volume[-1] <= 0:
        status = "untradable"
    elif close[-1] < lower_now - buffer or close[-1] < close[pole_end] - best["gain"] * cfg.max_retracement:
        status = "failed"
    elif close[-1] > upper_now + buffer:
        status = "breakout" if volume_ratio >= breakout_vol_mult else "unconfirmed_breakout"
    else:
        status = "forming"

    def date_at(idx: int) -> str:
        return pd.Timestamp(bars["trade_date"].iloc[idx]).strftime("%Y-%m-%d")

    return {
        "version": FLAG_VERSION,
        "id": f"flag:{date_at(pole_start)}:{date_at(pole_end)}",
        "direction": best["direction"], "direction_label": FLAG_DIRECTIONS[best["direction"]], "status": status,
        "pole_start_date": date_at(pole_start), "pole_end_date": date_at(pole_end),
        "flag_start_date": date_at(start), "fit_end_date": date_at(end), "signal_date": date_at(n - 1),
        "pole_bars": pole_end - pole_start, "flag_bars": end - start + 1,
        "pole_gain_pct": round(float(best["gain"] / close[pole_start] * 100), 3),
        "pole_efficiency": round(float(best["efficiency"]), 4),
        "slope_ratio": round(float(best["slope_ratio"]), 4),
        "flag_slope_pct_per_bar": round(float(best["slope"] / close[pole_end] * 100), 4),
        "retracement_pct": round(float(best["retracement"] * 100), 3),
        "width_to_pole_pct": round(float(best["width_ratio"] * 100), 3),
        "volume_contraction_ratio": round(float(best["contraction"]), 4),
        "breakout_volume_ratio": round(volume_ratio, 4), "breakout_volume_threshold": breakout_vol_mult,
        "upper_price": round(upper_now, 6), "lower_price": round(lower_now, 6),
        "breakout_buffer": round(buffer, 6),
        "upper_touch_count": len(best["peaks"]), "lower_touch_count": len(best["valleys"]),
        "pole": [
            {"date": date_at(pole_start), "price": float(close[pole_start])},
            {"date": date_at(pole_end), "price": float(close[pole_end])},
        ],
        "channel": [
            {"date": date_at(start), "upper": float(upper[0]), "lower": float(lower[0])},
            {"date": date_at(n - 1), "upper": upper_now, "lower": lower_now},
        ],
        "parameters": asdict(cfg),
    }
