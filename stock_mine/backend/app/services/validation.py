"""Locked-rule evaluation helpers, independent of signal discovery."""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

import numpy as np


VALIDATION_EXECUTION_MODEL = "shared_calendar_daily_proxy_v1"


def daily_proxy_blocked(row: dict[str, Any] | None, previous: float | None, buying: bool) -> bool:
    if row is None:
        return True
    try:
        values = [float(row[key]) for key in ("open", "high", "low", "close", "volume")]
    except (ValueError, KeyError, TypeError):
        return True
    if any(not isfinite(value) or value <= 0 for value in values):
        return True
    flat = isclose(float(row["high"]), float(row["low"]), rel_tol=0, abs_tol=1e-8)
    return flat and (buying or previous is None or float(row["close"]) <= previous)


def trade_on_calendar(code: str, signal: dict, prices: dict[str, dict], calendar: list[str],
                      horizon: int, cost_bps: float = 20) -> dict:
    """A daily-bar feasibility proxy, NOT a simulation of opening-auction queues."""
    if not isfinite(cost_bps) or cost_bps < 0 or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("invalid execution horizon or cost")
    signal_date = signal["signal_date"]
    signal_index = bisect_right(calendar, signal_date) - 1
    entry_index = signal_index + 1
    record = {
        "stock_code": code, "signal_date": signal_date, "horizon": horizon,
        "structure_score": signal.get("structure_score", 0), "signal_event_id": signal.get("signal_event_id"),
        "entry_status": "pending", "entry_date": None, "entry_price": None,
        "exit_date": None, "exit_price": None, "net_return_pct": None,
        "execution_model": VALIDATION_EXECUTION_MODEL,
    }
    if signal_index < 0 or calendar[signal_index] != signal_date:
        record.update(entry_status="blocked", entry_reason="signal_is_not_a_calendar_session")
        return record
    if entry_index >= len(calendar):
        return record
    entry_date = calendar[entry_index]
    record["entry_date"] = entry_date
    entry = prices.get(entry_date)
    previous = prices.get(signal_date, {}).get("close")
    if daily_proxy_blocked(entry, previous, True):
        record.update(entry_status="blocked", entry_reason="missing_or_flat_or_zero_volume")
        return record
    record.update(entry_status="filled", entry_price=float(entry["open"]))
    planned_index = max(signal_index + horizon, entry_index + 1)
    record["planned_exit_date"] = calendar[planned_index] if planned_index < len(calendar) else None
    last_close = float(entry["close"])
    for index in range(entry_index + 1, len(calendar)):
        stamp = calendar[index]
        row = prices.get(stamp)
        if index >= planned_index and not daily_proxy_blocked(row, last_close, False):
            gross = (float(row["close"]) / record["entry_price"] - 1) * 100
            record.update(exit_date=stamp, exit_price=float(row["close"]),
                          net_return_pct=gross - cost_bps / 100,
                          exit_delay_sessions=index - planned_index)
            break
        if row is not None:
            last_close = float(row["close"])
    return record


def trade_statistics(records: list[dict]) -> dict:
    mature = [row for row in records if row["entry_status"] != "pending"]
    entered = [row for row in records if row["entry_status"] == "filled"]
    values = [float(row["net_return_pct"]) for row in entered if row["net_return_pct"] is not None]
    gains = [value for value in values if value > 0]
    losses = [-value for value in values if value < 0]
    return {
        "signal_count": len(records), "mature_entry_count": len(mature),
        "filled_entry_count": len(entered), "blocked_entry_count": sum(row["entry_status"] == "blocked" for row in records),
        "pending_entry_count": sum(row["entry_status"] == "pending" for row in records),
        "closed_trade_count": len(values), "pending_exit_count": len(entered) - len(values),
        "entry_feasibility_pct": 100 * len(entered) / len(mature) if mature else None,
        "net_win_rate_pct": 100 * len(gains) / len(values) if values else None,
        "mean_net_return_pct": float(np.mean(values)) if values else None,
        "median_net_return_pct": float(np.median(values)) if values else None,
        "profit_factor": sum(gains) / sum(losses) if losses else None,
        "payoff_ratio": float(np.mean(gains)) / float(np.mean(losses)) if gains and losses else None,
        "profit_factor_status": "finite" if losses else "no_losses" if values else "no_closed_trades",
    }


@dataclass
class Position:
    units: float
    principal: float
    entry_date: str
    planned_exit: str | None
    last_close: float


def portfolio_curve(records: list[dict], frames: dict[str, dict[str, dict]], calendar: list[str],
                    start: str, end: str, max_positions: int = 10, cost_bps: float = 20) -> dict:
    """Chronological cash accounting with fractional shares and a fixed position cap.

    Each side's cost is charged on entry principal, matching the trade-return proxy.
    Closing proceeds cannot finance entries at the same morning's open.
    """
    if max_positions < 1 or cost_bps < 0:
        raise ValueError("invalid portfolio assumptions")
    entries = defaultdict(list)
    for record in records:
        if record["entry_status"] == "filled" and start <= record["entry_date"] <= end:
            entries[record["entry_date"]].append(record)
    cash, prior_nav, peak = 1.0, 1.0, 1.0
    positions: dict[str, Position] = {}
    curve = []
    half_cost = cost_bps / 20000
    executions, missed, realized = 0, 0, []
    exposure = []
    max_drawdown = 0.0
    for stamp in (day for day in calendar if start <= day <= end):
        daily_orders = sorted(entries[stamp], key=lambda row: (-float(row["structure_score"]), row["stock_code"]))
        for order in daily_orders:
            code = order["stock_code"]
            if code in positions or len(positions) >= max_positions or cash <= 1e-10:
                missed += 1
                continue
            budget = min(cash, prior_nav / max_positions)
            principal = budget / (1 + half_cost)
            units = principal / order["entry_price"]
            cash -= budget
            positions[code] = Position(units, principal, stamp, order.get("planned_exit_date"), order["entry_price"])
            executions += 1
        for code, position in list(positions.items()):
            row = frames[code].get(stamp)
            if (position.planned_exit and stamp >= position.planned_exit and stamp > position.entry_date
                    and not daily_proxy_blocked(row, position.last_close, False)):
                proceeds = position.units * float(row["close"]) - position.principal * half_cost
                cash += proceeds
                realized.append((float(row["close"]) / (position.principal / position.units) - 1) * 100 - cost_bps / 100)
                del positions[code]
            elif row is not None:
                position.last_close = float(row["close"])
        invested = sum(position.units * position.last_close for position in positions.values())
        nav = cash + invested
        peak = max(peak, nav)
        max_drawdown = max(max_drawdown, (peak - nav) / peak)
        exposure.append(invested / nav if nav > 0 else 0)
        curve.append({"date": stamp, "nav": round(nav, 9), "cash": round(cash, 9), "positions": len(positions)})
        prior_nav = nav
        assert cash >= -1e-8
    return {
        "net_return_pct": (prior_nav - 1) * 100 if executions else None,
        "max_drawdown_pct": max_drawdown * 100 if executions else None,
        "executed_entries": executions, "orders_skipped_for_capital_or_existing_position": missed,
        "closed_positions": len(realized), "open_positions": len(positions),
        "average_exposure_pct": 100 * float(np.mean(exposure)) if exposure else 0,
        "max_positions": max_positions, "curve": curve,
    }


def block_mean_difference(left: list[dict], right: list[dict], calendar: list[str],
                          block_size: int = 20, repetitions: int = 1000, seed: int = 20260910):
    """Synchronous block resampling of entry dates for differently sized signal sets."""
    positions = {stamp: index // block_size for index, stamp in enumerate(calendar)}
    blocks = sorted({positions[row["entry_date"]] for row in left + right
                     if row.get("net_return_pct") is not None and row.get("entry_date") in positions})
    if not blocks:
        return None
    aggregates = []
    for rows in (left, right):
        totals = defaultdict(lambda: [0.0, 0])
        for row in rows:
            if row.get("net_return_pct") is not None and row.get("entry_date") in positions:
                bucket = totals[positions[row["entry_date"]]]
                bucket[0] += float(row["net_return_pct"])
                bucket[1] += 1
        aggregates.append(np.asarray([totals[block] for block in blocks]))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(blocks), len(blocks))
        a, b = (array[indices].sum(axis=0) for array in aggregates)
        if a[1] and b[1]:
            differences.append(a[0] / a[1] - b[0] / b[1])
    return [float(value) for value in np.percentile(differences, [2.5, 97.5])] if differences else None
