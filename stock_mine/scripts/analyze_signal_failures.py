from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import numpy as np
import pandas as pd

from app.schemas import production_strategy_config
from app.services.backtest import evaluate_bars
from app.services.data_provider import create_provider


_WORKER_PROVIDER: Any = None


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    family: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze when historical buy alerts fail within five trading days.")
    parser.add_argument("--sample", type=int, default=240, help="Deterministic market-wide sample size; 0 means all stocks.")
    parser.add_argument("--lookback", type=int, default=500, help="Trading days replayed per stock.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--provider", default="akshare")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--checkpoint", help="JSONL checkpoint path; omitted means no checkpoint is written.")
    parser.add_argument("--resume", action="store_true", help="Resume completed stock attempts from --checkpoint.")
    return parser.parse_args()


def checkpoint_identity(args: argparse.Namespace, selected: pd.DataFrame) -> dict[str, Any]:
    codes = selected["stock_code"].astype(str).str.zfill(6).tolist()
    return {
        "version": 1,
        "provider": args.provider,
        "lookback": args.lookback,
        "sample": args.sample,
        "stock_count": len(codes),
        "stock_codes_sha256": hashlib.sha256("\n".join(codes).encode("ascii")).hexdigest(),
    }


def prepare_checkpoint(
    path: Path,
    identity: dict[str, Any],
    resume: bool,
) -> tuple[dict[str, dict[str, Any]], Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    completed: dict[str, dict[str, Any]] = {}

    if resume:
        if not path.exists() or not metadata_path.exists():
            raise RuntimeError(f"无法续跑：检查点或元数据不存在: {path}")
        stored_identity = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored_identity != identity:
            raise RuntimeError("无法续跑：检查点参数或股票池与本次运行不一致")
        with path.open("r", encoding="utf-8") as checkpoint_file:
            for line_number, line in enumerate(checkpoint_file, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"忽略检查点末尾的不完整记录（第 {line_number} 行）", flush=True)
                    break
                code = str(item.get("stock_code", "")).zfill(6)
                if code:
                    completed[code] = item
    else:
        metadata_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        path.write_text("", encoding="utf-8")

    return completed, path.open("a", encoding="utf-8", buffering=1)


def systematic_sample(stocks: pd.DataFrame, limit: int) -> pd.DataFrame:
    stocks = stocks.sort_values("stock_code").drop_duplicates("stock_code").reset_index(drop=True)
    if limit <= 0 or limit >= len(stocks):
        return stocks
    positions = np.linspace(0, len(stocks) - 1, num=limit).round().astype(int)
    return stocks.iloc[positions].drop_duplicates("stock_code").reset_index(drop=True)


def initialize_worker(provider_name: str) -> None:
    global _WORKER_PROVIDER
    _WORKER_PROVIDER = create_provider(provider_name)
    if hasattr(_WORKER_PROVIDER, "_history_source"):
        _WORKER_PROVIDER._history_source = "tencent"


def evaluate_stock_worker(payload: tuple[str, str, str, int, int]) -> dict[str, Any]:
    code, name, exchange, years, lookback = payload
    try:
        bars = _WORKER_PROVIDER.get_full_daily_bars(code, years=years)
        if bars is None or bars.empty:
            raise ValueError("未获取到历史行情")
        # 生产默认配置（联合模式、platform_events 引擎），而不是 StrategyConfig() 的 legacy 默认值。
        stock_report = evaluate_bars(code, name, bars, lookback_days=lookback,
                                     config=production_strategy_config(include_chart=False))
        return {
            "stock_code": code,
            "stock_name": name,
            "exchange": exchange,
            "signals": stock_report["signals"],
            "error": None,
        }
    except Exception as exc:
        return {
            "stock_code": code,
            "stock_name": name,
            "exchange": exchange,
            "signals": [],
            "error": str(exc),
        }


def load_benchmark() -> pd.DataFrame:
    raw = ak.stock_zh_index_daily(symbol="sh000300")
    benchmark = raw.rename(columns={"date": "trade_date"}).copy()
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark = benchmark.sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark["ma20"] = benchmark["close"].rolling(20).mean()
    benchmark["market_return_5_pct"] = benchmark["close"].pct_change(5) * 100
    benchmark["market_return_20_pct"] = benchmark["close"].pct_change(20) * 100
    benchmark["market_t5_return_pct"] = benchmark["close"].shift(-5).div(benchmark["close"]).sub(1).mul(100)
    benchmark["market_distance_to_ma20_pct"] = benchmark["close"].div(benchmark["ma20"]).sub(1).mul(100)
    benchmark["market_ma20_slope_5_pct"] = benchmark["ma20"].div(benchmark["ma20"].shift(5)).sub(1).mul(100)
    return benchmark.set_index("trade_date")


def benchmark_features(benchmark: pd.DataFrame, signal_date: str, signal: dict[str, Any]) -> dict[str, Any]:
    date = pd.Timestamp(signal_date)
    if date not in benchmark.index:
        available = benchmark.index[benchmark.index <= date]
        if available.empty:
            return {}
        date = available[-1]
    row = benchmark.loc[date]
    market_return_20 = finite(row.get("market_return_20_pct"))
    stock_return_20 = finite(signal.get("prior_return_20_pct"))
    market_t5 = finite(row.get("market_t5_return_pct"))
    stock_t5 = finite(signal.get("forward_return_5"))
    return {
        "market_date": date.strftime("%Y-%m-%d"),
        "market_return_5_pct": finite(row.get("market_return_5_pct")),
        "market_return_20_pct": market_return_20,
        "market_distance_to_ma20_pct": finite(row.get("market_distance_to_ma20_pct")),
        "market_ma20_slope_5_pct": finite(row.get("market_ma20_slope_5_pct")),
        "market_t5_return_pct": market_t5,
        "relative_strength_20_pct": round(stock_return_20 - market_return_20, 3) if stock_return_20 is not None and market_return_20 is not None else None,
        "excess_return_5_pct": round(stock_t5 - market_t5, 3) if stock_t5 is not None and market_t5 is not None else None,
    }


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return (center - margin) * 100, (center + margin) * 100


def cluster_bootstrap_context(frame: pd.DataFrame, iterations: int = 5000) -> dict[str, Any]:
    valid = frame[frame["forward_return_5"].notna()].copy()
    valid["down"] = (valid["forward_return_5"] < 0).astype(int)
    stocks = sorted(valid["stock_code"].astype(str).unique())
    grouped = valid.groupby(valid["stock_code"].astype(str))["down"].agg(["sum", "count"]).reindex(stocks, fill_value=0)
    draws = np.random.default_rng(20260820).integers(0, len(stocks), size=(iterations, len(stocks)))
    base_down = grouped["sum"].to_numpy()[draws].sum(axis=1)
    base_total = grouped["count"].to_numpy()[draws].sum(axis=1)
    return {
        "valid": valid,
        "stocks": stocks,
        "draws": draws,
        "base_rates": base_down / base_total * 100,
    }


def cluster_lift_interval(mask: pd.Series, context: dict[str, Any]) -> tuple[float, float]:
    valid = context["valid"]
    selected = valid[mask.reindex(valid.index, fill_value=False)]
    grouped = selected.groupby(selected["stock_code"].astype(str))["down"].agg(["sum", "count"])
    grouped = grouped.reindex(context["stocks"], fill_value=0)
    draws = context["draws"]
    condition_down = grouped["sum"].to_numpy()[draws].sum(axis=1)
    condition_total = grouped["count"].to_numpy()[draws].sum(axis=1)
    usable = condition_total > 0
    lifts = condition_down[usable] / condition_total[usable] * 100 - context["base_rates"][usable]
    low, high = np.quantile(lifts, [0.025, 0.975])
    return float(low), float(high)


def stats(frame: pd.DataFrame) -> dict[str, Any]:
    sample = frame[frame["forward_return_5"].notna()]
    total = len(sample)
    down = int((sample["forward_return_5"] < 0).sum())
    low, high = wilson_interval(down, total)
    executable = sample["forward_return_5_from_next_open"].dropna()
    relative = sample["excess_return_5_pct"].dropna()
    drawdowns = sample["mae_5"].dropna()
    return {
        "n": total,
        "down": down,
        "down_rate": down / total * 100 if total else None,
        "ci_low": low if total else None,
        "ci_high": high if total else None,
        "mean_return": float(sample["forward_return_5"].mean()) if total else None,
        "median_return": float(sample["forward_return_5"].median()) if total else None,
        "next_open_down_rate": float((executable < 0).mean() * 100) if len(executable) else None,
        "underperform_rate": float((relative < 0).mean() * 100) if len(relative) else None,
        "drawdown_3_rate": float((drawdowns <= -3).mean() * 100) if len(drawdowns) else None,
    }


def pct(value: Any, digits: int = 1) -> str:
    return "-" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}%"


def num(value: Any, digits: int = 2) -> str:
    return "-" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def pp(value: Any, digits: int = 1) -> str:
    return "-" if value is None or not math.isfinite(float(value)) else f"{float(value):+.{digits}f}"


def conditions() -> list[Condition]:
    return [
        Condition(
            "market_weak_regime",
            "沪深300近20日为负或MA20下行",
            "market_regime",
            lambda d: (d["market_return_20_pct"] < 0) | (d["market_ma20_slope_5_pct"] < 0),
        ),
        Condition("market_below_ma20", "沪深300低于MA20", "market_trend", lambda d: d["market_distance_to_ma20_pct"] < 0),
        Condition("market_ma20_falling", "沪深300 MA20近5日下行", "market_trend", lambda d: d["market_ma20_slope_5_pct"] < 0),
        Condition("market_20d_negative", "沪深300近20日收益为负", "market_momentum", lambda d: d["market_return_20_pct"] < 0),
        Condition("relative_weak", "个股近20日跑输沪深300超过3%", "relative_strength", lambda d: d["relative_strength_20_pct"] < -3),
        Condition("prior_5d_negative", "个股信号前5日仍在下跌", "stock_momentum", lambda d: d["prior_return_5_pct"] < 0),
        Condition("prior_20d_overextended", "个股信号前20日涨幅超过10%", "stock_momentum", lambda d: d["prior_return_20_pct"] > 10),
        Condition("below_ma5", "信号收盘低于MA5", "moving_average", lambda d: d["distance_to_ma5_pct"] < 0),
        Condition("ma5_below_ma20", "MA5仍低于MA20", "moving_average", lambda d: d["ma5_above_ma20"] == False),  # noqa: E712
        Condition("weak_close", "信号日收在日内区间下35%", "signal_candle", lambda d: d["close_location_pct"] < 35),
        Condition("bearish_signal_day", "信号日收阴", "signal_candle", lambda d: d["signal_body_pct"] < 0),
        Condition("large_range", "信号日振幅超过4%", "signal_volatility", lambda d: d["signal_range_pct"] > 4),
        Condition("high_volatility", "20日波动率超过3%", "signal_volatility", lambda d: d["volatility_20_pct"] > 3),
        Condition("under_key_line", "信号收盘仍在精准线下", "key_line", lambda d: d["distance_to_line_pct"] < 0),
        Condition("extended_from_line", "信号距精准线超过2%", "key_line", lambda d: d["distance_to_line_pct"] > 2),
        Condition("weak_breakout_volume", "突破量比低于1.8", "breakout_volume", lambda d: d["breakout_volume_ratio"] < 1.8),
        Condition("high_pullback_volume", "回踩量比高于0.60", "pullback_volume", lambda d: d["pullback_volume_ratio"] > 0.6),
        Condition("fast_pullback", "突破后1至2日即回踩", "breakout_timing", lambda d: d["bars_since_breakout"] <= 2),
        Condition("late_pullback", "突破6日后才回踩", "breakout_timing", lambda d: d["bars_since_breakout"] >= 6),
        Condition("minimum_line_strength", "精准线仅3次确认", "line_strength", lambda d: d["line_strength"] <= 3),
        Condition("young_line", "精准线形成不足60个交易日", "line_age", lambda d: d["line_age_bars"] < 60),
    ]


def condition_table(frame: pd.DataFrame, condition_list: list[Condition]) -> tuple[list[dict[str, Any]], pd.Timestamp, tuple[float, float]]:
    valid = frame[frame["forward_return_5"].notna()].copy()
    cutoff = valid["signal_date"].sort_values().iloc[len(valid) // 2]
    early = valid[valid["signal_date"] <= cutoff]
    recent = valid[valid["signal_date"] > cutoff]
    base = stats(valid)
    early_base = stats(early)
    recent_base = stats(recent)
    cluster_context = cluster_bootstrap_context(valid)
    cluster_base_interval = tuple(float(value) for value in np.quantile(cluster_context["base_rates"], [0.025, 0.975]))
    minimum = max(20, math.ceil(len(valid) * 0.05))
    rows: list[dict[str, Any]] = []
    for condition in condition_list:
        mask = condition.predicate(valid).fillna(False)
        subset = valid[mask]
        early_subset = early[condition.predicate(early).fillna(False)]
        recent_subset = recent[condition.predicate(recent).fillna(False)]
        overall_stats = stats(subset)
        early_stats = stats(early_subset)
        recent_stats = stats(recent_subset)
        if overall_stats["n"] < minimum:
            continue
        lift = overall_stats["down_rate"] - base["down_rate"]
        early_lift = (early_stats["down_rate"] - early_base["down_rate"]) if early_stats["n"] >= 10 else None
        recent_lift = (recent_stats["down_rate"] - recent_base["down_rate"]) if recent_stats["n"] >= 10 else None
        stable = early_lift is not None and recent_lift is not None and early_lift > 0 and recent_lift > 0
        cluster_lift_low, cluster_lift_high = cluster_lift_interval(mask, cluster_context)
        rows.append(
            {
                "key": condition.key,
                "label": condition.label,
                "family": condition.family,
                **overall_stats,
                "lift": lift,
                "early_rate": early_stats["down_rate"],
                "early_lift": early_lift,
                "recent_rate": recent_stats["down_rate"],
                "recent_lift": recent_lift,
                "stable": stable,
                "cluster_lift_low": cluster_lift_low,
                "cluster_lift_high": cluster_lift_high,
                "cluster_robust": cluster_lift_low > 0,
                "mask": mask,
            }
        )
    rows.sort(key=lambda row: (row["stable"], row["lift"], row["n"]), reverse=True)
    return rows, pd.Timestamp(cutoff), cluster_base_interval


def bucket_rows(frame: pd.DataFrame, feature: str, buckets: list[tuple[str, Callable[[pd.Series], pd.Series]]]) -> list[dict[str, Any]]:
    rows = []
    for label, predicate in buckets:
        subset = frame[predicate(frame[feature]).fillna(False)]
        if len(subset) < 10:
            continue
        rows.append({"label": label, **stats(subset)})
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "暂无满足最低样本量的分组。"
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def generate_report(
    signals: pd.DataFrame,
    stock_count: int,
    universe_count: int,
    failures: list[dict[str, str]],
    args: argparse.Namespace,
    generated_at: datetime,
) -> tuple[str, dict[str, Any]]:
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    baseline = stats(signals)
    full_universe = stock_count == universe_count
    scope_label = "全市场股票" if full_universe else "抽样股票"
    coverage_note = (
        "本次覆盖当前股票池全部沪深非 ST A 股，但仍不包含退市、现已 ST 股票。"
        if full_universe
        else "本次为系统化抽样，不是全市场普查。"
    )
    condition_list = conditions()
    condition_rows, cutoff, cluster_base_interval = condition_table(signals, condition_list)
    baseline["cluster_ci_low"], baseline["cluster_ci_high"] = cluster_base_interval
    stable_risks = [row for row in condition_rows if row["stable"] and row["lift"] >= 3]

    # Do not count correlated market rules as independent risk dimensions.
    composite_conditions = [
        next(item for item in condition_list if item.key == "extended_from_line"),
        next(item for item in condition_list if item.key == "market_weak_regime"),
        next(item for item in condition_list if item.key == "high_pullback_volume"),
    ]

    signals["risk_count"] = 0
    for condition in composite_conditions:
        signals["risk_count"] += condition.predicate(signals).fillna(False).astype(int)
    composite_rows = []
    for risk_count in range(len(composite_conditions) + 1):
        subset = signals[signals["risk_count"] == risk_count]
        if len(subset) < 10:
            continue
        row = {"label": f"{risk_count}项", **stats(subset)}
        row["early_rate"] = stats(subset[subset["signal_date"] <= cutoff])["down_rate"]
        row["recent_rate"] = stats(subset[subset["signal_date"] > cutoff])["down_rate"]
        composite_rows.append(row)

    bucket_specs = [
        ("精准线确认次数", "line_strength", [("3次", lambda s: s <= 3), ("4-5次", lambda s: (s >= 4) & (s <= 5)), ("6次及以上", lambda s: s >= 6)]),
        ("突破量比", "breakout_volume_ratio", [("<1.8", lambda s: s < 1.8), ("1.8-2.5", lambda s: (s >= 1.8) & (s < 2.5)), (">=2.5", lambda s: s >= 2.5)]),
        ("回踩量比", "pullback_volume_ratio", [("<=0.35", lambda s: s <= 0.35), ("0.35-0.60", lambda s: (s > 0.35) & (s <= 0.60)), (">0.60", lambda s: s > 0.60)]),
        ("距精准线", "distance_to_line_pct", [("线下", lambda s: s < 0), ("0-1%", lambda s: (s >= 0) & (s < 1)), ("1-2%", lambda s: (s >= 1) & (s < 2)), (">=2%", lambda s: s >= 2)]),
        ("前20日涨跌", "prior_return_20_pct", [("<0", lambda s: s < 0), ("0-10%", lambda s: (s >= 0) & (s <= 10)), (">10%", lambda s: s > 10)]),
        ("沪深300位置", "market_distance_to_ma20_pct", [("MA20下", lambda s: s < 0), ("MA20上", lambda s: s >= 0)]),
    ]

    top_lines = []
    row_by_key = {row["key"]: row for row in condition_rows}
    top_risk_keys = ["extended_from_line", "market_weak_regime", "high_pullback_volume", "fast_pullback", "large_range"]
    for key in top_risk_keys:
        row = row_by_key.get(key)
        if not row or not row["stable"] or row["lift"] < 2:
            continue
        evidence = "稳健" if row["cluster_lift_low"] >= 1 else "边界" if row["cluster_lift_low"] > 0 else "探索"
        top_lines.append(
            f"- **{row['label']}**（{evidence}）：T+5 下跌率 {pct(row['down_rate'])}（{row['n']} 个），"
            f"较基准高 {pp(row['lift'])} 个百分点；前半段 {pct(row['early_rate'])}，后半段 {pct(row['recent_rate'])}；"
            f"按股票聚类的提升区间 {pp(row['cluster_lift_low'])} 至 {pp(row['cluster_lift_high'])} 个百分点。"
        )
    if not top_lines:
        top_lines.append("- 没有条件同时满足最低样本量、前后两段均高于基准且提升至少 3 个百分点；不应据此新增硬过滤。")

    condition_md_rows = [
        [
            row["label"], str(row["n"]), pct(row["down_rate"]),
            f"{pct(row['ci_low'])}-{pct(row['ci_high'])}", pp(row["lift"]),
            f"{pp(row['cluster_lift_low'])}至{pp(row['cluster_lift_high'])}",
            pct(row["early_rate"]), pct(row["recent_rate"]), pct(row["mean_return"], 2), pct(row["drawdown_3_rate"]),
        ]
        for row in condition_rows[:12]
    ]
    composite_md = markdown_table(
        ["风险项数", "样本", "T+5下跌率", "早期", "近期", "T+5中位数", "平均T+5", "5日回撤>=3%"],
        [[
            row["label"], str(row["n"]), pct(row["down_rate"]), pct(row["early_rate"]), pct(row["recent_rate"]),
            pct(row["median_return"], 2), pct(row["mean_return"], 2), pct(row["drawdown_3_rate"]),
        ] for row in composite_rows],
    )

    bucket_sections = []
    for title, feature, buckets in bucket_specs:
        rows = bucket_rows(signals, feature, buckets)
        table = markdown_table(
            ["分组", "样本", "T+5下跌率", "平均T+5", "5日回撤>=3%"],
            [[row["label"], str(row["n"]), pct(row["down_rate"]), pct(row["mean_return"], 2), pct(row["drawdown_3_rate"])] for row in rows],
        )
        bucket_sections.append(f"### {title}\n\n{table}")

    selected_labels = "、".join(condition.label for condition in composite_conditions)
    composite_monotonic = all(
        current["down_rate"] >= previous["down_rate"]
        and current["early_rate"] >= previous["early_rate"]
        and current["recent_rate"] >= previous["recent_rate"]
        for previous, current in zip(composite_rows, composite_rows[1:])
    )
    composite_advice = (
        "本样本中组合风险在全样本及前后时间段均随风险项数增加而恶化；达到2项时可先降级为观察。"
        if composite_monotonic
        else "组合风险没有呈现单调恶化，暂不应按风险项数增加硬过滤。"
    )
    recommendation_two = (
        "2. 当上述三个独立风险维度达到 2 项及以上时，可先把 `BUY_ALERT` 降为 `WATCH`，等待下一根 K 线重新站稳；这是候选规则，仍需下一轮样本外验证后再固化。"
        if composite_monotonic
        else "2. 上述组合风险未呈现单调恶化，不按风险项数直接降级；优先单独观察本报告中跨前后时段且聚类区间稳健的风险条件。"
    )
    if baseline["cluster_ci_low"] > 50:
        edge_conclusion = "聚类区间整体高于 50%，当前信号在一周方向上表现弱于随机基准。"
    elif baseline["cluster_ci_high"] < 50:
        edge_conclusion = "聚类区间整体低于 50%，当前信号在一周方向上显示出统计优势。"
    else:
        edge_conclusion = "聚类区间覆盖 50%，当前信号的一周方向胜率没有统计上的明确优势。"
    if baseline["mean_return"] > 0 and baseline["median_return"] < 0:
        distribution_conclusion = "平均收益为正而中位数为负，收益依赖少数较大的上涨样本。"
    elif baseline["mean_return"] > 0 and baseline["median_return"] == 0:
        distribution_conclusion = "平均收益为正而中位数为零，正收益主要来自分布右侧较大的上涨样本。"
    elif baseline["mean_return"] > 0 and baseline["median_return"] > 0:
        distribution_conclusion = "平均收益与中位数均为正，收益分布的中心位于零以上。"
    elif baseline["mean_return"] < 0 and baseline["median_return"] < 0:
        distribution_conclusion = "平均收益与中位数均为负，收益分布的中心位于零以下。"
    else:
        distribution_conclusion = "平均收益与中位数方向不一致，收益分布存在明显偏斜。"
    report = f"""# A股历史预警一周失效分析报告

生成时间：{generated_at.strftime('%Y-%m-%d %H:%M:%S')}  
策略：Stock Mine 当前 `BUY_ALERT` 逻辑  
数据源：AKShare/Tencent 前复权日线；沪深300作为大盘代理

## 结论摘要

- {scope_label} {stock_count} 只（当前沪深非 ST 股票池 {universe_count} 只），逐股回放最近 {args.lookback} 个交易日；成功 {stock_count - len(failures)} 只，失败 {len(failures)} 只。
- 其中 {signals['stock_code'].astype(str).nunique()} 只股票产生过信号，共 {len(signals)} 个去重历史预警，具备完整 T+5 的样本 {baseline['n']} 个。以信号日收盘为基准，T+5 收盘下跌概率为 **{pct(baseline['down_rate'])}**；按股票聚类 bootstrap 的 95% 区间为 {pct(baseline['cluster_ci_low'])}-{pct(baseline['cluster_ci_high'])}。
- T+5 平均收益 {pct(baseline['mean_return'], 2)}，中位数 {pct(baseline['median_return'], 2)}；按下一交易日开盘作为可执行买点，下跌概率 {pct(baseline['next_open_down_rate'])}。
- {edge_conclusion}{distribution_conclusion}
- 未来 5 日内最大回撤达到 3% 的概率 {pct(baseline['drawdown_3_rate'])}；T+5 跑输沪深300的概率 {pct(baseline['underperform_rate'])}。
- 时间稳定性分割点为 {cutoff.strftime('%Y-%m-%d')}。下列“高风险条件”要求早期和近期样本都比各自基准差，不只看全样本最高值。

## 高风险条件

{chr(10).join(top_lines)}

这些是相关性，不是因果。建议先将稳定高风险条件用于“降级为观察”或仓位约束，不直接据此反向做空。

## 条件统计

{markdown_table(
    ['条件', '样本', 'T+5下跌率', 'Wilson区间', '高于基准(百分点)', '聚类提升95%区间(百分点)', '早期', '近期', '平均T+5', '5日回撤>=3%'],
    condition_md_rows,
)}

## 组合风险

组合使用的互不重复风险维度：{selected_labels}。

{composite_md}

{composite_advice}

## 关键变量分组

{chr(10).join(bucket_sections)}

## 对当前选股逻辑的建议

1. 保留“突破后缩量回踩 + MA20向上 + 精准线附近”作为结构入口，不因本报告直接改动核心判定。
{recommendation_two}
3. 大盘条件必须使用信号日的历史状态。当前行业分只保存最新快照，不能回填到历史信号，本报告没有把它伪装成历史行业因子。
4. 执行评估应优先看“下一交易日开盘到 T+5”，因为收盘后产生的预警无法保证按信号收盘价成交。
5. 精准线确认次数和突破量比是否有效，以本报告的分组统计为准；不要在没有样本外验证时直接提高 `min_line_strength` 或 `breakout_vol_mult`。
6. 下一步应做全市场、滚动年度和行业分层复盘，再将候选过滤规则固定后进行真正的样本外验证。

## 方法与限制

- 每个交易日仅向策略传入当日及此前数据；同一股票 20 个交易日内连续预警只保留首个。
- “一周下跌”主口径为第 5 个交易日收盘价低于信号日收盘；另报告下一日开盘执行口径与 5 日内最大回撤。
- 股票来自当前上市公司名单，存在幸存者偏差；退市、现已 ST 股票未进入样本。
- 前复权历史可能带来公司行为调整偏差；未计佣金、印花税、滑点、涨跌停导致的不可成交。
- {coverage_note}条件筛选仍有多重比较风险；95% 区间只描述单一比例的不确定性。
- 没有可靠的历史行业成分与当时行业风险快照，因此只使用沪深300历史行情，不评价行业因子的历史增益。

## 数据文件

- 逐信号明细 CSV：与本报告同目录、同时间戳的 `历史预警失效明细_*.csv`
- 机器可读摘要 JSON：与本报告同目录、同时间戳的 `历史预警失效摘要_*.json`
"""
    summary = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "parameters": vars(args),
        "universe_count": universe_count,
        "sample_stock_count": stock_count,
        "full_universe": full_universe,
        "stocks_with_signals": int(signals["stock_code"].astype(str).nunique()),
        "failed_stock_count": len(failures),
        "baseline": baseline,
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "stable_risks": [{key: value for key, value in row.items() if key != "mask"} for row in stable_risks],
        "composite_conditions": [condition.key for condition in composite_conditions],
        "composite_monotonic": composite_monotonic,
        "composite": composite_rows,
        "failures": failures,
    }
    return report, summary


def main() -> None:
    args = parse_args()
    if args.resume and not args.checkpoint:
        raise RuntimeError("--resume 必须与 --checkpoint 一起使用")
    started = time.monotonic()
    provider = create_provider(args.provider)
    if hasattr(provider, "_history_source"):
        provider._history_source = "tencent"
    stocks = provider.get_stock_list().copy()
    stocks["stock_code"] = stocks["stock_code"].astype(str).str.zfill(6)
    universe_count = len(stocks)
    selected = systematic_sample(stocks, args.sample)
    benchmark = load_benchmark()
    years = max(3, math.ceil(args.lookback / 220) + 2)
    all_signals: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    completed: dict[str, dict[str, Any]] = {}
    checkpoint_file = None
    if args.checkpoint:
        completed, checkpoint_file = prepare_checkpoint(
            Path(args.checkpoint), checkpoint_identity(args, selected), args.resume
        )
        for item in completed.values():
            all_signals.extend(item.get("records", []))
            if item.get("error"):
                failures.append(
                    {
                        "stock_code": str(item["stock_code"]).zfill(6),
                        "stock_name": str(item.get("stock_name", "")),
                        "error": str(item["error"]),
                    }
                )

    selected_codes = selected["stock_code"].astype(str).str.zfill(6)
    pending = selected[~selected_codes.isin(completed)].copy()
    progress = {"done": len(completed), "session_done": 0}
    if completed:
        print(
            f"从检查点恢复：completed={len(completed)} signals={len(all_signals)} "
            f"failures={len(failures)} pending={len(pending)}",
            flush=True,
        )

    payloads = [
        (
            str(row.stock_code).zfill(6),
            str(row.short_name),
            str(getattr(row, "exchange", "")),
            years,
            args.lookback,
        )
        for row in pending.itertuples(index=False)
    ]

    def record_result(result: dict[str, Any]) -> None:
        code = str(result["stock_code"]).zfill(6)
        name = str(result["stock_name"])
        error = result.get("error")
        records = [
            {
                "stock_code": code,
                "stock_name": name,
                "exchange": str(result.get("exchange", "")),
                **signal,
                **benchmark_features(benchmark, signal["signal_date"], signal),
            }
            for signal in result.get("signals", [])
        ]
        all_signals.extend(records)
        if error:
            failures.append({"stock_code": code, "stock_name": name, "error": error})
        if checkpoint_file is not None:
            checkpoint_file.write(
                json.dumps(
                    {
                        "stock_code": code,
                        "stock_name": name,
                        "records": records,
                        "error": error,
                    },
                    ensure_ascii=False,
                    default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
                )
                + "\n"
            )
        progress["done"] += 1
        progress["session_done"] += 1
        done = progress["done"]
        if done == 1 or done % 10 == 0 or done == len(selected):
            elapsed = time.monotonic() - started
            session_done = progress["session_done"]
            rate = session_done / elapsed if elapsed > 0 else 0
            eta = (len(selected) - done) / rate if rate > 0 else 0
            print(
                f"[{done}/{len(selected)}] signals={len(all_signals)} failures={len(failures)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
            if checkpoint_file is not None:
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())

    try:
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            initializer=initialize_worker,
            initargs=(args.provider,),
        ) as executor:
            futures = [executor.submit(evaluate_stock_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                record_result(future.result())
    finally:
        if checkpoint_file is not None:
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
            checkpoint_file.close()

    if not all_signals:
        raise RuntimeError("No historical alerts were produced; report cannot be generated.")
    signals = pd.DataFrame(all_signals)
    generated_at = datetime.now()
    report, summary = generate_report(signals, len(selected), universe_count, failures, args, generated_at)
    stamp = generated_at.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"历史预警失效分析_{stamp}.md"
    csv_path = output_dir / f"历史预警失效明细_{stamp}.csv"
    json_path = output_dir / f"历史预警失效摘要_{stamp}.json"
    report_path.write_text(report, encoding="utf-8")
    signals.sort_values(["signal_date", "stock_code"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "csv": str(csv_path), "summary": str(json_path), "signals": len(signals)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
