from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.backtest import evaluate_bars, summarize_signals
from app.services.strategy import STRATEGY_VERSION, StrategyConfig
from sweep_line_strength import load_universe


VARIANTS = {
    "baseline": "修正后基线",
    "macd_rising": "MACD 柱连续两次改善",
    "normal_volume": "回踩量不高于突破前20日均量",
    "recovery_close": "收阳且收盘位于当日振幅上40%",
}


def accepts(row: dict, variant: str) -> bool:
    if variant == "baseline":
        return True
    if variant == "macd_rising":
        return row.get("macd_hist_rising") is True
    if variant == "normal_volume":
        ratio = row.get("pullback_vs_prebreakout_volume")
        return ratio is not None and ratio <= 1.0
    if variant == "recovery_close":
        return row.get("signal_body_pct", -1) >= 0 and row.get("close_location_pct", 0) >= 60
    raise ValueError(f"unknown variant: {variant}")


def filter_and_dedupe(rows: list[dict], variant: str, min_gap: int = 20) -> list[dict]:
    last: dict[str, int] = {}
    selected = []
    for row in sorted(rows, key=lambda item: (item["stock_code"], item["signal_index"])):
        code, idx = row["stock_code"], row["signal_index"]
        # 条件先于去重，不能先取基线的第一天再筛，遗漏后续才满足条件的预警。
        if accepts(row, variant) and (code not in last or idx - last[code] >= min_gap):
            selected.append(row)
            last[code] = idx
    return selected


def replay(payload: tuple[str, str, str, int]) -> dict:
    code, name, cache_dir, lookback = payload
    try:
        cache = json.loads((Path(cache_dir) / f"{code}.json").read_text(encoding="utf-8"))
        report = evaluate_bars(
            code, name, pd.DataFrame(cache["bars"]), lookback_days=lookback,
            min_signal_gap=0, config=StrategyConfig(min_line_strength=3, breakout_vol_mult=1.5),
        )
        rows = [{"stock_code": code, "stock_name": name, **row} for row in report["signals"]]
        return {"rows": rows, "error": None}
    except Exception as exc:
        return {"rows": [], "error": {"stock_code": code, "error": str(exc)}}


def comparison_ci(baseline: list[dict], variant: list[dict], iterations: int = 2000) -> dict:
    def daily(rows):
        frame = pd.DataFrame([
            {"date": row["signal_date"], "pnl": row["net_return_5"], "win": int(row["net_return_5"] > 0)}
            for row in rows if row.get("net_return_5") is not None
        ])
        if frame.empty:
            return pd.DataFrame(columns=["pnl", "win", "count"])
        return frame.groupby("date").agg(pnl=("pnl", "sum"), win=("win", "sum"), count=("pnl", "count"))

    base, other = daily(baseline), daily(variant)
    if base.empty or other.empty:
        return {"net_mean_delta_95": None, "winrate_delta_pp_95": None}
    dates = base.index.union(other.index).sort_values()
    blocks = np.arange(len(dates)) // 20
    left = base.reindex(dates, fill_value=0).groupby(blocks).sum().to_numpy()
    right = other.reindex(dates, fill_value=0).groupby(blocks).sum().to_numpy()
    rng = np.random.default_rng(20260905)
    samples = []
    for _ in range(iterations):
        indices = rng.integers(0, len(left), size=len(left))
        b, v = left[indices].sum(axis=0), right[indices].sum(axis=0)
        if b[2] and v[2]:
            samples.append((v[0] / v[2] - b[0] / b[2], (v[1] / v[2] - b[1] / b[2]) * 100))
    if not samples:
        return {"net_mean_delta_95": None, "winrate_delta_pp_95": None}
    intervals = np.percentile(np.array(samples), [2.5, 97.5], axis=0)
    return {
        "net_mean_delta_95": intervals[:, 0].round(3).tolist(),
        "winrate_delta_pp_95": intervals[:, 1].round(2).tolist(),
        "date_block_count": len(left),
    }


def build_report(rows: list[dict]) -> dict:
    dates = sorted({row["signal_date"] for row in rows})
    split = dates[len(dates) * 2 // 3] if dates else ""
    early_end = dates[max(0, len(dates) * 2 // 3 - 20)] if dates else ""
    by_variant = {key: filter_and_dedupe(rows, key) for key in VARIANTS}
    report = {"split_date": split, "early_signal_cutoff_exclusive": early_end, "variants": {}}
    for key, selected in by_variant.items():
        early = [
            row for row in selected if row["signal_date"] < early_end
            and row.get("exit_date_20") is not None and row["exit_date_20"] < split
        ]
        late = [row for row in selected if row["signal_date"] >= split]
        report["variants"][key] = {
            "label": VARIANTS[key], "all": summarize_signals(selected),
            "early": summarize_signals(early), "late": summarize_signals(late),
            "comparison": comparison_ci(by_variant["baseline"], selected),
        }
    return report


def markdown(report: dict) -> str:
    def value(x):
        return "-" if x is None else str(x)
    lines = [
        "# 选股附加条件对照评估", "",
        f"生成时间：{report['generated_at']}；策略版本：`{STRATEGY_VERSION}`", "",
        "## 方法", "",
        f"- 当前缓存股票池均匀抽样 {report['stock_count']} 只，每只重放最近 {report['parameters']['lookback']} 根日线。",
        "- 基线：最低阻力点3、突破量比1.5；仅修复实现，不增加形态条件。",
        "- 全部候选日先判断附加条件，再按同股20根日线去重，防止事后切片遗漏信号。",
        "- 下一根有效日线开盘入场、T+5/10/20固定退出，费用与滑点合计假设20bps。",
        "- 一字或零成交量日不买入，卖出受阻则延后；未完成交易不计入净收益分母，单独计数。",
        f"- 后段从 {report['split_date']} 开始；前段去掉边界前20个信号日期，并要求退出日在边界之前。",
        "- 差值区间按连续20个信号日期分块、同步重采样基线与候选条件，2000次；没有多重检验校正。",
        "- 这是已使用历史数据上的探索性比较，不是全新样本外验证，也不是相对大盘的超额收益证明。", "",
        "## 结果", "",
        "| 条件 | 信号数 | T+5成交样本 | T+5净胜率 | T+5平均净收益 | T+10平均净收益 | T+20平均净收益 | 后段T+5净胜率 | 后段T+5均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["variants"].values():
        a, late = item["all"], item["late"]
        lines.append(
            f"| {item['label']} | {a['signal_count']} | {a.get('net_t5_sample_count', 0)} | "
            f"{value(a.get('net_t5_positive_rate'))}% | {value(a.get('net_t5_average_return'))}% | "
            f"{value(a.get('net_t10_average_return'))}% | {value(a.get('net_t20_average_return'))}% | "
            f"{value(late.get('net_t5_positive_rate'))}% | {value(late.get('net_t5_average_return'))}% |"
        )
    lines += ["", "## 相对基线的不确定性", ""]
    for key, item in report["variants"].items():
        if key != "baseline":
            ci = item["comparison"]
            lines.append(
                f"- {item['label']}：T+5均值差95%区间 {ci['net_mean_delta_95']}；"
                f"净胜率差（百分点）95%区间 {ci['winrate_delta_pp_95']}。"
            )
    lines += [
        "", "## 决策边界", "",
        "- 本报告不自动修改生产参数。均值、净胜率、后段表现和样本覆盖需共同评估，不能只取最高胜率。",
        "- 通过探索筛选也不代表有效，仍需锁定规则后在未使用的数据或前向模拟中验证。",
        "- 股票池为当前上市且有缓存的股票，存在幸存者偏差；本次不声称覆盖全部A股。",
        "- 复权行情缺少历史订单簿、完整停牌日历及历史风险警示状态，实际成交可能更差。",
        f"- 数据读取或回放失败 {len(report['failures'])} 只；详细排除与待完成样本计数见同名JSON。", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--lookback", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", default="data/bars_cache")
    parser.add_argument("--stock-list", default="data/stock_list_cache.json")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    stocks = load_universe(Path(args.stock_list), Path(args.cache_dir), args.sample)
    if stocks.empty:
        raise ValueError("no cached stocks")
    rows, failures = [], []
    started = time.monotonic()
    payloads = [(str(r.stock_code), str(r.short_name), args.cache_dir, args.lookback) for r in stocks.itertuples()]
    with ProcessPoolExecutor(max_workers=max(1, min(4, args.workers))) as pool:
        futures = [pool.submit(replay, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            rows.extend(result["rows"])
            if result["error"]:
                failures.append(result["error"])
            if done % 10 == 0 or done == len(futures):
                print(f"{done}/{len(futures)} raw_signals={len(rows)} failures={len(failures)} elapsed={time.monotonic()-started:.0f}s", flush=True)
    report = {
        **build_report(rows), "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": vars(args), "stock_count": len(stocks), "selected_codes": stocks.stock_code.tolist(),
        "strategy_version": STRATEGY_VERSION, "failures": failures,
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / "backend/app/services" / name).read_bytes()).hexdigest()
            for name in ("strategy.py", "zigzag.py", "backtest.py")
        },
    }
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"signal_filter_comparison_{datetime.now():%Y%m%d_%H%M%S}"
    stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(str(stem.with_suffix(".md")), flush=True)


if __name__ == "__main__":
    main()
