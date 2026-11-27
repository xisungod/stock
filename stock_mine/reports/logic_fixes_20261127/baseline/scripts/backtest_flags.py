from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import pandas as pd

from app.services.backtest import aggregate_reports, evaluate_bars
from app.services.flag_pattern import FlagConfig, FLAG_VERSION
from app.services.strategy import STRATEGY_VERSION, StrategyConfig
from sweep_line_strength import load_universe


def replay(payload):
    code, name, folder, lookback = payload
    try:
        cached = json.loads((Path(folder) / f"{code}.json").read_text(encoding="utf-8"))
        bars = pd.DataFrame(cached["bars"])
        report = evaluate_bars(
            code, name, bars, lookback_days=lookback,
            config=StrategyConfig(strategy_mode="flag_breakout", signal_offset=0),
        )
        report["data_end_date"] = str(bars.trade_date.max())
        return report, None
    except Exception as exc:
        return None, {"stock_code": code, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Frozen-rule, point-in-time flag replay; no network or trading.")
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
    started = time.monotonic()
    reports, failures = [], []
    payloads = [(str(r.stock_code), str(r.short_name), args.cache_dir, args.lookback) for r in stocks.itertuples()]
    with ProcessPoolExecutor(max_workers=max(1, min(4, args.workers))) as executor:
        tasks = [executor.submit(replay, payload) for payload in payloads]
        for done, task in enumerate(as_completed(tasks), 1):
            report, error = task.result()
            if report:
                reports.append(report)
            if error:
                failures.append(error)
            if done % 10 == 0 or done == len(tasks):
                count = sum(r["summary"]["signal_count"] for r in reports)
                print(f"{done}/{len(tasks)} signals={count} failures={len(failures)} elapsed={time.monotonic()-started:.0f}s", flush=True)
    summary = aggregate_reports(reports)
    events = [p for r in reports for p in r.get("pattern_events", [])]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_version": STRATEGY_VERSION, "flag_version": FLAG_VERSION,
        "parameters": vars(args), "flag_parameters": asdict(FlagConfig()),
        "breakout_volume_multiplier": 1.5,
        "selected_codes": stocks.stock_code.tolist(), "summary": summary,
        "observed_events": dict(Counter(p["status"] for p in events)),
        "data_end_dates": dict(Counter(r["data_end_date"] for r in reports)),
        "failures": failures, "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"flag_backtest_{datetime.now():%Y%m%d_%H%M%S}"
    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    records = [
        {"stock_code": r["stock_code"], "stock_name": r["stock_name"], **signal,
         "flag_pattern": json.dumps(signal["flag_pattern"], ensure_ascii=False)}
        for r in reports for signal in r["signals"]
    ]
    pd.DataFrame(records).to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    def pct(value):
        return "-" if value is None else f"{value:.3f}%"
    lines = [
        "# 三类旗形独立回测", "", f"生成时间：{payload['generated_at']}；版本：{FLAG_VERSION}", "",
        f"- 当前上市且有缓存的股票{'全部覆盖' if args.sample <= 0 else '均匀抽样'} {len(stocks)} 只，每只回放最近 {args.lookback} 根日线。",
        "- 原精准线策略不参与旗形筛选；方向为下倾、水平、上倾，所有类别使用相同规则。",
        "- 旗杆3～15根，涨幅至少10%及启动前4倍ATR，净上涨/累计绝对涨跌至少0.65。",
        "- 旗面4～20根；独立上下轨触碰各至少2次；带宽不超过旗杆高度50%；斜率绝对值不超过旗杆的50%。",
        "- 上下轨斜率差形成的全段宽度变化不超过平均带宽30%；至少90%的高低价在容差带内。",
        "- 后/前半段成交量中位数不超过0.8，后半段日内振幅中位数不超过前半段1.2倍。",
        "- 信号日不参与拟合；收盘高于上轨0.2倍ATR、量比至少1.5才触发。仅事前结构决定采用哪个候选。",
        "- 同股预警20根日线去重，再按事件当时的方向分组；成交模型与现有策略相同，成本假设20bps。",
        "- 不可成交入场和未完成样本单独计数；一字或零成交量买入不算成交，受阻卖出延后。",
        "- 前复权日线、现存股票池和日线成交近似存在局限；不构成样本外或实际盈利证明。", "",
        "| 旗面 | 预警数 | T+5样本 | T+5净胜率 | T+5平均净收益 | T+10平均净收益 | T+20平均净收益 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("flag_directions", []):
        lines.append(
            f"| {row['label']} | {row['signal_count']} | {row.get('net_t5_sample_count', 0)} | "
            f"{pct(row.get('net_t5_positive_rate'))} | {pct(row.get('net_t5_average_return'))} | "
            f"{pct(row.get('net_t10_average_return'))} | {pct(row.get('net_t20_average_return'))} |"
        )
    lines += [
        "", "## 解释边界", "",
        f"- 失败股票：{len(failures)}；不可入场预警：{summary.get('blocked_entry_count', 0)}；待入场预警：{summary.get('pending_entry_count', 0)}。",
        f"- 识别事件计数（非独立交易数）：{json.dumps(payload['observed_events'], ensure_ascii=False)}。",
        "- 失效事件及亏损交易不会因为后续失败而从回放中删去；未识别出来的结构不纳入这些事件计数。",
        "- 本轮未根据收益调参。少量样本的高胜率尤其不可靠，应锁定规则继续独立前向验证。",
        "- 同名JSON含规则、行情截止日期分布和分型完整统计，CSV保留每个交易事件的原始通道。", "",
    ]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(stem.with_suffix(".md"), flush=True)


if __name__ == "__main__":
    main()
