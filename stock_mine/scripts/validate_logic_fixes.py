"""2026-11-27 选股逻辑修复：真实行情对照验证。

规则、样本、分段、成本、判定标准全部写在本文件里，先于结果固定；看到结果后不改参数。
- 样本：当前沪深 A 股（0/3/6 开头、名称不含 ST/退）按代码哈希排序取前 N 只，不按收益挑选。
- 数据：腾讯前复权日线；沪深300 指数用于大盘环境。
- 信号：platform_events 精准线引擎，逐日只用当日及以前的数据；同一冻结突破事件只取首个预警；
  每组规则都用真实配置独立重放（引擎每日只输出最优线，不能用旧结果事后过滤代替）。
- 成交：信号次日开盘买入（次日一字涨停则买不到），第 h 个交易日收盘卖出，双边成本 20bps。
- 基准：同一信号日、同一持有期的全部样本股等权收益（同口径计算）；超额 = 信号收益 − 基准。
- 分段：开发段 2019-01-01～2023-12-31；留出段 2024-01-01～评估截止日（末尾 25 个交易日不出信号）。
- 判定（事先声明）：某组「有效」须在开发段与留出段的 T+5 超额均值都 > 0，
  且按月分块 bootstrap 的 95% 区间下限都 > 0。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import pandas as pd

from app.schemas import production_strategy_config
from app.services.backtest import evaluate_bars
from app.services.market_regime import regime_on, regime_table
from app.services.strategy import _normalize_bars
from app.services.tencent_bars import fetch_tencent_bars

HORIZONS = (5, 10, 20)
COST_PCT = 0.20
SEED = "logic-fix-20261127"
EVAL_START = "2019-01-01"
PERIODS = (("development", "2019-01-01", "2023-12-31"), ("holdout", "2024-01-01", None))
PURGE_SESSIONS = 25
NORMAL_VOLUME, RUNUP, EVENT_AGE = 1.0, 0.10, 750

OLD_RULES = dict(pullback_max_normal_volume_ratio=None, max_post_breakout_runup_pct=None,
                 max_event_age_bars=None, market_filter="off")
NEW_RULES = dict(pullback_max_normal_volume_ratio=NORMAL_VOLUME, max_post_breakout_runup_pct=RUNUP,
                 max_event_age_bars=EVENT_AGE, market_filter="off")
ARM_RULES = {
    "old": OLD_RULES,
    "only_normal_volume": {**OLD_RULES, "pullback_max_normal_volume_ratio": NORMAL_VOLUME},
    "only_runup": {**OLD_RULES, "max_post_breakout_runup_pct": RUNUP},
    "only_event_age": {**OLD_RULES, "max_event_age_bars": EVENT_AGE},
    "new": NEW_RULES,
}
ARMS = {
    "old": "修复前规则",
    "only_normal_volume": "仅加：回踩量≤常态量",
    "only_runup": "仅加：突破后冲高≤10%",
    "only_event_age": "仅加：证据时效750根",
    "new": "修复后规则（三条全加）",
    # 实盘降级只改已选中结果的类别、不重新挑线，所以由 new 组按信号日大盘状态过滤即与线上等价。
    "new_downgrade": "修复后 + 大盘偏弱不买",
}


def config(rules: dict):
    return production_strategy_config(strategy_mode="precision_pullback", include_chart=False, **rules)


def symbol(code: str) -> str:
    return ("sh" if code.startswith("6") else "sz") + code


def stock_universe() -> pd.DataFrame:
    from app.services.data_provider import create_provider

    # 与线上扫描同一来源：交易所名单失败时退回东财，再退回本地缓存。
    frame = create_provider("akshare").get_stock_list().rename(columns={"short_name": "stock_name"})
    frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    keep = frame["stock_code"].str[0].isin(["0", "3", "6"]) & ~frame["stock_name"].str.contains("ST|退", regex=True)
    frame = frame[keep].copy()
    frame["rank"] = frame["stock_code"].map(lambda code: hashlib.sha256(f"{SEED}:{code}".encode()).hexdigest())
    return frame.sort_values("rank").reset_index(drop=True)


def load_bars(code: str, cache: Path) -> pd.DataFrame:
    path = cache / f"{code}.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text())["bars"])
    for attempt in range(3):
        try:
            bars = fetch_tencent_bars(symbol(code))
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    path.write_text(json.dumps({"code": code, "bars": bars.to_dict("records")}))
    return bars


def load_settled_bars(path: Path, as_of: str) -> pd.DataFrame:
    """与 evaluate_bars 完全相同的清洗（剔除前复权产生的非正价格等无效行），行号才能对上信号。"""
    bars = _normalize_bars(pd.DataFrame(json.loads(Path(path).read_text())["bars"]))
    bars = bars[bars["trade_date"] <= pd.Timestamp(as_of)].reset_index(drop=True)
    bars["trade_date"] = bars["trade_date"].dt.strftime("%Y-%m-%d")
    return bars


def forward_returns(df: pd.DataFrame, index: int) -> dict[int, float | None]:
    """次日开盘买入、第 h 根收盘卖出的净收益（%）；次日一字涨停视为买不到。"""
    entry = index + 1
    if entry >= len(df):
        return {h: None for h in HORIZONS}
    o, high, low, close = (float(df[name].iloc[entry]) for name in ("open", "high", "low", "close"))
    if o == high == low == close and close > float(df["close"].iloc[index]):
        return {h: None for h in HORIZONS}
    out = {}
    for horizon in HORIZONS:
        exit_index = index + horizon
        out[horizon] = ((float(df["close"].iloc[exit_index]) / o - 1) * 100 - COST_PCT
                        if exit_index < len(df) and o > 0 else None)
    return out


def replay(task: tuple[str, str, str, str]) -> dict:
    code, name, path, as_of = task
    try:
        bars = load_settled_bars(Path(path), as_of)
        if len(bars) < 200 or bars["trade_date"].iloc[-1] < EVAL_START:
            return {"code": code, "signals": [], "returns": {}, "error": "history too short"}
        start = int((bars["trade_date"] >= EVAL_START).to_numpy().argmax())
        dates = bars["trade_date"].tolist()
        signals = []
        for arm, rules in ARM_RULES.items():
            report = evaluate_bars(code, name, bars, lookback_days=len(bars) - start, config=config(rules), as_of=as_of)
            for record in report["signals"]:
                index = record["signal_index"]
                if dates[index] != record["signal_date"]:
                    raise AssertionError(f"{code} 行号 {index} 对应 {dates[index]}，信号日为 {record['signal_date']}")
                breakout = dates.index(record["breakout_date"])
                peak = float(bars["high"].iloc[breakout:index + 1].max())
                signals.append({
                    "arm": arm, "stock_code": code, "signal_date": record["signal_date"], "signal_index": index,
                    "event_id": record["signal_event_id"], "line_price": record["key_line_price"],
                    "line_points": record["line_strength"], "breakout_date": record["breakout_date"],
                    "limit_up_breakout": record.get("breakout_limit_up"),
                    "normal_volume_ratio": record.get("pullback_normal_volume_ratio"),
                    "runup_pct": round((peak / record["key_line_price"] - 1) * 100, 3),
                    **{f"ret_{h}": value for h, value in forward_returns(bars, index).items()},
                })
        returns = {}
        for index in range(start, len(bars) - 1):
            values = forward_returns(bars, index)
            returns[dates[index]] = [values[h] for h in HORIZONS]
        return {"code": code, "signals": signals, "returns": returns, "error": None}
    except Exception as exc:
        return {"code": code, "signals": [], "returns": {}, "error": f"{type(exc).__name__}: {exc}"}


def select_arm(signals: pd.DataFrame, key: str) -> pd.DataFrame:
    if key == "new_downgrade":
        return signals[(signals.arm == "new") & signals.market_ok]
    return signals[signals.arm == key]


def block_bootstrap(frame: pd.DataFrame, column: str, iterations: int = 4000) -> list[float] | None:
    """按自然月分块重采样：同月信号受同一行情驱动，不能当独立样本。"""
    valid = frame[["signal_date", column]].dropna()
    if valid["signal_date"].nunique() < 10:
        return None
    months = pd.to_datetime(valid["signal_date"]).dt.to_period("M")
    blocks = valid.groupby(months)[column].agg(["sum", "count"])
    values, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    rng = np.random.default_rng(20261127)
    picks = rng.integers(0, len(blocks), size=(iterations, len(blocks)))
    means = values[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return [round(float(x), 3) for x in np.percentile(means, [2.5, 97.5])]


def summarize(arm: pd.DataFrame) -> dict:
    out = {"signals": int(len(arm)), "stocks": int(arm["stock_code"].nunique()) if len(arm) else 0,
           "signal_days": int(arm["signal_date"].nunique()) if len(arm) else 0}
    for h in HORIZONS:
        net = arm[f"ret_{h}"].dropna() if len(arm) else pd.Series(dtype=float)
        excess = arm[f"excess_{h}"].dropna() if len(arm) else pd.Series(dtype=float)
        out[f"t{h}"] = {
            "n": int(len(net)),
            "mean_net": round(float(net.mean()), 3) if len(net) else None,
            "median_net": round(float(net.median()), 3) if len(net) else None,
            "win_rate": round(float((net > 0).mean() * 100), 1) if len(net) else None,
            "mean_excess": round(float(excess.mean()), 3) if len(excess) else None,
            "beat_benchmark_rate": round(float((excess > 0).mean() * 100), 1) if len(excess) else None,
            "excess_ci95": block_bootstrap(arm, f"excess_{h}") if len(arm) else None,
        }
    return out


def fmt(value, suffix="") -> str:
    return "-" if value is None else f"{value}{suffix}"


def method_lines() -> list[str]:
    lines: list[str] = []
    for raw in __doc__.strip().splitlines()[3:]:
        if raw.startswith("- "):
            lines.append(raw)
        elif raw.startswith("  ") and lines:
            lines[-1] += raw.strip()
    return lines


def markdown(meta: dict, results: dict, verdicts: dict) -> str:
    lines = [
        "# 选股逻辑修复：真实行情对照验证", "",
        f"生成时间：{meta['generated_at']}；评估截止：{meta['as_of']}；耗时 {meta['elapsed_min']} 分钟。", "",
        "## 方法（先于结果固定）", "", *method_lines(),
        f"- 实际样本：成功 {meta['succeeded']} / 抽取 {meta['sampled']} 只；失败 {meta['failed']} 只。", "",
    ]
    for period, arms in results.items():
        lines += [f"## {period}", "",
                  "| 组 | 信号 | 股票 | T+5 净均值 | T+5 胜率 | T+5 超额均值 | 超额95%区间 | 跑赢基准 | T+10 超额 | T+20 超额 |",
                  "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
        for key, item in arms.items():
            t5, t10, t20 = item["t5"], item["t10"], item["t20"]
            ci = t5["excess_ci95"]
            lines.append(
                f"| {ARMS[key]} | {item['signals']} | {item['stocks']} | {fmt(t5['mean_net'], '%')} | "
                f"{fmt(t5['win_rate'], '%')} | {fmt(t5['mean_excess'], '%')} | "
                f"{'-' if ci is None else f'{ci[0]}%～{ci[1]}%'} | {fmt(t5['beat_benchmark_rate'], '%')} | "
                f"{fmt(t10['mean_excess'], '%')} | {fmt(t20['mean_excess'], '%')} |"
            )
        lines.append("")
    lines += ["## 判定（按事先声明的标准）", ""]
    lines += [f"- {ARMS[key]}：{verdict}" for key, verdict in verdicts.items()]
    lines += ["", "## 限制", "",
              "- 股票池为当前上市、非 ST 的公司，存在幸存者偏差；退市股的失败案例不在样本里。",
              "- 前复权价格不等于当年真实成交价；未模拟跌停卖不出、停牌、滑点随成交额变化。",
              "- 留出段只是事先划定的时间段，此前的研究已用过同期行情，不等于真正的前向验证。",
              "- 多组同时比较存在多重检验风险；只有事先声明的判定标准才作为结论。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--as-of", required=True, help="评估截止日（最近已收盘交易日）")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/logic_fixes_20261127/validation")
    args = parser.parse_args()
    started = time.monotonic()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    sample = stock_universe().head(args.sample)
    names = dict(zip(sample.stock_code, sample.stock_name))
    fetched, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(load_bars, code, args.cache): code for code in sample.stock_code}
        for done, future in enumerate(as_completed(futures), 1):
            code = futures[future]
            try:
                future.result()
                fetched.append(code)
            except Exception as exc:
                failures.append({"code": code, "stage": "fetch", "error": str(exc)})
            if done % 50 == 0:
                print(f"fetch {done}/{len(futures)} failures={len(failures)}", flush=True)

    tasks = [(code, names[code], str(args.cache / f"{code}.json"), args.as_of) for code in sorted(fetched)]
    raw, returns = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(replay, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            item = future.result()
            if item["error"]:
                failures.append({"code": item["code"], "stage": "replay", "error": item["error"]})
            raw.extend(item["signals"])
            returns[item["code"]] = item["returns"]
            if done % 25 == 0:
                print(f"replay {done}/{len(futures)} signals={len(raw)} elapsed={time.monotonic() - started:.0f}s", flush=True)

    index = regime_table(fetch_tencent_bars("sh000300"))
    index = index[index.index <= args.as_of]
    signals = pd.DataFrame(raw)
    signals["market_weak"] = signals["signal_date"].map(lambda day: (regime_on(index, day) or {}).get("weak"))
    signals["market_ok"] = signals["market_weak"] != True  # noqa: E712 —— 缺指数数据时不过滤，与实盘一致

    rows = [(day, *values) for series in returns.values() for day, values in series.items()]
    bench = pd.DataFrame(rows, columns=["signal_date", *(f"bench_{h}" for h in HORIZONS)]).groupby("signal_date").mean()
    signals = signals.join(bench, on="signal_date")
    for h in HORIZONS:
        signals[f"excess_{h}"] = signals[f"ret_{h}"] - signals[f"bench_{h}"]

    sessions = sorted(bench.index)
    purge_cutoff = sessions[-PURGE_SESSIONS - 1]
    results, by_period = {}, {}
    for period, start, end in PERIODS:
        end = min(end or purge_cutoff, purge_cutoff)
        window = signals[(signals.signal_date >= start) & (signals.signal_date <= end)]
        label = f"{period}（{start}～{end}）"
        results[label] = by_period[period] = {key: summarize(select_arm(window, key)) for key in ARMS}

    verdicts = {}
    for key in ARMS:
        ok = []
        for period in ("development", "holdout"):
            t5 = by_period[period][key]["t5"]
            ci = t5["excess_ci95"]
            ok.append(t5["mean_excess"] is not None and t5["mean_excess"] > 0 and ci is not None and ci[0] > 0)
        verdicts[key] = ("有效：开发段与留出段 T+5 超额均为正且区间下限>0" if all(ok) else
                         f"未达标（开发段{'达标' if ok[0] else '未达标'}，留出段{'达标' if ok[1] else '未达标'}）")

    meta = {"generated_at": datetime.now().isoformat(timespec="seconds"), "as_of": args.as_of,
            "sampled": len(sample), "succeeded": len(fetched) - sum(f["stage"] == "replay" for f in failures),
            "failed": len(failures), "elapsed_min": round((time.monotonic() - started) / 60, 1)}
    signals.to_csv(args.output / "signals.csv", index=False, encoding="utf-8-sig")
    (args.output / "summary.json").write_text(json.dumps(
        {"meta": meta, "results": results, "verdicts": verdicts, "failures": failures,
         "sample_codes": sample.stock_code.tolist()}, ensure_ascii=False, indent=2))
    (args.output / "report.md").write_text(markdown(meta, results, verdicts), encoding="utf-8")
    print(f"REPORT {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
