"""主线 B：精准线回踩打开已有的突破放量门槛，两套规则各自重放。

改前是当前默认：require_breakout_volume=false，突破日量只记录、不作为入选条件。
改后把这个已有开关打开：确认突破的当日成交量必须达到此前 10 日均量的 1.5 倍。
1.5 和 10 日是策略里的 breakout_vol_mult、vol_lookback，不另扫倍数。
这不是三角整理上已否决的放量，也不是把 0.75 用到回踩常态量，更不是把冲高上限收到 3.5%。

两套配置各自跑 platform_events。引擎可以因此换一条线，不能用旧信号事后过滤代替。
样本是三角扫描已经冻结的同一 600 个代码、同一份截到 2026-11-26 的行情。
成交、超额、分段与 validate_logic_fixes.py 相同：次日开盘、T+5、20bps、同日样本等权；
开发段到 2023-12-31，留出段 2024-01-01～2026-10-21。
两段 T+5 超额均值 > 0 且 95% 下限 > 0，才把开关留进新扫描的默认。
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from app.schemas import production_strategy_config
from app.services.backtest import evaluate_bars
from validate_logic_fixes import COST_PCT, HORIZONS, forward_returns
from validate_triangle_scan import CACHE, ci_text, load_bars, metrics, passes

OUT = ROOT / "reports/precision_breakout_volume_20261127"
AS_OF = "2026-11-26"
PERIODS = (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21"))


def config(require_volume: bool):
    return production_strategy_config(
        strategy_mode="precision_pullback", include_chart=False, require_breakout_volume=require_volume,
    )


def replay(path: str) -> dict:
    code = Path(path).stem
    try:
        bars = load_bars(Path(path))
        if len(bars) < 200 or bars["trade_date"].iloc[-1] < "2019-01-01":
            return {"code": code, "signals": [], "returns": {}, "error": "history too short"}
        dates = bars["trade_date"].tolist()
        signals = []
        for arm, require_volume in (("before", False), ("after", True)):
            report = evaluate_bars(code, code, bars, lookback_days=0, config=config(require_volume),
                                   as_of=AS_OF, calendar=dates)
            for record in report["signals"]:
                index = int(record["signal_index"])
                if dates[index] != record["signal_date"]:
                    raise AssertionError(f"{code} {index} {dates[index]} != {record['signal_date']}")
                if record["signal_date"] < "2019-01-01" or record["signal_date"] > "2026-10-21":
                    continue
                signals.append({
                    "arm": arm, "stock_code": code, "signal_date": record["signal_date"], "signal_index": index,
                    "event_id": record.get("signal_event_id"),
                    "breakout_volume_ratio": record.get("breakout_volume_ratio"),
                    **{f"ret_{horizon}": value for horizon, value in forward_returns(bars, index).items()},
                })
        returns = {}
        start = int((bars["trade_date"] >= "2019-01-01").to_numpy().argmax())
        for index in range(start, len(bars) - 1):
            if dates[index] > "2026-10-21":
                break
            values = forward_returns(bars, index)
            returns[dates[index]] = [values[horizon] for horizon in HORIZONS]
        return {"code": code, "signals": signals, "returns": returns, "error": None}
    except Exception as exc:
        return {"code": code, "signals": [], "returns": {}, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    if (OUT / "signals.csv").exists():
        raise SystemExit(f"{OUT / 'signals.csv'} 已存在，不覆盖。")
    before_cfg, after_cfg = config(False), config(True)
    if before_cfg.require_breakout_volume or not after_cfg.require_breakout_volume:
        raise SystemExit("默认开关状态不对，先停下来。")
    if before_cfg.breakout_vol_mult != 1.5 or after_cfg.breakout_vol_mult != 1.5:
        raise SystemExit("突破量比不是 1.5，不要改阈值。")
    if before_cfg.strategy_mode != "precision_pullback" or after_cfg.market_filter != before_cfg.market_filter:
        raise SystemExit("两套配置除放量开关外不应不同。")
    paths = sorted(CACHE.glob("*.json"))
    raw, returns, failures = [], {}, []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(replay, str(path)) for path in paths]
        for done, future in enumerate(as_completed(futures), 1):
            item = future.result()
            if item["error"]:
                failures.append({"code": item["code"], "error": item["error"]})
            else:
                raw.extend(item["signals"])
                returns[item["code"]] = item["returns"]
            if done % 25 == 0 or done == len(futures):
                print(f"replay {done}/{len(futures)} signals={len(raw)} fail={len(failures)} "
                      f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    bench_rows = [(day, *values) for series in returns.values() for day, values in series.items()]
    bench = pd.DataFrame(bench_rows, columns=["signal_date", *(f"bench_{h}" for h in HORIZONS)]).groupby("signal_date").mean()
    signals = pd.DataFrame(raw)
    if signals.empty:
        raise SystemExit("没有信号。")
    signals = signals.join(bench, on="signal_date")
    for horizon in HORIZONS:
        signals[f"excess_{horizon}"] = signals[f"ret_{horizon}"] - signals[f"bench_{horizon}"]
    rows = []
    for arm, variant in (("before", "改前：当前默认，不要求突破放量"), ("after", "改后：突破日量 ≥ 1.5 倍前 10 日均量")):
        part = signals[signals.arm == arm]
        for label, start, end in PERIODS:
            window = part[(part.signal_date >= start) & (part.signal_date <= end)]
            record = metrics(window)
            record.update({"label": label, "variant": variant, "arm": arm})
            rows.append(record)
    after_rows = [row for row in rows if row["arm"] == "after"]
    valid = all(passes(row["t5"]) for row in after_rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": AS_OF,
        "cost_pct": COST_PCT,
        "codes": len(paths),
        "replayed": len(returns),
        "failures": failures,
        "multiple": 1.5,
        "lookback": 10,
        "meets_validity_standard": valid,
        "retained_as_default": False,
        "rows": rows,
        "elapsed_min": round((time.monotonic() - started) / 60, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} valid={valid}")


def render(payload: dict) -> str:
    lines = [
        "# 主线 B：精准线回踩打开突破放量",
        "",
        f"生成时间：{payload['generated_at']}；行情截止 {payload['as_of']}；耗时 {payload['elapsed_min']} 分钟。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 B：精准线回踩。三角整理的默认规则保持不变，已完成的基线、1.5 倍放量和收盘离上沿 3.5% 都不重做。",
        "也不再做精准线上已否决的两件事：把突破后最高越线收到 3.5%，以及把 0.75 缩量用到突破前均量。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：当前新扫描的精准线回踩不要求突破日放量。`require_breakout_volume=false`，突破日量只记录。",
        "改后：同一引擎把这个已有开关打开。突破成立还要求当日成交量 ≥ 此前 10 日均量的 1.5 倍。",
        "均量不含突破日本身。均量不足或成交无效，这次突破不算数，引擎可以改选另一条线。",
        "所以这不是把旧信号表里量不够的行删掉。两套配置各自从头重放。",
        "1.5 倍和 10 日是 `breakout_vol_mult` 与 `vol_lookback` 的现成值，没有再试别的倍数。",
        "旧批次若保存了自己的开关，仍按保存的值。这一轮默认不改，除非改后两段都达标。",
        "",
        "## 3. 改前改后",
        "",
        f"样本是已经冻结的 {payload['codes']} 个代码里成功重放的 {payload['replayed']} 只，名单与三角扫描相同。",
        "分段固定为开发段 2019-01-01～2023-12-31、留出段 2024-01-01～2026-10-21。",
        "成交是信号次日开盘买入，一字涨停买不到，第 5 个交易日收盘卖出，双边 20bps。",
        "超额是信号净收益减去同一天全部样本股的等权收益。",
        "",
        "| 口径 | 时段 | 信号 | 股票 | 有 T+5 成交 | 净胜率 | 平均净收益 | 中位净收益 | T+5 超额 | 超额95%区间 | 该时段达标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        t5 = row["t5"]
        lines.append(
            f"| {row['variant']} | {row['label']} | {row['signals']} | {row['stocks']} | {t5['n']} | "
            f"{t5['win_rate']}% | {t5['mean_net']}% | {t5['median_net']}% | {t5['mean_excess']}% | "
            f"{ci_text(t5['excess_ci95'])} | {'是' if passes(t5) else '否'} |"
        )
    lines += ["", "## 是否留下", ""]
    if payload["meets_validity_standard"]:
        lines.append("两段都达标。新扫描的精准线默认打开突破放量。旧批次已保存的开关不改。三角扫描和联合筛选里的旗形规则不改。")
    else:
        before = [row for row in payload["rows"] if row["arm"] == "before"]
        after = [row for row in payload["rows"] if row["arm"] == "after"]
        lines += [
            "不把突破放量留进新扫描的默认规则。三角扫描默认也不改。",
            "开发段和留出段没有同时达到 T+5 超额均值大于 0 且 95% 下限大于 0。",
            (
                f"开发段超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%，"
                f"留出段从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。"
            ),
            "开关仍可显式打开。当前默认继续只记录突破日量、不拿它卡入选。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        "- 原精准线验证的日线文件没有入库。这里用的是同一代码名单、同一截止日期重新冻结的前复权行情，价格可能与 2026-11-26 那份旧缓存有复权差异。区间和名单没有另选。",
        "- 因此改前数字不必等于更早那份 `logic_fixes` 账本。本轮对照是同一份行情上的两次独立重放。",
        f"- 重放失败 {len(payload['failures'])} 只，没有用别的股票填上。",
        "- 前复权、幸存者偏差、留出段已被先前研究看过，这些限制仍在。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
