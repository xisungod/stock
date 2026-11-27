from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.backtest import summarize_signals, evaluate_bars
from app.schemas import production_strategy_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare min_line_strength settings with point-in-time replay.")
    parser.add_argument("--sample", type=int, default=120, help="Systematic A-share sample; 0 means all cached stocks.")
    parser.add_argument("--lookback", type=int, default=120, help="Historical signal days replayed per stock.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--strengths", default="2,3,4")
    # 默认扫生产引擎：platform_events 的「已验证事件数」；legacy 才是「最低阻力峰数」。
    parser.add_argument("--engine", choices=["platform_events", "legacy"], default="platform_events")
    parser.add_argument("--cache-dir", default="data/bars_cache")
    parser.add_argument("--stock-list", default="data/stock_list_cache.json")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def systematic_sample(stocks: pd.DataFrame, limit: int) -> pd.DataFrame:
    stocks = stocks.sort_values("stock_code").drop_duplicates("stock_code").reset_index(drop=True)
    if limit <= 0 or limit >= len(stocks):
        return stocks
    positions = np.linspace(0, len(stocks) - 1, num=limit).round().astype(int)
    return stocks.iloc[positions].drop_duplicates("stock_code").reset_index(drop=True)


def load_universe(stock_list: Path, cache_dir: Path, sample: int) -> pd.DataFrame:
    payload = json.loads(stock_list.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", [])
    stocks = pd.DataFrame(records).rename(columns={"code": "stock_code", "name": "short_name"})
    stocks["stock_code"] = stocks["stock_code"].astype(str).str.zfill(6)
    available = {path.stem for path in cache_dir.glob("*.json")}
    stocks = stocks[stocks["stock_code"].isin(available)].copy()
    return systematic_sample(stocks, sample)


def strength_config(engine: str, strength: int):
    """生产默认配置，只改被扫的那一个参数（精准线单独评估，不叠加旗形）。"""
    field = "confirmed_event_count" if engine == "platform_events" else "min_line_strength"
    return production_strategy_config(strategy_mode="precision_pullback", precision_engine=engine, **{field: strength})


def evaluate_one(payload: tuple[str, str, str, int, tuple[int, ...], str]) -> dict[str, Any]:
    code, name, path, lookback, strengths, engine = payload
    try:
        cached = json.loads(Path(path).read_text(encoding="utf-8"))
        bars = pd.DataFrame(cached.get("bars", []))
        if bars.empty:
            raise ValueError("empty bars cache")
        reports: dict[str, dict[str, Any]] = {}
        for strength in strengths:
            reports[str(strength)] = evaluate_bars(
                code,
                name,
                bars,
                lookback_days=lookback,
                config=strength_config(engine, strength),
            )
        return {"stock_code": code, "stock_name": name, "reports": reports, "error": None}
    except Exception as exc:
        return {"stock_code": code, "stock_name": name, "reports": {}, "error": str(exc)}


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return round((center - margin) * 100, 1), round((center + margin) * 100, 1)


def setting_summary(signals: list[dict[str, Any]], stock_count: int) -> dict[str, Any]:
    summary = summarize_signals(signals)
    t5 = [float(item["forward_return_5"]) for item in signals if item.get("forward_return_5") is not None]
    positive = sum(value > 0 for value in t5)
    low, high = wilson(positive, len(t5))
    summary.update(
        {
            "stock_count": stock_count,
            "stocks_with_signals": len({item["stock_code"] for item in signals}),
            "t5_down_rate": round(sum(value < 0 for value in t5) / len(t5) * 100, 1) if t5 else None,
            "t5_positive_wilson_95": [low, high],
        }
    )
    return summary


def overlap_summary(by_strength: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    keys = {
        strength: {(item["stock_code"], item["signal_date"]) for item in signals}
        for strength, signals in by_strength.items()
    }
    out: dict[str, Any] = {}
    strengths = sorted(keys)
    for left, right in zip(strengths, strengths[1:]):
        common = keys[left] & keys[right]
        out[f"{left}_vs_{right}"] = {
            "common": len(common),
            f"only_{left}": len(keys[left] - keys[right]),
            f"only_{right}": len(keys[right] - keys[left]),
        }
    return out


def markdown_report(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    summaries: dict[int, dict[str, Any]],
    overlaps: dict[str, Any],
    failures: list[dict[str, str]],
    elapsed: float,
) -> str:
    rows = []
    for strength, item in summaries.items():
        interval = item.get("t5_positive_wilson_95") or [None, None]
        interval_text = "-" if interval[0] is None else f"{interval[0]}%~{interval[1]}%"
        rows.append(
            f"| {strength} | {item.get('signal_count', 0)} | {item.get('stocks_with_signals', 0)} | "
            f"{item.get('t5_positive_rate', '-')}% | {interval_text} | {item.get('t5_average_return', '-')}% | "
            f"{item.get('t10_positive_rate', '-')}% | {item.get('t20_positive_rate', '-')}% | "
            f"{item.get('target_5pct_hit_rate', '-')}% | {item.get('target_before_stop_rate', '-')}% |"
        )
    overlap_lines = [f"- `{name}`: {json.dumps(value, ensure_ascii=False)}" for name, value in overlaps.items()]
    field = "confirmed_event_count（已验证事件数）" if args.engine == "platform_events" else "min_line_strength（最低阻力峰数）"
    return f"""# {field} 参数对照回测

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 方法

- 股票池：当前 A 股缓存中的全市场均匀样本 {len(selected)} 只
- 历史回放：每只最近 {args.lookback} 个交易日
- 引擎：{args.engine}；其余规则为当前新建扫描的默认配置
- 参数：{args.strengths}
- 口径：逐日只使用当日及此前 K 线；同股信号 20 日内去重
- 数据：本地前复权日线缓存，不请求网络
- 耗时：{elapsed / 60:.1f} 分钟；失败 {len(failures)} 只

## 结果

| 参数值 | 信号数 | 有信号股票 | T+5正收益 | 95%区间 | T+5平均 | T+10正收益 | T+20正收益 | 20日触及+5% | +5%先于-5% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## 信号重合

{chr(10).join(overlap_lines)}

## 解读限制

- 这是参数本身的重新回放，不是对强度 3 旧信号做事后分组。
- 当前上市股票池存在幸存者偏差，未计佣金、滑点、涨跌停不可成交。
- 最终参数应同时考虑收益质量、信号覆盖和跨时段稳定性，不能只选最高命中率。
"""


def main() -> None:
    args = parse_args()
    strengths = tuple(sorted({int(value) for value in args.strengths.split(",")}))
    cache_dir = Path(args.cache_dir)
    selected = load_universe(Path(args.stock_list), cache_dir, args.sample)
    payloads = [
        (
            str(row.stock_code).zfill(6),
            str(row.short_name),
            str(cache_dir / f"{str(row.stock_code).zfill(6)}.json"),
            args.lookback,
            strengths,
            args.engine,
        )
        for row in selected.itertuples(index=False)
    ]
    started = time.monotonic()
    by_strength: dict[int, list[dict[str, Any]]] = {value: [] for value in strengths}
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(evaluate_one, payload) for payload in payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["error"]:
                failures.append({"stock_code": result["stock_code"], "error": result["error"]})
            for strength, report in result["reports"].items():
                for signal in report.get("signals", []):
                    by_strength[int(strength)].append(
                        {"setting": int(strength), "stock_code": result["stock_code"], "stock_name": result["stock_name"], **signal}
                    )
            if done == 1 or done % 10 == 0 or done == len(futures):
                elapsed = time.monotonic() - started
                eta = elapsed / done * (len(futures) - done) if done else 0
                counts = ", ".join(f"s{k}={len(v)}" for k, v in by_strength.items())
                print(f"[{done}/{len(futures)}] {counts} failures={len(failures)} eta={eta / 60:.1f}m", flush=True)

    elapsed = time.monotonic() - started
    summaries = {
        strength: setting_summary(signals, len(selected))
        for strength, signals in by_strength.items()
    }
    overlaps = overlap_summary(by_strength)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = output_dir / f"line_strength_sweep_{stamp}"
    all_signals = [item for signals in by_strength.values() for item in signals]
    pd.DataFrame(all_signals).to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": vars(args),
        "selected_codes": selected["stock_code"].tolist(),
        "summaries": summaries,
        "overlaps": overlaps,
        "failures": failures,
        "elapsed_seconds": round(elapsed, 1),
    }
    stem.with_suffix(".json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(
        markdown_report(args, selected, summaries, overlaps, failures, elapsed), encoding="utf-8"
    )
    print(json.dumps({"report": str(stem.with_suffix('.md')), "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
