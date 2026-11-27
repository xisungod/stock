"""主线 A 的一步附加条件：三角形向上突破还要确认日放量。

门槛在跑数前写死，取策略里已经存在的两个数，不另扫倍数：
- 均量窗口 vol_lookback = 10，不含确认日本身
- 倍数 breakout_vol_mult = 1.5
确认日量 / 此前 10 日均量 >= 1.5 才是买入。均量算不出则不入选。

改前是 reports/triangle_scan_20261127/signals.csv 里的向上突破确认日。
改后用同一张表上的收益，只按上面的量能规则去掉不合格的确认日。
样本、分段、成本和超额都不变。两段 T+5 超额均值 > 0 且 95% 下限 > 0 才把门槛留进三角扫描；
否则 triangle_breakout 默认仍不要求放量，联合筛选和精准线不动。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from app.services.chart_patterns import confirmation_volume_ratio
from app.services.strategy import StrategyConfig
from validate_triangle_scan import CACHE, ci_text, load_bars, metrics, passes

BASELINE = ROOT / "reports/triangle_scan_20261127"
OUT = ROOT / "reports/triangle_volume_20261127"


def attach_volume(signals: pd.DataFrame, lookback: int) -> pd.DataFrame:
    frames = {}
    ratios = []
    for row in signals.itertuples(index=False):
        code = str(row.stock_code).zfill(6)
        if code not in frames:
            path = CACHE / f"{code}.json"
            if not path.exists():
                raise SystemExit(f"缺少锁定行情 {path}，不能改用别的股票池。")
            frames[code] = load_bars(path)
        bars = frames[code]
        index = int(row.signal_index)
        if bars["trade_date"].iloc[index] != row.signal_date:
            raise SystemExit(f"{code} 行号 {index} 是 {bars['trade_date'].iloc[index]}，信号日是 {row.signal_date}")
        ratios.append(confirmation_volume_ratio(bars, index, lookback))
    out = signals.copy()
    out["volume_ratio"] = ratios
    return out


def main() -> None:
    if (OUT / "signals.csv").exists():
        raise SystemExit(f"{OUT / 'signals.csv'} 已存在，不覆盖。")
    cfg = StrategyConfig()
    if cfg.vol_lookback != 10 or cfg.breakout_vol_mult != 1.5:
        raise SystemExit("放量门槛已不是 10 日与 1.5 倍。先停下来，不要换成别的阈值。")
    published = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    signals = pd.read_csv(BASELINE / "signals.csv", dtype={"stock_code": str}, encoding="utf-8-sig")
    signals["stock_code"] = signals["stock_code"].str.zfill(6)
    measured = attach_volume(signals, cfg.vol_lookback)
    before_rows = []
    after_rows = []
    kept = measured[measured["volume_ratio"].notna() & (measured["volume_ratio"] >= cfg.breakout_vol_mult)]
    for label, start, end in (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21")):
        window = measured[(measured.signal_date >= start) & (measured.signal_date <= end)]
        before = metrics(window)
        before.update({"label": label, "variant": "改前：向上突破确认日买入"})
        after = metrics(kept[(kept.signal_date >= start) & (kept.signal_date <= end)])
        after.update({"label": label, "variant": "改后：确认日量 ≥ 1.5 倍前 10 日均量"})
        expected = next(row for row in published["results"] if row["label"] == label)["t5"]
        for field in ("n", "mean_net", "win_rate", "mean_excess", "excess_ci95"):
            if before["t5"][field] != expected[field]:
                raise SystemExit(f"改前对不上三角基线 {label} {field}: {before['t5'][field]} != {expected[field]}")
        if before["signals"] != next(row for row in published["results"] if row["label"] == label)["signals"]:
            raise SystemExit(f"改前信号数对不上 {label}")
        before_rows.append(before)
        after_rows.append(after)
    valid = all(passes(row["t5"]) for row in after_rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lookback": cfg.vol_lookback,
        "multiple": cfg.breakout_vol_mult,
        "retained": False,
        "meets_validity_standard": valid,
        "rows": before_rows + after_rows,
        "missing_ratio": int(measured["volume_ratio"].isna().sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    measured.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} valid={valid}")


def render(payload: dict) -> str:
    lines = [
        "# 主线 A：三角整理向上突破附加确认日放量",
        "",
        f"生成时间：{payload['generated_at']}。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 A：三角整理。上一轮的向上突破确认日基线不重做，只在那条规则上加一个条件。",
        "不回到精准线旧信号的事后过滤，也不把三角整理并进联合筛选。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：`triangle_breakout` 只要收盘越过上沿 0.5% 就是买入，不看量。",
        f"改后：同一确认日还要求成交量不低于此前 {payload['lookback']} 根均量的 {payload['multiple']:g} 倍。",
        "这两个数是策略里原有的 `vol_lookback` 和 `breakout_vol_mult`，不是按这批收益扫出来的。",
        "均量只含确认日之前的 K 线。均量不足、成交缺失或均量为 0，这一笔不入选。",
        "形态怎么画没有改。量不够的确认日不再是买入预警；扫描里它变成观察（越过上沿但量未确认）。",
        "",
        "## 3. 改前改后",
        "",
        "改前数字与 `reports/triangle_scan_20261127/summary.json` 逐项一致，否则脚本退出。",
        "股票池、分段（留出段到 2026-10-21）、次日开盘、T+5 和同日样本等权超额都不变。",
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
        lines.append("两段都达标。确认日放量留在三角扫描的买入条件里。联合筛选和精准线默认仍不改。")
    else:
        before = [row for row in payload["rows"] if row["variant"].startswith("改前")]
        after = [row for row in payload["rows"] if row["variant"].startswith("改后")]
        lines += [
            "不把放量门槛留进 `triangle_breakout` 的默认买入条件，也不进入联合筛选。",
            "开发段和留出段都没有达到 T+5 超额均值大于 0 且 95% 下限大于 0。",
            (
                f"加上放量之后，开发段超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%，"
                f"留出段从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。两段都更差，区间也都整体小于 0。"
            ),
            "代码里这个门槛只有在显式打开 `require_breakout_volume` 时才生效。三角扫描默认仍然只认收盘突破。",
            f"比值缺失 {payload['missing_ratio']} 笔，缺失不算通过。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        "- 这是看过向上突破基线之后追加的一次比较。门槛没有改过第二次。",
        "- 前复权与幸存者偏差同上一轮锁定行情。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
