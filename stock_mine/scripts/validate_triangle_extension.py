"""主线 A 的一步位置条件：三角形确认日收盘不能离上沿超过近线带。

门槛在跑数前写死，用策略里已有的 max_extension_pct = 0.035，不另扫分档。
这是确认日收盘相对三角形上沿的距离，不是精准线回踩里「突破后最高价越线不超过 3.5%」那条已否决的规则。

改前是 reports/triangle_scan_20261127 的收盘突破买入。
改后只留下收盘离上沿不超过 3.5% 的确认日。样本、分段、成交和超额都不变。
两段都达标才把这个上限留进三角扫描的默认买入；否则默认仍只认收盘突破。
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

from app.services.chart_patterns import triangle_up_breakouts
from app.services.strategy import StrategyConfig
from validate_triangle_scan import CACHE, ci_text, load_bars, metrics, passes

BASELINE = ROOT / "reports/triangle_scan_20261127"
OUT = ROOT / "reports/triangle_extension_20261127"


def attach_extension(signals: pd.DataFrame) -> pd.DataFrame:
    frames = {}
    uppers = {}
    distances = []
    for row in signals.itertuples(index=False):
        code = str(row.stock_code).zfill(6)
        if code not in frames:
            path = CACHE / f"{code}.json"
            if not path.exists():
                raise SystemExit(f"缺少锁定行情 {path}，不能改用别的股票池。")
            bars = load_bars(path)
            frames[code] = bars
            uppers[code] = {item["event_id"]: float(item["upper"]) for item in triangle_up_breakouts(bars)}
        bars = frames[code]
        index = int(row.signal_index)
        if bars["trade_date"].iloc[index] != row.signal_date:
            raise SystemExit(f"{code} 行号 {index} 是 {bars['trade_date'].iloc[index]}，信号日是 {row.signal_date}")
        upper = uppers[code].get(row.event_id)
        if upper is None or upper <= 0:
            raise SystemExit(f"{code} {row.event_id} 在锁定行情上对不上上沿")
        close = float(bars["close"].iloc[index])
        distances.append((close / upper - 1) * 100)
    out = signals.copy()
    out["close_extension_pct"] = distances
    return out


def main() -> None:
    if (OUT / "signals.csv").exists():
        raise SystemExit(f"{OUT / 'signals.csv'} 已存在，不覆盖。")
    cfg = StrategyConfig()
    if abs(cfg.max_extension_pct - 0.035) > 1e-12:
        raise SystemExit(f"近线带已不是 3.5%：{cfg.max_extension_pct}。先停下来，不要换成别的阈值。")
    limit = cfg.max_extension_pct * 100
    published = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    signals = pd.read_csv(BASELINE / "signals.csv", dtype={"stock_code": str}, encoding="utf-8-sig")
    signals["stock_code"] = signals["stock_code"].str.zfill(6)
    measured = attach_extension(signals)
    kept = measured[measured["close_extension_pct"] <= limit]
    rows = []
    for label, start, end in (("开发段", "2019-01-01", "2023-12-31"), ("留出段", "2024-01-01", "2026-10-21")):
        window = measured[(measured.signal_date >= start) & (measured.signal_date <= end)]
        before = metrics(window)
        before.update({"label": label, "variant": "改前：收盘越过上沿就买入"})
        after = metrics(kept[(kept.signal_date >= start) & (kept.signal_date <= end)])
        after.update({"label": label, "variant": "改后：收盘离上沿不超过 3.5%"})
        expected_row = next(item for item in published["results"] if item["label"] == label)
        for field in ("n", "mean_net", "win_rate", "mean_excess", "excess_ci95"):
            if before["t5"][field] != expected_row["t5"][field]:
                raise SystemExit(f"改前对不上三角基线 {label} {field}: {before['t5'][field]} != {expected_row['t5'][field]}")
        if before["signals"] != expected_row["signals"]:
            raise SystemExit(f"改前信号数对不上 {label}")
        rows.extend((before, after))
    after_rows = [row for row in rows if row["variant"].startswith("改后")]
    valid = all(passes(row["t5"]) for row in after_rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "limit_pct": limit,
        "retained": False,
        "meets_validity_standard": valid,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    measured.to_csv(OUT / "signals.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    print(f"REPORT {OUT / 'report.md'} valid={valid}")


def render(payload: dict) -> str:
    lines = [
        "# 主线 A：三角整理确认日收盘不得离上沿过远",
        "",
        f"生成时间：{payload['generated_at']}。",
        "",
        "## 1. 本轮主线",
        "",
        "主线 A：三角整理。不重做向上突破基线，也不重做 1.5 倍放量。",
        "不回到精准线旧信号，也不把三角整理并进联合筛选。",
        "",
        "## 2. 规则改了哪一处",
        "",
        "改前：`triangle_breakout` 只要收盘高于上沿 0.5% 就是买入。",
        f"改后：同一确认日的收盘还不得高于上沿 {payload['limit_pct']:g}% 以上。",
        "3.5% 是策略里已有的 `max_extension_pct`，用在三角形上沿，不是按收益再找的切点。",
        "它量的是确认日收盘离形态上沿有多远。精准线上已否决的那条，量的是回踩前最高价离开精准线有多远，两者不是同一条规则。",
        "上沿用确认日已经确定的边界。收盘离得更远的确认日不再是买入，扫描里标成观察。",
        "默认不打开这个上限。只有两段都达标，才把它留进三角扫描的默认买入。",
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
        lines.append("两段都达标。确认日收盘离上沿不超过 3.5% 留在三角扫描的默认买入条件里。联合筛选和精准线仍不改。")
    else:
        before = [row for row in payload["rows"] if row["variant"].startswith("改前")]
        after = [row for row in payload["rows"] if row["variant"].startswith("改后")]
        lines += [
            "不把这条位置上限留进 `triangle_breakout` 的默认买入，也不进入联合筛选。",
            "开发段和留出段没有同时达到 T+5 超额均值大于 0 且 95% 下限大于 0。",
            (
                f"加上位置上限之后，开发段超额从 {before[0]['t5']['mean_excess']}% 变成 {after[0]['t5']['mean_excess']}%，"
                f"留出段从 {before[1]['t5']['mean_excess']}% 变成 {after[1]['t5']['mean_excess']}%。"
            ),
            "代码里只有显式设置 `triangle_max_close_extension_pct` 才生效。三角扫描默认仍然只认收盘突破 0.5%。",
        ]
    lines += [
        "",
        "## 限制",
        "",
        "- 这是看过向上突破基线之后追加的一次比较。阈值没有改过第二次。",
        "- 前复权与幸存者偏差同锁定行情。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
