"""成功（先涨10%）与失败（先跌3%）的特征差异：事后分析，只产生假设，不改规则。

口径：
- 样本：当前规则（修复后 + 大盘偏弱不买）在 591 只抽样股票上的信号，持有 20 个交易日；
  与 target_stop_backtest.py 完全相同的竞速判定。未分胜负、未完成、买不到的不参与比较。
- 特征：只用信号日收盘时已知的数据；「次日开盘跳空」单列为开盘时可知。
- 区分力：AUC = 随机抽一笔成功、一笔失败，成功者该特征更大的概率（0.5 = 无区分）。
- 稳定：开发段（2019–2023）与留出段（2024–2026）的 AUC 都在 0.5 的同一侧且各偏离 ≥0.03，
  且全样本 |z| ≥ 2.6（约等于 25 个特征做 Bonferroni 校正后的 5% 水平）。
- 对照：同一批股票、大盘不弱的日子随机抽 40000 个买点，算同一特征的 AUC；
  对照组里同样有效的特征说明是普遍规律，不是这套信号特有的。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from app.services.market_regime import regime_table
from app.services.tencent_bars import fetch_tencent_bars
from target_stop_backtest import race
from validate_logic_fixes import load_settled_bars

WINDOW = 20
CONTROL_SAMPLE = 40000
FEATURES = {
    "distance_to_line_pct": "信号日收盘距精准线 %",
    "bars_since_breakout": "距突破交易日数",
    "breakout_volume_ratio": "突破日量比（/前10日均量）",
    "breakout_day_return_pct": "突破日涨幅 %",
    "runup_pct": "突破后最高越线 %",
    "pullback_normal_volume_ratio": "回踩日量 / 突破前10日均量",
    "pullback_vs_breakout_volume": "回踩日量 / 突破日量",
    "line_points": "精准线独立验证事件数",
    "prior_return_5_pct": "信号前5日涨跌 %",
    "prior_return_20_pct": "信号前20日涨跌 %",
    "prior_return_60_pct": "信号前60日涨跌 %",
    "prior_return_120_pct": "信号前120日涨跌 %",
    "volatility_20_pct": "20日波动率 %",
    "atr_20_pct": "ATR20 / 收盘 %",
    "distance_to_ma20_pct": "距MA20 %",
    "distance_to_ma60_pct": "距MA60 %",
    "ma20_slope_5_pct": "MA20 近5日斜率 %",
    "ma60_slope_20_pct": "MA60 近20日斜率 %",
    "signal_body_pct": "信号日实体（收/开）%",
    "close_location_pct": "信号日收盘在日内区间位置 %",
    "signal_range_pct": "信号日振幅 %",
    "signal_day_return_pct": "信号日涨跌 %",
    "volume_ratio_20": "信号日量 / 前20日均量",
    "distance_to_250d_high_pct": "距250日最高 %",
    "log_amount_20": "20日平均成交额（对数，前复权近似）",
    "market_return_20_pct": "沪深300近20日 %",
    "market_distance_to_ma20_pct": "沪深300距MA20 %",
    "entry_gap_pct": "次日开盘跳空 %（开盘时可知）",
}
STOCK_LEVEL = {"breakout_volume_ratio", "breakout_day_return_pct", "runup_pct", "pullback_normal_volume_ratio",
               "pullback_vs_breakout_volume", "line_points", "bars_since_breakout", "distance_to_line_pct"}


def series_features(bars: pd.DataFrame, index_table: pd.DataFrame) -> pd.DataFrame:
    """每根K线收盘时可知的通用特征（不依赖精准线）。"""
    o, h, l, c, v = (bars[name].astype(float) for name in ("open", "high", "low", "close", "volume"))
    previous = c.shift(1)
    true_range = pd.concat([h - l, (h - previous).abs(), (l - previous).abs()], axis=1).max(axis=1)
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    day_range = (h - l).replace(0, np.nan)
    out = pd.DataFrame({
        "prior_return_5_pct": (c / c.shift(5) - 1) * 100,
        "prior_return_20_pct": (c / c.shift(20) - 1) * 100,
        "prior_return_60_pct": (c / c.shift(60) - 1) * 100,
        "prior_return_120_pct": (c / c.shift(120) - 1) * 100,
        "volatility_20_pct": c.pct_change().rolling(20).std() * 100,
        "atr_20_pct": true_range.rolling(20).mean() / c * 100,
        "distance_to_ma20_pct": (c / ma20 - 1) * 100,
        "distance_to_ma60_pct": (c / ma60 - 1) * 100,
        "ma20_slope_5_pct": (ma20 / ma20.shift(5) - 1) * 100,
        "ma60_slope_20_pct": (ma60 / ma60.shift(20) - 1) * 100,
        "signal_body_pct": (c / o - 1) * 100,
        "close_location_pct": ((c - l) / day_range * 100).fillna(50),
        "signal_range_pct": (h - l) / previous * 100,
        "signal_day_return_pct": (c / previous - 1) * 100,
        "volume_ratio_20": v / v.shift(1).rolling(20).mean(),
        "distance_to_250d_high_pct": (c / h.rolling(250, min_periods=60).max() - 1) * 100,
        "log_amount_20": np.log10((c * v).rolling(20).mean().clip(lower=1)),
        "entry_gap_pct": (o.shift(-1) / c - 1) * 100,
    })
    market = index_table.reindex(bars["trade_date"].to_numpy())
    out["market_return_20_pct"] = market["return_20_pct"].to_numpy()
    out["market_distance_to_ma20_pct"] = market["distance_to_ma20_pct"].to_numpy()
    return out


def signal_features(bars: pd.DataFrame, row: pd.Series) -> dict:
    c, v = bars["close"].astype(float).to_numpy(), bars["volume"].astype(float).to_numpy()
    i = int(row.signal_index)
    b = int(np.flatnonzero(bars["trade_date"].to_numpy() == row.breakout_date)[0])
    average = v[max(0, b - 10):b].mean() if b >= 5 else np.nan
    return {
        "distance_to_line_pct": (c[i] / row.line_price - 1) * 100,
        "bars_since_breakout": i - b,
        "breakout_volume_ratio": v[b] / average if average and average > 0 else np.nan,
        "breakout_day_return_pct": (c[b] / c[b - 1] - 1) * 100,
        "runup_pct": row.runup_pct,
        "pullback_normal_volume_ratio": row.normal_volume_ratio,
        "pullback_vs_breakout_volume": v[i] / v[b] if v[b] > 0 else np.nan,
        "line_points": row.line_points,
    }


def auc(success: pd.Series, failure: pd.Series) -> tuple[float | None, float | None]:
    a, b = success.dropna().to_numpy(), failure.dropna().to_numpy()
    if len(a) < 20 or len(b) < 20:
        return None, None
    ranks = pd.Series(np.r_[a, b]).rank().to_numpy()
    value = (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    z = (value - 0.5) / math.sqrt((len(a) + len(b) + 1) / (12 * len(a) * len(b)))
    return round(float(value), 3), round(float(z), 2)


def quintiles(frame: pd.DataFrame, feature: str, edges: np.ndarray) -> list[dict]:
    rows = []
    bins = pd.cut(frame[feature], edges, include_lowest=True, duplicates="drop")
    for interval, group in frame.groupby(bins, observed=True):
        rows.append({"range": f"{interval.left:.2f}～{interval.right:.2f}", "n": len(group),
                     "success_rate": round(float((group.outcome == "success").mean() * 100), 1)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--signals", type=Path, default=ROOT / "reports/logic_fixes_20261127/validation/signals.csv")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/logic_fixes_20261127/success_features")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(args.signals, dtype={"stock_code": str}, encoding="utf-8-sig")
    signals["stock_code"] = signals["stock_code"].str.zfill(6)
    current = signals[(signals.arm == "new") & (signals.market_ok == True)]  # noqa: E712
    index_table = regime_table(fetch_tencent_bars("sh000300"))
    index_table = index_table[index_table.index <= args.as_of]
    weak_days = set(index_table.index[(index_table.return_20_pct < 0) | (index_table.ma20_slope_5_pct < 0)])

    trades, control = [], []
    rng = np.random.default_rng(20261127)
    codes = sorted(set(signals.stock_code.unique()))
    per_stock_control = max(1, CONTROL_SAMPLE // len(codes))
    for code in codes:
        path = args.cache / f"{code}.json"
        if not path.exists():
            continue
        bars = load_settled_bars(path, args.as_of)
        outcome = race(bars, WINDOW)
        generic = series_features(bars, index_table)
        board = "科创板" if code.startswith("68") else "创业板" if code.startswith("30") else "主板"
        for _, row in current[current.stock_code == code].iterrows():
            i = int(row.signal_index)
            if bars["trade_date"].iloc[i] != row.signal_date:
                raise AssertionError(f"{code} 行号 {i} 与信号日 {row.signal_date} 不一致")
            trades.append({"stock_code": code, "signal_date": row.signal_date, "board": board,
                           "outcome": outcome.outcome.iloc[i], **generic.iloc[i].to_dict(),
                           **signal_features(bars, row)})
        dates = bars["trade_date"].to_numpy()
        eligible = np.flatnonzero((dates >= "2019-01-01") & ~np.isin(dates, list(weak_days)))
        eligible = eligible[outcome.outcome.iloc[eligible].isin(["success", "failure"]).to_numpy()]
        for i in rng.choice(eligible, size=min(per_stock_control, len(eligible)), replace=False):
            control.append({"stock_code": code, "signal_date": dates[i], "board": board,
                            "outcome": outcome.outcome.iloc[i], **generic.iloc[i].to_dict()})

    trades = pd.DataFrame(trades)
    trades = trades[trades.outcome.isin(["success", "failure"])].reset_index(drop=True)
    control = pd.DataFrame(control)
    trades.to_csv(args.output / "trades.csv", index=False, encoding="utf-8-sig")
    periods = {"dev": trades.signal_date <= "2023-12-31", "holdout": trades.signal_date >= "2024-01-01"}

    rows = []
    for feature, label in FEATURES.items():
        win, lose = trades.outcome == "success", trades.outcome == "failure"
        overall, z = auc(trades.loc[win, feature], trades.loc[lose, feature])
        dev, _ = auc(trades.loc[win & periods["dev"], feature], trades.loc[lose & periods["dev"], feature])
        hold, _ = auc(trades.loc[win & periods["holdout"], feature], trades.loc[lose & periods["holdout"], feature])
        ctl = None
        if feature in control:
            ctl, _ = auc(control.loc[control.outcome == "success", feature], control.loc[control.outcome == "failure", feature])
        stable = (overall is not None and dev is not None and hold is not None and abs(z) >= 2.6
                  and (dev - 0.5) * (hold - 0.5) > 0 and min(abs(dev - 0.5), abs(hold - 0.5)) >= 0.03)
        rows.append({
            "feature": feature, "label": label, "auc": overall, "z": z, "auc_dev": dev, "auc_holdout": hold,
            "auc_control": ctl, "stable": stable,
            "median_success": round(float(trades.loc[win, feature].median()), 3),
            "median_failure": round(float(trades.loc[lose, feature].median()), 3),
        })
    table = pd.DataFrame(rows).sort_values("auc", key=lambda s: (s - 0.5).abs(), ascending=False)

    lines = ["# 成功与失败案例的特征差异（事后分析）", "", "## 口径", ""]
    lines += [line if line.startswith("- ") else "  " + line.strip() for line in __doc__.strip().splitlines()[3:]
              if line.strip()]
    lines += ["", f"成功 {int((trades.outcome == 'success').sum())} 笔，失败 {int((trades.outcome == 'failure').sum())} 笔；"
              f"对照随机买点 {len(control)} 个（成功率 {(control.outcome == 'success').mean() * 100:.1f}%）。", "",
              "## 全部特征（按区分力排序）", "",
              "| 特征 | 成功中位数 | 失败中位数 | AUC | z | 开发段AUC | 留出段AUC | 对照组AUC | 稳定 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.label} | {row.median_success} | {row.median_failure} | {row.auc} | {row.z} | "
                     f"{row.auc_dev} | {row.auc_holdout} | {'-' if row.auc_control is None or pd.isna(row.auc_control) else row.auc_control} | "
                     f"{'是' if row.stable else ''} |")
    lines += ["", "## 板块", "", "| 板块 | 笔数 | 成功率 | 对照成功率 |", "|---|---:|---:|---:|"]
    for board, group in trades.groupby("board"):
        ctl = control[control.board == board]
        lines.append(f"| {board} | {len(group)} | {(group.outcome == 'success').mean() * 100:.1f}% | "
                     f"{(ctl.outcome == 'success').mean() * 100:.1f}% |")
    lines += ["", "## 稳定特征分五档的成功率（信号组 vs 对照组，同一组分档边界）", ""]
    for row in table[table.stable].itertuples():
        edges = np.unique(np.nanquantile(trades[row.feature], np.linspace(0, 1, 6)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        sig = quintiles(trades, row.feature, edges)
        ctl = quintiles(control, row.feature, edges) if row.feature in control else []
        lines += [f"### {row.label}", "", "| 区间 | 信号笔数 | 信号成功率 | 对照成功率 |", "|---|---:|---:|---:|"]
        for k, item in enumerate(sig):
            control_rate = f"{ctl[k]['success_rate']}%" if k < len(ctl) else "-"
            lines.append(f"| {item['range']} | {item['n']} | {item['success_rate']}% | {control_rate} |")
        lines.append("")
    lines += ["## 解读限制", "",
              "- 这是看过结果后的分析，多个特征同时比较，即使有统计校正，仍需用新行情前向验证后才能写进规则。",
              "- 对照组 AUC 与信号组相近的特征，说明它对任何买点都成立（通常与波动率、趋势有关），不是这套形态的特点。",
              "- 前复权价格与当前上市股票池会让成交额、涨幅类特征有偏差。", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    table.to_csv(args.output / "features.csv", index=False, encoding="utf-8-sig")
    print(f"REPORT {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
