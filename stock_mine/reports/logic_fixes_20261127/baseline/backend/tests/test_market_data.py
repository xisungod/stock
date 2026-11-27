from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.services.detail import build_market_snapshot
from app.services.market_data import _normalize_industry_bars, resolve_ths_industry


def test_resolve_ths_industry_maps_sina_names() -> None:
    assert resolve_ths_industry("酿酒行业") == "白酒"
    assert resolve_ths_industry("金融行业") == "银行"
    assert resolve_ths_industry("医疗器械") == "医疗器械"


def test_industry_bars_normalize_for_chart_pipeline() -> None:
    frame = pd.DataFrame(
        {"日期": ["2026-01-01"], "开盘价": [10], "最高价": [11], "最低价": [9], "收盘价": [10.5], "成交量": [1000]}
    )
    normalized = _normalize_industry_bars(frame)
    assert list(normalized.columns) == ["trade_date", "open", "high", "low", "close", "volume"]


def test_market_snapshot_contains_key_quote_metrics() -> None:
    closes = [10 + index * 0.1 for index in range(30)]
    bars = pd.DataFrame(
        {
            "trade_date": [date(2026, 1, 1) + timedelta(days=index) for index in range(30)],
            "open": closes,
            "high": [value + 0.2 for value in closes],
            "low": [value - 0.2 for value in closes],
            "close": closes,
            "volume": [1000] * 29 + [2000],
        }
    )
    payload = build_market_snapshot(bars)
    assert payload["close"] == closes[-1]
    assert payload["change_pct"] > 0
    assert payload["volume_ratio_20"] > 1
