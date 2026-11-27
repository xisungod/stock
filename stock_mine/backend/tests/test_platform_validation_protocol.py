import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from validate_platform_v2 import match_controls, partition_rows, select_sample


def test_partition_purges_market_sessions_not_only_signal_dates():
    calendar = pd.bdate_range("2024-01-01", periods=50).strftime("%Y-%m-%d").tolist()
    part = {"start": calendar[0], "end": calendar[-1]}
    rows = [{"signal_date": calendar[index]} for index in [0, 29, 30, 49]]
    assert partition_rows(rows, part, calendar) == rows[:2]


def test_sample_is_deterministic_and_does_not_read_prices_or_future_returns(tmp_path):
    rows = [{"stock_code": code} for code in ["000001", "000002", "600001", "300001", "688001"]]
    for row in rows:
        (tmp_path / f"{row['stock_code']}.json").write_text("prices are deliberately not readable")
    selected = select_sample(rows, tmp_path, 1)
    assert len(selected) == 4
    assert selected == select_sample(list(reversed(rows)), tmp_path, 1)


def test_controls_exclude_all_raw_alerts_and_require_causal_history_without_peeking_at_tomorrow():
    calendar = pd.bdate_range("2024-01-01", periods=121).strftime("%Y-%m-%d").tolist()
    frames = {code: {day: {"volume": 100} for day in calendar} for code in ["a", "b", "c", "d"]}
    stamp = calendar[-2]
    frames["c"].pop(calendar[-1])
    results = [{"code": code, "engines": {"platform_events": {"raw_alert_dates": [stamp] if code in ["a", "b"] else []}}}
               for code in frames]
    signals = [{"stock_code": "a", "signal_date": stamp}, {"stock_code": "b", "signal_date": stamp}]
    matched, controls, missing = match_controls(signals, results, frames)
    assert missing == 0 and len(matched) == 2
    assert {row["stock_code"] for row in controls} == {"c", "d"}
    assert all(row["signal_date"] == stamp for row in controls)
    assert match_controls(signals, list(reversed(results)), frames) == (matched, controls, missing)
