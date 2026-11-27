import json
from datetime import datetime

import pandas as pd
import pytest

from app.services.company import CompanyService, _shareholder_payload, sanitize_legacy_shareholders


def rows(missing=False, previous_missing=False):
    output = []
    for stamp, gap in (("2025-12-31", previous_missing), ("2026-06-30", missing)):
        for i, value in enumerate([50, 10, 5, 4, 3, 2, 1, 1, .5, .5]):
            output.append({"截至日期": stamp, "股东名称": f"holder-{i}", "持股比例": None if gap and i == 0 else value})
    return pd.DataFrame(output)


def counts():
    return pd.DataFrame([{"股东户数统计截止日": "2026-06-30", "股东户数-本次": 100, "股东户数-增减比例": -5}])


def test_missing_largest_holder_is_not_a_fifty_point_decline():
    result = _shareholder_payload(rows(missing=True), counts())
    assert result["top10_concentration_pct"] is None
    assert result["top10_concentration_change_pct"] is None
    assert result["sentiment"] == "unknown"
    assert not any("50.00" in message for message in result["signals"])
    assert result["data_quality"]["valid_ratio_count"] == 9


def test_missing_prior_period_does_not_produce_an_increase():
    result = _shareholder_payload(rows(previous_missing=True), counts())
    assert result["top10_concentration_pct"] == 77
    assert result["top10_concentration_change_pct"] is None
    assert result["sentiment"] == "unknown"


def test_complete_percent_strings_are_normalized():
    frame = rows()
    frame["持股比例"] = frame["持股比例"].map(lambda value: f"{value}%")
    result = _shareholder_payload(frame, counts())
    assert result["top10_concentration_pct"] == 77
    assert result["top10_concentration_change_pct"] == 0
    assert result["sentiment"] == "positive"
    assert result["top_holders"][0]["ratio_pct"] == 50


@pytest.mark.parametrize("bad", [float("inf"), -1, 101, "--"])
def test_invalid_ratio_invalidates_the_aggregate(bad):
    frame = rows()
    frame["持股比例"] = frame["持股比例"].astype(object)
    frame.loc[10, "持股比例"] = bad
    assert _shareholder_payload(frame, counts())["top10_concentration_pct"] is None


def test_incomplete_or_duplicate_rows_are_not_top_ten():
    frame = rows().iloc[1:].copy()
    assert _shareholder_payload(frame, counts())["top10_concentration_change_pct"] is None
    frame = rows()
    frame.loc[10, "股东名称"] = "holder-1"
    assert _shareholder_payload(frame, counts())["top10_concentration_pct"] is None


def test_legacy_cache_cannot_reuse_an_unverifiable_delta(tmp_path):
    old = {"updated_at": datetime.now().isoformat(), "company": {"industry": "邮政业"},
           "shareholders": {"top_holders": [{"name": "one", "ratio_pct": None}], "top10_concentration_change_pct": -50, "sentiment": "negative"}}
    (tmp_path / "002120.json").write_text(json.dumps(old))
    service = CompanyService(tmp_path)
    assert service._load("002120") is None
    safe = service._load("002120", ignore_age=True)
    assert safe["company"]["industry"] == "邮政业"
    assert safe["shareholders"]["top10_concentration_change_pct"] is None
    assert safe["shareholders"]["sentiment"] == "unknown"
    assert sanitize_legacy_shareholders({})["top10_concentration_pct"] is None
