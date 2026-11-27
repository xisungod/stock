from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta

from app.services.industry import (
    IndustryContext,
    IndustryService,
    aggregate_industries,
    apply_industry_context,
    score_industry,
)
from app.services.strategy import SignalCategory, SignalResult


def _signal() -> SignalResult:
    return SignalResult(
        stock_code="000001",
        stock_name="平安银行",
        category=SignalCategory.BUY_ALERT,
        score=82.0,
        suggestion="测试",
        key_line_price=10.0,
        line_strength=4,
        signal_date="2026-08-07",
    )


def test_industry_score_distinguishes_strong_and_high_risk_breadth() -> None:
    strong = score_industry(change_pct=2.2, up_count=38, down_count=8)
    weak = score_industry(change_pct=-2.8, up_count=6, down_count=40)

    assert strong[0] >= 70
    assert strong[1] == "low"
    assert weak[0] < 40
    assert weak[1] == "high"


def test_industry_adjustment_is_bounded_and_preserves_structure_score() -> None:
    result = _signal()
    context = IndustryContext(
        name="银行",
        score=10,
        risk_level="high",
        adjustment=-15,
        change_pct=-3.1,
        up_count=3,
        down_count=38,
        source="新浪行业",
        message="行业显著走弱",
    )

    apply_industry_context(result, context)

    assert result.structure_score == 82.0
    assert result.industry_adjustment == -15
    assert result.score == 67.0
    assert result.industry == "银行"
    assert any("行业" in item for item in result.risks)


def test_industry_aggregation_counts_selected_stocks() -> None:
    one = _signal()
    two = _signal()
    two.stock_code = "600000"
    two.category = SignalCategory.WATCH
    context = IndustryContext("银行", 74, "low", 10, 1.5, 30, 8, "新浪行业", "行业偏强")
    apply_industry_context(one, context)
    apply_industry_context(two, context)

    rows = aggregate_industries([one.to_dict(), two.to_dict()])

    assert rows[0]["industry"] == "银行"
    assert rows[0]["selected_count"] == 2
    assert rows[0]["buy_alert_count"] == 1


def _context() -> IndustryContext:
    return IndustryContext("银行", 74, "low", 10, 1.5, 30, 8, "测试", "行业偏强")


def test_in_memory_contexts_expire_and_refresh(tmp_path, monkeypatch) -> None:
    clock = [datetime(2026, 9, 4, 9)]
    service = IndustryService(tmp_path / "industry.json", now=lambda: clock[0])
    calls = []

    def fetch():
        calls.append(clock[0])
        return {"000001": _context()}

    monkeypatch.setattr(service, "_fetch", fetch)
    service.prepare()
    clock[0] += timedelta(hours=1)
    service.prepare()
    assert len(calls) == 1
    clock[0] += timedelta(hours=8)
    service.prepare()
    assert len(calls) == 2
    assert service.get_context("000001").updated_at == clock[0].isoformat(timespec="seconds")


def test_detail_loading_empty_cache_does_not_disable_scan_fetch(tmp_path, monkeypatch) -> None:
    service = IndustryService(tmp_path / "industry.json")
    assert service.get_context("000001") is None
    monkeypatch.setattr(service, "_fetch", lambda: {"000001": _context()})
    assert service.prepare()["000001"].stale is False


def test_failed_refresh_never_restamps_old_cache_and_can_retry(tmp_path, monkeypatch) -> None:
    path = tmp_path / "industry.json"
    original = {"updated_at": "2026-08-31T16:00:00", "stocks": {"000001": asdict(_context())}}
    path.write_text(json.dumps(original), encoding="utf-8")
    service = IndustryService(path, now=lambda: datetime(2026, 9, 5, 16))
    assert service.get_context("000001").stale is True

    def fail():
        raise RuntimeError("offline")

    monkeypatch.setattr(service, "_fetch", fail)
    assert service.prepare()["000001"].stale is True
    assert json.loads(path.read_text()) == original
    monkeypatch.setattr(service, "_fetch", lambda: {"000001": _context()})
    assert service.prepare()["000001"].stale is False


def test_expired_context_never_boosts_the_composite_score() -> None:
    from dataclasses import replace

    signal = _signal()
    apply_industry_context(signal, replace(_context(), stale=True, updated_at="2026-08-31"))
    assert signal.score == signal.structure_score == 82
    assert signal.industry_adjustment == 0
    assert signal.industry_score is None
    assert signal.industry_stale is True
    assert "过期" in signal.industry_risk_message


def test_conflicting_company_identity_never_uses_the_wrong_industry_score(tmp_path):
    service = IndustryService(tmp_path / "industry.json", profile_lookup=lambda code: "邮政业")
    raw = IndustryContext("家具行业", 33, "high", -6.8, -.34, 2, 11, "新浪行业", "wrong sector")
    corrected = service._validate_context("002120", raw)
    assert corrected.name == "邮政业"
    assert corrected.source_industry == "家具行业"
    assert corrected.validation_status == "conflict"
    assert service._validate_context("002120", corrected).validation_status == "conflict"
    signal = apply_industry_context(_signal(), corrected)
    assert signal.score == 82
    assert signal.industry_score is None
    assert signal.industry_adjustment == 0
    assert signal.industry_risk_level == "uncovered"
    assert "冲突" in signal.industry_risk_message


def test_unverified_real_vendor_classification_cannot_give_a_bonus(tmp_path):
    service = IndustryService(tmp_path / "industry.json", profile_lookup=lambda code: None)
    context = IndustryContext("电力行业", 78, "low", 10, 1.2, 50, 8, "新浪行业", "strong")
    unverified = service._validate_context("600780", context)
    assert unverified.validation_status == "unverified"
    assert apply_industry_context(_signal(), unverified).score == 82


def test_compatible_cached_profile_verifies_the_vendor_family(tmp_path):
    service = IndustryService(tmp_path / "industry.json", profile_lookup=lambda code: "电力、热力生产和供应业")
    context = IndustryContext("电力行业", 78, "low", 10, 1.2, 50, 8, "新浪行业", "strong")
    verified = service._validate_context("600780", context)
    assert verified.validation_status == "verified"
    assert apply_industry_context(_signal(), verified).score == 92


def test_expired_profile_is_not_used_for_classification(tmp_path):
    from app.services.industry_identity import cached_company_industry
    (tmp_path / "002120.json").write_text(json.dumps({
        "updated_at": "2025-01-01", "company": {"industry": "邮政业"}}))
    assert cached_company_industry("002120", tmp_path, datetime(2026, 9, 10)) is None
