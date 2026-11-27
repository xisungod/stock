from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.schemas import BacktestRequest, ScanRequest
from app.services.strategy import SignalCategory, SignalResult


@pytest.mark.parametrize("endpoint", ["/api/scans/run", "/api/backtests/run"])
def test_invalid_provider_is_rejected_before_starting_a_job(tmp_path, endpoint):
    client = TestClient(create_app(tmp_path / "invalid-provider.db", FakeScanner()))
    response = client.post(endpoint, json={"provider": "akshrae"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "provider"]


class FakeScanner:
    provider_name = "fake"
    last_scan_stats = {
        "universe_count": 20,
        "attempted_count": 2,
        "analyzed_count": 2,
        "failed_count": 0,
        "current_stock": "",
    }

    def __init__(self):
        self.last_provider_name = "fake"
        self.last_backtest_stats = {
            "target_count": 0,
            "processed_count": 0,
            "failed_count": 0,
            "signal_count": 0,
        }

    def run_scan(self, params, on_progress=None, should_stop=None):
        results = [
            SignalResult(
                stock_code="000001",
                stock_name="平安银行",
                category=SignalCategory.BUY_ALERT,
                score=88.0,
                suggestion="买入预警：测试信号",
                key_line_price=10.1,
                line_strength=4,
                signal_date="2026-05-13",
                breakout_date="2026-05-10",
                pullback_date="2026-05-13",
                current_price=10.2,
                distance_to_line_pct=0.99,
                reasons=["放量突破", "缩量回踩"],
                risks=["技术预警不构成投资建议"],
                chart={"bars": [], "key_line": 10.1, "markers": []},
            ),
            SignalResult(
                stock_code="000002",
                stock_name="万科A",
                category=SignalCategory.WATCH,
                score=58.0,
                suggestion="观察机会：测试信号",
                key_line_price=8.8,
                line_strength=3,
                signal_date="2026-05-13",
                current_price=8.9,
                distance_to_line_pct=1.14,
                reasons=["接近关键线"],
                risks=[],
                chart={"bars": [], "key_line": 8.8, "markers": []},
            ),
        ]
        if on_progress is not None:
            for item in results:
                on_progress(dict(self.last_scan_stats), item)
        return results

    def analyze_stock(self, stock_code: str, stock_name: str | None = None):
        return self.run_scan({})[0]

    def diagnose_stock(self, stock_code: str, stock_name: str | None = None, scan_params=None):
        if stock_code == "000001":
            return {
                "stock_code": "000001",
                "stock_name": "平安银行",
                "found": True,
                "stage": "ok",
                "message": "买入预警：测试信号",
                "result": {"stock_code": "000001", "key_line_price": 10.1},
                "chart": {},
            }
        return {
            "stock_code": stock_code,
            "stock_name": stock_code,
            "found": False,
            "stage": "no_key_line",
            "message": "近 240 日未找到精准线",
            "result": None,
            "chart": {},
        }

    def latest_prices(self, stock_codes):
        return [
            {
                "stock_code": code,
                "latest_price": 11.22,
                "latest_price_at": "2026-05-14",
                "data_source": "fake",
                "error": None,
            }
            for code in stock_codes
        ]

    def backtest_stock(self, stock_code, stock_name=None, lookback_days=0, scan_params=None):
        return {
            "stock_code": stock_code,
            "stock_name": stock_name or stock_code,
            "lookback_days": lookback_days,
            "received_params": scan_params or {},
            "signals": [],
            "summary": {"signal_count": 0},
        }

    def run_backtest(self, params, on_progress=None):
        self.last_provider_name = params.get("provider", "fake")
        self.last_backtest_stats = {
            "target_count": 0,
            "processed_count": 0,
            "failed_count": 0,
            "signal_count": 0,
        }
        return [], {"stock_count": 0, "stocks_with_signals": 0, "signal_count": 0}


def _client(tmp_path: Path) -> TestClient:
    app = create_app(db_path=tmp_path / "stock_mine.db", scanner=FakeScanner())
    return TestClient(app)


def test_health_reports_ok(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_flag_request_validation_and_legacy_defaults() -> None:
    assert ScanRequest().strategy_mode == "combined"
    assert ScanRequest(strategy_mode="flag_breakout", flag_direction="ascending").flag_direction == "ascending"
    for payload in (
        {"strategy_mode": "unknown"},
        {"strategy_mode": "flag_breakout", "flag_direction": "unknown"},
        {"strategy_mode": "flag_breakout", "min_days_since_breakout": 2},
    ):
        with pytest.raises(ValidationError):
            ScanRequest(**payload)


def test_prebreakout_defaults_to_fifty_percent_thirty_days_and_two_percent_buffer():
    for model in (ScanRequest, BacktestRequest):
        assert model().max_prebreakout_extension_pct == .5
        assert model().prebreakout_reject_above_days == 30
        assert model().prebreakout_above_buffer_pct == .02
        assert model(max_prebreakout_extension_pct=0).max_prebreakout_extension_pct == 0
        assert model(max_prebreakout_extension_pct=None).max_prebreakout_extension_pct is None
        assert model(prebreakout_reject_above_days=None).prebreakout_reject_above_days is None
        assert model(prebreakout_reject_above_days=1).prebreakout_reject_above_days == 1
        assert model(prebreakout_above_buffer_pct=0).prebreakout_above_buffer_pct == 0
        for value in (-.1, 1.1, 20, float("nan")):
            with pytest.raises(ValidationError):
                model(max_prebreakout_extension_pct=value)
            with pytest.raises(ValidationError):
                model(prebreakout_above_buffer_pct=value)
        for value in (0, -1, 251, 9.5, float("inf")):
            with pytest.raises(ValidationError):
                model(prebreakout_reject_above_days=value)


def test_advertised_path_defaults_match_scan_and_backtest_models(tmp_path):
    advertised = _client(tmp_path).get("/api/config/defaults").json()
    for name in ("max_prebreakout_extension_pct", "prebreakout_reject_above_days", "prebreakout_above_buffer_pct"):
        assert advertised[name] == getattr(ScanRequest(), name) == getattr(BacktestRequest(), name)


@pytest.mark.parametrize("saved_limit", [None, .2, .35])
@pytest.mark.parametrize("duration_params", [{}, {"prebreakout_reject_above_days": None},
    {"prebreakout_reject_above_days": 10, "prebreakout_above_buffer_pct": .02},
    {"prebreakout_reject_above_days": 30, "prebreakout_above_buffer_pct": .02},
    {"prebreakout_reject_above_days": 8, "prebreakout_above_buffer_pct": 0}])
def test_prebreakout_limit_is_inherited_by_single_and_batch_replays(tmp_path, saved_limit, duration_params):
    from app.db import Database
    captured = []
    scanner = FakeScanner()
    original = scanner.run_backtest
    def recording(params, on_progress=None):
        captured.append(params)
        return original(params, on_progress)
    scanner.run_backtest = recording
    path = tmp_path / "limit.db"
    client = TestClient(create_app(db_path=path, scanner=scanner))
    params = {"provider": "fake", "strategy_mode": "precision_pullback"}
    if saved_limit is not None:
        params["max_prebreakout_extension_pct"] = saved_limit
    params.update(duration_params)
    run_id = Database(path).create_scan_run("fake", params)
    single = client.get(f"/api/stocks/000001/backtest?scan_id={run_id}&lookback_days=120")
    assert single.json()["received_params"]["max_prebreakout_extension_pct"] == saved_limit
    assert single.json()["received_params"]["prebreakout_reject_above_days"] == duration_params.get("prebreakout_reject_above_days")
    assert single.json()["received_params"]["prebreakout_above_buffer_pct"] == duration_params.get("prebreakout_above_buffer_pct", .02)
    submitted = client.post("/api/backtests/run", json={
        "scan_id": run_id, "lookback_days": 120, "max_prebreakout_extension_pct": .9,
        "prebreakout_reject_above_days": 99, "prebreakout_above_buffer_pct": .08,
    }).json()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get(f"/api/backtests/{submitted['run_id']}/progress").json()
        if state["status"] != "running":
            break
        time.sleep(.01)
    assert state["status"] == "completed"
    assert captured[0]["max_prebreakout_extension_pct"] == saved_limit
    assert captured[0]["prebreakout_reject_above_days"] == duration_params.get("prebreakout_reject_above_days")
    assert captured[0]["prebreakout_above_buffer_pct"] == duration_params.get("prebreakout_above_buffer_pct", .02)


def test_flag_batch_params_reach_single_and_batch_backtests(tmp_path: Path) -> None:
    from app.db import Database
    captured = []
    scanner = FakeScanner()
    original = scanner.run_backtest
    def recording(params, on_progress=None):
        captured.append(params)
        return original(params, on_progress)
    scanner.run_backtest = recording
    path = tmp_path / "flag.db"
    client = TestClient(create_app(db_path=path, scanner=scanner))
    db = Database(path)
    run_id = db.create_scan_run("fake", {
        "provider": "fake", "strategy_mode": "flag_breakout", "flag_direction": "ascending",
    })
    single = client.get(f"/api/stocks/000001/backtest?scan_id={run_id}&lookback_days=120")
    assert single.status_code == 200
    assert single.json()["received_params"]["strategy_mode"] == "flag_breakout"
    response = client.post("/api/backtests/run", json={"scan_id": run_id, "lookback_days": 120})
    assert response.status_code == 200
    deadline = time.monotonic() + 2
    while not captured and time.monotonic() < deadline:
        time.sleep(.01)
    assert captured[0]["strategy_mode"] == "flag_breakout"
    assert captured[0]["flag_direction"] == "ascending"


def test_legacy_batch_cannot_be_silently_replayed_as_flags(tmp_path: Path) -> None:
    from app.db import Database
    captured = []
    scanner = FakeScanner()
    original = scanner.run_backtest
    def recording(params, on_progress=None):
        captured.append(params)
        return original(params, on_progress)
    scanner.run_backtest = recording
    path = tmp_path / "old.db"
    client = TestClient(create_app(db_path=path, scanner=scanner))
    run_id = Database(path).create_scan_run("fake", {"provider": "fake"})
    client.post("/api/backtests/run", json={"scan_id": run_id, "strategy_mode": "flag_breakout", "flag_direction": "ascending"})
    deadline = time.monotonic() + 2
    while not captured and time.monotonic() < deadline:
        time.sleep(.01)
    assert captured[0]["strategy_mode"] == "precision_pullback"
    assert captured[0]["flag_direction"] == "all"


def test_scan_freshness_counts_survive_storage_and_legacy_results_are_identified(tmp_path: Path) -> None:
    from app.db import Database
    client = _client(tmp_path)
    db = Database(tmp_path / "stock_mine.db")
    run_id = db.create_scan_run("fake", {})
    legacy = FakeScanner().run_scan({})[0].to_dict()
    legacy.pop("strategy_version")
    db.complete_scan_run(run_id, "fake", {}, [legacy], {
        "attempted_count": 3, "analyzed_count": 1, "failed_count": 2, "stale_count": 1,
    })
    payload = client.get("/api/scans/latest").json()
    assert payload["stale_count"] == 1
    assert payload["failed_count"] == 2
    assert payload["strategy_outdated"] is True
    assert len(payload["results"]) == 1, "不能因为是旧策略结果就删掉历史预警"


def test_watchlist_crud(tmp_path: Path) -> None:
    client = _client(tmp_path)

    created = client.post(
        "/api/watchlist",
        json={
            "stock_code": "000001",
            "stock_name": "平安银行",
            "note": "重点跟踪",
            "entry_price": 10.2,
            "entry_date": "2026-05-13",
        },
    )
    assert created.status_code == 200
    assert created.json()["stock_code"] == "000001"
    assert created.json()["entry_price"] == 10.2
    assert created.json()["return_pct"] == 0

    listed = client.get("/api/watchlist")
    assert listed.status_code == 200
    assert listed.json()[0]["note"] == "重点跟踪"

    updated = client.patch("/api/watchlist/000001", json={"status": "paused", "note": "等待回踩"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "paused"

    refreshed = client.post("/api/watchlist/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["updated_count"] == 1
    assert refreshed.json()["items"][0]["latest_price"] == 11.22
    assert refreshed.json()["items"][0]["price_change"] == 1.02
    assert refreshed.json()["items"][0]["return_pct"] == 10.0

    repeated = client.post(
        "/api/watchlist",
        json={"stock_code": "000001", "stock_name": "平安银行", "entry_price": 12.5},
    )
    assert repeated.json()["entry_price"] == 10.2, "重复加入不能重置最初加入价"
    assert repeated.json()["latest_price"] == 11.22, "旧筛选价不能覆盖已经刷新的行情"

    deleted = client.delete("/api/watchlist/000001")
    assert deleted.status_code == 200
    assert client.get("/api/watchlist").json() == []


def _wait_done(client, scan_id):
    for _ in range(300):
        payload = client.get(f"/api/scans/{scan_id}/progress").json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("扫描未在超时时间内结束")


def _wait_backtest_done(client, run_id):
    for _ in range(300):
        payload = client.get(f"/api/backtests/{run_id}/progress").json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("复盘未在超时时间内结束")


def test_run_scan_persists_latest_results(tmp_path: Path) -> None:
    client = _client(tmp_path)

    run = client.post("/api/scans/run", json={"limit": 2, "provider": "fake"})
    assert run.status_code == 200
    scan_id = run.json()["scan_id"]

    progress = _wait_done(client, scan_id)
    assert progress["status"] == "completed"

    latest = client.get("/api/scans/latest")
    assert latest.status_code == 200
    body = latest.json()
    assert body["id"] == scan_id
    assert body["buy_alert_count"] == 1
    assert body["watch_count"] == 1
    assert body["attempted_count"] == 2
    assert body["failed_count"] == 0
    assert body["results"][0]["stock_code"] == "000001"

    results = client.get(f"/api/scans/{scan_id}/results")
    assert results.status_code == 200
    assert len(results.json()) == 2


def test_batch_backtest_inherits_the_source_scan_strategy(tmp_path: Path) -> None:
    client = _client(tmp_path)
    scan = client.post(
        "/api/scans/run",
        json={
            "provider": "fake",
            "limit": 2,
            "signal_offset": 0,
            "breakout_vol_mult": 1.2,
            "require_breakout": True,
            "min_days_since_breakout": 2,
            "max_days_since_breakout": 6,
        },
    ).json()
    _wait_done(client, scan["scan_id"])

    started = client.post(
        "/api/backtests/run",
        json={"scan_id": scan["scan_id"], "lookback_days": 0, "stock_codes": ["000001"]},
    )
    assert started.status_code == 200
    _wait_backtest_done(client, started.json()["run_id"])
    params = client.get("/api/backtests/latest").json()["parameters"]

    assert params["scan_id"] == scan["scan_id"]
    assert params["lookback_days"] == 0
    assert params["provider"] == "fake"
    assert params["signal_offset"] == 0
    assert params["breakout_vol_mult"] == 1.2
    assert params["require_breakout"] is True
    assert params["min_days_since_breakout"] == 2
    assert params["max_days_since_breakout"] == 6


def test_run_scan_returns_immediately_with_scan_id(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/api/scans/run", json={"provider": "fake", "limit": 2})

    assert response.status_code == 200
    payload = response.json()
    assert "scan_id" in payload
    assert payload["status"] in {"running", "completed"}


def test_scan_progress_endpoint_returns_state(tmp_path: Path) -> None:
    client = _client(tmp_path)
    scan_id = client.post("/api/scans/run", json={"provider": "fake", "limit": 2}).json()["scan_id"]

    payload = _wait_done(client, scan_id)

    assert payload["status"] == "completed"
    assert payload["scan_id"] == scan_id
    assert isinstance(payload["results"], list)


def test_progress_returns_404_for_unknown_scan(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/api/scans/999999/progress").status_code == 404


def test_active_scan_returns_null_when_idle(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/api/scans/active").json() is None


def test_defaults_limit_is_full_market(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/api/config/defaults").json()["limit"] == 0


def test_scan_request_without_limit_defaults_to_full_market(tmp_path: Path) -> None:
    client = _client(tmp_path)

    run = client.post("/api/scans/run", json={"provider": "fake"})

    assert run.status_code == 200
    scan_id = run.json()["scan_id"]
    _wait_done(client, scan_id)

    latest = client.get("/api/scans/latest").json()
    # db.get_scan 已经把 parameters_json 反序列化过（db.py 的 _scan_row），这里拿到的就是 dict
    assert isinstance(latest["parameters"], dict)
    assert latest["parameters"]["limit"] == 0, "不带 limit 提交时应默认全量扫描"


def test_stock_analysis_returns_diagnosis_when_no_signal(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/api/stocks/000002/analysis")

    assert response.status_code == 200
    payload = response.json()
    assert payload["found"] is False
    assert payload["stage"] == "no_key_line"
    assert "精准线" in payload["message"]


def test_stock_analysis_returns_result_when_found(tmp_path: Path) -> None:
    client = _client(tmp_path)

    payload = client.get("/api/stocks/000001/analysis").json()

    assert payload["found"] is True
    assert payload["result"]["key_line_price"] == 10.1


def test_startup_marks_stale_running_scans_as_failed(tmp_path: Path) -> None:
    """中断任务要标记失败，但不能覆盖最近一次成功筛选。"""
    from app.db import Database

    db_path = tmp_path / "stock_mine.db"
    db = Database(db_path)
    completed_id = db.create_scan_run("akshare", {"limit": 2})
    db.complete_scan_run(completed_id, "akshare", {"limit": 2}, [], {})
    stale_id = db.create_scan_run("akshare", {"limit": 0})

    # 模拟重启：同一个库上重新建 app
    client = TestClient(create_app(db_path=db_path, scanner=FakeScanner()))

    latest = client.get("/api/scans/latest").json()
    assert latest["id"] == completed_id
    stale = Database(db_path).get_scan(stale_id)
    assert stale is not None
    assert stale["status"] == "failed"
    assert stale["error"], "应写明失败原因"


def test_bulk_add_inserts_every_stock_and_reports_counts(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/watchlist/bulk",
        json={"items": [
            {"stock_code": "000001", "stock_name": "平安银行", "note": "批量"},
            {"stock_code": "600519", "stock_name": "贵州茅台"},
            {"stock_code": "1", "stock_name": "补零测试"},
        ]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["added"] == 2, "「1」补零后与 000001 重复"
    assert payload["skipped"] == 1
    codes = {item["stock_code"] for item in payload["items"]}
    assert codes == {"000001", "600519"}, f"「1」应补零成 000001 并与已有的合并，实得 {codes}"
    assert len(client.get("/api/watchlist").json()) == 2


def test_bulk_add_is_idempotent_for_stocks_already_in_the_pool(tmp_path: Path) -> None:
    """重复批量加入不该产生重复行，也不该报错。"""
    client = _client(tmp_path)
    client.post("/api/watchlist", json={"stock_code": "000001", "stock_name": "平安银行"})

    payload = client.post(
        "/api/watchlist/bulk",
        json={"items": [{"stock_code": "000001", "stock_name": "平安银行"},
                        {"stock_code": "600519", "stock_name": "贵州茅台"}]},
    ).json()

    assert payload["added"] == 1
    assert payload["skipped"] == 1
    assert len(client.get("/api/watchlist").json()) == 2


def test_bulk_delete_removes_only_the_listed_stocks(tmp_path: Path) -> None:
    client = _client(tmp_path)
    for code in ("000001", "600519", "300750"):
        client.post("/api/watchlist", json={"stock_code": code, "stock_name": code})

    payload = client.post("/api/watchlist/bulk-delete", json={"stock_codes": ["000001", "300750"]}).json()

    assert payload["deleted"] == 2
    remaining = [item["stock_code"] for item in client.get("/api/watchlist").json()]
    assert remaining == ["600519"]


def test_bulk_delete_ignores_codes_that_are_not_in_the_pool(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/api/watchlist", json={"stock_code": "000001", "stock_name": "平安银行"})

    payload = client.post("/api/watchlist/bulk-delete", json={"stock_codes": ["000001", "999999"]}).json()

    assert payload["deleted"] == 1
    assert client.get("/api/watchlist").json() == []


def test_bulk_endpoints_reject_an_empty_list(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/api/watchlist/bulk", json={"items": []}).status_code == 422
    assert client.post("/api/watchlist/bulk-delete", json={"stock_codes": []}).status_code == 422


def test_cancel_endpoint_stops_a_running_scan(tmp_path: Path) -> None:
    client = _client(tmp_path)
    started = client.post("/api/scans/run", json={"provider": "fake", "limit": 2}).json()

    response = client.post(f"/api/scans/{started['scan_id']}/cancel")

    assert response.status_code == 200
    assert response.json()["cancelled"] in (True, False)


def test_cancelling_an_unknown_scan_is_404(tmp_path: Path) -> None:
    assert _client(tmp_path).post("/api/scans/999999/cancel").status_code == 404


def test_an_unknown_scan_id_is_rejected_instead_of_silently_defaulting(tmp_path: Path) -> None:
    """批次号无效时要报 404，不能悄悄退回默认配置。

    URL 写错或批次被删除时，页面会展示另一套筛选条件下的结论，而用户以为
    自己看的还是那一批。
    """
    client = _client(tmp_path)
    assert client.get("/api/stocks/000001/detail?scan_id=999999999").status_code == 404
    assert client.get("/api/stocks/000001/analysis?scan_id=999999999").status_code == 404
    # 不带 scan_id 仍然正常
    assert client.get("/api/stocks/000001/analysis").status_code == 200


def test_breakout_window_rejects_an_impossible_range() -> None:
    with pytest.raises(ValidationError):
        ScanRequest(min_days_since_breakout=10, max_days_since_breakout=5)
    with pytest.raises(ValidationError):
        ScanRequest(min_days_since_breakout=10, max_days_since_breakout=0)
    with pytest.raises(ValidationError):
        ScanRequest(min_days_since_breakout=0, max_days_since_breakout=10)
    assert ScanRequest(min_days_since_breakout=9, max_days_since_breakout=0)
    # 上下限相等 = "恰好第 5 天"，是合法筛选，边界不能收紧成 >=
    assert ScanRequest(min_days_since_breakout=5, max_days_since_breakout=5)
    # 必须留一组落在 0～9 合法区间内的逆序值：上面几组都被 le=MAX_BREAKOUT_AGE
    # 先拦下了，走不到 min>max 那条分支——只靠它们，整条范围校验删掉也全绿。
    with pytest.raises(ValidationError):
        ScanRequest(min_days_since_breakout=7, max_days_since_breakout=3)


def test_backtest_request_accepts_zero_as_all_history() -> None:
    assert BacktestRequest(lookback_days=0).lookback_days == 0
    with pytest.raises(ValidationError):
        BacktestRequest(lookback_days=60)


def test_single_stock_backtest_rejects_the_same_short_window_as_batch(tmp_path: Path) -> None:
    response = _client(tmp_path).get("/api/stocks/000001/backtest?lookback_days=60")

    assert response.status_code == 422


def test_single_stock_backtest_receives_the_scan_parameters(tmp_path: Path) -> None:
    client = _client(tmp_path)
    started = client.post(
        "/api/scans/run",
        json={"provider": "fake", "limit": 2, "breakout_vol_mult": 1.2, "require_breakout": True},
    ).json()
    _wait_done(client, started["scan_id"])

    response = client.get(f"/api/stocks/000001/backtest?lookback_days=120&scan_id={started['scan_id']}")

    assert response.status_code == 200
    assert response.json()["received_params"]["breakout_vol_mult"] == 1.2
    assert response.json()["received_params"]["require_breakout"] is True


def test_legacy_breakout_age_cap_tracks_the_legacy_search_window() -> None:
    """旧批次的 MAX_BREAKOUT_AGE 从 breakout_window 推出来，不能各自漂移。

    _find_breakout 的搜索起点是 signal_idx - breakout_window + 1，所以它能报出的
    age 上限恒为 breakout_window - 1。两个常量分别写在 schemas.py 和 strategy.py 里，
    此前只靠注释维系：把 breakout_window 改成 5 或 20，全量测试都照样全绿，
    改小的那个方向会让 min_days_since_breakout=9 重新变成"永远选不出东西"的死区。
    """
    from app.schemas import MAX_BREAKOUT_AGE
    from app.services.strategy import StrategyConfig

    legacy = StrategyConfig(require_pullback=False, max_pullback_days=None)
    assert MAX_BREAKOUT_AGE == legacy.breakout_window - 1, (
        f"改了旧版 breakout_window 就要同步兼容参数 MAX_BREAKOUT_AGE："
        f"window={legacy.breakout_window}, cap={MAX_BREAKOUT_AGE}"
    )
