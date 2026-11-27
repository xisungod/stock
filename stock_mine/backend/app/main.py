from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.db import Database
from app.schemas import (
    BacktestRequest,
    RuleDefaults,
    ScanRequest,
    WatchBulkCreate,
    WatchBulkDelete,
    WatchCreate,
    WatchUpdate,
    validate_backtest_lookback,
    PrecisionEngine,
)
from app.services.backtest_job_manager import BacktestJobManager
from app.services.industry import aggregate_industries
from app.services.job_manager import ScanJobManager
from app.services.scanner import Scanner
from app.services.strategy import STRATEGY_VERSION, version_for_engine
from app.services.alert_monitor import AlertMonitor, MonitorFilters
from app.services.market_calendar import CalendarUnavailable, china_now


DEFAULT_DB_PATH = Path(
    os.environ.get(
        "STOCK_MINE_DB_PATH",
        str(Path(__file__).resolve().parents[2] / "data" / "stock_mine.db"),
    )
)


def create_app(
    db_path: str | Path = DEFAULT_DB_PATH, scanner: Any | None = None, *,
    monitor_quotes=None, monitor_clock=china_now, monitor_schedule_enabled: bool | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        monitor_service.start()
        try:
            yield
        finally:
            monitor_service.stop()

    app = FastAPI(title="Stock Mine API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    db = Database(db_path)
    # 上次进程退出时若有扫描在跑，它会永远停在 running（进度只在内存里），启动时清理
    db.fail_stale_running_scans()
    scan_service = scanner or Scanner()
    job_manager = ScanJobManager(db=db, scanner=scan_service)
    backtest_manager = BacktestJobManager(db=db, scanner=scan_service)
    monitor_service = AlertMonitor(
        db, quotes=monitor_quotes, clock=monitor_clock,
        schedule_enabled=(
            os.environ.get("STOCK_MINE_MONITOR_SCHEDULE_ENABLED", "0") == "1"
            if monitor_schedule_enabled is None else monitor_schedule_enabled
        ),
    )
    app.state.alert_monitor = monitor_service

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config/defaults")
    def defaults() -> dict[str, Any]:
        return {
            "provider": "auto",
            "limit": 0,
            "signal_offset": 0,
            "signal_offset_help": "0 = 最近一个已收盘交易日（盘中未收盘K线自动去掉），与回测一致。",
            **RuleDefaults().model_dump(),
            "limit_help": "0 表示扫描全部股票；正数表示均匀覆盖全市场的抽样数量。",
            "min_days_since_breakout": 0,
            "max_days_since_breakout": 0,
            "require_breakout": False,
            "require_pullback": True,
            "min_pullback_days": 1,
            "max_pullback_days": 10,
            "pullback_days_help": "突破次日为第1个交易日，信号日须在设置范围内完成缩量回踩；可设置1～60天。",
            "breakout_vol_mult": 1.5,
            "require_breakout_volume": False,
            "strategy_mode": "combined",
            "precision_engine": "platform_events",
            "precision_engines": ["platform_events", "pivot_fixed", "legacy"],
            "strategy_version": STRATEGY_VERSION,
            "flag_direction": "all",
            "max_prebreakout_extension_pct": 0.50,
            "prebreakout_reject_above_days": 30,
            "prebreakout_above_buffer_pct": 0.02,
            "breakout_age_help": "旧版兼容参数（0～9）；新筛选使用回调天数范围（1～60个交易日）。",
            "disclaimer": "所有提示均为技术分析预警，不构成投资建议。",
        }

    @app.get("/api/alert-monitor")
    def alert_monitor(filters: Annotated[MonitorFilters, Query()]):
        try:
            return monitor_service.listing(filters)
        except CalendarUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/alert-monitor/status")
    def alert_monitor_status():
        return monitor_service.status()

    @app.post("/api/alert-monitor/refresh", status_code=202)
    def refresh_alert_monitor():
        try:
            return monitor_service.submit()
        except CalendarUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/alert-monitor/{entry_id}/history")
    def alert_monitor_history(entry_id: int):
        rows = monitor_service.store.history(entry_id)
        if rows is None:
            raise HTTPException(status_code=404, detail="跟踪记录不存在")
        return {"entry_id": entry_id, "items": rows, "price_basis": "unadjusted"}

    @app.post("/api/scans/run")
    def run_scan(request: ScanRequest) -> dict[str, Any]:
        scan_id = job_manager.submit({**request.model_dump(), "strategy_version": version_for_engine(request.precision_engine)})
        state = job_manager.get(scan_id) or {"scan_id": scan_id, "status": "running"}
        return state

    @app.post("/api/scans/{scan_id}/cancel")
    def cancel_scan(scan_id: int) -> dict[str, Any]:
        """请求停止扫描。已命中的结果会照常落库，不会白跑。"""
        if job_manager.get(scan_id) is None:
            raise HTTPException(status_code=404, detail="scan not found")
        return {"cancelled": job_manager.cancel(scan_id)}

    @app.get("/api/scans/active")
    def active_scan() -> dict[str, Any] | None:
        return job_manager.active()

    @app.get("/api/scans/{scan_id}/progress")
    def scan_progress(scan_id: int) -> dict[str, Any]:
        state = job_manager.get(scan_id)
        if state is None:
            raise HTTPException(status_code=404, detail="scan not found")
        return state

    @app.get("/api/scans/latest")
    def latest_scan() -> dict[str, Any]:
        latest = db.latest_scan()
        if latest is None:
            return {
                "results": [],
                "total_count": 0,
                "buy_alert_count": 0,
                "watch_count": 0,
                "universe_count": 0,
                "attempted_count": 0,
                "analyzed_count": 0,
                "failed_count": 0,
                "industry_summary": [],
            }
        latest["industry_summary"] = aggregate_industries(latest.get("results", []))
        expected_version = version_for_engine((latest.get("parameters") or {}).get("precision_engine", "legacy"))
        latest["strategy_outdated"] = any(item.get("strategy_version") != expected_version for item in latest.get("results", []))
        return latest

    @app.get("/api/scans/{scan_id}/results")
    def scan_results(scan_id: int) -> list[dict[str, Any]]:
        if db.get_scan(scan_id) is None:
            raise HTTPException(status_code=404, detail="scan not found")
        return db.get_scan_results(scan_id)

    @app.get("/api/watchlist")
    def watchlist() -> list[dict[str, Any]]:
        return db.list_watchlist()

    @app.post("/api/watchlist")
    def add_watch(item: WatchCreate) -> dict[str, Any]:
        stock_name = item.stock_name or item.stock_code
        return db.upsert_watch(
            item.stock_code.zfill(6),
            stock_name,
            item.note,
            item.status,
            item.entry_price,
            item.entry_date,
        )

    @app.post("/api/watchlist/bulk")
    def bulk_add_watch(payload: WatchBulkCreate) -> dict[str, Any]:
        """批量加入观测池。已在池中的按跳过计，不报错也不产生重复行。"""
        existing = {item["stock_code"] for item in db.list_watchlist()}
        added = 0
        skipped = 0
        for item in payload.items:
            code = item.stock_code.zfill(6)
            if code in existing:
                skipped += 1
                continue
            db.upsert_watch(
                code,
                item.stock_name or code,
                item.note,
                item.status,
                item.entry_price,
                item.entry_date,
            )
            existing.add(code)
            added += 1
        return {"added": added, "skipped": skipped, "items": db.list_watchlist()}

    @app.post("/api/watchlist/bulk-delete")
    def bulk_delete_watch(payload: WatchBulkDelete) -> dict[str, Any]:
        """批量移除。池中不存在的代码静默忽略，只统计真正删掉的数量。"""
        deleted = sum(1 for code in payload.stock_codes if db.delete_watch(code.zfill(6)))
        return {"deleted": deleted, "items": db.list_watchlist()}

    @app.post("/api/watchlist/refresh")
    def refresh_watchlist_prices() -> dict[str, Any]:
        watches = db.list_watchlist()
        if not watches:
            return {"items": [], "updated_count": 0, "failed_count": 0, "errors": []}
        if not hasattr(scan_service, "latest_prices"):
            raise HTTPException(status_code=503, detail="当前分析服务不支持行情刷新")
        quotes = scan_service.latest_prices([item["stock_code"] for item in watches])
        errors: list[dict[str, str]] = []
        updated_count = 0
        for quote in quotes:
            if quote.get("latest_price") is None:
                errors.append(
                    {
                        "stock_code": quote["stock_code"],
                        "message": quote.get("error") or "行情获取失败",
                    }
                )
                continue
            db.update_watch_price(
                quote["stock_code"],
                quote["latest_price"],
                quote.get("latest_price_at"),
            )
            updated_count += 1
        return {
            "items": db.list_watchlist(),
            "updated_count": updated_count,
            "failed_count": len(errors),
            "errors": errors,
        }

    @app.patch("/api/watchlist/{stock_code}")
    def update_watch(stock_code: str, item: WatchUpdate) -> dict[str, Any]:
        updated = db.update_watch(stock_code.zfill(6), note=item.note, status=item.status)
        if updated is None:
            raise HTTPException(status_code=404, detail="watch item not found")
        return updated

    @app.delete("/api/watchlist/{stock_code}")
    def delete_watch(stock_code: str) -> dict[str, bool]:
        db.delete_watch(stock_code.zfill(6))
        return {"deleted": True}

    def _scan_params(scan_id: int | None) -> dict[str, Any] | None:
        """按批次号取回那次筛选用的参数，让详情与列表保持同一套判定。

        批次不存在时报 404 而不是退回默认配置——否则 URL 写错或批次被删除后，
        页面会展示另一套条件下的结论，而用户以为看的还是那一批。
        """
        if not scan_id:
            return None
        run = db.get_scan(scan_id)
        if run is None:
            raise HTTPException(status_code=404, detail="scan not found")
        params = dict(run.get("parameters") or {})
        params.setdefault("precision_engine", "legacy")
        params.setdefault("max_prebreakout_extension_pct", None)
        params.setdefault("prebreakout_reject_above_days", None)
        params.setdefault("prebreakout_above_buffer_pct", 0.02)
        params.setdefault("require_pullback", False)
        params.setdefault("min_pullback_days", 1)
        params.setdefault("max_pullback_days", None)
        params.setdefault("require_breakout_volume", True)
        # 2026-11-27 之前的批次没有这些规则：缺失即未启用，结论不被静默改写。
        for name in RuleDefaults.model_fields:
            params.setdefault(name, "off" if name == "market_filter" else None)
        return params

    def _analysis_params(scan_id, precision_engine=None, as_of=None):
        params = _scan_params(scan_id)
        if precision_engine is not None or as_of is not None:
            if params is None:
                params = asdict(scan_service.strategy_config()) if hasattr(scan_service, "strategy_config") else ScanRequest().model_dump()
            else:
                params = dict(params)
            if precision_engine is not None:
                params["precision_engine"] = precision_engine
            if as_of is not None:
                if as_of > date.today():
                    raise HTTPException(status_code=422, detail="历史分析日期不能晚于今天")
                params["_analysis_as_of"] = as_of.isoformat()
        return params

    @app.get("/api/stocks/{stock_code}/analysis")
    def stock_analysis(stock_code: str, stock_name: str | None = None,
                       scan_id: int | None = None) -> dict[str, Any]:
        params = _scan_params(scan_id)
        return scan_service.diagnose_stock(stock_code.zfill(6), stock_name, params)

    @app.get("/api/stocks/{stock_code}/detail")
    def stock_detail(stock_code: str, stock_name: str | None = None, years: int = 0,
                     scan_id: int | None = None, precision_engine: PrecisionEngine | None = None,
                     as_of: date | None = None) -> dict[str, Any]:
        # 批次校验放在 try 之外：它要抛的是 404，不该被下面的兜底吞成 503
        params = _analysis_params(scan_id, precision_engine, as_of)
        try:
            history_years = 0 if years <= 0 else max(3, min(40, years))
            payload = scan_service.stock_detail(stock_code.zfill(6), stock_name, history_years, params)
            if scan_id:
                source_version = (params or {}).get("strategy_version")
                source_engine = (_scan_params(scan_id) or {}).get("precision_engine", "legacy")
                payload["scan_context"] = {
                    "scan_id": scan_id, "source_strategy_version": source_version,
                    "recomputed_with_new_strategy": source_version != version_for_engine(params.get("precision_engine", "legacy")),
                    "engine_override": precision_engine is not None and precision_engine != source_engine,
                }
            return payload
        except HTTPException:
            # 404（批次不存在）之类的语义化错误要原样传出，别被下面的兜底吞成 503
            raise
        except ValueError as exc:
            if as_of:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/stocks/{stock_code}/backtest")
    def stock_backtest(stock_code: str, stock_name: str | None = None, lookback_days: int = 0,
                       scan_id: int | None = None, precision_engine: PrecisionEngine | None = None,
                       as_of: date | None = None) -> dict[str, Any]:
        params = _analysis_params(scan_id, precision_engine, as_of)
        try:
            try:
                history_days = validate_backtest_lookback(lookback_days)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return scan_service.backtest_stock(stock_code.zfill(6), stock_name, history_days, params)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/stocks/{stock_code}/strategy-comparison")
    def stock_strategy_comparison(stock_code: str, stock_name: str | None = None,
                                  scan_id: int | None = None, as_of: date | None = None):
        params = _analysis_params(scan_id, None, as_of)
        if not hasattr(scan_service, "compare_stock_strategies"):
            raise HTTPException(status_code=503, detail="当前分析服务不支持引擎对照")
        return scan_service.compare_stock_strategies(stock_code.zfill(6), stock_name, params)

    @app.get("/api/stocks/{stock_code}/company")
    def stock_company(stock_code: str) -> dict[str, Any]:
        return scan_service.stock_company(stock_code.zfill(6))

    @app.get("/api/stocks/{stock_code}/news")
    def stock_news(stock_code: str) -> dict[str, Any]:
        return scan_service.stock_news(stock_code.zfill(6))

    @app.get("/api/stocks/{stock_code}/related-board")
    def stock_related_board(stock_code: str, years: int = 3) -> dict[str, Any]:
        try:
            return scan_service.stock_related_board(stock_code.zfill(6), years=max(1, min(10, years)))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/stocks/{stock_code}/industry-market")
    def stock_industry_market(stock_code: str, years: int = 10) -> dict[str, Any]:
        return scan_service.stock_industry_market(stock_code.zfill(6), max(3, min(20, years)))

    @app.post("/api/backtests/run")
    def run_backtest(request: BacktestRequest) -> dict[str, Any]:
        params = request.model_dump()
        scan_params = _scan_params(request.scan_id)
        if scan_params:
            # 股票范围和回看长度来自本次复盘请求；策略与数据源必须沿用原筛选批次。
            params["strategy_mode"] = scan_params.get("strategy_mode", "precision_pullback")
            params["precision_engine"] = scan_params.get("precision_engine", "legacy")
            params["flag_direction"] = scan_params.get("flag_direction", "all")
            params["max_prebreakout_extension_pct"] = scan_params.get("max_prebreakout_extension_pct")
            params["prebreakout_reject_above_days"] = scan_params.get("prebreakout_reject_above_days")
            params["prebreakout_above_buffer_pct"] = scan_params["prebreakout_above_buffer_pct"]
            params["require_pullback"] = scan_params["require_pullback"]
            params["min_pullback_days"] = scan_params["min_pullback_days"]
            params["max_pullback_days"] = scan_params["max_pullback_days"]
            params["require_breakout_volume"] = scan_params["require_breakout_volume"]
            for name in RuleDefaults.model_fields:
                params[name] = scan_params[name]
            for name in (
                "provider", "signal_offset", "min_days_since_breakout",
                "max_days_since_breakout", "require_breakout", "breakout_vol_mult",
                "strategy_mode", "flag_direction",
            ):
                if name in scan_params:
                    params[name] = scan_params[name]
        run_id = backtest_manager.submit(params)
        return backtest_manager.get(run_id) or {"run_id": run_id, "status": "running"}

    @app.get("/api/backtests/active")
    def active_backtest() -> dict[str, Any] | None:
        return backtest_manager.active()

    @app.get("/api/backtests/latest")
    def latest_backtest() -> dict[str, Any]:
        return db.latest_backtest() or {"status": "empty", "summary": {}, "results": []}

    @app.get("/api/backtests/{run_id}/progress")
    def backtest_progress(run_id: int) -> dict[str, Any]:
        payload = backtest_manager.get(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="backtest not found")
        return payload

    return app


app = create_app()
