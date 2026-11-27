from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import pandas as pd

from app.services.akshare_guard import call_akshare, get_akshare
from app.services.industry_identity import cached_company_industry, industry_family


INDUSTRY_WORKERS = 4
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "industry_cache.json"


@dataclass(frozen=True)
class IndustryContext:
    name: str
    score: float
    risk_level: str
    adjustment: float
    change_pct: float | None
    up_count: int
    down_count: int
    source: str
    message: str
    updated_at: str | None = None
    stale: bool = False
    validation_status: str = "verified"
    source_industry: str | None = None
    company_industry: str | None = None


def score_industry(
    change_pct: float | None,
    up_count: int,
    down_count: int,
    net_flow: float | None = None,
) -> tuple[float, str, float, str]:
    """将行业涨幅和涨跌家数压缩为可解释、有限幅度的风险修正。"""
    score = 50.0
    if change_pct is not None:
        if change_pct >= 2:
            score += 18
        elif change_pct >= 0.5:
            score += 10
        elif change_pct > -0.5:
            score += 3
        elif change_pct >= -2:
            score -= 10
        else:
            score -= 22

    breadth_total = max(0, up_count) + max(0, down_count)
    breadth = up_count / breadth_total if breadth_total else None
    if breadth is not None:
        if breadth >= 0.7:
            score += 18
        elif breadth >= 0.55:
            score += 8
        elif breadth <= 0.3:
            score -= 20
        elif breadth <= 0.45:
            score -= 8
    if net_flow is not None:
        score += 6 if net_flow > 0 else -6 if net_flow < 0 else 0

    score = round(max(0.0, min(100.0, score)), 1)
    adjustment = round(max(-15.0, min(10.0, (score - 50.0) * 0.4)), 1)
    if score >= 70:
        risk_level, label = "low", "行业偏强"
    elif score >= 45:
        risk_level, label = "medium", "行业中性"
    else:
        risk_level, label = "high", "行业风险偏高"
    breadth_text = f"，上涨 {up_count} / 下跌 {down_count}" if breadth_total else ""
    change_text = f"，当日涨跌 {change_pct:+.2f}%" if change_pct is not None else ""
    return score, risk_level, adjustment, f"{label}{change_text}{breadth_text}"


def apply_industry_context(result: Any, context: IndustryContext | None) -> Any:
    result.structure_score = float(result.score if result.structure_score is None else result.structure_score)
    result.score = result.structure_score
    result.industry_adjustment = 0.0
    if context is None:
        result.industry_risk_level = "uncovered"
        result.industry_risk_message = "行业数据暂未覆盖，综合分未作行业修正"
        return result

    result.industry = context.name
    result.industry_data_as_of = context.updated_at
    result.industry_stale = context.stale
    result.industry_source = context.source
    if context.validation_status != "verified":
        result.industry_score = None
        result.industry_risk_level = "uncovered"
        result.industry_risk_message = context.message
        result.risks.append(context.message)
        return result
    if context.stale:
        result.industry_score = None
        result.industry_risk_level = "uncovered"
        result.industry_risk_message = f"行业快照已过期（{context.updated_at or '时间未知'}），未计入综合分"
        result.risks.append(result.industry_risk_message)
        return result
    result.industry_score = context.score
    # 行业分来自当日快照：无法回填历史、从未被回测验证，且与信号 K 线不是同一时点。
    # 只作环境提示，不再加减排序分（score 恒等于结构分）。
    result.industry_adjustment = 0.0
    result.industry_risk_level = context.risk_level
    result.industry_risk_message = context.message
    result.industry_source = context.source
    if context.risk_level == "high":
        result.risks.append(f"行业风险：{context.message}")
    elif context.risk_level == "low":
        result.reasons.append(f"行业环境：{context.message}")
    return result


def aggregate_industries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        groups.setdefault(item.get("industry") or "未覆盖", []).append(item)
    rows: list[dict[str, Any]] = []
    for name, items in groups.items():
        covered = [item for item in items if item.get("industry_score") is not None]
        scores = [float(item.get("score", 0)) for item in items]
        rows.append(
            {
                "industry": name,
                "selected_count": len(items),
                "buy_alert_count": sum(item.get("category") == "buy_alert" for item in items),
                "watch_count": sum(item.get("category") == "watch" for item in items),
                "average_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "industry_score": covered[0].get("industry_score") if covered else None,
                "industry_adjustment": covered[0].get("industry_adjustment") if covered else 0,
                "risk_level": covered[0].get("industry_risk_level") if covered else "uncovered",
                "risk_message": covered[0].get("industry_risk_message") if covered else "行业数据暂未覆盖",
                "source": covered[0].get("industry_source") if covered else None,
            }
        )
    return sorted(rows, key=lambda row: (-row["selected_count"], -row["average_score"], row["industry"]))


class IndustryService:
    def __init__(
        self, cache_path: str | Path = DEFAULT_CACHE_PATH, client: Any | None = None,
        ttl_hours: int = 8, now: Callable[[], datetime] = datetime.now,
        profile_lookup: Callable[[str], str | None] | None = None,
    ):
        self.cache_path = Path(cache_path)
        self.client = client
        self.ttl = timedelta(hours=ttl_hours)
        self.now = now
        self.profile_lookup = profile_lookup or (lambda code: cached_company_industry(code, now=self.now()))
        self._updated_at: datetime | None = None
        self._contexts: dict[str, IndustryContext] | None = None
        self._lock = Lock()

    def prepare(self, force: bool = False) -> dict[str, IndustryContext]:
        with self._lock:
            if self._contexts and self._is_current() and not force:
                self._contexts = self._validate_contexts(self._contexts)
                return self._contexts
            cached = self._load_cache() if not force else None
            if cached is not None:
                self._contexts = self._validate_contexts(cached)
                return self._contexts
            try:
                contexts = self._fetch()
                if not contexts:
                    raise ValueError("行业接口返回空数据")
            except Exception:
                previous = self._contexts or self._load_cache(ignore_age=True) or {}
                self._contexts = self._validate_contexts({code: replace(value, stale=True) for code, value in previous.items()})
                return self._contexts
            self._updated_at = self.now()
            stamp = self._updated_at.isoformat(timespec="seconds")
            self._contexts = {
                code: replace(value, updated_at=stamp, stale=False) for code, value in contexts.items()
            }
            self._contexts = self._validate_contexts(self._contexts)
            self._save_cache(self._contexts)
            return self._contexts

    def get_context(self, stock_code: str) -> IndustryContext | None:
        # 单股详情优先即时返回已有行业映射。行业全量缓存需要约 49 次网络请求，
        # 过期刷新应由全市场扫描触发，不能把打开详情页阻塞数分钟。
        with self._lock:
            if self._contexts is None:
                self._contexts = self._load_cache(ignore_age=True) or {}
            context = self._contexts.get(str(stock_code).zfill(6))
            if context:
                context = self._validate_context(str(stock_code).zfill(6), context)
                return replace(context, stale=context.stale or not self._is_current())
            return None

    def _validate_contexts(self, contexts: dict[str, IndustryContext]) -> dict[str, IndustryContext]:
        return {code: self._validate_context(code, context) for code, context in contexts.items()}

    def _validate_context(self, code: str, context: IndustryContext) -> IndustryContext:
        if context.source != "新浪行业":
            return context
        raw = context.source_industry or context.name
        profile = self.profile_lookup(code)
        expected, actual = industry_family(profile), industry_family(raw)
        if context.validation_status == "ambiguous":
            return context
        if expected is not None and expected == actual:
            _, _, _, message = score_industry(context.change_pct, context.up_count, context.down_count)
            return replace(context, name=raw, source_industry=raw, company_industry=profile,
                           validation_status="verified", message=message)
        status = "conflict" if expected is not None and actual is not None else "unverified"
        message = (
            f"行业分类冲突：公司资料为{profile}，行情来源为{raw}；未计入行业分"
            if status == "conflict" else f"行业归属尚未核验（行情来源：{raw}）；未计入行业分，不代表低风险"
        )
        return replace(context, name=profile or "未核验", source_industry=raw, company_industry=profile,
                       validation_status=status, message=message)

    def _is_current(self) -> bool:
        if self._updated_at is None:
            return False
        age = self.now() - self._updated_at
        return timedelta(0) <= age < self.ttl and self.now().date() == self._updated_at.date()

    def _ak(self) -> Any:
        if self.client is None:
            self.client = get_akshare()
        return self.client

    def _fetch(self) -> dict[str, IndustryContext]:
        client = self._ak()
        spot = call_akshare(lambda: client.stock_sector_spot(indicator="新浪行业"))
        if spot is None or spot.empty:
            return {}
        rows = spot.to_dict("records")

        def load_sector(row: dict[str, Any]) -> tuple[IndustryContext, list[str]]:
            label = str(row.get("label", ""))
            name = str(row.get("板块", label))
            change = _number(row.get("涨跌幅"))
            details = call_akshare(lambda: client.stock_sector_detail(sector=label))
            changes = pd.to_numeric(details.get("changepercent"), errors="coerce") if details is not None else pd.Series(dtype=float)
            up_count = int((changes > 0).sum())
            down_count = int((changes < 0).sum())
            score, risk, adjustment, message = score_industry(change, up_count, down_count)
            context = IndustryContext(name, score, risk, adjustment, change, up_count, down_count, "新浪行业", message)
            codes = [] if details is None or details.empty else details["code"].astype(str).str.extract(r"(\d{6})", expand=False).dropna().tolist()
            return context, codes

        candidates: dict[str, list[IndustryContext]] = {}
        with ThreadPoolExecutor(max_workers=INDUSTRY_WORKERS) as executor:
            for context, codes in executor.map(load_sector, rows):
                for code in codes:
                    candidates.setdefault(code, []).append(context)
        contexts = {}
        for code, choices in candidates.items():
            unique = {item.name: item for item in choices}
            if len(unique) == 1:
                contexts[code] = choices[0]
                continue
            profile = self.profile_lookup(code)
            family = industry_family(profile)
            matching = [item for item in unique.values() if family is not None and industry_family(item.name) == family]
            if len(matching) == 1:
                contexts[code] = matching[0]
            else:
                contexts[code] = replace(choices[0], name=profile or "未核验", validation_status="ambiguous",
                    source_industry=" / ".join(unique), company_industry=profile,
                    message=f"同一股票存在多个行业归属（{'、'.join(unique)}），未计入行业分")
        return contexts

    def _load_cache(self, ignore_age: bool = False) -> dict[str, IndustryContext] | None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            updated_at = datetime.fromisoformat(payload["updated_at"])
            age = self.now() - updated_at
            current = timedelta(0) <= age < self.ttl and updated_at.date() == self.now().date()
            if not ignore_age and not current:
                return None
            contexts = {
                code: replace(IndustryContext(**value), updated_at=payload["updated_at"], stale=not current)
                for code, value in payload.get("stocks", {}).items()
            }
            self._updated_at = updated_at
            return contexts
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _save_cache(self, contexts: dict[str, IndustryContext]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": self._updated_at.isoformat(timespec="seconds"),
                "stocks": {code: asdict(context) for code, context in contexts.items()},
            }
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None
