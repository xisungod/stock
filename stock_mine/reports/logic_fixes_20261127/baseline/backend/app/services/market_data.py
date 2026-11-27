from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.akshare_guard import call_akshare, get_akshare
from app.services.detail import build_full_chart


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "market_cache"

SINA_TO_THS_INDUSTRY = {
    "邮政业": "物流",
    "资本市场服务": "证券",
    "货币金融服务": "银行",
    "电力、热力生产和供应业": "电力",
    "家具制造业": "家居用品",
    "玻璃行业": "建筑材料",
    "船舶制造": "军工装备",
    "传媒娱乐": "文化传媒",
    "电力行业": "电力",
    "电器行业": "白色家电",
    "电子器件": "其他电子",
    "电子信息": "软件开发",
    "房地产": "房地产",
    "发电设备": "其他电源设备",
    "飞机制造": "军工装备",
    "纺织行业": "纺织制造",
    "纺织机械": "专用设备",
    "服装鞋类": "服装家纺",
    "公路桥梁": "公路铁路运输",
    "供水供气": "燃气",
    "钢铁行业": "钢铁",
    "环保行业": "环境治理",
    "化工行业": "化学制品",
    "化纤行业": "化学纤维",
    "家电行业": "白色家电",
    "酒店旅游": "旅游及酒店",
    "家具行业": "家居用品",
    "金融行业": "银行",
    "交通运输": "物流",
    "机械行业": "通用设备",
    "建筑建材": "建筑材料",
    "开发区": "房地产",
    "酿酒行业": "白酒",
    "摩托车": "汽车整车",
    "煤炭行业": "煤炭开采加工",
    "农林牧渔": "种植业与林业",
    "农药化肥": "农化制品",
    "汽车制造": "汽车整车",
    "其它行业": "综合",
    "塑料制品": "塑料制品",
    "水泥行业": "建筑材料",
    "食品行业": "食品加工制造",
    "次新股": "综合",
    "生物制药": "生物制品",
    "商业百货": "零售",
    "石油行业": "油气开采及服务",
    "陶瓷行业": "建筑材料",
    "物资外贸": "贸易",
    "医疗器械": "医疗器械",
    "仪器仪表": "自动化设备",
    "印刷包装": "包装印刷",
    "有色金属": "工业金属",
    "综合行业": "综合",
    "造纸行业": "造纸",
}


def resolve_ths_industry(industry_name: str | None) -> str | None:
    if not industry_name or industry_name in {"未核验", "未覆盖"}:
        return None
    name = str(industry_name).strip()
    return SINA_TO_THS_INDUSTRY.get(name, name.removesuffix("行业"))


class MarketDataService:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR, client: Any | None = None):
        self.cache_dir = Path(cache_dir)
        self.client = client

    def get_news(self, stock_code: str, limit: int = 20) -> dict[str, Any]:
        code = str(stock_code).zfill(6)
        cache_key = f"news_{code}"
        cached = self._load(cache_key, timedelta(minutes=30))
        if cached is not None:
            return cached
        try:
            frame = call_akshare(lambda: self._ak().stock_news_em(symbol=code))
            items = _normalize_news(frame, limit)
            payload = {
                "items": items,
                "source": "东方财富资讯（经 AKShare）",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "message": "" if items else "暂未获取到相关新闻",
            }
            self._save(cache_key, payload)
            return payload
        except Exception as exc:
            stale = self._load(cache_key, None)
            if stale is not None:
                stale["message"] = "新闻接口暂不可用，当前展示缓存内容"
                stale["stale"] = True
                return stale
            return {"items": [], "source": None, "updated_at": None, "message": f"新闻暂不可用：{exc}"}

    def get_industry_chart(self, industry_name: str | None, years: int = 10) -> dict[str, Any]:
        ths_name = resolve_ths_industry(industry_name)
        if not ths_name:
            return {"name": None, "mapped_name": None, "chart": {"bars": [], "markers": []}, "message": "未识别所属行业"}
        cache_key = f"industry_{ths_name}_{years}"
        cached = self._load(cache_key, timedelta(hours=8))
        if cached is not None:
            return cached
        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=max(3, years) * 366)).strftime("%Y%m%d")
        try:
            frame = call_akshare(
                lambda: self._ak().stock_board_industry_index_ths(
                    symbol=ths_name,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            bars = _normalize_industry_bars(frame)
            if bars.empty:
                raise ValueError("接口返回空数据")
            payload = {
                "name": industry_name,
                "mapped_name": ths_name,
                "chart": build_full_chart(bars),
                "source": "同花顺行业指数（经 AKShare）",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "message": "" if industry_name == ths_name else f"按同花顺“{ths_name}”行业指数展示",
            }
            self._save(cache_key, payload)
            return payload
        except Exception as exc:
            stale = self._load(cache_key, None)
            if stale is not None:
                stale["message"] = "板块行情接口暂不可用，当前展示缓存内容"
                stale["stale"] = True
                return stale
            return {
                "name": industry_name,
                "mapped_name": ths_name,
                "chart": {"bars": [], "markers": []},
                "source": None,
                "updated_at": None,
                "message": f"所属板块 K 线暂不可用：{exc}",
            }

    def _ak(self) -> Any:
        if self.client is None:
            self.client = get_akshare()
        return self.client

    def _path(self, key: str) -> Path:
        safe = "".join(char for char in key if char.isalnum() or char in {"_", "-"})
        return self.cache_dir / f"{safe}.json"

    def _load(self, key: str, max_age: timedelta | None) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._path(key).read_text(encoding="utf-8"))
            if max_age is not None:
                updated = datetime.fromisoformat(payload["updated_at"])
                if datetime.now() - updated > max_age:
                    return None
            return payload
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _save(self, key: str, payload: dict[str, Any]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def _normalize_news(frame: pd.DataFrame | None, limit: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    work = frame.copy()
    work["发布时间"] = pd.to_datetime(work.get("发布时间"), errors="coerce")
    work = work.sort_values("发布时间", ascending=False).head(max(1, limit))
    items = []
    for _, row in work.iterrows():
        published = row.get("发布时间")
        content = str(row.get("新闻内容") or "").strip()
        items.append(
            {
                "title": str(row.get("新闻标题") or "未命名资讯").strip(),
                "summary": content[:240],
                "published_at": published.strftime("%Y-%m-%d %H:%M") if pd.notna(published) else None,
                "source": str(row.get("文章来源") or "").strip(),
                "url": str(row.get("新闻链接") or "").strip(),
            }
        )
    return items


def _normalize_industry_bars(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    normalized = frame.rename(
        columns={
            "日期": "trade_date",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
        }
    )
    needed = ["trade_date", "open", "high", "low", "close", "volume"]
    if any(column not in normalized for column in needed):
        return pd.DataFrame()
    return normalized[needed].copy()
