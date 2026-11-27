from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.related_board import correlation, rank_related_boards


def _series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


def _walk(seed: int, n: int = 120, drift: float = 0.0) -> pd.Series:
    rng = np.random.RandomState(seed)
    return _series(list(100 * np.exp(np.cumsum(rng.randn(n) * 0.01 + drift))))


def test_identical_movement_correlates_perfectly() -> None:
    stock = _walk(1)
    assert correlation(stock, stock * 3.7) == 1.0, "同一条走势缩放后仍应完全相关"


def test_mirrored_movement_correlates_negatively() -> None:
    stock = _walk(2)
    mirror = _series(list(200 - stock.to_numpy()), start="2026-01-01")
    assert correlation(stock, mirror) < -0.9


def test_correlation_uses_daily_returns_not_price_levels() -> None:
    """两条都在上涨但日间涨跌毫无关系的序列，不该被判为高度相关。

    直接对收盘价求相关是伪相关：任意两条上升趋势的价格序列都接近 1，
    那样选出来的"最相关板块"只是"同样在涨的板块"，没有信息量。
    """
    a = _walk(3, drift=0.004)
    b = _walk(4, drift=0.004)

    level_corr = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])
    return_corr = correlation(a, b)

    assert level_corr > 0.8, f"构造前提：价格水平确实高度相关（{level_corr:.2f}）"
    assert abs(return_corr) < 0.3, f"日收益率相关应接近 0，实得 {return_corr:.2f}"


def test_only_overlapping_trading_days_are_compared() -> None:
    """个股停牌或板块指数起始日不同，只能用交集，不能用位置对齐。"""
    stock = _walk(5, n=120)
    board = stock.iloc[30:].copy() * 2.0
    assert correlation(stock, board) == 1.0


def test_too_little_overlap_returns_none_instead_of_a_number() -> None:
    stock = _walk(6, n=120)
    board = stock.iloc[-10:].copy()
    assert correlation(stock, board, min_overlap=60) is None


def test_flat_board_index_is_skipped_not_ranked_as_nan() -> None:
    """常数序列没有波动，相关系数是 NaN；它不能混进排序结果。"""
    stock = _walk(7)
    flat = _series([50.0] * len(stock))
    assert correlation(stock, flat) is None


def test_ranking_is_by_correlation_descending_and_respects_top_n() -> None:
    stock = _walk(8)
    boards = {
        "强相关": stock * 1.5,
        "弱相关": _walk(9),
        "反相关": _series(list(300 - stock.to_numpy())),
        "无波动": _series([7.0] * len(stock)),
    }
    ranked = rank_related_boards(stock, boards, top_n=2)

    assert [item["name"] for item in ranked] == ["强相关", "弱相关"], f"实得 {ranked}"
    assert ranked[0]["correlation"] > ranked[1]["correlation"]
    assert all(item["overlap_days"] > 0 for item in ranked)


# ---- 板块指数仓库 ----

class _StubClient:
    """假的 akshare：记录调用次数，不触网。"""

    def __init__(self, broken: set[str] | None = None):
        self.index_calls: list[str] = []
        self.name_calls = 0
        self.broken = broken or set()

    def stock_board_industry_name_ths(self):
        self.name_calls += 1
        return pd.DataFrame({"name": ["银行", "半导体"], "code": ["881", "882"]})

    def stock_board_concept_name_ths(self):
        self.name_calls += 1
        return pd.DataFrame({"name": ["数字货币"], "code": ["883"]})

    def _bars(self, symbol: str):
        self.index_calls.append(symbol)
        if symbol in self.broken:
            raise RuntimeError(f"{symbol} 接口挂了")
        dates = pd.bdate_range("2026-01-01", periods=90)
        return pd.DataFrame({
            "日期": dates.strftime("%Y-%m-%d"),
            "开盘价": np.arange(90, dtype=float) + 10,
            "最高价": np.arange(90, dtype=float) + 11,
            "最低价": np.arange(90, dtype=float) + 9,
            "收盘价": np.arange(90, dtype=float) + 10,
            "成交量": [1000] * 90,
        })

    def stock_board_industry_index_ths(self, symbol, **kwargs):
        return self._bars(symbol)

    def stock_board_concept_index_ths(self, symbol, **kwargs):
        return self._bars(symbol)


def test_repository_fetches_industries_and_concepts_once_then_serves_from_cache(tmp_path) -> None:
    from app.services.related_board import BoardIndexRepository

    client = _StubClient()
    repo = BoardIndexRepository(cache_dir=tmp_path, client=client)

    first = repo.closes()
    assert set(first) == {"银行", "半导体", "数字货币"}
    assert len(client.index_calls) == 3
    assert repo.kinds()["数字货币"] == "concept"

    # 换一个实例，模拟进程重启：应当直接读盘，不再触网
    reopened = BoardIndexRepository(cache_dir=tmp_path, client=_StubClient())
    assert set(reopened.closes()) == {"银行", "半导体", "数字货币"}
    assert reopened.client.index_calls == [], "命中磁盘缓存时不应再调接口"


def test_one_broken_board_does_not_sink_the_whole_universe(tmp_path) -> None:
    """465 个板块里挂掉一两个是常态，不能因此整个功能不可用。"""
    from app.services.related_board import BoardIndexRepository

    repo = BoardIndexRepository(cache_dir=tmp_path, client=_StubClient(broken={"半导体"}))
    closes = repo.closes()

    assert set(closes) == {"银行", "数字货币"}


def test_closes_are_a_dated_series_so_suspended_days_align(tmp_path) -> None:
    from app.services.related_board import BoardIndexRepository

    closes = BoardIndexRepository(cache_dir=tmp_path, client=_StubClient()).closes()["银行"]

    assert isinstance(closes.index, pd.DatetimeIndex), "必须按日期索引，否则无法与个股取交集"
    assert len(closes) == 90


def test_chart_bars_are_normalized_ohlcv_for_the_chart_builder(tmp_path) -> None:
    """选中的板块要能直接喂给 build_full_chart 画 K 线。"""
    from app.services.detail import build_full_chart
    from app.services.related_board import BoardIndexRepository

    repo = BoardIndexRepository(cache_dir=tmp_path, client=_StubClient())
    bars = repo.chart_bars("银行", "industry")

    assert list(bars.columns) == ["trade_date", "open", "high", "low", "close", "volume"]
    assert len(build_full_chart(bars)["bars"]) == 90


def test_cold_cache_returns_not_ready_immediately_instead_of_blocking(tmp_path) -> None:
    """首次构建要拉 465 个板块指数（实测 4.5 分钟）。

    HTTP 请求不能挂在那里等——浏览器早就超时了。第一次问就先答"还没好"，
    同时在后台开始构建。
    """
    from app.services.related_board import BoardIndexRepository

    repo = BoardIndexRepository(cache_dir=tmp_path, client=_StubClient())
    assert repo.ensure_ready(background=True) is False
    repo.wait_for_build(timeout=5)
    assert repo.ensure_ready(background=True) is True


def test_concurrent_requests_do_not_start_a_second_build(tmp_path) -> None:
    from app.services.related_board import BoardIndexRepository

    client = _StubClient()
    repo = BoardIndexRepository(cache_dir=tmp_path, client=client)
    for _ in range(5):
        repo.ensure_ready(background=True)
    repo.wait_for_build(timeout=5)

    assert len(client.index_calls) == 3, f"三个板块只应各取一次，实得 {client.index_calls}"


def test_building_flag_reports_progress_to_the_caller(tmp_path) -> None:
    from app.services.related_board import BoardIndexRepository

    repo = BoardIndexRepository(cache_dir=tmp_path, client=_StubClient())
    repo.ensure_ready(background=True)
    repo.wait_for_build(timeout=5)

    assert repo.is_building is False


def test_suspended_days_do_not_misalign_the_returns() -> None:
    """个股停牌时，必须先对齐共同交易日再算收益。

    先各自 pct_change 再取日期交集的话，复牌当日个股的收益跨了整个停牌区间，
    而板块那天只跨一天——两个不同跨度的收益被直接相比，相关系数是错的。
    """
    dates = pd.bdate_range("2026-01-01", periods=80)
    board = pd.Series(100 * np.linspace(1.0, 1.5, 80), index=dates)

    # 个股走势与板块完全一致，只是中间停牌 10 天（那些交易日缺失）
    stock_full = board * 0.5
    keep = [i for i in range(80) if not (30 <= i < 40)]
    stock = stock_full.iloc[keep]

    value = correlation(stock, board, min_overlap=30)

    assert value is not None
    assert value > 0.99, f"停牌只是缺了几天，走势本身完全相同，实得 {value}"


def test_returns_are_computed_after_alignment_not_before() -> None:
    """同一份数据，先对齐再算收益 与 先算收益再对齐，结果必须以前者为准。"""
    dates = pd.bdate_range("2026-01-01", periods=60)
    rng = np.random.RandomState(5)
    board = pd.Series(100 * np.exp(np.cumsum(rng.randn(60) * 0.02)), index=dates)
    stock = board.iloc[[i for i in range(60) if i not in (20, 21, 22)]] * 2.0

    assert correlation(stock, board, min_overlap=20) > 0.99


def test_a_totally_failed_build_is_not_cached_as_done(tmp_path) -> None:
    """板块接口全挂时不能把空结果落盘。

    否则 ensure_ready() 会认为"已经构建完成"，24 小时内都不再重试，
    最相关板块这块功能会一直空着。
    """
    from app.services.related_board import BoardIndexRepository

    class AllBroken(_StubClient):
        def _bars(self, symbol):
            raise RuntimeError("接口全挂")

    repo = BoardIndexRepository(cache_dir=tmp_path, client=AllBroken())
    assert repo.closes() == {}
    assert not repo.cache_path.exists(), "全失败不该落盘"

    # 换成正常的客户端应当立刻重试成功，而不是被空缓存挡住
    healthy = BoardIndexRepository(cache_dir=tmp_path, client=_StubClient())
    assert set(healthy.closes()) == {"银行", "半导体", "数字货币"}


def test_a_mostly_failed_build_is_also_rejected(tmp_path) -> None:
    """只成功一两个也算失败——那点样本选不出有意义的相关板块。"""
    from app.services.related_board import BoardIndexRepository

    repo = BoardIndexRepository(cache_dir=tmp_path, client=_StubClient(broken={"银行", "半导体"}))
    repo.closes()
    assert not repo.cache_path.exists(), "成功比例过低不该当作构建完成"


def test_the_board_universe_is_fetched_only_once_per_build(tmp_path) -> None:
    """名单请求一次就够。请求两次不仅多一趟网络，第二次若失败会返回 0，
    让成功率门槛失效、把残缺结果当成定稿缓存 24 小时。"""
    from app.services.related_board import BoardIndexRepository

    client = _StubClient()
    BoardIndexRepository(cache_dir=tmp_path, client=client).closes()

    assert client.name_calls == 2, f"行业+概念各一次，共 2 次；实得 {client.name_calls}"


def test_a_missing_whole_board_category_is_not_cached(tmp_path) -> None:
    """概念或行业名单整个失败时，不能拿剩下一个类别冒充完整板块池。"""
    from app.services.related_board import BoardIndexRepository

    class MissingConceptNames(_StubClient):
        def stock_board_concept_name_ths(self):
            self.name_calls += 1
            raise ConnectionError("概念名单接口失败")

    repo = BoardIndexRepository(cache_dir=tmp_path, client=MissingConceptNames())

    assert repo.closes() == {}
    assert not repo.cache_path.exists()
