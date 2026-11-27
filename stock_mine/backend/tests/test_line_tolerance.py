"""同一条精准线的价格容差：默认 2%，收到 1% 后相差更大的点不能画在同一条线上。"""
from app.services.price_bands import bounded_band


def test_two_percent_tolerance_keeps_prices_two_and_a_half_percent_apart():
    assert bounded_band([10.0, 10.25], 0.02) is not None


def test_one_percent_tolerance_splits_prices_two_and_a_half_percent_apart():
    assert bounded_band([10.0, 10.25], 0.01) is None
    assert bounded_band([10.0, 10.15], 0.01) is not None
