from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from compare_signal_filters import accepts, comparison_ci, filter_and_dedupe


def test_filters_run_before_dedupe_so_later_qualifying_days_are_not_lost():
    rows = [
        {"stock_code": "000001", "signal_index": 100, "macd_hist_rising": False},
        {"stock_code": "000001", "signal_index": 101, "macd_hist_rising": True},
        {"stock_code": "000001", "signal_index": 120, "macd_hist_rising": True},
        {"stock_code": "000001", "signal_index": 121, "macd_hist_rising": True},
    ]
    assert [r["signal_index"] for r in filter_and_dedupe(rows, "macd_rising")] == [101, 121]
    assert [r["signal_index"] for r in filter_and_dedupe(rows, "baseline")] == [100, 120]


def test_unknown_volume_is_not_treated_as_low_volume():
    assert accepts({"pullback_vs_prebreakout_volume": None}, "normal_volume") is False
    assert accepts({"pullback_vs_prebreakout_volume": 1.0}, "normal_volume") is True


def test_comparison_resamples_overlapping_groups_together():
    rows = [{"signal_date": f"2026-01-{i:02}", "net_return_5": i - 10} for i in range(1, 30)]
    ci = comparison_ci(rows, rows, iterations=100)
    assert ci["net_mean_delta_95"] == [0, 0]
    assert ci["winrate_delta_pp_95"] == [0, 0]
