# Platform Events V2

## Status

- [x] Inspect current pipeline and archive the baseline source.
- [x] Fix price-band tolerance, incomplete shareholder data, and industry conflicts.
- [x] Implement causal local turns, independent band tests, and immutable line versions.
- [x] Separate candidate, observation, confirmed pullback, and invalidation states.
- [x] Integrate API, legacy batch compatibility, chart evidence, and event-based replay.
- [x] Freeze the implementation and run old/new point and signal comparisons.
- [x] Run predeclared historical holdouts with net return, profit factor, drawdown, and entry feasibility.

## Constraints

- Baseline source: `reports/platform_v2_validation/baseline`.
- Do not overwrite saved scans, historical results, or the watchlist.
- Preserve the original peak strategy as an explicit legacy comparison mode.
- Preserve the existing large-scale ZigZag for trend display.
- Confirmation uses only the current and earlier bars. Candidate extrema may move; confirmed events do not.
- Every price contribution must satisfy the final band's tolerance. An event's complete adverse/favorable extremes remain recorded.
- A breakout uses the line version that existed before its candle; its pullback cannot retroactively provide formation evidence.
- Candidate/observation signals are not buy alerts.
- A failed breakout invalidates that attempt, not the historical price evidence. Another attempt requires a fresh excursion/retest and reruns path filters.
- Validation parameters and partitions are frozen before reading new-model evaluation results.
- Historical holdouts are not presented as entirely unseen data: prior versions have already used this cache. Genuine forward performance requires subsequent market data.

## Verification

- Initial regression: 449 backend tests and 127 frontend tests passed.
- Added explicit band-departure independence and zero-volume signal guards before freezing evaluation rules.
- Added a standalone freeze/run validator: deterministic cross-market sample, three engines, chronological
  holdouts with market-session purging, same-date non-alert controls, fixed fees, cash-constrained daily NAV.
- Initial locked run completed 24/24 stocks. The date audit revealed candidate creation incorrectly required
  a closing-price excursion, missing wick-led local turns. Corrected candidate/departure extrema with two
  synthetic regressions; retained the initial report and froze a separate v3 engineering revision.
- Final run: 24/24 stocks, no failed replays; all 27 input snapshots identical across revisions.
  Validated all 368 retained event signals / 358 frozen versions for uniqueness, tolerance, and confirmation timing.
- Historical detail no longer waits for current network quotes when same-provider cache covers the date.
  API test: 002120 as of 2016-03-23, HTTP 200 in 0.56 seconds.
- Regression after backend changes: 459 backend and 128 frontend tests passed; production build passed.
- Report: `reports/platform_v2_validation/locked_20260910_r2/report.md`;
  concise Chinese acceptance notes: `reports/platform_v2_validation/验收说明.md`.
- Profitability not established. Historical holdouts are not genuinely untouched data; forward validation remains necessary.
