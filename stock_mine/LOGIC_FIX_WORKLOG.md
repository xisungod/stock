# Logic Fixes, 2026-11-27

## Status

- [x] Archive the pre-fix source in `reports/logic_fixes_20261127/baseline` (backend, frontend source, scripts, previous `dist`).
- [x] Signal bar: `signal_offset=0` now means the latest settled session; unsettled intraday bars are dropped
      using the verified `expected_bar_date`. New scans default to 0, the same bar backtests replay.
- [x] Pullback volume: a "shrinking" pullback must also be at or below the pre-breakout 10-day average
      (`pullback_max_normal_volume_ratio`, default 1.0), not only below the breakout-day spike.
- [x] Give-back is not a pullback: if the post-breakout high ran more than 10% above the line
      (`max_post_breakout_runup_pct`), a return to the line no longer triggers a buy alert.
- [x] Limit-up breakouts are flagged (`breakout_limit_up`, risk text); board- and date-aware limits.
- [x] Evidence age: buy alerts need enough independent confirmations within 750 sessions before the breakout
      (`max_event_age_bars`); older forward-adjusted evidence is shown but cannot upgrade an alert.
- [x] Ranking: industry snapshots no longer change the score (`score_type: structure`); ties sort by distance to line.
- [x] Market regime: CSI 300 (`sh000300`) point-in-time regime on every signal; `market_filter` = risk
      (default: annotate only) / downgrade (buy alert becomes watch; not validated) / off. Backtests apply
      the same rule per signal day; missing index data means "not evaluated, not downgraded" in both scans and
      backtests (`market_regime_available: false`), and a direct `evaluate_bars` call refuses downgrade without data.
- [x] Research scripts build their configuration with `production_strategy_config()`; labels are derived from
      the effective config (the old "突破量比1.5" label described a filter that was never enabled).
- [x] Frontend: settled-session default, market selector, structure-score labels, new evidence facts; `dist` rebuilt.

## Constraints kept

- Saved batches: every new rule is absent in old parameters and therefore disabled (`None` / `off`);
  `signal_offset>=1` keeps its literal "drop the last N rows" meaning. Old frozen protocols reload unchanged.
- New thresholds were chosen from the rule's meaning before any evaluation, not tuned on returns.
- `market_filter`: new scans/backtests default to `risk` (annotate only); saved batches keep `off`. It was briefly
  `downgrade` on the invalid validation numbers and was reverted at the user's request (see the end of this file).

## Verification

- Baseline before changes: 522 backend and 145 frontend tests passed.
- After changes: 550 backend tests (28 new in `tests/test_logic_fixes_20261127.py`, including the real
  300016 北陆药业 2026-08-21～28 bars) and 146 frontend tests passed; production build passed.
- Real-data validation: `scripts/validate_logic_fixes.py`, report in `reports/logic_fixes_20261127/validation/`.

## Real-data validation result (591/600 hash-sampled stocks, 2019-01 ～ 2026-10, as of 2026-11-26)

**INVALID — computed with the index bug described under "Correction". Superseded below.**

Pre-declared pass rule: T+5 excess over the same-day equal-weight sample benchmark > 0 with a monthly-block
bootstrap 95% lower bound > 0 in both development (2019–2023) and holdout (2024–2026). Only one arm passed:

| Arm | Dev T+5 excess (95% CI) | Holdout T+5 excess (95% CI) | Verdict |
|---|---|---|---|
| Pre-fix rules | +0.21% (−0.11～0.53) | +0.06% (−0.14～0.26) | fail |
| Fixed rules | +0.46% (+0.02～1.03) | +0.45% (−0.06～0.92) | fail (holdout) |
| Fixed rules + skip weak market | +0.84% (+0.11～1.91) | +0.76% (+0.14～1.31) | pass |

Robustness of the passing arm: positive in 7 of 8 calendar years (2020: −0.03%); 66% of months positive;
median excess −0.34% (the mean comes from the right tail — dropping the best 1% of signals leaves +0.09%);
still positive without the top 10 contributing stocks (+0.23%) and at 60/80 bps cost (+0.49%/+0.29% net).
Limit-up breakouts inside the passing arm: −0.05% over 80 signals (hypothesis only; not acted upon).

Not established: forward performance. The sample is today's listed, non-ST companies (survivorship bias),
the holdout period was already seen by earlier research, six arms were compared, and exits ignore limit-down
locks. Per user decision, `market_filter=downgrade` is now the default; forward tracking is still required.

- After the default change: 551 backend and 146 frontend tests passed; production build passed.

## Target/stop race (+10% before −3%, `scripts/target_stop_backtest.py`, `reports/logic_fixes_20261127/target_stop/`)

**INVALID — computed with the index bug described under "Correction". Superseded below.**

Current rules, 20-session window: 658 of 2631 decided trades (25.0%) reached +10% first, 1804 (68.6%) hit −3% first,
169 (6.4%) neither; +0.41% net per trade. Buying every stock on the same non-weak market days gives 24.6% / +0.35%:
the difference (+0.4 pp, z≈0.5) is not significant. With an entry-based −3% stop the stock signal adds almost
nothing beyond the market filter; most trades are stopped within days (A-share daily volatility ≈ 3%).

## Correction: analysis-script index bug (found 2026-11-27, during the success/failure feature analysis)

`evaluate_bars` normalizes bars (drops non-positive forward-adjusted prices etc.) and reports `signal_index` in
that normalized frame, but `validate_logic_fixes.py`, `target_stop_backtest.py` and the first feature script
indexed the raw bars with it. 116 of the 600 sampled stocks lose rows to normalization (median 828 early rows),
so their signal returns were taken from the wrong — much earlier — dates. Every number in the two sections
above ("Real-data validation result" and "Target/stop race") is therefore invalid; the affected reports are kept
as `reports/logic_fixes_20261127/*_buggy_index/` for audit only. All three scripts now load bars through
`load_settled_bars()` (same normalization as the engine) and assert that each index maps to its signal date.
Production code and the in-app backtest were not affected (they index and price the same normalized frame).
The `market_filter=downgrade` default was decided on the invalid numbers and is re-checked below.

## Corrected results (585/600 stocks after normalization, same frozen rules, as of 2026-11-26)

T+5 excess over the same-day equal-weight benchmark (monthly-block 95% CI):

| Arm | Development 2019–2023 | Holdout 2024–2026 | Verdict |
|---|---|---|---|
| Pre-fix rules | +0.12% (−0.07～0.30) | −0.02% (−0.24～0.18) | fail |
| Fixed rules | +0.17% (−0.06～0.41) | +0.22% (−0.18～0.56) | fail |
| Fixed rules + skip weak market | +0.37% (+0.03～0.72) | +0.40% (−0.05～0.79) | fail (holdout lower bound < 0) |

No arm passes the pre-declared rule. The best arm is positive in both periods but not established.

Target/stop race, current rules, 20 sessions: 729 of 3014 decided trades (24.2%) reached +10% first,
2085 (69.2%) hit −3% first, 200 (6.6%) neither; +0.32% net per trade. Random buys on the same non-weak days:
24.6% / +0.35%. The stock signal adds nothing under an entry-based −3% stop.

Success vs failure features (`reports/logic_fixes_20261127/success_features/`): none of 28 signal-day features
passes the stability rule (both periods on the same side, |z| ≥ 2.6). The largest differences — CSI 300 distance
to MA20 (AUC 0.536) and signal-day volume ratio (AUC 0.532) — are equally present for random buys (control AUC
0.538 / 0.521), i.e. generic, not specific to this pattern. STAR/ChiNext signals succeed more often than main-board
ones (32.6% / 29.8% vs 24.3%; controls 25.5% / 27.6% / 25.7%) — post-hoc and small samples, a hypothesis only.

The `market_filter=downgrade` default no longer had the evidence it was adopted on. At the user's request it was
reverted to `risk` (annotate only): `RuleDefaults.market_filter`, the frontend form default and payload fallback,
and the selector labels (downgrade is now marked 未经验证). `downgrade` remains available as an explicit option.
After the revert: 551 backend and 146 frontend tests passed; production build passed (`dist/assets/index-SNI2nrVE.js`).

## Line-break exit vs matched-stop random entries (`scripts/line_stop_backtest.py`, `reports/logic_fixes_20261127/line_stop/`)

Protocol fixed before running (see the script docstring): next-open entry; exit at the next non-locked open after a
close below line × (1 − 1.5%) (the engine's own breakout-failure level), else at the close of day N (primary N=60).
Control: for each signal, 20 random other sample stocks bought on the same day with the identical stop distance and
exit rule; paired difference = signal return − mean of its controls. Pass rule: primary group (current default,
fixed rules, market annotate-only), N=60, paired difference > 0 with monthly-block 95% lower bound > 0 in both
development (2019-01～2023-09, 60-session purge) and holdout (2024-01～). 580 stocks, as of 2026-11-26.

Result: **fail**. Current default, N=60: development +0.28% (−0.35～0.92) over 3095 trades, holdout +0.42%
(−0.31～1.09) over 2070. The trade profile is generic trend-following: ~15% winners averaging +25～28%, ~85% stopped
within a median 7 days averaging −4.2～4.5% (median initial risk 2.4%; close-confirmed exits slip ~2 pp past the
stop), profit factor 1.02 / 1.18 — and random entries with the same stop distance do about as well (−0.20% / +0.22%).
The paired difference of the fixed rules is positive in both periods at all of N=20/60/120 (+0.28～0.50%), never
significant; the pre-fix rules are at or below random (holdout N=60 −0.38%, CI −0.72～−0.02).
