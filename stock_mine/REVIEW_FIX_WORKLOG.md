# Review Fixes, 2026-09-10

- [x] Archive the pre-fix source in `reports/review_fixes_20260910/baseline`.
- [x] Add regression cases for trial ownership, delayed quotes, invalid providers, and calendar gaps.
- [x] Make complete platform trials own their validation evidence.
- [x] Verify quote freshness against completed exchange sessions.
- [x] Reject unknown providers instead of returning sample data.
- [x] Share calendar-based execution between app backtests and research reports.
- [x] Run all tests, validate the UI, and rerun the frozen comparison.

Do not change thresholds to improve observed returns. Keep previous reports and
saved scan/watch records. Retain raw confirmed local turns, but do not allow
them to bypass a platform trial's full-extreme validation. Preserve completed
trial intervals to prevent a late local confirmation from bypassing rejection.

The nine initial regression cases failed on the archived source before implementation.

- Added a bounded XSHG calendar dependency, pinned at exchange-calendars 4.13.2.
- Historical reports keep their old model identifiers; new reports share the execution
  ledger, including missing next-session entries and delayed exits.
- The rerun will reuse the previous 27 input snapshots and 24 evaluated stocks. Prior
  v3 signals will be repriced on the same calendar and costs, not compared using mixed ledgers.

## Completed verification

- 488 backend tests passed; frontend expanded to 134 passing tests.
- Frozen rerun completed 24/24 stocks with no failures. Source/input hashes checked;
  282 v4 signals checked for bounded contributions, causal confirmation and unique trial IDs.
- Rerun: `reports/platform_v2_validation/locked_20260910_r3`.
- User-requested layout update: first-view decision/risk summary, full-width chart, collapsed
  parameter evidence, four research tabs, compact empty sector states, keyboard tab navigation.
- Desktop/mobile UI verified; chart-panel bottom gap is 10px, no horizontal page overflow.
- Implementation and evaluation notes: `reports/review_fixes_20260910/验收说明.md`.
