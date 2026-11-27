# Stock Scan Progress Design

> **⚠️ 已废弃（2026-08-07）。本文档描述的是未实施的旧方案，与当前代码不符，仅作历史记录保留。**
>
> 现行方案见 `docs/superpowers/specs/2026-08-06-stock-mine-scan-ux-design.md`（仓库根目录下的 `docs/`，不是本目录）。
>
> 主要差异：
>
> | 本文档（旧方案） | 当前实现 |
> |---|---|
> | `POST /api/scans/start` 返回 202 | 复用 `POST /api/scans/run`，返回 200 |
> | `GET /api/scans/{id}` | `GET /api/scans/{id}/progress` |
> | 并发提交返回 **409** | 返回正在运行的 `scan_id` |
> | 进度**持久化**到 `scan_runs`，新增 `target_count` 列 | 进度只在**进程内内存**（`ScanJobManager`），不加列 |
> | 启动时把僵尸 running 记录标失败 | 未实现（后端重启会丢运行中任务的进度，属刻意取舍） |
>
> 唯一被当前方案采纳的概念是 **`target_count`**（抽样后的实际扫描数量）——它作为进度分母，
> 避免了「limit=80 却显示 80/4997」的问题。当前实现把它放在 `Scanner.last_scan_stats` 里而非数据库列。

## Goal

Clicking "开始筛选" must show real scan progress rather than an indefinite animation. The page must report how many stocks have been processed, the total target count, the percentage, successful analyses, and failures. Reloading the page during a scan must reconnect to the active scan.

## Architecture

The existing synchronous `POST /api/scans/run` endpoint remains available for compatibility. The frontend uses two new endpoints:

- `POST /api/scans/start`: creates a scan record, starts one background worker, and immediately returns the running scan record with HTTP 202.
- `GET /api/scans/{scan_id}`: returns the current scan state and includes results after completion.

The backend permits only one scan worker at a time because `Scanner` keeps mutable provider and statistics state. A concurrent start request returns HTTP 409 with a clear message. A worker always releases the lock after success or failure.

## Progress Data

`scan_runs` gains a `target_count` integer column. Existing counters are updated while the scan runs:

- `universe_count`: eligible A-share universe size before sampling.
- `target_count`: stocks selected for this run after applying the limit.
- `attempted_count`: stocks fully processed so far.
- `analyzed_count`: stocks with usable bars and a completed analysis.
- `failed_count`: stocks whose data fetch or analysis failed.

Before the universe is available, the UI displays "正在获取股票列表". Once `target_count` is known, percentage is calculated as `attempted_count / target_count`, clamped to 0-100. Completion always displays 100%, including an empty universe.

## Scanner Contract

`Scanner.run_scan` accepts an optional progress callback. It calls the callback:

1. After the universe is filtered and the target sample is selected.
2. After every stock finishes, whether it succeeds or fails.

The callback receives a copy of the current statistics, including `target_count`. Existing callers that omit the callback keep the current behavior.

## Frontend Flow

1. Start the scan and store the returned scan id.
2. Poll the status endpoint every second.
3. While the universe is loading, show an indeterminate bar and the loading message.
4. While stocks are being processed, show a determinate bar, percentage, `processed/target`, successful analyses, and failures.
5. On completion, replace the displayed results and retain the completion summary.
6. On failure, stop polling and show the backend error.
7. On initial page load, if the latest scan is still running, resume polling it automatically.

Controls remain disabled while a scan is active, preventing duplicate starts from the same page.

## Failure Handling

- Polling tolerates two consecutive transient request failures and then surfaces the error.
- A backend scan exception marks the database record as failed with `finished_at` and an error message.
- A stale running scan left by a backend restart is marked failed during application startup so the frontend cannot poll forever.
- The previous completed results stay visible until the new scan completes.

## Testing

- Scanner test verifies initial and per-stock progress callbacks, including failures.
- Database/API tests verify progress persistence, immediate background start, running status, completion, and concurrent-start rejection.
- Frontend unit tests verify percentage calculation and status copy.
- Browser verification covers loading-universe, determinate progress, completion, reload recovery, and console errors.

## Out of Scope

- Scheduling and automatic daily execution.
- Cancelling an active scan.
- Multi-process distributed workers. A durable queue can replace the local worker when the application is deployed with multiple backend processes.
