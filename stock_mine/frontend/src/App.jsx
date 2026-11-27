import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BookmarkPlus,
  CheckCircle2,
  ChevronDown,
  Database,
  Eye,
  Flag,
  LineChart,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Star,
  Trash2
} from 'lucide-react';
import { api } from './api.js';
import { sortTags, tagTone } from './statusTags.js';
import { isAllSelected, keepExisting, toggleAll, toggleCode } from './selection.js';
import { POLL_INTERVAL_MS, MAX_POLL_FAILURES, isTerminal, formatProgress, progressPercent } from './scanProgress.js';
import StockDetailPage from './StockDetailPage.jsx';
import { backtestMetric, isNetBacktest } from './backtestPresentation.js';
import { flagDirectionLabels, precisionEngineLabels, referencePrice, scanPayload, strategyLabels } from './flagPatterns.js';
import FlagBreakdown from './components/FlagBreakdown.jsx';
import PatternTags from './components/PatternTags.jsx';
import { selectionEvidence } from './resultEvidence.js';
import { ExpandedEvidence, ReasonPreview, ResistanceSummary } from './components/ResultEvidence.jsx';
import AlertMonitorPanel from './components/AlertMonitorPanel.jsx';

const emptyScan = {
  results: [],
  total_count: 0,
  buy_alert_count: 0,
  watch_count: 0,
  universe_count: 0,
  attempted_count: 0,
  analyzed_count: 0,
  failed_count: 0,
  stale_count: 0
};

const providerLabels = {
  sample: '示例数据',
  auto: '自动选择',
  akshare: 'AKShare',
  adata: 'AData'
};

function formatScanTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', { hour12: false });
}

function completedStatus(scan, prefix = '筛选完成') {
  const provider = providerLabels[scan.provider] || scan.provider || '未知数据源';
  const finishedAt = formatScanTime(scan.finished_at);
  const timeText = finishedAt ? ` · ${finishedAt}` : '';
  return {
    phase: 'success',
    title: prefix,
    message: `${provider} · 成功分析 ${scan.analyzed_count || 0}/${scan.attempted_count || 0} 只 · 入选 ${scan.total_count || 0} 只 · 失败 ${scan.failed_count || 0} 只${scan.stale_count ? `（含行情未更新 ${scan.stale_count} 只）` : ''}${timeText}${scan.strategy_outdated ? ' · 旧策略批次' : ''}`
  };
}

function categoryLabel(category) {
  if (category === 'buy_alert') return '买入预警';
  if (category === 'watch') return '观察机会';
  return '暂不行动';
}

function categoryClass(category) {
  if (category === 'buy_alert') return 'tag tag-alert';
  if (category === 'watch') return 'tag tag-watch';
  return 'tag';
}

// 带上批次号：详情要按"打开它的那次筛选"的参数来判定，否则改了条件或
// 重启后端之后，点进去看到的结论会跟列表不一致。
function openStockPage(stockCode, stockName = '', scanId = null) {
  const params = new URLSearchParams({ stock: String(stockCode).padStart(6, '0') });
  if (stockName) params.set('name', stockName);
  if (scanId) params.set('scan_id', String(scanId));
  window.open(`/?${params.toString()}`, '_blank', 'noopener,noreferrer');
}

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const stockCode = params.get('stock');
  if (stockCode) {
    return (
      <StockDetailPage
        stockCode={stockCode.padStart(6, '0')}
        stockName={params.get('name') || ''}
        scanId={params.get('scan_id') || ''}
        precisionEngine={params.get('precision_engine') || ''}
        asOf={params.get('as_of') || ''}
      />
    );
  }
  return <Dashboard />;
}

function Dashboard() {
  const [workspaceView, setWorkspaceView] = useState(() =>
    new URLSearchParams(window.location.search).get('view') === 'alert-monitor' ? 'alert-monitor' : 'workbench');
  const [scan, setScan] = useState(emptyScan);
  const [watchlist, setWatchlist] = useState([]);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  // 批次复盘的回看天数：跟个股详情用同一组档位，别写死 500
  const [batchLookback, setBatchLookback] = useState(500);
  const [watchMessage, setWatchMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [form, setForm] = useState({ provider: 'auto', limit: 0, signal_offset: 0, market_filter: 'risk', require_breakout: false, require_pullback: true, min_pullback_days: 1, max_pullback_days: 10, breakout_vol_mult: 1.5, require_breakout_volume: false, strategy_mode: 'combined', precision_engine: 'platform_events', flag_direction: 'all', prebreakout_extension_percent: 50, prebreakout_reject_above_days: 30, prebreakout_above_buffer_percent: 2 });
  const [scanStatus, setScanStatus] = useState({
    phase: 'idle',
    title: '筛选状态',
    message: '正在读取上次筛选结果...'
  });
  const [progress, setProgress] = useState(null);
  const [activeScanId, setActiveScanId] = useState(null);
  const [singleCode, setSingleCode] = useState('');
  const [backtest, setBacktest] = useState({ status: 'empty', summary: {}, results: [] });
  const [backtestProgress, setBacktestProgress] = useState(null);
  const [backtestRunId, setBacktestRunId] = useState(null);
  const [backtestRunning, setBacktestRunning] = useState(false);
  const [watchRefreshing, setWatchRefreshing] = useState(false);
  const [watchRefreshMessage, setWatchRefreshMessage] = useState('');
  const initializedRef = useRef(false);

  const buyAlerts = useMemo(() => (scan.results || []).filter((item) => item.category === 'buy_alert'), [scan]);
  const watchCandidates = useMemo(() => (scan.results || []).filter((item) => item.category === 'watch'), [scan]);

  async function refreshWatchPrices() {
    setWatchRefreshing(true);
    setWatchRefreshMessage('');
    try {
      const result = await api.refreshWatchlist();
      setWatchlist(result.items || []);
      setWatchRefreshMessage(
        result.failed_count
          ? `已更新 ${result.updated_count} 只，${result.failed_count} 只暂时失败`
          : `已更新 ${result.updated_count} 只`
      );
    } catch (err) {
      setWatchRefreshMessage(`行情刷新失败：${err.message}`);
    } finally {
      setWatchRefreshing(false);
    }
  }

  async function refresh() {
    setError('');
    const [latestResult, watchesResult, backtestResult] = await Promise.allSettled([
      api.latestScan(),
      api.watchlist(),
      api.latestBacktest()
    ]);
    const failures = [];
    if (latestResult.status === 'fulfilled') {
      const latest = latestResult.value;
      setScan(latest.results ? latest : emptyScan);
      setScanStatus(latest.id
        ? completedStatus(latest, '当前显示上次筛选')
        : { phase: 'idle', title: '尚未筛选', message: '选择参数后点击“开始筛选”。' });
    } else {
      failures.push(`筛选结果：${latestResult.reason.message}`);
      setScanStatus({ phase: 'error', title: '筛选结果读取失败', message: latestResult.reason.message });
    }
    if (watchesResult.status === 'fulfilled') {
      setWatchlist(watchesResult.value);
    } else {
      failures.push(`观察池：${watchesResult.reason.message}`);
    }
    if (backtestResult.status === 'fulfilled') setBacktest(backtestResult.value);
    else failures.push(`复盘：${backtestResult.reason.message}`);
    if (failures.length) {
      setError(failures.join('；'));
    }
  }

  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;
    refresh();
    api.activeScan().then((state) => {
      if (state && state.scan_id) {
        setActiveScanId(state.scan_id);
        setProgress(state);
        setLoading(true);
        setScanStatus({ phase: 'running', title: '正在筛选', message: '检测到后台仍在运行的筛选任务，已接续显示进度。' });
      }
    }).catch(() => {});
    api.activeBacktest().then((state) => {
      if (state?.run_id) {
        setBacktestRunId(state.run_id);
        setBacktestProgress(state);
        setBacktestRunning(true);
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!activeScanId) return undefined;
    let failures = 0;
    let cancelled = false;

    const timer = setInterval(async () => {
      try {
        const state = await api.scanProgress(activeScanId);
        if (cancelled) return;
        failures = 0;
        setProgress(state);
        if (state.results) {
          // 归零上一轮的分类计数，让指标卡回落到本轮实时结果的派生值，避免与下方表格互相矛盾
          setScan((prev) => ({
            ...prev,
            results: state.results,
            total_count: state.results.length,
            buy_alert_count: 0,
            watch_count: 0
          }));
        }
        if (isTerminal(state)) {
          clearInterval(timer);
          setActiveScanId(null);
          setLoading(false);
          if (state.status === 'failed') {
            setError(state.error || '筛选失败');
            setScanStatus({ phase: 'error', title: '筛选失败', message: state.error || '未知错误' });
          } else {
            const latest = await api.latestScan();
            setScan(latest);
            setScanStatus(completedStatus(latest));
          }
        }
      } catch (err) {
        failures += 1;
        if (failures >= MAX_POLL_FAILURES) {
          clearInterval(timer);
          setActiveScanId(null);
          setLoading(false);
          setError(`进度查询失败：${err.message}`);
        }
      }
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeScanId]);

  useEffect(() => {
    if (!backtestRunId) return undefined;
    let cancelled = false;
    const timer = setInterval(async () => {
      try {
        const state = await api.backtestProgress(backtestRunId);
        if (cancelled) return;
        setBacktestProgress(state);
        if (state.status === 'completed' || state.status === 'failed') {
          clearInterval(timer);
          setBacktestRunning(false);
          setBacktestRunId(null);
          if (state.status === 'completed') setBacktest(await api.latestBacktest());
          else setError(state.error || '批次复盘失败');
        }
      } catch (err) {
        clearInterval(timer);
        setBacktestRunning(false);
        setBacktestRunId(null);
        setError(`复盘进度查询失败：${err.message}`);
      }
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [backtestRunId]);

  async function runScan() {
    const limit = Math.max(0, Number(form.limit) || 0);
    const targetText = form.provider === 'sample'
      ? '3 只示例股票'
      : limit === 0 ? '全部 A 股' : `${limit} 只 A 股样本`;
    setLoading(true);
    setError('');
    setProgress(null);
    setScanStatus({
      phase: 'running',
      title: '正在筛选',
      message: `正在连接${providerLabels[form.provider] || form.provider}并分析${targetText}。当前使用 4 个线程，耗时取决于数据源限流；可以切走做别的事，回来仍能看到进度。`
    });
    try {
      const submitted = await api.runScan(scanPayload(form));
      setActiveScanId(submitted.scan_id);
    } catch (err) {
      setError(err.message);
      setScanStatus({ phase: 'error', title: '筛选失败', message: err.message });
      setLoading(false);
    }
  }

  async function analyzeSingle() {
    const code = singleCode.trim().padStart(6, '0');
    if (!/^\d{6}$/.test(code)) {
      setError('请输入 6 位股票代码');
      return;
    }
    openStockPage(code);
  }

  async function addToWatch(item) {
    if (!item) return;
    setError('');
    try {
      await api.addWatch({
        stock_code: item.stock_code,
        stock_name: item.stock_name,
        note: item.suggestion || '',
        status: 'active',
        entry_price: item.current_price || undefined,
        entry_date: item.signal_date || undefined
      });
      setWatchlist(await api.watchlist());
    } catch (err) {
      setError(err.message);
    }
  }

  async function removeWatch(stockCode) {
    setError('');
    try {
      await api.removeWatch(stockCode);
      setWatchlist(await api.watchlist());
    } catch (err) {
      setError(err.message);
    }
  }

  async function stopScan() {
    if (!activeScanId) return;
    setCancelling(true);
    try {
      await api.cancelScan(activeScanId);
    } catch (err) {
      setError(err.message);
    } finally {
      setCancelling(false);
    }
  }

  async function bulkAddWatch(items) {
    setError('');
    setBulkBusy(true);
    try {
      const payload = await api.bulkAddWatch(items.map((item) => ({
        stock_code: item.stock_code,
        stock_name: item.stock_name || item.stock_code,
        entry_price: item.current_price ?? undefined
      })));
      setWatchlist(payload.items);
      setWatchMessage(`已加入 ${payload.added} 只` + (payload.skipped ? `，${payload.skipped} 只已在池中` : ''));
    } catch (err) {
      setError(err.message);
    } finally {
      setBulkBusy(false);
    }
  }

  async function bulkRemoveWatch(stockCodes) {
    setError('');
    setBulkBusy(true);
    try {
      const payload = await api.bulkDeleteWatch(stockCodes);
      setWatchlist(payload.items);
      setWatchMessage(`已移除 ${payload.deleted} 只`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBulkBusy(false);
    }
  }

  async function manualAddWatch(stockCode, note) {
    setError('');
    setBulkBusy(true);
    try {
      await api.addWatch({ stock_code: stockCode, stock_name: stockCode, note });
      setWatchlist(await api.watchlist());
      setWatchMessage(`已手动加入 ${stockCode}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBulkBusy(false);
    }
  }

  function openWatch(item) {
    // 观察池是手动跟踪，跟当前这批筛选没有关系——不该套用页面上的 scan.id
    openStockPage(item.stock_code, item.stock_name);
  }

  function openResult(item) {
    // 筛选结果必须带上它所属的批次，详情才会用同一套条件判定
    openStockPage(item.stock_code, item.stock_name, activeScanId || scan.id || null);
  }

  function changeWorkspace(view) {
    setWorkspaceView(view);
    const url = new URL(window.location.href);
    if (view === 'alert-monitor') url.searchParams.set('view', view);
    else url.searchParams.delete('view');
    window.history.replaceState({}, '', url);
  }

  async function runSelectedBacktest() {
    if (!scan.results?.length) return;
    setError('');
    setBacktestRunning(true);
    setBacktestProgress(null);
    try {
      const submitted = await api.runBacktest({
        provider: scan.parameters?.provider || (form.provider === 'sample' ? 'sample' : 'auto'),
        limit: 0,
        lookback_days: Number(batchLookback),
        stock_codes: scan.results.map((item) => item.stock_code),
        scan_id: activeScanId || scan.id || undefined
      });
      setBacktestRunId(submitted.run_id);
      setBacktestProgress(submitted);
    } catch (err) {
      setBacktestRunning(false);
      setError(err.message);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">中国 A 股技术选股工作台</div>
          <h1>Stock Mine</h1>
        </div>
        {workspaceView === 'workbench' && <div className="topbar-actions">
          <button className="ghost-button" onClick={refresh} disabled={loading}>
            <RefreshCw size={16} /> 刷新
          </button>
          <button className="primary-button" onClick={runScan} disabled={loading}>
            {loading ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
            <span>{loading ? '筛选中...' : '开始筛选'}</span>
          </button>
        </div>}
      </header>

      <nav className="workspace-navigation" aria-label="工作台视图">
        <button className={workspaceView === 'workbench' ? 'active' : ''} aria-current={workspaceView === 'workbench' ? 'page' : undefined}
          onClick={() => changeWorkspace('workbench')}><Search size={16} />选股工作台</button>
        <button className={workspaceView === 'alert-monitor' ? 'active' : ''} aria-current={workspaceView === 'alert-monitor' ? 'page' : undefined}
          onClick={() => changeWorkspace('alert-monitor')}><Activity size={16} />预警跟踪</button>
      </nav>
      {workspaceView === 'alert-monitor' ? <AlertMonitorPanel
        onOpen={row => openStockPage(row.stock_code, row.stock_name, row.first_scan_id)} /> : <>
      {error && scanStatus.phase !== 'error' && (
        <div className="error-strip">
          <AlertTriangle size={16} /> {error}
        </div>
      )}

      <div className={`scan-status ${scanStatus.phase}`} role="status" aria-live="polite">
        <div className="scan-status-icon">
          {scanStatus.phase === 'running' && <Loader2 className="spin" size={18} />}
          {scanStatus.phase === 'success' && <CheckCircle2 size={18} />}
          {scanStatus.phase === 'error' && <AlertTriangle size={18} />}
          {scanStatus.phase === 'idle' && <Activity size={18} />}
        </div>
        <div className="scan-status-copy">
          <strong>{scanStatus.title}</strong>
          <span>{scanStatus.message}</span>
        </div>
        {scanStatus.phase === 'running' && <div className="scan-status-pulse" aria-hidden="true"><span /></div>}
      </div>

      {progress && loading && (
        <div className="scan-progress">
          <div className="scan-progress-bar">
            <div className="scan-progress-fill" style={{ width: `${progressPercent(progress)}%` }} />
          </div>
          <div className="scan-progress-row">
            <div className="scan-progress-text">{formatProgress(progress)}</div>
            <button className="danger-button" onClick={stopScan} disabled={cancelling}>
              {cancelling ? '停止中...' : '停止筛选'}
            </button>
          </div>
        </div>
      )}

      {!error && scan.failed_count > 0 && (
        <div className="warning-strip">
          <AlertTriangle size={16} /> 本次有 {scan.failed_count} 只股票数据获取或分析失败，结果可能不完整。
          {scan.stale_count > 0 && <span>其中 {scan.stale_count} 只行情未更新，已排除。</span>}
        </div>
      )}

      <section className="quick-lookup">
        <label htmlFor="quick-code">查个股</label>
        <input
          id="quick-code"
          value={singleCode}
          onChange={(event) => setSingleCode(event.target.value)}
          onKeyDown={(event) => { if (event.key === 'Enter') analyzeSingle(); }}
          placeholder="输入 6 位代码，如 600519"
          maxLength={6}
        />
        <button className="primary-button" onClick={analyzeSingle} disabled={!singleCode.trim()}>打开详情</button>
        <span className="quick-lookup-hint">不受下方筛选条件影响，直接查看任意股票</span>
      </section>

      <section className="strategy-picker">
        <div className="result-tabs" role="group" aria-label="形态策略">
          <button type="button" disabled={loading} className={form.strategy_mode === 'combined' ? 'active' : ''} aria-pressed={form.strategy_mode === 'combined'} onClick={() => setForm({ ...form, strategy_mode: 'combined' })}><Search size={15} />联合筛选</button>
          <button type="button" disabled={loading} className={form.strategy_mode === 'precision_pullback' ? 'active' : ''} aria-pressed={form.strategy_mode === 'precision_pullback'} onClick={() => setForm({ ...form, strategy_mode: 'precision_pullback' })}><LineChart size={15} />精准线回调</button>
          <button type="button" disabled={loading} className={form.strategy_mode === 'flag_breakout' ? 'active' : ''} aria-pressed={form.strategy_mode === 'flag_breakout'} onClick={() => setForm({ ...form, strategy_mode: 'flag_breakout' })}><Flag size={15} />旗形突破</button>
        </div>
        {form.strategy_mode !== 'precision_pullback' && <label>旗面方向
          <select disabled={loading} value={form.flag_direction} onChange={event => setForm({ ...form, flag_direction: event.target.value })}>
            <option value="all">全部方向</option>
            {Object.entries(flagDirectionLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>}
        <label>点位引擎
          <select aria-label="点位引擎" disabled={loading || form.strategy_mode === 'flag_breakout'}
            value={form.precision_engine} onChange={event => setForm({ ...form, precision_engine: event.target.value })}>
            {Object.entries(precisionEngineLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <span className="strategy-current">当前结果：{strategyLabels[scan.parameters?.strategy_mode || 'precision_pullback']} · {precisionEngineLabels[scan.parameters?.precision_engine || 'legacy']}</span>
      </section>
      <section className="control-band">
        <label>
          数据源
          <select disabled={loading} value={form.provider} onChange={(event) => setForm({ ...form, provider: event.target.value })}>
            <option value="auto">自动选择真实数据源</option>
            <option value="akshare">akshare</option>
            <option value="adata">adata</option>
            <option value="sample">示例数据（仅演示）</option>
          </select>
        </label>
        <label>
          扫描数量（0=全部）
          <input disabled={loading} type="number" min="0" max="6000" value={form.limit} onChange={(event) => setForm({ ...form, limit: event.target.value })} />
        </label>
        {form.strategy_mode !== 'flag_breakout' && form.precision_engine !== 'platform_events' && <label className="checkbox-field">
          <input
            disabled={loading}
            type="checkbox"
            checked={form.require_pullback ?? true}
            onChange={event => setForm({ ...form, require_pullback: event.target.checked })}
            aria-label="精准线只选突破后回踩"
          />
          精准线只选突破后回踩
        </label>}
        {(form.strategy_mode !== 'precision_pullback' || !(form.require_pullback ?? true)) && <label className="checkbox-field">
          <input
            disabled={loading}
            type="checkbox"
            checked={form.require_breakout}
            onChange={(event) => setForm({ ...form, require_breakout: event.target.checked })}
          />
          {form.strategy_mode === 'flag_breakout' || (form.strategy_mode === 'combined' && (form.require_pullback ?? true)) ? '旗形只选已突破' : '只要已突破的股票'}
        </label>}
        {form.strategy_mode !== 'flag_breakout' && <label className="checkbox-field">
          <input
            disabled={loading}
            type="checkbox"
            checked={form.require_breakout_volume ?? false}
            onChange={event => setForm({ ...form, require_breakout_volume: event.target.checked })}
            aria-label="精准线启用突破量比过滤"
          />
          精准线启用突破量比过滤
        </label>}
        {(form.strategy_mode !== 'precision_pullback' || form.require_breakout_volume) && <label>
          {form.strategy_mode === 'flag_breakout' || !form.require_breakout_volume ? '旗形突破量比门槛（成交量）' : '突破量比门槛（成交量）'}
          <input
            disabled={loading}
            type="number"
            min="1"
            max="5"
            step="0.1"
            value={form.breakout_vol_mult}
            onChange={(event) => setForm({ ...form, breakout_vol_mult: event.target.value })}
            aria-label="突破量比门槛"
          />
          <small className="field-hint">{form.strategy_mode === 'combined' && form.require_breakout_volume ? '精准线以此前10日均量为基准，旗形以此前20日均量为基准' : `突破当日成交量 ÷ 前 ${form.strategy_mode === 'precision_pullback' ? 10 : 20} 日均量`}</small>
        </label>}
        {form.strategy_mode !== 'flag_breakout' && <label>
          突破前越线上限（%）
          <input
            disabled={loading}
            type="number"
            min="0"
            max="100"
            step="1"
            value={form.prebreakout_extension_percent ?? 50}
            onChange={event => setForm({ ...form, prebreakout_extension_percent: event.target.value })}
            aria-label="突破前越线上限百分比"
          />
          <small className="field-hint">最高价 ≤ 线价 × (1 + 上限%)</small>
        </label>}
        {form.strategy_mode !== 'flag_breakout' && <label>
          连续站上排除（交易日）
          <input
            disabled={loading}
            type="number"
            min="1"
            max="250"
            step="1"
            value={form.prebreakout_reject_above_days ?? 30}
            onChange={event => setForm({ ...form, prebreakout_reject_above_days: event.target.value })}
            aria-label="突破前连续站上排除天数"
          />
          <small className="field-hint">突破前连续有效收线上天数 ≥ 阈值即排除</small>
        </label>}
        {form.strategy_mode !== 'flag_breakout' && <label>
          有效收线上缓冲（%）
          <input
            disabled={loading}
            type="number"
            min="0"
            max="100"
            step="0.1"
            value={form.prebreakout_above_buffer_percent ?? 2}
            onChange={event => setForm({ ...form, prebreakout_above_buffer_percent: event.target.value })}
            aria-label="有效收线上缓冲百分比"
          />
          <small className="field-hint">收盘价 &gt; 线价 × (1 + 缓冲%)</small>
        </label>}
        {form.strategy_mode !== 'flag_breakout' && <label>
          回调天数（交易日）
          <div className="range-input">
            <input
              disabled={loading}
              type="number"
              min="1"
              max="60"
              step="1"
              value={form.min_pullback_days ?? 1}
              onChange={(event) => setForm({ ...form, min_pullback_days: event.target.value })}
              aria-label="回调天数下限"
            />
            <span>~</span>
            <input
              disabled={loading}
              type="number"
              min="1"
              max="60"
              step="1"
              value={form.max_pullback_days ?? 10}
              onChange={(event) => setForm({ ...form, max_pullback_days: event.target.value })}
              aria-label="回调天数上限"
            />
          </div>
          <small className="field-hint">突破次日为第1天；信号日回踩确认，非连续阴线天数</small>
        </label>}
        <label>
          信号 K 线
          <select disabled={loading} value={form.signal_offset} onChange={(event) => setForm({ ...form, signal_offset: event.target.value })}>
            <option value="0">最近已收盘交易日（推荐，与回测一致）</option>
            <option value="1">最新K线的前一根（旧版默认）</option>
          </select>
          <small className="field-hint">盘中未收盘的K线会自动去掉，不会用半天的数据判定回踩</small>
        </label>
        <label>
          大盘环境（沪深300）
          <select disabled={loading} value={form.market_filter} onChange={(event) => setForm({ ...form, market_filter: event.target.value })}>
            <option value="risk">偏弱时仅提示，不降级（默认）</option>
            <option value="downgrade">偏弱时买入预警降为观察（未经验证）</option>
            <option value="off">不评估</option>
          </select>
          <small className="field-hint">偏弱 = 近20日收益为负或MA20近5日下行；降级在回测中未通过验证；缺当日指数时不降级并提示</small>
        </label>
        <div className="disclaimer">技术预警仅用于复盘和观察，不构成投资建议。</div>
      </section>

      <section className="metrics-grid">
        <Metric
          icon={<Activity size={18} />}
          label="入选结果"
          value={scan.total_count || 0}
          hint={scan.attempted_count ? `成功分析 ${scan.analyzed_count}/${scan.attempted_count}，失败 ${scan.failed_count}` : '尚未筛选'}
        />
        <Metric icon={<CheckCircle2 size={18} />} label="买入预警" value={scan.buy_alert_count || buyAlerts.length} hint="所选形态条件确认" tone="alert" />
        <Metric icon={<Eye size={18} />} label="观察机会" value={scan.watch_count || watchCandidates.length} hint="形态尚待确认" tone="watch" />
        <Metric icon={<Star size={18} />} label="观测池" value={watchlist.length} hint="手动跟踪" />
      </section>

      <WatchPanel
        rows={watchlist}
        refreshing={watchRefreshing}
        refreshMessage={watchRefreshMessage}
        bulkMessage={watchMessage}
        onRefresh={refreshWatchPrices}
        onOpen={openWatch}
        onRemove={removeWatch}
        onBulkRemove={bulkRemoveWatch}
        onManualAdd={manualAddWatch}
        bulkBusy={bulkBusy}
      />

      <section className="workspace-stack">
        <ResultPanel
          buyRows={buyAlerts}
          watchRows={watchCandidates}
          onSelect={openResult}
          onWatch={addToWatch}
          onBulkWatch={bulkAddWatch}
          bulkBusy={bulkBusy}
        />
        <IndustryPanel rows={scan.industry_summary || []} />
      </section>
      <BatchBacktestPanel
        report={backtest}
        progress={backtestProgress}
        running={backtestRunning}
        selectedCount={scan.results?.length || 0}
        onRun={runSelectedBacktest}
        lookback={batchLookback}
        onLookbackChange={setBatchLookback}
        sourceScanId={backtest?.parameters?.scan_id || activeScanId || scan.id || null}
      />
      </>}
    </main>
  );
}

function Metric({ icon, label, value, hint, tone = '' }) {
  return (
    <div className={`metric ${tone}`}>
      <div className="metric-icon">{icon}</div>
      <div>
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>
        <div className="metric-hint">{hint}</div>
      </div>
    </div>
  );
}

function ResultPanel({ buyRows, watchRows, onSelect, onWatch, onBulkWatch, bulkBusy }) {
  const [activeTab, setActiveTab] = useState('buy_alert');
  const [visibleCount, setVisibleCount] = useState(50);
  const [picked, setPicked] = useState([]);
  const [expanded, setExpanded] = useState([]);
  const viewportRef = useRef(null);
  const rows = activeTab === 'buy_alert' ? buyRows : watchRows;
  const visibleRows = useMemo(() => rows.slice(0, visibleCount), [rows, visibleCount]);
  const evidenceByCode = useMemo(() => new Map(
    visibleRows.map(row => [row.stock_code, selectionEvidence(row)])
  ), [visibleRows]);

  // 结果刷新或切换分页后，清掉已经不在列表里的选中项，避免对幽灵代码执行批量操作
  useEffect(() => { setPicked((current) => keepExisting(current, rows)); }, [rows]);
  useEffect(() => { setExpanded([]); }, [rows]);
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    // clientWidth excludes the scrollbar, unlike container query units.
    const sizeEvidence = () => viewport.style.setProperty('--result-viewport-width', `${viewport.clientWidth}px`);
    sizeEvidence();
    const observer = new ResizeObserver(sizeEvidence);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [rows.length]);

  function selectTab(tab) {
    setActiveTab(tab);
    setVisibleCount(50);
    setPicked([]);
    setExpanded([]);
  }

  async function bulkWatch() {
    const items = rows.filter((row) => picked.includes(row.stock_code));
    if (!items.length) return;
    await onBulkWatch(items);
    setPicked([]);
  }

  return (
    <section className="data-panel result-panel">
      <div className="panel-heading">
        <h2>{activeTab === 'buy_alert' ? <AlertTriangle size={17} /> : <Search size={17} />}筛选结果</h2>
        <div className="result-tabs" role="tablist" aria-label="筛选结果分类">
          <button className={activeTab === 'buy_alert' ? 'active' : ''} onClick={() => selectTab('buy_alert')} role="tab" aria-selected={activeTab === 'buy_alert'}>
            买入预警 <b>{buyRows.length}</b>
          </button>
          <button className={activeTab === 'watch' ? 'active' : ''} onClick={() => selectTab('watch')} role="tab" aria-selected={activeTab === 'watch'}>
            观察机会 <b>{watchRows.length}</b>
          </button>
        </div>
      </div>
      {picked.length > 0 && (
        <div className="bulk-bar">
          <span>已选 <b>{picked.length}</b> 只</span>
          <button className="primary-button" onClick={bulkWatch} disabled={bulkBusy}>
            <BookmarkPlus size={15} /> 批量加入观测池
          </button>
          <button className="ghost-button" onClick={() => setPicked([])}>取消选择</button>
        </div>
      )}
      {rows.length === 0 ? (
        <div className="empty-row">暂无数据</div>
      ) : (
        <div className="table-wrap result-table-wrap" ref={viewportRef}>
          <table className="result-table">
            <colgroup>
              {[32, 64, 78, 92, 76, 114, 54, 78, 80, 114, 118, 272, 58, 48]
                .map((width, index) => <col key={index} style={{ width }} />)}
            </colgroup>
            <thead>
              <tr>
                <th className="pick-col">
                  <input
                    type="checkbox"
                    checked={isAllSelected(picked, visibleRows)}
                    onChange={() => setPicked(toggleAll(picked, visibleRows))}
                    aria-label="全选当前页"
                  />
                </th>
                <th>代码</th>
                <th>名称</th>
                <th>信号日</th>
                <th>类型</th>
                <th>形态</th>
                <th>结构分</th>
                <th>行业</th>
                <th>参考位</th>
                <th>形态状态</th>
                <th>阻力点</th>
                <th>选股理由</th>
                <th>距离</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row) => {
                const evidence = evidenceByCode.get(row.stock_code);
                const isExpanded = expanded.includes(row.stock_code);
                const evidenceId = `result-evidence-${row.stock_code}`;
                return <Fragment key={`${row.stock_code}-${row.category}`}>
                <tr className={isExpanded ? 'evidence-open' : ''} onClick={() => onSelect(row)}>
                  <td className="pick-col" onClick={(event) => event.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={picked.includes(row.stock_code)}
                      onChange={() => setPicked(toggleCode(picked, row.stock_code))}
                      aria-label={`选择 ${row.stock_code}`}
                    />
                  </td>
                  <td>{row.stock_code}</td>
                  <td>{row.stock_name}</td>
                  <td>{row.signal_date || '-'}</td>
                  <td><span className={categoryClass(row.category)}>{categoryLabel(row.category)}</span></td>
                  <td className="pattern-cell"><PatternTags row={row} /></td>
                  <td title="形态结构分；行业只作提示、不计分；不代表胜率">{Math.round(row.score)}</td>
                  <td>{row.industry || '未覆盖'}</td>
                  <td className="key-line-cell">
                    {formatPrice(referencePrice(row))}
                    {row.pattern_type === 'flag_breakout' && <small>上轨</small>}
                    {row.line_strength ? <small>{row.line_strength} 点</small> : null}
                  </td>
                  <td><div className="status-tags">
                    {row.breakout_status?.tags?.length
                      ? sortTags(row.breakout_status.tags).map((tag) => (
                          <span key={tag} className={`status-tag tone-${tagTone(tag)}`}>{tag}</span>
                        ))
                      : <span className="status-tag tone-neutral">未突破</span>}
                  </div></td>
                  <td className="resistance-cell"><ResistanceSummary evidence={evidence.resistance} /></td>
                  <td className="reason-cell">
                    <div className="reason-cell-content">
                      <ReasonPreview groups={evidence.groups} />
                      <button type="button" className="icon-button evidence-toggle"
                        aria-label={`${isExpanded ? '收起' : '展开'}${row.stock_name}的选股依据`}
                        title={`${isExpanded ? '收起' : '展开'}选股理由与阻力点`}
                        aria-expanded={isExpanded} aria-controls={evidenceId}
                        onClick={(event) => {
                          event.stopPropagation();
                          setExpanded(current => toggleCode(current, row.stock_code));
                        }}>
                        <ChevronDown size={16} />
                      </button>
                    </div>
                  </td>
                  <td>{row.distance_to_line_pct ?? '-'}%</td>
                  <td>
                    <button className="icon-button" onClick={(event) => { event.stopPropagation(); onWatch(row); }} title="加入观测池">
                      <BookmarkPlus size={15} />
                    </button>
                  </td>
                </tr>
                {isExpanded && <tr className="result-evidence-row">
                  <td colSpan={14}>
                    <ExpandedEvidence row={row} evidence={evidence} id={evidenceId} onOpen={onSelect} />
                  </td>
                </tr>}
                </Fragment>;
              })}
            </tbody>
          </table>
          {visibleRows.length < rows.length && (
            <button className="load-more-button" onClick={() => setVisibleCount((count) => count + 50)}>
              <ChevronDown size={15} /> 再显示 {Math.min(50, rows.length - visibleRows.length)} 只
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function WatchPanel({ rows, refreshing, refreshMessage, bulkMessage, onRefresh, onOpen, onRemove, onBulkRemove, onManualAdd, bulkBusy }) {
  const [picked, setPicked] = useState([]);
  const [manualCode, setManualCode] = useState('');
  const [manualNote, setManualNote] = useState('');

  useEffect(() => { setPicked((current) => keepExisting(current, rows)); }, [rows]);

  async function removePicked() {
    if (!picked.length) return;
    await onBulkRemove(picked);
    setPicked([]);
  }

  async function addManual() {
    const code = manualCode.trim();
    if (!/^\d{1,6}$/.test(code)) return;
    await onManualAdd(code.padStart(6, '0'), manualNote.trim());
    setManualCode('');
    setManualNote('');
  }

  return (
    <section className="data-panel watch-panel">
      <div className="panel-heading">
        <h2><Star size={17} />观测池</h2>
        <div className="watch-panel-actions">
          {(refreshMessage || bulkMessage) && <span>{refreshMessage || bulkMessage}</span>}
          <button className="icon-button" onClick={onRefresh} disabled={refreshing || rows.length === 0} title="刷新观察池行情">
            <RefreshCw className={refreshing ? 'spin' : ''} size={15} />
          </button>
          <b>{rows.length} 只</b>
        </div>
      </div>
      <div className="watch-manual-add">
        <input
          value={manualCode}
          onChange={(event) => setManualCode(event.target.value)}
          onKeyDown={(event) => { if (event.key === 'Enter') addManual(); }}
          placeholder="股票代码"
          maxLength={6}
          aria-label="手动加入的股票代码"
        />
        <input
          value={manualNote}
          onChange={(event) => setManualNote(event.target.value)}
          onKeyDown={(event) => { if (event.key === 'Enter') addManual(); }}
          placeholder="备注（可选）"
          aria-label="备注"
        />
        <button className="ghost-button" onClick={addManual} disabled={!/^\d{1,6}$/.test(manualCode.trim())}>
          <BookmarkPlus size={15} /> 手动加入
        </button>
      </div>
      {picked.length > 0 && (
        <div className="bulk-bar">
          <span>已选 <b>{picked.length}</b> 只</span>
          <button className="danger-button" onClick={removePicked} disabled={bulkBusy}>
            <Trash2 size={15} /> 批量移除
          </button>
          <button className="ghost-button" onClick={() => setPicked([])}>取消选择</button>
        </div>
      )}
      {rows.length === 0 ? (
        <div className="empty-row">暂无手动跟踪股票</div>
      ) : (
        <div className="table-wrap watch-table-wrap">
          <table className="watch-table">
            <thead><tr><th className="pick-col"><input type="checkbox" checked={isAllSelected(picked, rows)} onChange={() => setPicked(toggleAll(picked, rows))} aria-label="全选观测池" /></th><th>股票</th><th>加入价</th><th>当前价</th><th>持有表现</th><th>价格日期</th><th>备注</th><th>操作</th></tr></thead>
            <tbody>
              {rows.map((row) => {
                const performance = Number(row.return_pct || 0);
                const tone = row.return_pct === null || row.return_pct === undefined
                  ? 'flat'
                  : performance > 0 ? 'gain' : performance < 0 ? 'loss' : 'flat';
                return (
                  <tr key={row.stock_code} onClick={() => onOpen(row)}>
                    <td className="pick-col" onClick={(event) => event.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={picked.includes(row.stock_code)}
                        onChange={() => setPicked(toggleCode(picked, row.stock_code))}
                        aria-label={`选择 ${row.stock_code}`}
                      />
                    </td>
                    <td><div className="watch-stock"><strong>{row.stock_name}</strong><span>{row.stock_code}</span></div></td>
                    <td>
                      <strong>{formatPrice(row.entry_price)}</strong>
                      <small>{entrySourceLabel(row)}</small>
                    </td>
                    <td><strong>{formatPrice(row.latest_price)}</strong></td>
                    <td><div className={`watch-return ${tone}`}><strong>{formatSignedPercent(row.return_pct)}</strong><small>{formatSignedPrice(row.price_change)}</small></div></td>
                    <td>{row.latest_price_at || '-'}</td>
                    <td className="watch-note">{row.note || '无备注'}</td>
                    <td>
                      <button className="icon-button danger" onClick={(event) => { event.stopPropagation(); onRemove(row.stock_code); }} title="移出观测池">
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function formatPrice(value) {
  return value === null || value === undefined ? '-' : Number(value).toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

function formatSignedPercent(value) {
  if (value === null || value === undefined) return '待刷新';
  const numeric = Number(value);
  return `${numeric > 0 ? '+' : ''}${numeric.toFixed(2)}%`;
}

function formatSignedPrice(value) {
  if (value === null || value === undefined) return '暂无对比';
  const numeric = Number(value);
  return `${numeric > 0 ? '+' : ''}${numeric.toFixed(3)}`;
}

function entrySourceLabel(row) {
  if (!row.entry_price) return '等待建立基准';
  if (row.entry_price_source === 'first_refresh') return '首次刷新基准';
  return row.entry_date || '加入时';
}

function IndustryPanel({ rows }) {
  return (
    <section className="data-panel">
      <div className="panel-heading">
        <h2><LineChart size={17} />行业分布与风险</h2>
        <span>{rows.length} 个行业</span>
      </div>
      {rows.length === 0 ? (
        <div className="empty-row">完成筛选后显示行业聚合</div>
      ) : (
        <div className="industry-list">
          {rows.slice(0, 12).map((row) => (
            <div className="industry-row" key={row.industry}>
              <div><strong>{row.industry}</strong><span>{row.selected_count} 只 · 预警 {row.buy_alert_count}</span></div>
              <div className={`risk-label risk-${row.risk_level}`}><b>{row.industry_score ?? '-'}</b><span>{row.risk_message}</span></div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function BatchBacktestPanel({ report, progress, running, selectedCount, onRun, lookback, onLookbackChange, sourceScanId }) {
  const summary = report?.summary || {};
  const net = isNetBacktest(summary);
  const metrics = [5, 10, 20].map((window) => backtestMetric(summary, window));
  const processed = progress?.processed || 0;
  const total = progress?.total || selectedCount || 0;
  const pct = total ? Math.min(100, processed / total * 100) : 0;
  return (
    <section className="data-panel batch-backtest-panel">
      <div className="panel-heading">
        <h2><Database size={17} />批次历史预警复盘</h2>
        <label className="batch-lookback">
          回看
          <select value={lookback} onChange={(event) => onLookbackChange(Number(event.target.value))} disabled={running}>
            <option value="250">1年</option>
            <option value="500">2年</option>
            <option value="750">3年</option>
            <option value="0">全部</option>
          </select>
        </label>
        <button className="primary-button" onClick={onRun} disabled={running || selectedCount === 0}>
          {running ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
          {running ? '复盘中...' : `复盘本次入选（${selectedCount}只）`}
        </button>
      </div>
      {running && (
        <div className="batch-progress">
          <div><span style={{ width: `${pct}%` }} /></div>
          <p>已处理 {processed}/{total} · 历史预警 {progress?.signal_count || 0} 个 · 失败 {progress?.failed || 0}{progress?.current_stock ? ` · ${progress.current_stock}` : ''}</p>
        </div>
      )}
      <div className="batch-metrics">
        <BatchMetric label="复盘股票" value={summary.stock_count ?? '-'} sub={`${summary.stocks_with_signals || 0} 只有历史预警`} />
        <BatchMetric label="历史预警数" value={summary.signal_count ?? '-'} sub="20个交易日内去重" />
        {metrics.map((item) => <BatchMetric key={item.label} label={item.label} value={formatRate(item.rate)} sub={`${item.count} 个样本${item.pending ? ` · ${item.pending} 个待退出` : ''}`} />)}
        <BatchMetric label={net ? 'T+5 平均净收益' : 'T+5 平均涨跌'} value={metrics[0].mean == null ? '-' : formatSignedPercent(metrics[0].mean)} sub={net ? '日线成交近似 · 扣成本' : '信号收盘价基准 · 未扣费'} />
      </div>
      <FlagBreakdown rows={summary.flag_directions} />
      {summary.signal_count > 0 && <p className="method-note batch-method-note">
        {net
          ? `开盘入场 · 固定周期退出 · 往返费用及滑点假设 20 bps · 不可入场 ${summary.blocked_entry_count || 0} 个 · 待入场 ${summary.pending_entry_count || 0} 个`
          : '旧版信号涨跌统计，未按开盘成交与交易成本重算'}
      </p>}
      {(report?.results || []).length > 0 && (
        <div className="table-wrap batch-table-wrap">
          <table>
            <thead><tr><th>代码</th><th>名称</th><th>历史预警</th>{metrics.map((item) => <th key={item.label}>{item.label}</th>)}</tr></thead>
            <tbody>
              {report.results.slice(0, 30).map((item) => (
                <tr key={item.stock_code} onClick={() => openStockPage(item.stock_code, item.stock_name, sourceScanId)}>
                  <td>{item.stock_code}</td><td>{item.stock_name}</td><td>{item.summary?.signal_count || 0}</td>
                  {[5, 10, 20].map((window) => <td key={window}>{formatRate(net && !isNetBacktest(item.summary) ? undefined : backtestMetric(item.summary, window).rate)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function BatchMetric({ label, value, sub }) {
  return <div><span>{label}</span><strong>{value}</strong><small>{sub}</small></div>;
}

function formatRate(value) {
  return value === null || value === undefined ? '-' : `${Number(value).toFixed(1)}%`;
}
