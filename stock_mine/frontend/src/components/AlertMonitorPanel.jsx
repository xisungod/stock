import { Fragment, useEffect, useRef, useState } from 'react';
import { AlertTriangle, ChevronLeft, ChevronRight, Clock3, History, Loader2, RefreshCw, RotateCcw, Search } from 'lucide-react';
import { api } from '../api.js';
import { emptyMonitorFilters, monitorPercent, monitorPrice, monitorQuery, monitorTone, validateMonitorFilters } from '../alertMonitor.js';
import './AlertMonitorPanel.css';

const statusLabels = { fresh: '已更新', pending: '待收盘价', stale: '未更新' };
const runLabels = { running: '更新中', completed: '更新完成', partial: '部分行情未更新', failed: '更新失败' };

function chinaTime(value) {
  if (!value) return '--';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
}

export default function AlertMonitorPanel({ onOpen }) {
  const [draft, setDraft] = useState({ ...emptyMonitorFilters });
  const [filters, setFilters] = useState({ ...emptyMonitorFilters });
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const [historyId, setHistoryId] = useState(null);
  const lastRunRef = useRef('');
  const runKey = run => JSON.stringify([run?.id, run?.status, run?.processed]);
  const query = monitorQuery(filters, page);

  useEffect(() => {
    let active = true;
    setBusy(true);
    setError('');
    api.alertMonitor(query).then(payload => {
      if (!active) return;
      const lastPage = Math.max(1, Math.ceil(payload.total_count / payload.page_size));
      if (page > lastPage) setPage(lastPage);
      lastRunRef.current = runKey(payload.run);
      setData(payload);
    }).catch(err => { if (active) setError(err.message); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [query, revision, page]);

  useEffect(() => {
    let active = true;
    let pending = false;
    const timer = setInterval(async () => {
      if (pending || document.visibilityState !== 'visible') return;
      pending = true;
      try {
        const status = await api.alertMonitorStatus();
        if (!active) return;
        setError(current => current.startsWith('更新状态读取失败') ? '' : current);
        const key = runKey(status.run);
        if (lastRunRef.current !== key) {
          lastRunRef.current = key;
          setRevision(value => value + 1);
        }
        setData(current => current ? { ...current, ...status } : current);
      } catch (err) {
        if (active) setError(`更新状态读取失败：${err.message}`);
      } finally {
        pending = false;
      }
    }, 4000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  const change = (key, value) => setDraft(current => ({ ...current, [key]: value }));
  const running = data?.run?.status === 'running';

  function applyFilters(event) {
    event.preventDefault();
    const problem = validateMonitorFilters(draft);
    if (problem) { setError(problem); return; }
    setError('');
    setPage(1);
    setHistoryId(null);
    setFilters({ ...draft });
  }

  function resetFilters() {
    setDraft({ ...emptyMonitorFilters });
    setFilters({ ...emptyMonitorFilters });
    setPage(1);
    setHistoryId(null);
    setError('');
  }

  async function updatePrices() {
    setRefreshing(true);
    setError('');
    try {
      const run = await api.refreshAlertMonitor();
      setData(current => current ? { ...current, run } : current);
      setRevision(value => value + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setRefreshing(false);
    }
  }

  const summary = data?.summary || {};
  const run = data?.run;
  const totalPages = Math.max(1, Math.ceil((data?.total_count || 0) / 50));

  return (
    <section className="alert-monitor" aria-label="买入预警涨跌跟踪">
      <header className="monitor-heading">
        <div>
          <h2>预警跟踪</h2>
          <div className="monitor-schedule">
            <Clock3 size={14} />
            <span>{data?.schedule?.enabled ? '交易日 16:00 自动更新 · 北京时间' : '自动更新未启用'}</span>
            {data?.as_of && <span>跟踪截至 {data.as_of}</span>}
          </div>
        </div>
        <div className="monitor-actions">
          <button className="icon-button" title="重新读取跟踪记录" aria-label="重新读取跟踪记录"
            disabled={busy} onClick={() => setRevision(value => value + 1)}><RefreshCw size={16} /></button>
          <button className="primary-button" onClick={updatePrices} disabled={running || refreshing || !data}>
            {running || refreshing ? <Loader2 size={16} className="spin" /> : <RefreshCw size={16} />}
            {running || refreshing ? '更新中' : '更新收盘价'}
          </button>
        </div>
      </header>

      <div className="monitor-overview">
        <div><span>筛选记录</span><strong>{data?.total_count ?? '--'}</strong></div>
        <div><span>上涨</span><strong className="monitor-gain">{summary.up_count ?? '--'}</strong></div>
        <div><span>下跌</span><strong className="monitor-loss">{summary.down_count ?? '--'}</strong></div>
        <div><span>待更新</span><strong>{summary.pending_count ?? '--'}</strong></div>
        <div className="monitor-updated"><span>最近更新</span><strong>{chinaTime(run?.finished_at)}</strong></div>
      </div>

      {run && <div className={`monitor-run ${run.status}`} role="status" aria-live="polite">
        {running ? <Loader2 size={15} className="spin" /> : <Clock3 size={15} />}
        <span>{runLabels[run.status] || run.status} · {run.market_date} · {run.processed}/{run.total} 只 · 更新 {run.updated} 条{run.failed ? ` · ${run.failed} 只未完整更新` : ''}{run.current_stock ? ` · ${run.current_stock}` : ''}</span>
        {running && <progress aria-label="收盘价更新进度" value={run.processed} max={Math.max(1, run.total)} />}
        {run.error && <span>{run.error}</span>}
      </div>}
      {error && <div className="error-strip" role="alert"><AlertTriangle size={16} />{error}</div>}

      <form className="monitor-filters" onSubmit={applyFilters}>
        <label className="monitor-search">股票 / 行业
          <input aria-label="跟踪股票搜索" value={draft.q} onChange={e => change('q', e.target.value)} placeholder="代码、名称或行业" maxLength={100} />
        </label>
        <label>记录范围
          <select aria-label="跟踪记录范围" value={draft.scope} onChange={e => change('scope', e.target.value)}>
            <option value="latest">每股最新入选</option><option value="all">全部入选记录</option>
          </select>
        </label>
        <fieldset><legend>入选后交易日数</legend><div className="monitor-range">
          <input type="number" min="0" max="20000" step="1" aria-label="跟踪交易日数下限" placeholder="不限" value={draft.min_days} onChange={e => change('min_days', e.target.value)} />
          <span>至</span>
          <input type="number" min="0" max="20000" step="1" aria-label="跟踪交易日数上限" placeholder="不限" value={draft.max_days} onChange={e => change('max_days', e.target.value)} />
        </div></fieldset>
        <fieldset><legend>较入选涨跌幅（%）</legend><div className="monitor-range">
          <input type="number" step="any" aria-label="跟踪涨跌幅下限" placeholder="不限" value={draft.min_return} onChange={e => change('min_return', e.target.value)} />
          <span>至</span>
          <input type="number" step="any" aria-label="跟踪涨跌幅上限" placeholder="不限" value={draft.max_return} onChange={e => change('max_return', e.target.value)} />
        </div></fieldset>
        <fieldset className="monitor-date-range"><legend>入选日期</legend><div className="monitor-range">
          <input type="date" aria-label="跟踪入选开始日期" value={draft.selected_from} onChange={e => change('selected_from', e.target.value)} />
          <span>至</span>
          <input type="date" aria-label="跟踪入选结束日期" value={draft.selected_to} onChange={e => change('selected_to', e.target.value)} />
        </div></fieldset>
        <label>行情状态
          <select aria-label="跟踪行情状态" value={draft.data_status} onChange={e => change('data_status', e.target.value)}>
            <option value="all">全部状态</option><option value="fresh">已更新</option>
            <option value="stale">行情未更新</option><option value="pending">待基准收盘价</option>
          </select>
        </label>
        <label>排序
          <select aria-label="跟踪排序" value={draft.sort} onChange={e => change('sort', e.target.value)}>
            <option value="selected_desc">最近入选优先</option><option value="return_desc">涨跌幅从高到低</option>
            <option value="return_asc">涨跌幅从低到高</option><option value="days_desc">交易日数从多到少</option>
            <option value="days_asc">交易日数从少到多</option>
          </select>
        </label>
        <div className="monitor-filter-actions">
          <button type="submit" className="primary-button"><Search size={15} />筛选</button>
          <button type="button" className="icon-button" aria-label="重置跟踪筛选" title="重置筛选" onClick={resetFilters}><RotateCcw size={16} /></button>
        </div>
      </form>

      <div className="monitor-table-wrap" aria-busy={busy}>
        <table className="monitor-table">
          <thead><tr>
            <th>股票</th>
            <th title="（最新未复权收盘价 ÷ 入选基准价 - 1）×100%；未计分红和费用，不是总收益率">较入选涨跌幅</th>
            <th>已过交易日</th><th>入选日期</th>
            <th title="入选当日未复权收盘价；非交易日使用上一交易日，基准日另行标注">入选收盘价</th>
            <th>最新收盘价</th><th>行业</th>
            <th>行情日期</th><th>状态</th><th>每日记录</th>
          </tr></thead>
          <tbody>
            {data?.items?.map(row => <Fragment key={row.id}>
              <tr>
                <td><button className="monitor-stock" onClick={() => onOpen(row)}><strong>{row.stock_name}</strong><span>{row.stock_code}</span></button></td>
                <td className={`monitor-return ${monitorTone(row.return_pct)}`}>{monitorPercent(row.return_pct)}</td>
                <td className="monitor-days">{row.trading_days}<small>交易日</small></td>
                <td>{row.selected_date}<small title={chinaTime(row.selected_at)}>批次 #{row.first_scan_id}{row.scan_count > 1 ? ` · 同日 ${row.scan_count} 批` : ''}</small></td>
                <td>{monitorPrice(row.baseline_close)}{row.baseline_date !== row.selected_date && <small>基准 {row.baseline_date}</small>}</td>
                <td>{monitorPrice(row.latest_close)}</td>
                <td>{row.industry || '--'}</td>
                <td>{row.quote_date || '--'}</td>
                <td><span className={`monitor-status ${row.data_status}`}>{statusLabels[row.data_status]}</span>
                  {row.last_error && <small className="monitor-row-error" title={row.last_error}>{row.last_error}</small>}</td>
                <td><button className="icon-button" title="查看每日跟踪记录" aria-label={`查看${row.stock_code}每日记录`}
                  aria-expanded={historyId === row.id} onClick={() => setHistoryId(current => current === row.id ? null : row.id)}><History size={16} /></button></td>
              </tr>
              {historyId === row.id && <tr className="monitor-history-row"><td colSpan={10}><DailyHistory entryId={row.id} revision={run?.finished_at} /></td></tr>}
            </Fragment>)}
            {!data?.items?.length && <tr><td colSpan={10} className="monitor-empty">{busy ? '正在读取预警跟踪…' : error ? '记录读取失败，请重试' : '暂无符合条件的预警记录'}</td></tr>}
          </tbody>
        </table>
      </div>
      <footer className="monitor-pagination">
        <span>共 {data?.total_count || 0} 条 · 第 {page} / {totalPages} 页</span>
        <div>
          <button className="icon-button" disabled={page <= 1 || busy} aria-label="跟踪上一页" title="上一页" onClick={() => setPage(value => value - 1)}><ChevronLeft size={17} /></button>
          <button className="icon-button" disabled={page >= totalPages || busy} aria-label="跟踪下一页" title="下一页" onClick={() => setPage(value => value + 1)}><ChevronRight size={17} /></button>
        </div>
      </footer>
    </section>
  );
}

function DailyHistory({ entryId, revision }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    setRows(null);
    setError('');
    api.alertMonitorHistory(entryId).then(payload => { if (active) setRows(payload.items); })
      .catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [entryId, revision]);
  if (error) return <div role="alert">{error}</div>;
  if (!rows) return <div>正在读取每日记录…</div>;
  if (!rows.length) return <div>尚无每日跟踪记录</div>;
  return <div className="monitor-history"><table>
    <thead><tr><th>收盘日期</th><th>入选后交易日</th><th>收盘价</th><th>较入选涨跌幅</th><th>更新时间</th></tr></thead>
    <tbody>{rows.slice().reverse().map(row => <tr key={row.market_date}><td>{row.market_date}</td><td>{row.trading_days}</td>
      <td>{monitorPrice(row.close)}</td><td className={monitorTone(row.return_pct)}>{monitorPercent(row.return_pct)}</td><td>{chinaTime(row.updated_at)}</td></tr>)}</tbody>
  </table></div>;
}
