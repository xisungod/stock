import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BookmarkPlus,
  Building2,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  Database,
  ExternalLink,
  LineChart,
  Loader2,
  Newspaper,
  RefreshCw,
  ShieldAlert,
  SlidersHorizontal,
  TrendingDown,
  TrendingUp,
  Users
} from 'lucide-react';
import { api } from './api.js';
import { backtestMetric, backtestReturn, entryLabel, isNetBacktest } from './backtestPresentation.js';
import KLineChart from './components/KLineChart.jsx';
import { lineKey, snapshotMatchesBars } from './components/chartLines.js';
import { flagDirectionLabels, patternMatches, precisionEngineLabels } from './flagPatterns.js';
import FlagBreakdown from './components/FlagBreakdown.jsx';
import PatternTags from './components/PatternTags.jsx';
import { resistanceEvidence } from './resultEvidence.js';
import StrategyComparison from './components/StrategyComparison.jsx';

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

function number(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return Number(value).toLocaleString('zh-CN', { maximumFractionDigits: digits, minimumFractionDigits: digits === 3 ? 2 : 0 });
}

function percent(value) {
  if (value === null || value === undefined) return '-';
  const numeric = Number(value);
  return `${numeric > 0 ? '+' : ''}${numeric.toFixed(2)}%`;
}

function rate(value) {
  if (value === null || value === undefined) return '-';
  return `${Number(value).toFixed(1)}%`;
}

function lineTypeLabel(type) {
  if (type === 'role_reversal') return '长周期阻力转支撑';
  if (type === 'resistance_support') return '支撑 / 阻力验证';
  if (type === 'support') return '支撑平台';
  return '阻力平台';
}

// Replay windows limit evaluated signal dates, not the history used to form lines.
const BACKTEST_WINDOWS = [
  { days: 250, label: '1年' },
  { days: 750, label: '3年' },
  { days: 0, label: '全部' }
];

// 补充板块的名单只写一处：早先初始 state 和 load() 里的重置各写了一遍，
// 新增 relatedBoard 时漏掉后者，JSX 读 sections.relatedBoard.status 直接抛错白屏。
const DETAIL_SECTIONS = ['company', 'news', 'industry', 'relatedBoard', 'backtest'];

function sectionStates(status, overrides = {}) {
  return Object.fromEntries(
    DETAIL_SECTIONS.map((name) => [name, { status: overrides[name] || status, error: '' }])
  );
}

function withAlertMarkers(chart, markers) {
  // 板块图不画个股价位，只保留日期标记
  if (!chart) return chart;
  return {
    ...chart,
    markers: [
      ...(chart.markers || []).filter((marker) => marker.type !== 'alert'),
      ...markers.map(({ price, ...marker }) => marker)
    ]
  };
}

function historicalMarkers(backtest) {
  return (backtest?.signals || []).map((item) => ({
    date: item.signal_date,
    type: 'alert',
    label: '历史预警',
    price: item.signal_price
  }));
}

function sectionMeta(section, fallback = '') {
  if (section?.status === 'loading') return '加载中...';
  if (section?.status === 'error') return `加载失败：${section.error || '请稍后刷新'}`;
  return fallback || '';
}

export default function StockDetailPage({ stockCode, stockName = '', scanId = '', precisionEngine = '', asOf = '' }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingSeconds, setLoadingSeconds] = useState(0);
  const [error, setError] = useState('');
  const [watchState, setWatchState] = useState('idle');
  const [researchTab, setResearchTab] = useState('boards');
  const [linkedRange, setLinkedRange] = useState(null);
  const [linkedCursor, setLinkedCursor] = useState(null);
  const [selectedLineKeys, setSelectedLineKeys] = useState([]);
  const [selectedFlag, setSelectedFlag] = useState(null);
  const [selectedPatternType, setSelectedPatternType] = useState('');
  const [engineChoice, setEngineChoice] = useState(precisionEngine);
  const [analysisDate, setAnalysisDate] = useState(asOf);
  const [comparison, setComparison] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [selectedHistoricalSignal, setSelectedHistoricalSignal] = useState(null);
  const historicalBasisMatches = useMemo(() => !selectedHistoricalSignal?.line_snapshot
    || snapshotMatchesBars(selectedHistoricalSignal.line_snapshot, detail?.chart?.bars || []),
  [selectedHistoricalSignal, detail?.chart?.bars]);
  const precisionEvidence = useMemo(() => resistanceEvidence(selectedHistoricalSignal?.line_snapshot ? {
    ...selectedHistoricalSignal, category: 'buy_alert', pattern_type: 'precision_pullback',
    key_lines: [selectedHistoricalSignal.line_snapshot], chart: detail?.chart
  } : { ...detail?.analysis?.result, chart: detail?.chart }),
  [detail?.analysis?.result, detail?.chart, selectedHistoricalSignal]);
  const stockChart = useMemo(() => {
    const type = selectedPatternType || detail?.analysis?.result?.pattern_type;
    if (selectedFlag || type === 'flag_breakout') {
      return { ...detail?.chart, key_line: null, key_lines: [], flag_pattern: selectedFlag || detail?.chart?.flag_pattern };
    }
    if (selectedHistoricalSignal?.line_snapshot && detail?.chart) {
      const frozen = selectedHistoricalSignal.line_snapshot;
      return {
        ...detail.chart, flag_pattern: null,
        key_line: historicalBasisMatches ? frozen.price : null,
        key_lines: historicalBasisMatches ? [{ ...frozen, is_primary: true }] : [],
        markers: [
          ...(detail.chart.markers || []).filter(marker => marker.type === 'alert'),
          ...(historicalBasisMatches && frozen.breakout_date ? [{ date: frozen.breakout_date, type: 'breakout', label: '历史突破' }] : []),
        ],
      };
    }
    return detail?.chart ? { ...detail.chart, flag_pattern: null } : undefined;
  }, [detail?.chart, detail?.analysis?.result?.pattern_type, selectedFlag, selectedPatternType, selectedHistoricalSignal, historicalBasisMatches]);
  const [showSwings, setShowSwings] = useState(false);
  const [showAlerts, setShowAlerts] = useState(false);
  const [backtestWindow, setBacktestWindow] = useState(250);
  const [backtestSeconds, setBacktestSeconds] = useState(0);
  const [sections, setSections] = useState(() => sectionStates('idle'));
  const loadSequenceRef = useRef(0);

  function toggleLine(line) {
    const key = lineKey(line);
    setSelectedLineKeys((current) =>
      current.includes(key) ? [] : [key]
    );
  }

  function selectFlag(pattern) {
    setSelectedHistoricalSignal(null);
    setSelectedFlag(pattern);
    if (pattern) setSelectedPatternType('flag_breakout');
    if (pattern) {
      setLinkedRange({ source: 'flag-selection', startDate: pattern.pole_start_date, endDate: pattern.signal_date });
    } else {
      const bars = detail?.chart?.bars || [];
      if (bars.length) setLinkedRange({ source: 'flag-selection', startDate: bars[Math.max(0, bars.length - 490)].date, endDate: bars[bars.length - 1].date });
    }
  }

  function selectPattern(type) {
    setSelectedHistoricalSignal(null);
    setSelectedPatternType(type);
    setSelectedFlag(null);
    const flag = detail?.analysis?.result?.flag_pattern;
    const bars = detail?.chart?.bars || [];
    if (type === 'flag_breakout' && flag) {
      setLinkedRange({ source: 'pattern-selection', startDate: flag.pole_start_date, endDate: flag.signal_date });
    } else if (bars.length) {
      setLinkedRange({ source: 'pattern-selection', startDate: bars[Math.max(0, bars.length - 490)].date, endDate: bars[bars.length - 1].date });
    }
  }

  async function load(options = {}) {
    const requestedEngine = options.engine ?? engineChoice;
    const requestedDate = options.asOf ?? analysisDate;
    const requestId = loadSequenceRef.current + 1;
    const previousSections = sections;
    const hadDetail = Boolean(detail);
    loadSequenceRef.current = requestId;
    setLoading(true);
    setError('');
    setWatchState('idle');
    setShowAlerts(false);
    setSelectedFlag(null);
    setSelectedPatternType('');
    setSelectedHistoricalSignal(null);
    setComparison(null);
    setLinkedRange(null);
    setLinkedCursor(null);
    // 复盘是按需触发的，别跟着首屏转圈
    setSections(sectionStates('loading', { backtest: 'idle' }));
    try {
      const core = await api.stockDetail(stockCode, stockName, scanId, requestedEngine, requestedDate);
      if (loadSequenceRef.current !== requestId) return;
      setDetail(core);
      setEngineChoice(core.precision_engine || 'legacy');
      setAnalysisDate(core.analysis_as_of || '');
      const primaryLine = core.chart?.key_lines?.find(line => line.is_primary);
      setSelectedLineKeys(primaryLine ? [lineKey(primaryLine)] : []);
      const stockBars = core.chart?.bars || [];
      if (stockBars.length) {
        const flag = core.analysis?.result?.pattern_type === 'flag_breakout' ? core.analysis.result.flag_pattern : null;
        const poleIndex = flag ? stockBars.findIndex(bar => bar.date >= flag.pole_start_date) : -1;
        const startIndex = Math.max(0, poleIndex >= 0 ? poleIndex - 5 : stockBars.length - 490);
        setLinkedRange({
          source: flag ? 'flag-initial' : 'stock',
          startDate: stockBars[startIndex].date,
          endDate: stockBars[stockBars.length - 1].date,
          presetBars: flag ? 0 : 490
        });
      }
      setLoading(false);
      if (core.analysis_as_of) {
        setSections(sectionStates('idle'));
      } else {
        void loadCompany(requestId);
        void loadNews(requestId);
        void loadIndustry(requestId);
        void loadRelatedBoard(requestId);
      }
    } catch (err) {
      if (loadSequenceRef.current !== requestId) return;
      setError(err.message);
      if (hadDetail) setSections(previousSections);
    } finally {
      if (loadSequenceRef.current === requestId) setLoading(false);
    }
  }

  async function loadComparison() {
    const requestId = loadSequenceRef.current;
    setComparing(true);
    try {
      const result = await api.stockComparison(stockCode, stockName, scanId,
        detail?.analysis_as_of || detail?.analysis?.result?.signal_date);
      if (requestId === loadSequenceRef.current) setComparison(result);
    } catch (err) {
      if (requestId === loadSequenceRef.current) setError(err.message);
    } finally {
      setComparing(false);
    }
  }

  function selectHistorical(item) {
    requestAnimationFrame(() => document.getElementById('price-chart')?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    if (!item.line_snapshot) {
      if (item.flag_pattern) selectFlag(item.flag_pattern);
      else {
        setAnalysisDate(item.signal_date);
        void load({ asOf: item.signal_date, engine: item.precision_engine || 'legacy' });
      }
      return;
    }
    setSelectedFlag(null);
    setSelectedPatternType('precision_pullback');
    setSelectedHistoricalSignal(item);
    setSelectedLineKeys([lineKey(item.line_snapshot)]);
    const bars = detail?.chart?.bars || [];
    const index = bars.findIndex(bar => bar.date === item.signal_date);
    if (index >= 0) setLinkedRange({
      source: 'historical-selection', startDate: bars[Math.max(0, index - 120)].date,
      endDate: bars[Math.min(bars.length - 1, index + 30)].date,
    });
  }

  function updateSection(name, status, error = '') {
    setSections((current) => ({ ...current, [name]: { status, error } }));
  }

  async function loadCompany(requestId) {
    try {
      const payload = await api.stockCompany(stockCode);
      if (loadSequenceRef.current !== requestId) return;
      setDetail((current) => current && ({
        ...current,
        company: payload.company || {},
        shareholders: payload.shareholders || current.shareholders,
        company_data: {
          source: payload.source,
          updated_at: payload.updated_at,
          stale: payload.stale || false,
          message: payload.message || ''
        }
      }));
      updateSection('company', 'ready');
    } catch (err) {
      if (loadSequenceRef.current !== requestId) return;
      setDetail((current) => current && ({
        ...current,
        company_data: { ...current.company_data, message: err.message || '公司与股东数据加载失败' }
      }));
      updateSection('company', 'error', err.message);
    }
  }

  async function loadNews(requestId) {
    try {
      const payload = await api.stockNews(stockCode);
      if (loadSequenceRef.current !== requestId) return;
      setDetail((current) => current && ({ ...current, news: payload }));
      updateSection('news', 'ready');
    } catch (err) {
      if (loadSequenceRef.current !== requestId) return;
      setDetail((current) => current && ({
        ...current,
        news: { items: [], source: null, updated_at: null, message: err.message || '新闻加载失败' }
      }));
      updateSection('news', 'error', err.message);
    }
  }

  async function loadRelatedBoard(requestId) {
    updateSection('relatedBoard', 'loading');
    try {
      const payload = await api.stockRelatedBoard(stockCode);
      if (loadSequenceRef.current !== requestId) return;
      setDetail((current) => current && ({
        ...current,
        related_board: { ...payload, chart: withAlertMarkers(payload.chart, historicalMarkers(current.backtest)) }
      }));
      updateSection('relatedBoard', 'ready');
    } catch (err) {
      if (loadSequenceRef.current !== requestId) return;
      updateSection('relatedBoard', 'error', err.message);
    }
  }

  async function loadIndustry(requestId) {
    updateSection('industry', 'loading');
    try {
      const payload = await api.stockIndustryMarket(stockCode);
      if (loadSequenceRef.current !== requestId) return;
      setDetail((current) => {
        if (!current) return current;
        const markers = historicalMarkers(current.backtest);
        return {
          ...current,
          industry_market: { ...payload, chart: withAlertMarkers(payload.chart, markers) }
        };
      });
      updateSection('industry', 'ready');
    } catch (err) {
      if (loadSequenceRef.current !== requestId) return;
      setDetail((current) => current && ({
        ...current,
        industry_market: { ...current.industry_market, message: err.message || '板块行情加载失败' }
      }));
      updateSection('industry', 'error', err.message);
    }
  }

  async function loadBacktest(lookbackDays = backtestWindow, requestId = loadSequenceRef.current) {
    if (sections.backtest.status === 'loading') return;
    setSelectedFlag(null);
    setSelectedHistoricalSignal(null);
    setBacktestWindow(lookbackDays);
    setShowAlerts(true);
    updateSection('backtest', 'loading');
    try {
      const payload = await api.stockBacktest(stockCode, lookbackDays, scanId,
        detail?.precision_engine, detail?.analysis_as_of);
      if (loadSequenceRef.current !== requestId) return;
      const markers = historicalMarkers(payload);
      setDetail((current) => current && ({
        ...current,
        backtest: payload,
        chart: {
          ...current.chart,
          markers: [
            ...(current.chart?.markers || []).filter((marker) => marker.type !== 'alert'),
            ...markers
          ]
        },
        industry_market: {
          ...current.industry_market,
          chart: withAlertMarkers(current.industry_market?.chart, markers)
        },
        related_board: {
          ...current.related_board,
          chart: withAlertMarkers(current.related_board?.chart, markers)
        }
      }));
      updateSection('backtest', 'ready');
    } catch (err) {
      if (loadSequenceRef.current !== requestId) return;
      setDetail((current) => current && ({
        ...current,
        backtest: { signals: [], summary: { signal_count: 0 }, methodology: err.message || '历史复盘加载失败' }
      }));
      updateSection('backtest', 'error', err.message);
    }
  }

  useEffect(() => {
    let started = false;
    const startWhenVisible = () => {
      if (started || document.visibilityState !== 'visible') return;
      started = true;
      load();
    };
    startWhenVisible();
    document.addEventListener('visibilitychange', startWhenVisible);
    return () => {
      document.removeEventListener('visibilitychange', startWhenVisible);
      loadSequenceRef.current += 1;
    };
  }, [stockCode, stockName, scanId]);

  useEffect(() => {
    if (!loading || detail) {
      setLoadingSeconds(0);
      return undefined;
    }
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setLoadingSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [loading, detail, stockCode]);

  useEffect(() => {
    if (sections.backtest.status !== 'loading') {
      setBacktestSeconds(0);
      return undefined;
    }
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setBacktestSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [sections.backtest.status, stockCode]);

  function toggleHistoricalAlerts(event) {
    const checked = event.target.checked;
    setShowAlerts(checked);
    if (checked && !['loading', 'ready'].includes(sections.backtest.status)) {
      void loadBacktest(backtestWindow);
    }
  }

  async function addToWatch() {
    const stock = detail?.stock;
    if (detail?.data_quality?.status === 'stale') return;
    if (!stock || detail?.analysis_as_of || loading) return;
    setWatchState('loading');
    try {
      await api.addWatch({
        stock_code: stock.stock_code,
        stock_name: stock.stock_name,
        note: detail.analysis?.result?.suggestion || '从个股详情加入',
        status: 'active',
        entry_price: detail.market?.close || undefined,
        entry_date: detail.market?.trade_date || undefined
      });
      setWatchState('done');
    } catch (err) {
      setWatchState('idle');
      setError(err.message);
    }
  }

  function goBack() {
    if (window.opener) window.close();
    else window.location.href = '/';
  }

  if (loading && !detail) {
    const loadingHint = loadingSeconds >= 20
      ? '数据源响应较慢，超过 45 秒会停止并显示重试按钮'
      : loadingSeconds >= 8
        ? '正在分析全历史精准线，首次打开通常需要 5-20 秒'
        : '正在获取完整日线并计算精准线';
    return (
      <main className="stock-page stock-page-loading">
        <Loader2 className="spin" size={28} />
        <strong>正在加载 {stockCode} 完整行情 · {loadingSeconds} 秒</strong>
        <span>{loadingHint}</span>
      </main>
    );
  }

  if (!detail) {
    return (
      <main className="stock-page">
        <header className="stock-nav">
          <button className="ghost-button" onClick={goBack}><ArrowLeft size={16} /> 返回工作台</button>
        </header>
        <div className="detail-error">
          <AlertTriangle size={24} />
          <div><strong>个股详情加载失败</strong><p>{error || '未知错误'}</p></div>
          <button className="primary-button" onClick={load}><RefreshCw size={16} /> 重试</button>
        </div>
      </main>
    );
  }

  const stock = detail.stock || {};
  const analysis = detail.analysis || {};
  const primarySignal = analysis.result;
  const matches = patternMatches(primarySignal);
  const activeType = selectedPatternType || primarySignal?.pattern_type || 'precision_pullback';
  const activeMatch = matches.find(match => match.pattern_type === activeType);
  const liveSignal = activeMatch ? { ...primarySignal, ...activeMatch } : primarySignal;
  const historicalMatch = selectedHistoricalSignal?.pattern_matches?.find(match => match.pattern_type === 'precision_pullback');
  const signal = selectedHistoricalSignal ? {
    ...selectedHistoricalSignal, ...historicalMatch,
    key_line_price: selectedHistoricalSignal.line_snapshot.price,
    line_strength: selectedHistoricalSignal.line_snapshot.point_count,
    key_lines: stockChart?.key_lines || [],
  } : liveSignal;
  const precisionLines = signal?.key_lines || detail.chart?.key_lines || [];
  const market = detail.market || {};
  const company = detail.company || {};
  const shareholders = detail.shareholders || {};
  const backtest = detail.backtest || { summary: {}, signals: [] };
  // 文案要跟着实际选的窗口走，否则选了 1 年却写着"全部历史"
  const backtestWindowLabel =
    BACKTEST_WINDOWS.find((item) => item.days === backtestWindow)?.label || '全部';
  const related = detail.related_board || {};
  const summary = backtest.summary || {};
  const flagMode = Boolean(selectedFlag) || activeType === 'flag_breakout' || detail.strategy_mode === 'flag_breakout';
  const displayedFlag = selectedFlag || signal?.flag_pattern;
  const netBacktest = isNetBacktest(summary);
  const backtestMetrics = [5, 10, 20].map((window) => backtestMetric(summary, window));
  const changePositive = Number(market.change_pct || 0) >= 0;
  const historicalView = Boolean(detail.analysis_as_of);
  const quoteIsStale = detail.data_quality?.status === 'stale';
  const activeResearchTab = historicalView ? 'backtest' : researchTab;
  const researchTabs = historicalView ? [{ id: 'backtest', label: '历史预警', Icon: Database }] : [
    { id: 'boards', label: '板块对照', Icon: LineChart },
    { id: 'backtest', label: '历史预警', Icon: Database },
    { id: 'company', label: '公司与股东', Icon: Users },
    { id: 'news', label: '新闻', Icon: Newspaper }
  ];
  const focusCategory = selectedHistoricalSignal ? 'buy_alert' : signal?.category;
  const focusTitle = quoteIsStale && !selectedHistoricalSignal ? '行情待更新，暂停预警' : ({
    candidate: '价位带待验证', confirmed: '等待有效突破', breakout_observation: '已上穿，等待突破确认',
    awaiting_pullback: '已突破，等待回踩', pullback_alert: '回踩预警已确认',
    invalid: '本次突破已失效', rejected: '突破前路径未通过', expired: '本轮观察已到期',
    reforming: '等待重新验证', no_trading: '当日无有效成交',
  }[signal?.platform_state] || signal?.suggestion || analysis.message || '尚无有效信号');
  const industryState = historicalView ? '未回填' : detail.industry?.stale ? '快照过期'
    : !detail.industry || detail.industry.validation_status !== 'verified' ? '未核验'
      : ({ high: '风险偏高', medium: '中性', low: '偏强' }[detail.industry.risk_level] || '未覆盖');
  const focusRisks = [...new Set([
    ...(signal?.risks || []),
    ...(!historicalView && ['未核验', '快照过期'].includes(industryState)
      ? [signal?.industry_risk_message || detail.industry?.message || '行业归属未核验，行业分未计入'] : [])
  ].filter(Boolean))].slice(0, 2);
  const evidenceReasons = (signal?.reasons || []).filter(reason =>
    !signal?.line_snapshot?.version_id || !reason.includes(signal.line_snapshot.version_id));

  return (
    <main className="stock-page">
      <header className="stock-nav">
        <button className="ghost-button" onClick={goBack}><ArrowLeft size={16} /> 返回工作台</button>
        <div className="stock-nav-brand"><LineChart size={18} /> Stock Mine 个股研究</div>
        <div className="stock-nav-actions">
          <button className="ghost-button" onClick={load} disabled={loading} aria-label="刷新数据" title="刷新数据">
            <RefreshCw className={loading ? 'spin' : ''} size={16} /> <span className="action-label">{loading ? '刷新中' : '刷新数据'}</span>
          </button>
          <button className="primary-button" onClick={addToWatch} disabled={watchState !== 'idle' || loading || historicalView || quoteIsStale}
            title={historicalView ? '历史价格不能作为当前观察池的加入价格' : '加入观测池'}>
            {watchState === 'loading' ? <Loader2 className="spin" size={16} /> : watchState === 'done' ? <CheckCircle2 size={16} /> : <BookmarkPlus size={16} />}
            {watchState === 'done' ? '已加入观测池' : '加入观测池'}
          </button>
        </div>
      </header>

      <section className="stock-identity">
        <div>
          <div className="stock-title-line">
            <h1>{stock.stock_name}</h1>
            <span>{stock.stock_code}</span>
            <span className={categoryClass(primarySignal?.category)}>{categoryLabel(primarySignal?.category)}</span>
          </div>
          <p>{company.full_name || company.main_business || '中国 A 股上市公司'}</p>
        </div>
        <div className={`quote-block ${changePositive ? 'quote-up' : 'quote-down'}`}>
          <strong>{number(market.close, 3)}</strong>
          <span>{changePositive ? <TrendingUp size={17} /> : <TrendingDown size={17} />}{percent(market.change_pct)}</span>
          <small>{historicalView ? '历史行情' : detail.data_quality?.provisional_last_bar ? '盘中行情' : '行情截至'} {market.trade_date || detail.end_date}</small>
        </div>
      </section>

      <section className={`decision-focus decision-${focusCategory || 'none'}`} aria-label="关键结论">
        <div className="decision-heading">
          <div>
            {focusCategory === 'buy_alert' ? <CheckCircle2 size={20} /> : <ShieldAlert size={20} />}
            <h2>{focusTitle}</h2>
            <span>{selectedHistoricalSignal ? '历史预警' : '判定日'} {signal?.signal_date || primarySignal?.signal_date || '-'}</span>
          </div>
          <span className="decision-score" title={signal?.score_type === 'structure' ? '形态结构分；行业只作提示、不计分；不代表胜率' : `旧批次综合分：结构分 ${number(signal?.structure_score ?? signal?.score, 0)} · 行业修正 ${number(signal?.industry_adjustment, 1)} · 不代表胜率`}>
            {signal?.score_type === 'structure' ? '结构分' : '综合分'} <b>{number(signal?.score ?? signal?.structure_score, 0)}</b>
          </span>
        </div>
        <div className="decision-facts">
          <Fact label={flagMode ? '旗面上轨' : '判定线'} value={number(flagMode ? displayedFlag?.upper_price : signal?.key_line_price, 3)} />
          <Fact label={selectedHistoricalSignal ? '预警日距线' : '判定日距线'} value={percent(signal?.distance_to_line_pct)} />
          <Fact label={flagMode ? '旗杆涨幅' : '独立验证'} value={flagMode ? percent(displayedFlag?.pole_gain_pct) : signal?.line_strength ? `${signal.line_strength} 点` : '未形成'} />
          <Fact label="突破确认" value={signal?.breakout_date || '尚未确认'} />
          <Fact label={flagMode ? '旗面长度' : '回踩状态'} value={flagMode ? displayedFlag ? `${displayedFlag.flag_bars} 根` : '-'
            : signal?.pullback_check?.passed ? '已确认' : signal?.breakout_date ? '等待回踩' : '尚未突破'} />
          <Fact label="行业环境" value={industryState} />
        </div>
        {focusRisks.length > 0 && <div className="decision-risks">
          <AlertTriangle size={14} />
          <div>{focusRisks.map(risk => <span key={risk}>{risk}</span>)}</div>
        </div>}
      </section>

      {matches.length > 0 && <section className="matched-pattern-band">
        <span>命中形态</span>
        <PatternTags row={primarySignal} activeType={activeType} onSelect={selectPattern} />
      </section>}

      <details className="analysis-settings">
        <summary><SlidersHorizontal size={14} /><span>分析口径</span>
          <small>{precisionEngineLabels[detail.precision_engine] || '-'} · {historicalView ? detail.analysis_as_of : primarySignal?.signal_date || '-'}</small>
          <ChevronDown size={14} />
        </summary>
        <section className="analysis-controls" aria-label="分析口径">
        <label>点位引擎
          <select aria-label="个股点位引擎" value={engineChoice} disabled={loading}
            onChange={event => { setEngineChoice(event.target.value); void load({ engine: event.target.value }); }}>
            {Object.entries(precisionEngineLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label>分析截至
          <input type="date" aria-label="历史分析日期" value={analysisDate}
            onChange={event => setAnalysisDate(event.target.value)} disabled={loading} />
        </label>
        <button type="button" className="icon-button" title="应用分析日期" aria-label="应用分析日期"
          onClick={() => load()} disabled={loading}><CalendarDays size={16} /></button>
        {historicalView && <button type="button" className="ghost-button" onClick={() => load({ asOf: '' })} disabled={loading}>
          <RefreshCw size={14} />最新分析
        </button>}
        <button type="button" className="ghost-button" onClick={loadComparison} disabled={loading || comparing}>
          {comparing ? <Loader2 size={14} className="spin" /> : <Database size={14} />}引擎对照
        </button>
        <span>{historicalView ? `历史时点 ${detail.analysis_as_of}` : `信号日 ${primarySignal?.signal_date || '-'}`} · {detail.strategy_version || '-'}</span>
      </section>
      </details>
      {error && <div className="error-strip detail-inline-error"><AlertTriangle size={16} />操作失败，当前页面数据仍可查看：{error}</div>}
      {historicalView && <div className="warning-strip"><CalendarDays size={16} />
        仅使用 {detail.end_date} 及以前日线；当前前复权缓存口径，不回填当前行业、股东和新闻数据。
      </div>}
      {selectedHistoricalSignal && <div className="historical-selection-strip">
        <span>历史信号 {selectedHistoricalSignal.signal_date} · 线版本 {selectedHistoricalSignal.line_snapshot.version_id}</span>
        <button type="button" className="icon-button" title="返回当前结构" onClick={() => {
          setSelectedHistoricalSignal(null);
          const primary = detail.chart?.key_lines?.find(line => line.is_primary);
          setSelectedLineKeys(primary ? [lineKey(primary)] : []);
        }}><RefreshCw size={14} /></button>
      </div>}
      {!historicalBasisMatches && <div className="warning-strip"><AlertTriangle size={16} />
        历史线与当前行情的复权基准不一致，已暂停叠加旧线；历史记录未改写。
      </div>}
      {detail.data_quality?.status === 'stale' && <div className="warning-strip" role="status">
        <AlertTriangle size={16} />行情更新失败，数据截至 {detail.data_quality.last_bar_date || '-'}，交易预警已暂停。
      </div>}
      {detail.scan_context?.recomputed_with_new_strategy && <div className="warning-strip">
        <Database size={16} />旧批次 #{detail.scan_context.scan_id} 的参数 · 当前策略版本重算，非原始批次结论。
      </div>}

      {comparison && <StrategyComparison comparison={comparison}
        onSelect={engine => { setEngineChoice(engine); void load({ engine }); }} />}

      <section className="quote-metrics">
        <QuoteMetric label={historicalView ? '当日开盘' : '今开'} value={number(market.open, 3)} />
        <QuoteMetric label="最高" value={number(market.high, 3)} tone="up" />
        <QuoteMetric label="最低" value={number(market.low, 3)} tone="down" />
        <QuoteMetric label="成交量" value={number(market.volume, 0)} />
      </section>

      <section className="stock-analysis-grid">
        <div className="section-panel chart-panel" id="price-chart">
          <SectionHeading icon={<Activity size={18} />} title="个股完整 K 线" meta={`${detail.start_date || '-'} 至 ${detail.end_date || '-'} · ${detail.bar_count || 0} 根`} />
          <div className="chart-toggles" role="group" aria-label="K线图层开关">
            <label>
              <input type="checkbox" checked={showAlerts} onChange={toggleHistoricalAlerts} />
              历史预警
            </label>
            <span className="backtest-picker">
              {sections.backtest.status === 'loading'
                ? <span className="chart-backtest-state"><Loader2 className="spin" size={14} /> 复盘中 {backtestSeconds} 秒</span>
                : sections.backtest.status === 'ready'
                  ? <span className="chart-backtest-state ok"><CheckCircle2 size={14} /> {summary.signal_count || 0} 个历史预警</span>
                  : <span className="chart-backtest-state"><Database size={14} /> 复盘</span>}
              {BACKTEST_WINDOWS.map((option) => (
                <button
                  type="button"
                  key={option.days}
                  className={sections.backtest.status === 'ready' && backtestWindow === option.days ? 'active' : ''}
                  onClick={() => loadBacktest(option.days)}
                  disabled={sections.backtest.status === 'loading'}
                  title={`回放 ${option.label} 历史预警`}
                >
                  {option.label}
                </button>
              ))}
            </span>
            <span className="chart-toggle-hint">
              {flagMode ? '旗形实验规则' : '阻力取最高价 · 支撑取最低价'}
            </span>
            {!flagMode && precisionLines.length > 0 && <label className="chart-line-picker">精准线
              <select aria-label="图表精准线" value={selectedLineKeys[0] || ''}
                onChange={event => setSelectedLineKeys(event.target.value ? [event.target.value] : [])}>
                <option value="">不显示精准点</option>
                {precisionLines.map(line => <option key={lineKey(line)} value={lineKey(line)}>
                  {number(line.price, 3)} · {line.point_count} 点
                </option>)}
              </select>
            </label>}
          </div>
          <KLineChart
            chart={stockChart}
            full
            chartId="stock"
            linkedRange={linkedRange}
            linkedCursor={linkedCursor}
            selectedLineKeys={selectedLineKeys}
            showSwings={showSwings}
            showAlerts={showAlerts}
            precisionEvidence={precisionEvidence}
            onRangeChange={setLinkedRange}
            onCursorChange={setLinkedCursor}
          />
        </div>
        <aside className={`section-panel signal-panel ${flagMode ? 'signal-panel-flag' : ''}`}>
          <SectionHeading icon={<ShieldAlert size={18} />} title={selectedHistoricalSignal ? '历史预警证据' : '关键证据'} />
          {flagMode && <details className="strategy-parameters">
            <summary><SlidersHorizontal size={14} />旗形参数与时间证据<ChevronDown size={14} /></summary>
            <div className="flag-detail">
            <div className="flag-detail-heading">
              <h3>{displayedFlag ? flagDirectionLabels[displayedFlag.direction] : '未找到旗形'}</h3>
              {selectedFlag && <button type="button" className="icon-button" title="返回当前旗形" onClick={() => selectFlag(null)}><RefreshCw size={15} /></button>}
            </div>
            {displayedFlag && <>
              <p className="method-note">{selectedFlag ? '历史结构' : '当前结构'} · {displayedFlag.signal_date}</p>
              <div className="signal-facts">
                <Fact label="旗杆涨幅" value={percent(displayedFlag.pole_gain_pct)} />
                <Fact label="整理长度" value={`${displayedFlag.flag_bars} 根`} />
                <Fact label="旗面上轨" value={number(displayedFlag.upper_price, 3)} />
                <Fact label="旗面下轨" value={number(displayedFlag.lower_price, 3)} />
                <Fact label="回撤/旗杆" value={percent(displayedFlag.retracement_pct)} />
                <Fact label="宽度/旗杆" value={percent(displayedFlag.width_to_pole_pct)} />
                <Fact label="整理后/前量比" value={number(displayedFlag.volume_contraction_ratio)} />
                <Fact label="突破量比" value={number(displayedFlag.breakout_volume_ratio)} />
                <Fact label="旗杆起点" value={displayedFlag.pole_start_date} />
                <Fact label="旗杆终点" value={displayedFlag.pole_end_date} />
                <Fact label="拟合截止日" value={displayedFlag.fit_end_date} />
                <Fact label="上轨/下轨触碰" value={`${displayedFlag.upper_touch_count} / ${displayedFlag.lower_touch_count}`} />
              </div>
            </>}
          </div></details>}
          {!flagMode && <>
          <details className="strategy-parameters">
          <summary><SlidersHorizontal size={14} />规则参数与时间证据<ChevronDown size={14} /></summary>
          <div className="signal-facts">
            <Fact label="结构分" value={number(signal?.structure_score ?? signal?.score, 0)} />
            {signal?.score_type === 'structure'
              ? <Fact label="行业环境（不计分）" value={signal?.industry_risk_message || '-'} wide />
              : <Fact label="行业修正" value={number(signal?.industry_adjustment, 1)} />}
            <Fact label="大盘环境" value={!signal?.market_regime ? '未评估' : signal.market_regime.status !== 'evaluated' ? '缺少当日指数' : `${signal.market_regime.weak ? '偏弱' : '正常'}（近20日 ${percent(signal.market_regime.return_20_pct)}）`} />
            <Fact label="回踩量/常态量" value={number(signal?.pullback_normal_volume_ratio)} />
            <Fact label="涨停突破" value={signal?.breakout_limit_up == null ? '-' : signal.breakout_limit_up ? '是' : '否'} />
            {signal?.platform_state && <>
              <Fact label="平台阶段" value={({ candidate: '候选', confirmed: '已验证', breakout_observation: '上穿观察',
                awaiting_pullback: '等待回踩', pullback_alert: '回踩预警', invalid: '突破失效',
                rejected: '风险过滤', expired: '观察到期', reforming: '重新验证' }[signal.platform_state] || signal.platform_state)} />
              <Fact label="独立事件数" value={signal.line_snapshot?.point_count ?? '-'} />
              <Fact label="线版本可用日" value={signal.line_snapshot?.version_created_at || '-'} />
              <Fact label="线版本" value={signal.line_snapshot?.version_id || '-'} wide />
              <Fact label="突破前已确认" value={signal.line_snapshot?.confirmed_before_breakout ? '是' : '否 / 尚未突破'} />
            </>}
            <Fact label="精准线" value={number(signal?.key_line_price, 3)} />
            <Fact label="当前距离" value={percent(signal?.distance_to_line_pct)} />
            <Fact label="主线精准点" value={signal?.line_strength ? `${signal.line_strength} 点` : '-'} />
            <Fact label="信号日" value={signal?.signal_date || '-'} />
            <Fact label="突破日" value={signal?.breakout_date || '-'} />
            <Fact label="回踩日" value={signal?.pullback_date || '-'} />
            <Fact label="回调天数范围" value={signal?.pullback_check ? `${signal.pullback_check.min_days}～${signal.pullback_check.max_days} 交易日${signal.pullback_check.range_mode === 'legacy' ? '（旧版）' : ''}` : '-'} />
            <Fact label="距突破交易日数" value={signal?.pullback_check?.days_since_breakout == null ? '-' : `${signal.pullback_check.days_since_breakout} 交易日`} />
            <Fact label={signal?.precision_engine === 'platform_events' ? '预警要求回踩' : '只选突破后回踩'} value={signal?.pullback_check ? signal.pullback_check.required ? '开启' : '关闭' : '-'} />
            <Fact label="信号日回踩确认" value={signal?.pullback_check ? ({ confirmed: '已确认', no_breakout: '尚未突破', outside_window: '不在天数范围', awaiting_pullback: '尚未确认' }[signal.pullback_check.status] || '-') : '-'} />
            <Fact label="突破量比要求" value={signal?.require_breakout_volume == null ? '未记录' : signal.require_breakout_volume ? `≥ ${number(signal.breakout_volume_threshold)} 倍` : '未启用'} />
            <Fact label="实际突破量比" value={number(signal?.breakout_volume_ratio)} />
            <Fact label="突破前越线上限" value={signal?.prebreakout_check?.limit_pct == null ? '未启用' : `${number(signal.prebreakout_check.limit_pct)}%`} />
            <Fact label="突破前最高越线" value={signal?.prebreakout_check?.status === 'no_breakout' ? '待突破' : signal?.prebreakout_check?.max_above_line_pct == null ? '-' : `${number(signal.prebreakout_check.max_above_line_pct)}%`} />
            <Fact label="连续站上排除阈值" value={signal?.prebreakout_check?.reject_above_days == null ? '未启用' : `≥ ${signal.prebreakout_check.reject_above_days} 交易日`} />
            <Fact label="突破前最长连续站上" value={signal?.prebreakout_check?.reject_above_days == null ? '未启用' : signal.prebreakout_check.status === 'no_breakout' ? '待突破' : signal.prebreakout_check.max_consecutive_above_days == null ? '-' : `${signal.prebreakout_check.max_consecutive_above_days} 交易日`} />
            <Fact label="有效收线上门槛" value={signal?.prebreakout_check?.above_buffer_pct == null ? '未启用' : `收盘 > 线价 × ${number(1 + signal.prebreakout_check.above_buffer_pct / 100, 3)}`} />
            <Fact label="检查起点（不含）" value={signal?.prebreakout_check?.anchor_date || '-'} />
            <Fact label="最长连续起日" value={signal?.prebreakout_check?.longest_run_start_date || '-'} />
            <Fact label="最长连续止日" value={signal?.prebreakout_check?.longest_run_end_date || '-'} />
          </div>
          </details>
          <div className="precision-lines">
            <div className="precision-lines-heading">
              <h3>精准线结构</h3>
              <label className="swing-toggle">
                <input type="checkbox" checked={showSwings} onChange={(event) => setShowSwings(event.target.checked)} />
                <i className="swing-key swing-key-peak" aria-hidden="true" />
                <i className="swing-key swing-key-valley" aria-hidden="true" />
                波峰波谷
              </label>
            </div>
            {!precisionLines.length && <p className="precision-lines-hint">未找到精准线</p>}
            {precisionLines.map((line) => {
                const picked = selectedLineKeys.includes(lineKey(line));
                return (
                  <button
                    type="button"
                    className={`precision-line-row ${line.is_primary ? 'primary' : ''} ${picked ? 'picked' : ''}`}
                    key={lineKey(line)}
                    aria-pressed={picked}
                    title={`${picked ? '隐藏' : '查看'} ${number(line.price, 3)} 元线的精准点`}
                    onClick={() => toggleLine(line)}
                  >
                    <div>
                      <strong>{number(line.price, 3)}</strong>
                      {line.is_primary && <span className="precision-primary">{selectedHistoricalSignal ? '历史预警线' : signal?.category === 'buy_alert' ? '本次预警线' : '当前参考线'}</span>}
                    </div>
                    <b>{line.point_count} 点</b>
                    <small>{lineTypeLabel(line.line_type)} · 阻力 {line.resistance_point_count} · 支撑 {line.support_point_count}</small>
                    <time>{line.last_pivot_date || line.formation_date || '-'}</time>
                  </button>
                );
            })}
          </div>
          </>}
          <div className="evidence-reasons">
          <DetailList title="支持条件" items={selectedFlag ? [`${selectedFlag.direction_label}，旗杆涨幅 ${selectedFlag.pole_gain_pct}%`, `旗面 ${selectedFlag.flag_bars} 根，后/前量比 ${selectedFlag.volume_contraction_ratio}`] : evidenceReasons} tone="positive" />
          <DetailList title="风险条件" items={selectedFlag ? ['历史形态不代表下一次交易的胜率；成交与成本见回测'] : signal?.risks || [analysis.message]} tone="risk" />
          <div className={`industry-risk risk-${detail.industry?.stale ? 'uncovered' : detail.industry?.risk_level || 'uncovered'}`}>
            <strong>{detail.industry?.name || '行业未覆盖'}</strong>
            <span>{detail.industry?.stale ? `行业快照已过期 · ${detail.industry.updated_at || '时间未知'} · 未计入综合分` : detail.industry?.message || '暂无行业风险数据'}</span>
          </div>
          </div>
          <details className="quote-statistics">
            <summary><Activity size={14} />行情统计<ChevronDown size={14} /></summary>
            <div className="quote-metrics">
              <QuoteMetric label="20日量比" value={number(market.volume_ratio_20)} />
              <QuoteMetric label="52周高" value={number(market.high_52w, 3)} />
              <QuoteMetric label="52周低" value={number(market.low_52w, 3)} />
              <QuoteMetric label="近一年" value={percent(market.return_1y_pct)} tone={Number(market.return_1y_pct || 0) >= 0 ? 'up' : 'down'} />
            </div>
          </details>
        </aside>
      </section>

      <nav className="research-tabs" role="tablist" aria-label="个股研究视图">
        {researchTabs.map(({ id, label, Icon }, index) => <button
          key={id} id={`research-tab-${id}`} type="button" role="tab" aria-selected={activeResearchTab === id}
          aria-controls="research-content" tabIndex={activeResearchTab === id ? 0 : -1}
          onClick={() => setResearchTab(id)}
          onKeyDown={event => {
            const offsets = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: researchTabs.length - 1 - index };
            if (!(event.key in offsets)) return;
            event.preventDefault();
            const next = (index + offsets[event.key] + researchTabs.length) % researchTabs.length;
            setResearchTab(researchTabs[next].id);
            document.getElementById(`research-tab-${researchTabs[next].id}`)?.focus();
          }}><Icon size={16} />{label}{id === 'backtest' && sections.backtest.status === 'ready' && <small>{summary.signal_count || 0}</small>}
        </button>)}
      </nav>
      <section className="stock-research" id="research-content" role="tabpanel" aria-labelledby={`research-tab-${activeResearchTab}`}>
      {!historicalView && <>
      {activeResearchTab === 'boards' && <><section className="section-panel sector-panel">
        <SectionHeading
          icon={<LineChart size={18} />}
          title={`${detail.industry_market?.mapped_name || detail.industry?.name || '所属行业'}板块 K 线`}
          meta={detail.industry_market?.chart?.bars?.length
            ? sectionMeta(sections.industry, detail.industry_market?.source) : ''}
        />
        {detail.industry_market?.chart?.bars?.length > 0 ? <KLineChart
          chart={detail.industry_market?.chart}
          full
          chartId="industry"
          linkedRange={linkedRange}
          linkedCursor={linkedCursor}
          onRangeChange={setLinkedRange}
          onCursorChange={setLinkedCursor}
        /> : <SectionState status={sections.industry.status}
          message={sections.industry.status === 'loading' ? '板块行情加载中' : detail.industry_market?.message || sections.industry.error || '暂无匹配的行业指数'}
          onRetry={() => loadIndustry(loadSequenceRef.current)} />}
      </section>

      <section className="section-panel sector-panel">
        <SectionHeading
          icon={<LineChart size={18} />}
          title={related.board ? `最相关板块：${related.board.name}` : '最相关板块'}
          meta={related.chart?.bars?.length ? sectionMeta(
            sections.relatedBoard,
            related.board
              ? `${related.board.kind === 'concept' ? '概念' : '行业'} · 与个股日涨跌相关系数 ${related.board.correlation} · 从 ${related.universe_size || 0} 个板块中选出`
              : (related.message || '正在从全部行业与概念板块中匹配')
          ) : ''}
        />
        {related.building || !related.chart?.bars?.length ? (
          <SectionState status={related.building ? 'loading' : sections.relatedBoard.status}
            message={related.building ? '板块索引更新中' : sections.relatedBoard.status === 'loading' ? '相关板块匹配中' : related.message || sections.relatedBoard.error || '暂无相关板块行情'}
            onRetry={() => loadRelatedBoard(loadSequenceRef.current)} />
        ) : (
          <KLineChart
            chart={related.chart}
            full
            chartId="related"
            linkedRange={linkedRange}
            linkedCursor={linkedCursor}
            onRangeChange={setLinkedRange}
            onCursorChange={setLinkedCursor}
          />
        )}
      </section></>}

      {activeResearchTab === 'company' && <>
      <section className="detail-two-column">
        <div className="section-panel company-panel">
          <SectionHeading icon={<Building2 size={18} />} title="公司主要资料" meta={sectionMeta(sections.company, detail.company_data?.source || detail.company_data?.message)} />
          <div className="company-facts">
            <Fact label="所属行业" value={company.industry || detail.industry?.name || '-'} />
            <Fact label="上市市场" value={company.market || '-'} />
            <Fact label="法人代表" value={company.chairman || '-'} />
            <Fact label="成立日期" value={company.established_date || '-'} />
            <Fact label="上市日期" value={company.listing_date || '-'} />
            <Fact label="注册地址" value={company.registered_address || '-'} wide />
          </div>
          <div className="company-copy"><strong>主营业务</strong><p>{company.main_business || '暂无'}</p></div>
          <div className="company-copy"><strong>公司简介</strong><p>{company.introduction || '暂无'}</p></div>
        </div>

        <div className="section-panel shareholder-panel">
          <SectionHeading icon={<Users size={18} />} title="最新股东信息" meta={sectionMeta(sections.company, shareholders.as_of ? `截至 ${shareholders.as_of}` : detail.company_data?.message)} />
          <div className="holder-summary">
            <div><span>股东户数</span><b>{number(shareholders.holder_count, 0)}</b><small>{percent(shareholders.holder_count_change_pct)} 较上期</small></div>
            <div><span>前十集中度</span><b>{shareholders.top10_concentration_pct === null ? '-' : `${number(shareholders.top10_concentration_pct)}%`}</b><small>{shareholders.top10_concentration_change_pct === null ? '-' : `${percent(shareholders.top10_concentration_change_pct)} 个百分点`}</small></div>
            <div className={`holder-sentiment sentiment-${shareholders.sentiment || 'unknown'}`}><span>筹码标记</span><b>{shareholders.sentiment_label || '数据不足'}</b><small>仅按户数与集中度</small></div>
          </div>
          <DetailList title="变化判断" items={shareholders.signals || []}
            tone={shareholders.sentiment === 'negative' ? 'risk' : shareholders.sentiment === 'positive' ? 'positive' : 'neutral'} />
          {shareholders.data_quality?.status && shareholders.data_quality.status !== 'complete'
            && <p className="source-note">股东比较数据不完整，综合标记暂不判断；仅展示可核实分项。</p>}
          <p className="source-note">{shareholders.disclaimer}</p>
        </div>
      </section>

      <section className="section-panel holders-table-panel">
        <SectionHeading icon={<Users size={18} />} title="前十大股东" meta={sectionMeta(sections.company, `${shareholders.top_holders?.length || 0} 条`)} />
        <div className="table-wrap detail-table-wrap">
          <table>
            <thead><tr><th>股东名称</th><th>持股数量</th><th>持股比例</th><th>股份性质</th></tr></thead>
            <tbody>
              {(shareholders.top_holders || []).map((holder, index) => (
                <tr key={`${holder.name}-${index}`}>
                  <td>{holder.name}</td><td>{number(holder.shares, 0)}</td><td>{holder.ratio_pct === null ? '-' : `${number(holder.ratio_pct)}%`}</td><td>{holder.nature || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!shareholders.top_holders?.length && <div className="empty-row">暂无前十大股东数据</div>}
        </div>
      </section></>}</>}

      {(activeResearchTab === 'backtest' || activeResearchTab === 'news') && <section className={`detail-two-column detail-lower-grid ${historicalView ? 'single-column' : ''}`}>
        {activeResearchTab === 'backtest' && <div className="section-panel backtest-panel">
          <SectionHeading
            icon={<Database size={18} />}
            title="历史预警复盘"
            meta={sectionMeta(sections.backtest, sections.backtest.status === 'ready' ? `${summary.signal_count || 0} 个去重信号` : '尚未计算')}
          />
          {sections.backtest.status !== 'ready' && (
            <div className={`backtest-callout ${sections.backtest.status === 'error' ? 'error' : ''}`}>
              <div>
                <strong>{sections.backtest.status === 'loading' ? `正在逐交易日复盘${backtestWindowLabel}` : sections.backtest.status === 'error' ? `${backtestWindowLabel}复盘未完成` : '历史预警按需复盘'}</strong>
                <span>{sections.backtest.status === 'loading' ? `已用时 ${backtestSeconds} 秒 · 预警同步到三张 K 线图` : sections.backtest.error || `回看 ${backtestWindowLabel} · 当前点位引擎`}</span>
              </div>
              <button type="button" className="secondary-button" onClick={() => loadBacktest()} disabled={sections.backtest.status === 'loading'}>
                {sections.backtest.status === 'loading' ? <Loader2 className="spin" size={15} /> : <Database size={15} />}
                {sections.backtest.status === 'loading' ? '复盘中' : sections.backtest.status === 'error' ? '重新复盘' : `开始复盘${backtestWindowLabel}`}
              </button>
            </div>
          )}
          {sections.backtest.status === 'ready' && <><div className="backtest-metrics">
            {backtestMetrics.map((item) => <BacktestMetric key={item.label} label={item.label} value={rate(item.rate)} sub={`${item.count} 个样本${item.pending ? ` · ${item.pending} 个待退出` : ''}`} />)}
            <BacktestMetric label={netBacktest ? 'T+5 平均净收益' : 'T+5 平均涨跌'} value={percent(backtestMetrics[0].mean)} sub={netBacktest ? '开盘入场 · 扣成本' : '旧口径 · 未扣费'} />
          </div>
          {netBacktest && <p className="method-note">不可入场 {summary.blocked_entry_count || 0} 个 · 待入场 {summary.pending_entry_count || 0} 个</p>}
          {netBacktest && <div className="backtest-metrics">
            <BacktestMetric label="T+5 盈利因子" value={number(summary.net_t5_profit_factor)}
              sub={summary.net_t5_profit_factor_status === 'no_losses' ? '尚无亏损样本' : '净盈利合计 / 净亏损合计'} />
            <BacktestMetric label="T+5 平均盈亏比" value={number(summary.net_t5_payoff_ratio)} sub="平均盈利 / 平均亏损" />
            <BacktestMetric label="日线可入场比例" value={rate(summary.entry_feasibility_rate)}
              sub={`${summary.entry_feasibility_sample_count || 0} 次已到期入场 · 非排队成交保证`} />
            <BacktestMetric label="独立事件" value={summary.signal_count ?? '-'}
              sub={backtest.deduplication === 'breakout_event' ? `原始 ${backtest.raw_signal_day_count || 0} 个信号日` : '旧版20个交易日去重'} />
          </div>}
          <p className="method-note">{backtest.methodology}</p>
          <FlagBreakdown rows={summary.flag_directions} />
          <div className="table-wrap backtest-table-wrap">
            <table>
              <thead><tr><th>预警日</th><th>形态</th><th>信号收盘</th><th>模拟入场价</th><th>{netBacktest ? 'T+5净收益' : 'T+5涨跌'}</th><th>{netBacktest ? 'T+10净收益' : 'T+10涨跌'}</th><th>{netBacktest ? 'T+20净收益' : 'T+20涨跌'}</th><th>成交状态</th></tr></thead>
              <tbody>
                {(backtest.signals || []).map((item) => (
                  <tr key={item.signal_date}>
                    <td><button className="history-flag-button" type="button" onClick={() => selectHistorical(item)}
                      title={item.line_snapshot ? '恢复当时冻结的精准线与验证点' : item.flag_pattern ? '查看历史旗杆与通道' : '按该历史日期重新分析'}>{item.signal_date}</button></td>
                    <td className="pattern-cell"><PatternTags row={item} /></td><td>{number(item.signal_price, 3)}</td>
                    <td title={item.entry_date || ''}>{number(item.entry_price, 3)}</td>
                    {[5, 10, 20].map((window) => <td key={window} title={item[`exit_date_${window}`] ? `退出 ${item[`exit_date_${window}`]}${item[`exit_delay_${window}`] ? `，延后 ${item[`exit_delay_${window}`]} 根日线` : ''}` : ''}>{percent(backtestReturn(item, window))}</td>)}
                    <td>{entryLabel(item.entry_status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>}</div>}

        {!historicalView && activeResearchTab === 'news' && <div className="section-panel news-panel">
          <SectionHeading icon={<Newspaper size={18} />} title="最新新闻" meta={sectionMeta(sections.news, detail.news?.source || detail.news?.message)} />
          <div className="news-list">
            {(detail.news?.items || []).map((item) => (
              <article key={`${item.published_at}-${item.title}`}>
                <div className="news-meta"><span><CalendarDays size={13} /> {item.published_at || '-'}</span><span>{item.source}</span></div>
                <h3>{item.url ? <a href={item.url} target="_blank" rel="noreferrer">{item.title}<ExternalLink size={13} /></a> : item.title}</h3>
                {item.summary && <p>{item.summary}</p>}
              </article>
            ))}
            {!detail.news?.items?.length && <div className="empty-row">{detail.news?.message || '暂无相关新闻'}</div>}
          </div>
        </div>}
      </section>}
      </section>

      <footer className="detail-disclaimer"><AlertTriangle size={15} /> {detail.disclaimer}</footer>
    </main>
  );
}

function SectionHeading({ icon, title, meta = '' }) {
  return <div className="section-heading"><h2>{icon}{title}</h2>{meta && <span>{meta}</span>}</div>;
}

function SectionState({ status, message, onRetry }) {
  return <div className="section-state" role="status">
    {status === 'loading' ? <Loader2 size={17} className="spin" /> : <LineChart size={17} />}
    <span>{message}</span>
    <button type="button" className="icon-button" title="重新加载" aria-label="重新加载"
      onClick={onRetry} disabled={status === 'loading'}><RefreshCw size={15} /></button>
  </div>;
}

function QuoteMetric({ label, value, tone = '' }) {
  return <div className={`quote-metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function Fact({ label, value, wide = false }) {
  return <div className={`fact ${wide ? 'wide' : ''}`}><span>{label}</span><strong>{value}</strong></div>;
}

function DetailList({ title, items, tone = '' }) {
  return (
    <div className={`detail-list ${tone}`}>
      <h3>{title}</h3>
      {items.length ? items.map((item) => <p key={item}>{item}</p>) : <p>暂无</p>}
    </div>
  );
}

function BacktestMetric({ label, value, sub }) {
  return <div><span>{label}</span><strong>{value}</strong><small>{sub}</small></div>;
}
