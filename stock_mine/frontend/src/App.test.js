import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { scanPayload } from './flagPatterns.js';

const appSource = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8');
const detailSource = readFileSync(new URL('./StockDetailPage.jsx', import.meta.url), 'utf8');
const chartSource = readFileSync(new URL('./components/KLineChart.jsx', import.meta.url), 'utf8');
const chartLinesSource = readFileSync(new URL('./components/chartLines.js', import.meta.url), 'utf8');

test('defaults to scanning the whole market with a real provider', () => {
  // 只锁这三项默认值，表单新增字段时不该误伤
  // signal_offset 0 = 最近已收盘交易日，与回测同一根信号K线
  assert.match(appSource, /useState\(\{ provider: 'auto', limit: 0, signal_offset: 0[,}]/);
});

test('uses explicit copy while a scan is running', () => {
  assert.match(appSource, /loading \? '筛选中\.\.\.' : '开始筛选'/);
});

test('renders a persistent live scan status', () => {
  assert.match(appSource, /role="status"/);
  assert.match(appSource, /aria-live="polite"/);
});

test('renders every visible precision line with its point count', () => {
  assert.match(chartSource, /chart\?\.key_lines/);
  // 线的文案与取色已收敛到 chartLines.js，图上标记与下方图例共用同一份
  assert.match(chartLinesSource, /line\.point_count/);
  assert.match(chartSource, /keyLineLabel\(line\)/);
  assert.match(detailSource, /阻力 \{line\.resistance_point_count\}/);
  assert.match(detailSource, /支撑 \{line\.support_point_count\}/);
});

test('precision points follow one selected line and initialize to the primary line after loading', () => {
  assert.match(detailSource, /const \[selectedLineKeys, setSelectedLineKeys\] = useState\(\[\]\)/);
  const loadStart = detailSource.indexOf('async function load(');
  const loadEnd = detailSource.indexOf('\n  function updateSection', loadStart);
  const loadBody = detailSource.slice(loadStart, loadEnd);
  assert.match(loadBody, /const primaryLine = core\.chart\?\.key_lines\?\.find\(line => line\.is_primary\)/);
  assert.match(loadBody, /setSelectedLineKeys\(primaryLine \? \[lineKey\(primaryLine\)\] : \[\]\)/);
  const companyStart = detailSource.indexOf('async function loadCompany');
  const companyEnd = detailSource.indexOf('\n  async function', companyStart + 1);
  assert.doesNotMatch(detailSource.slice(companyStart, companyEnd), /setSelectedLineKeys/);
  assert.match(detailSource, /current\.includes\(key\) \? \[\] : \[key\]/);
  assert.doesNotMatch(detailSource, /showPrecisionPoints|aria-label="显示精准点"/);
  assert.match(detailSource, /precisionEvidence=\{precisionEvidence\}/);
  assert.match(chartSource, /selectedPrecisionLines\(drawnLines, selectedLineKeys\)/);
  assert.match(chartSource, /buildPrecisionPointSeries\(pointLines, bars/);
  assert.doesNotMatch(chartSource, /buildPrecisionPointSeries\(drawnLines/);
  assert.match(chartSource, /buildFormationMarkers\(pointLines, dates\)/);
  assert.doesNotMatch(chartSource, /buildFormationMarkers\(drawnLines/);
  assert.match(chartSource, /full && pointLines\.length > 0/);
  assert.match(chartSource, /data-precision-point-count=\{precisionPoints\.length\}/);
  assert.match(chartSource, /aria-label="精准点图例"/);
  assert.match(chartSource, /tooltip: \{ trigger: 'axis', confine: true/);
  assert.match(chartSource, /instance\.resize\(\);[\s\n]*instance\.dispatchAction\(\{ type: 'hideTip' \}\)/);
});

test('watchlist records the entry price and renders performance', () => {
  assert.match(appSource, /entry_price: item\.current_price/);
  assert.match(detailSource, /entry_price: detail\.market\?\.close/);
  assert.match(appSource, /row\.return_pct/);
  assert.match(appSource, /row\.price_change/);
});

test('detail supplementary sections load independently and backtest is opt-in', () => {
  assert.match(detailSource, /void loadCompany\(requestId\)/);
  assert.match(detailSource, /void loadNews\(requestId\)/);
  assert.match(detailSource, /void loadIndustry\(requestId\)/);
  // 复盘必须仍是按需触发：首屏不能跑它
  assert.doesNotMatch(detailSource, /void loadBacktest\(requestId\)/);
  assert.match(detailSource, /api\.stockBacktest\(stockCode, lookbackDays, scanId,[\s\n]*detail\?\.precision_engine, detail\?\.analysis_as_of\)/);
  // 回放代价随窗口线性上涨，用户要能自己选，且「全部」（days: 0）仍然可达
  assert.match(detailSource, /BACKTEST_WINDOWS/);
  assert.match(detailSource, /\{ days: 0, label: '全部'/);
  assert.match(detailSource, /document\.visibilityState !== 'visible'/);
  assert.doesNotMatch(detailSource, /Promise\.allSettled/);
});

test('checking historical alerts starts a one-year replay and shows elapsed time', () => {
  assert.match(detailSource, /const \[showAlerts, setShowAlerts\] = useState\(false\)/);
  assert.match(detailSource, /const \[backtestWindow, setBacktestWindow\] = useState\(250\)/);
  assert.match(detailSource, /onChange=\{toggleHistoricalAlerts\}/);
  assert.match(detailSource, /void loadBacktest\(backtestWindow\)/);
  assert.match(detailSource, /backtestSeconds/);
});

test('workbench does not refresh every watch price during initial load', () => {
  assert.doesNotMatch(appSource, /if \(watchesResult\.value\.length\) void refreshWatchPrices\(\)/);
});

test('detail section list is declared once so a new section cannot be half-registered', () => {
  // 初始 state 与 load() 的重置若各写一份名单，新增板块时漏掉一处就会
  // 让 sections.<新名字> 变成 undefined，JSX 一读就抛错白屏。
  assert.match(detailSource, /const DETAIL_SECTIONS = \[/);
  assert.match(detailSource, /useState\(\(\) => sectionStates\('idle'\)\)/);
  assert.match(detailSource, /setSections\(sectionStates\('loading'/);
  assert.equal(
    (detailSource.match(/company: \{ status:/g) || []).length,
    0,
    '不应再出现逐个字面量列举 section 的写法'
  );
});

test('scan form exposes the confirmed-pullback window and submits it', () => {
  // 这是选股条件而非页面筛选：必须随扫描请求一起提交，否则调了等于没调
  assert.match(appSource, /require_pullback: true, min_pullback_days: 1, max_pullback_days: 10/);
  assert.match(appSource, /api\.runScan\(scanPayload\(form\)\)/);
  const request = scanPayload({ min_pullback_days: '12', max_pullback_days: '20' });
  assert.equal(request.min_pullback_days, 12);
  assert.equal(request.max_pullback_days, 20);
  assert.equal(request.require_pullback, true);
  assert.match(appSource, /回调天数/);
  assert.doesNotMatch(appSource, /aria-label="距突破天数/);
  const inputFor = (label) => {
    const end = appSource.indexOf(`aria-label="${label}"`);
    const start = appSource.lastIndexOf('<input', end);
    return appSource.slice(start, end);
  };
  for (const label of ['回调天数下限', '回调天数上限']) {
    assert.match(inputFor(label), /min="1"/);
    assert.match(inputFor(label), /max="60"/);
    assert.match(inputFor(label), /step="1"/);
  }
  assert.match(appSource, /aria-label="精准线只选突破后回踩"/);
  assert.match(detailSource, /pullback_check\.days_since_breakout/);
  assert.match(detailSource, /pullback_check\.min_days/);
  assert.match(detailSource, /pullback_check\.max_days/);
});

test('scan form exposes the breakout gating controls', () => {
  // 这三项决定"哪些股票算已突破"，都要能人工调并随请求提交
  const request = scanPayload({ require_breakout: true, breakout_vol_mult: '1.8' });
  assert.equal(request.require_breakout, true);
  assert.equal(request.breakout_vol_mult, 1.8);
  assert.match(appSource, /只要已突破的股票/);
  assert.match(appSource, /突破量比门槛/);
  assert.match(appSource, /require_breakout_volume: false/);
  assert.match(appSource, /aria-label="精准线启用突破量比过滤"/);
  assert.match(appSource, /form\.strategy_mode !== 'precision_pullback' \|\| form\.require_breakout_volume/);
  assert.match(appSource, /旗形突破量比门槛/);
  assert.match(detailSource, /signal\?\.require_breakout_volume/);
  assert.match(detailSource, /signal\.breakout_volume_threshold/);
});

test('combined is the default and historical flags can restore their frozen channel', () => {
  assert.match(appSource, /strategy_mode: 'combined'/);
  assert.match(appSource, /aria-label="形态策略"/);
  assert.match(detailSource, /selectFlag\(item\.flag_pattern\)/);
  assert.match(detailSource, /chart=\{stockChart\}/);
  assert.match(chartSource, /buildFlagSeries\(chart\?\.flag_pattern, bars\)/);
  assert.match(chartSource, /legend: full[\s\S]*?type: 'scroll'/);
});

test('results, detail and history render all matched-pattern tags', () => {
  assert.match(appSource, /<PatternTags row=\{row\}/);
  assert.match(detailSource, /<PatternTags row=\{primarySignal\}/);
  assert.match(detailSource, /<PatternTags row=\{item\}/);
  assert.match(appSource, /联合筛选/);
});

test('prebreakout path thresholds are shown in the form and precision diagnostics', () => {
  assert.match(appSource, /prebreakout_extension_percent: 50/);
  assert.match(appSource, /prebreakout_reject_above_days: 30/);
  assert.match(appSource, /prebreakout_above_buffer_percent: 2/);
  assert.match(appSource, /aria-label="突破前越线上限百分比"/);
  assert.match(appSource, /aria-label="突破前连续站上排除天数"/);
  assert.match(appSource, /aria-label="有效收线上缓冲百分比"/);
  assert.match(detailSource, /prebreakout_check\?\.limit_pct/);
  assert.match(detailSource, /prebreakout_check\.max_above_line_pct/);
  assert.match(detailSource, /prebreakout_check\.max_consecutive_above_days/);
  assert.match(detailSource, /prebreakout_check\?\.longest_run_start_date/);
  assert.match(detailSource, /prebreakout_check\?\.longest_run_end_date/);
});

test('results table renders the post-breakout status tags', () => {
  assert.match(appSource, /breakout_status\?\.tags/);
  assert.match(appSource, /sortTags\(row\.breakout_status\.tags\)/);
  assert.match(appSource, /tone-\$\{tagTone\(tag\)\}/);
});

test('screening results expose resistance and reasons with a separate expand control', () => {
  assert.match(appSource, /<th>阻力点<\/th>/);
  assert.match(appSource, /<th>选股理由<\/th>/);
  assert.match(appSource, /<ResistanceSummary evidence=\{evidence\.resistance\}/);
  assert.match(appSource, /<ReasonPreview groups=\{evidence\.groups\}/);
  assert.match(appSource, /visibleRows\.map\(row => \[row\.stock_code, selectionEvidence\(row\)\]\)/);
  assert.match(appSource, /aria-expanded=\{isExpanded\} aria-controls=\{evidenceId\}/);
  assert.match(appSource, /event\.stopPropagation\(\);[\s\n]*setExpanded\(current => toggleCode\(current, row\.stock_code\)\)/);
  assert.match(appSource, /<td colSpan=\{14\}>/);
  assert.match(appSource, /<ExpandedEvidence row=\{row\} evidence=\{evidence\} id=\{evidenceId\} onOpen=\{onSelect\}/);
});

test('single-stock lookup lives outside the scan filter form', () => {
  // 它跟筛选条件无关，挤在一起会让人以为受筛选影响
  assert.match(appSource, /className="quick-lookup"/);
  const band = appSource.slice(appSource.indexOf('className="control-band"'));
  assert.doesNotMatch(band.slice(0, band.indexOf('</section>')), /singleCode/);
});

test('results and watchlist both support bulk actions', () => {
  assert.match(appSource, /onBulkWatch=\{bulkAddWatch\}/);
  assert.match(appSource, /onBulkRemove=\{bulkRemoveWatch\}/);
  assert.match(appSource, /api\.bulkAddWatch/);
  assert.match(appSource, /api\.bulkDeleteWatch/);
  assert.match(appSource, /批量加入观测池/);
  assert.match(appSource, /批量移除/);
});

test('workbench uses the full width for an expanded results table', () => {
  assert.doesNotMatch(appSource, /className="detail-panel"/);
  assert.doesNotMatch(appSource, /function StockDetail\(/);
  assert.match(appSource, /className="table-wrap result-table-wrap"/);
  assert.match(appSource, /useState\(50\)/);
  assert.match(appSource, /count \+ 50/);
});

test('watchlist supports adding a stock by hand', () => {
  assert.match(appSource, /onManualAdd=\{manualAddWatch\}/);
  assert.match(appSource, /手动加入/);
});

test('stale selections are dropped when the underlying rows change', () => {
  // 否则批量操作会打到已经不在列表里的代码上
  assert.match(appSource, /setPicked\(\(current\) => keepExisting\(current, rows\)\)/);
});

test('a running scan can be stopped from the progress bar', () => {
  // 之前只能杀后端进程，会把已扫出来的结果一起丢掉
  assert.match(appSource, /api\.cancelScan\(activeScanId\)/);
  assert.match(appSource, /停止筛选/);
});

test('historical alerts reach every chart on the detail page', () => {
  // 个股、所属行业、最相关板块三张图都要标出历史预警，否则对照时缺一张
  assert.match(detailSource, /function withAlertMarkers/);
  const backtest = detailSource.slice(detailSource.indexOf('async function loadBacktest'));
  const body = backtest.slice(0, backtest.indexOf('\n  async function'));
  assert.match(body, /industry_market:[\s\S]*withAlertMarkers/);
  assert.match(body, /related_board:[\s\S]*withAlertMarkers/);
});

test('opening a stock from the results carries the scan batch id', () => {
  // 详情要按"打开它的那次筛选"判定；只靠后端进程内的全局配置，
  // 重启、开新扫描、或改完参数看旧批次时都会与列表不一致
  assert.match(appSource, /function openStockPage\(stockCode, stockName = '', scanId = null\)/);
  assert.match(appSource, /params\.set\('scan_id', String\(scanId\)\)/);
  // 这里必须按函数体断言。之前只用整文件正则，匹配到了 openWatch 那行就通过了，
  // 而真正该带批次号的 openResult 反倒没带——典型的假绿灯。
  const bodyOf = (name) => {
    const start = appSource.indexOf(`function ${name}(`);
    const rest = appSource.slice(start + 1);
    const end = rest.indexOf('\n  function ');
    return rest.slice(0, end === -1 ? rest.length : end);
  };
  assert.match(bodyOf('openResult'),
    /openStockPage\(item\.stock_code, item\.stock_name, activeScanId \|\| scan\.id/);
  // 只看调用本身，别把注释里出现的字样也算上
  const callIn = (body) => (body.match(/openStockPage\([^)]*\)/) || [''])[0];
  assert.doesNotMatch(callIn(bodyOf('openWatch')), /scan\.id/,
    '观察池是手动跟踪，不该套用当前页面的批次号');
  assert.match(detailSource, /api\.stockDetail\(stockCode, stockName, scanId, requestedEngine, requestedDate\)/);
});

test('backtest copy and range stay in sync', () => {
  // 选了 1 年却写着"全部历史"会让人以为跑的是全量
  assert.match(detailSource, /backtestWindowLabel/);
  assert.doesNotMatch(detailSource, /开始全历史复盘/);
  assert.match(detailSource, /同步到三张 K 线图/);
  // 批次复盘不再写死 500 日
  assert.match(appSource, /lookback_days: Number\(batchLookback\)/);
  assert.doesNotMatch(appSource, /lookback_days: Number\(batchLookback\) \|\| 500/);
  assert.match(appSource, /scan_id: activeScanId \|\| scan\.id/);
  assert.match(appSource, /onLookbackChange=\{setBatchLookback\}/);
});

test('both views use the net-or-legacy presentation contract', () => {
  assert.match(appSource, /backtestMetric\(summary, window\)/);
  assert.match(detailSource, /backtestMetric\(summary, window\)/);
  assert.match(detailSource, /backtestReturn\(item, window\)/);
  assert.match(appSource, /旧版信号涨跌统计/);
  assert.match(detailSource, /detail\.data_quality\?\.status === 'stale'/);
  assert.match(detailSource, /detail\.industry\?\.stale/);
});

test('engine choice and historical cutoff are explicit and historical charts use frozen versions', () => {
  assert.match(appSource, /precision_engine: 'platform_events'/);
  assert.match(appSource, /aria-label="点位引擎"/);
  assert.match(detailSource, /aria-label="个股点位引擎"/);
  assert.match(detailSource, /aria-label="历史分析日期"/);
  assert.match(detailSource, /if \(core\.analysis_as_of\)/);
  assert.match(detailSource, /!historicalView &&/);
  assert.match(detailSource, /snapshotMatchesBars/);
  assert.match(detailSource, /selectedHistoricalSignal\.line_snapshot/);
  assert.match(detailSource, /summary\.net_t5_profit_factor/);
  assert.match(detailSource, /summary\.entry_feasibility_rate/);
  assert.match(appSource, /form\.precision_engine !== 'platform_events' && <label className="checkbox-field">/);
  assert.match(detailSource, /type === 'support'\) return '支撑平台'/);
  assert.match(detailSource, /watchState !== 'idle' \|\| loading \|\| historicalView/);
  assert.match(detailSource, /if \(!stock \|\| detail\?\.analysis_as_of \|\| loading\) return/);
});

test('wheel zoom is handled before ECharts consumes the event and is cleaned up on remount', () => {
  assert.match(chartSource, /zoomOnMouseWheel: false, moveOnMouseWheel: false/);
  assert.match(chartSource, /addEventListener\('wheel', handleWheel, \{ passive: false, capture: true \}\)/);
  assert.match(chartSource, /removeEventListener\('wheel', handleWheel, true\)/);
  assert.match(chartSource, /rightAnchoredWheelRange\(dates, current, pendingWheelDelta\)/);
  assert.match(chartSource, /cancelAnimationFrame\(wheelFrame\)/);
});
