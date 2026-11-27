import test from 'node:test';
import assert from 'node:assert/strict';
import { lineKey, defaultVisibleLines, resolveVisibleLines, selectedPrecisionLines, lineTouchPoints } from './components/chartLines.js';
import { snapshotMatchesBars } from './components/chartLines.js';

const LINES = [
  { price: 20.0, is_primary: false, formation_date: '2024-01-05', resistance_dates: ['2024-01-05'], support_dates: [] },
  { price: 11.2, is_primary: true, formation_date: '2024-03-11', resistance_dates: ['2024-03-11', '2024-05-02'], support_dates: ['2024-07-19'] },
  { price: 10.6, is_primary: false, formation_date: '2024-02-20', resistance_dates: ['2024-02-20'], support_dates: [] },
  { price: 9.8, is_primary: false, formation_date: '2023-11-08', resistance_dates: ['2023-11-08'], support_dates: [] },
  { price: 4.1, is_primary: false, formation_date: '2023-06-01', resistance_dates: ['2023-06-01'], support_dates: [] }
];
const CURRENT = 11.0;

test('lineKey 区分同价位但形成日不同的两条线', () => {
  const a = { price: 10.6, formation_date: '2024-02-20' };
  const b = { price: 10.6, formation_date: '2025-02-20' };
  assert.notEqual(lineKey(a), lineKey(b));
});

test('line versions keep distinct identities even when the price stays unchanged', () => {
  assert.notEqual(lineKey({ price: 10, version_id: 'one:v1' }), lineKey({ price: 10, version_id: 'one:v2' }));
});

test('a historical snapshot is not overlaid on a changed adjustment basis', () => {
  const snapshot = { resistance_points: [{ date: '2026-01-05', price: 10.7, price_basis: 'high' }],
    support_points: [{ date: '2026-01-06', price: 10, price_basis: 'low' }] };
  assert.equal(snapshotMatchesBars(snapshot, BARS), true);
  assert.equal(snapshotMatchesBars(snapshot, BARS.map(bar => ({ ...bar, high: bar.high * 1.1, low: bar.low * 1.1 }))), false);
  assert.equal(snapshotMatchesBars(snapshot, []), false);
});

test('默认只画离现价最近的两根', () => {
  const visible = defaultVisibleLines(LINES, CURRENT, 2);
  assert.equal(visible.length, 2);
  assert.deepEqual(visible.map((line) => line.price).sort((x, y) => x - y), [10.6, 11.2]);
});

test('主线必定入选，即使它不是离现价最近的', () => {
  const lines = [
    { price: 11.05, is_primary: false, formation_date: 'a' },
    { price: 10.95, is_primary: false, formation_date: 'b' },
    { price: 12.4, is_primary: true, formation_date: 'c' }
  ];
  const visible = defaultVisibleLines(lines, CURRENT, 2);
  assert.ok(visible.some((line) => line.is_primary), `主线应在 ${JSON.stringify(visible.map((l) => l.price))} 中`);
  assert.equal(visible.length, 2);
});

test('点选的线会追加显示，且不挤掉默认那两根', () => {
  const picked = lineKey(LINES[4]);           // 4.1，离现价很远
  const visible = resolveVisibleLines(LINES, CURRENT, [picked], 2);
  const prices = visible.map((line) => line.price);
  assert.ok(prices.includes(4.1), `点选的 4.1 应显示，实得 ${prices}`);
  assert.ok(prices.includes(11.2) && prices.includes(10.6), `默认两根应保留，实得 ${prices}`);
});

test('重复点选默认已显示的线不会画两遍', () => {
  const visible = resolveVisibleLines(LINES, CURRENT, [lineKey(LINES[1])], 2);
  assert.equal(visible.length, 2);
});

test('输出按价格从高到低排序，保证标签位置稳定', () => {
  const visible = resolveVisibleLines(LINES, CURRENT, [lineKey(LINES[0]), lineKey(LINES[4])], 2);
  const prices = visible.map((line) => line.price);
  assert.deepEqual(prices, [...prices].sort((x, y) => y - x));
});

test('未选线时没有精准点所属线，默认主线也不会自动入选', () => {
  const drawn = resolveVisibleLines(LINES, CURRENT);
  assert.equal(drawn.length, 2);
  assert.deepEqual(selectedPrecisionLines(drawn), []);
  assert.deepEqual(selectedPrecisionLines(drawn, []), []);
});

test('仅返回选中线，切换后不保留前一条线的点', () => {
  const drawn = resolveVisibleLines(LINES, CURRENT);
  assert.deepEqual(selectedPrecisionLines(drawn, [lineKey(LINES[1])]), [LINES[1]]);
  assert.deepEqual(selectedPrecisionLines(drawn, [lineKey(LINES[2])]), [LINES[2]]);
});

test('选择额外的线只显示它的点，不带上默认两根线的点', () => {
  const key = lineKey(LINES[4]);
  const drawn = resolveVisibleLines(LINES, CURRENT, [key]);
  assert.equal(drawn.length, 3);
  assert.deepEqual(selectedPrecisionLines(drawn, [key]), [LINES[4]]);
});

test('失效选择不回退到主线，同价不同形成日也不能串线', () => {
  const other = { ...LINES[1], formation_date: '2026-01-01' };
  assert.deepEqual(selectedPrecisionLines([LINES[1], other], [lineKey(other)]), [other]);
  assert.deepEqual(selectedPrecisionLines(LINES, ['missing']), []);
  assert.deepEqual(selectedPrecisionLines(null, [lineKey(other)]), []);
});

test('lineTouchPoints 保留日期和类型，缺少真实点价时不拿线价代替', () => {
  const points = lineTouchPoints(LINES[1]);
  assert.deepEqual(points.map((p) => p.date), ['2024-03-11', '2024-05-02', '2024-07-19']);
  assert.deepEqual(points.map((p) => p.kind), ['resistance', 'resistance', 'support']);
  assert.ok(points.every((p) => p.price === null));
  assert.ok(points.every((p) => p.line_price === 11.2));
});

import { buildPrecisionPointSeries, precisionPointTooltip } from './components/chartLines.js';

const POINT_BARS = [
  { date: '2026-04-07', open: 6.16, close: 6.2, high: 6.24, low: 6.08 },
  { date: '2026-08-04', open: 6.1, close: 6.21, high: 6.22, low: 6.05 },
  { date: '2026-09-07', open: 6.2, close: 6.25, high: 6.3, low: 6.15 },
];
const POINT_LINE = {
  price: 6.15, is_primary: true, resistance_dates: ['2025-11-11', '2026-08-04'],
  support_dates: ['2026-04-07'],
};
const POINT_EVIDENCE = {
  linePrice: 6.15, anchor: { date: '2026-08-04', price: 6.22, confirmed_date: '2026-08-13', role: 'before_breakout' },
};

test('仅为传入的线生成精准点，不把普通波峰自动当成精准点', () => {
  const series = buildPrecisionPointSeries([POINT_LINE], POINT_BARS);
  assert.deepEqual(series.map(item => item.id), ['precision-resistance', 'precision-support']);
  assert.deepEqual(series.flatMap(item => item.data).map(item => item.value),
    [['2026-08-04', 6.22], ['2026-04-07', 6.08]]);
});

test('未选择的主线阻力点不会混入另一条线的精准点', () => {
  const other = { price: 6.2, support_dates: ['2026-04-07'] };
  const selected = selectedPrecisionLines([POINT_LINE, other], [lineKey(other)]);
  const series = buildPrecisionPointSeries(selected, POINT_BARS, { evidence: POINT_EVIDENCE });
  assert.deepEqual(series.map(item => item.id), ['precision-support']);
  assert.deepEqual(series[0].data.map(item => item.precisionPoint.lines), [[6.2]]);
  assert.deepEqual(buildPrecisionPointSeries(
    selectedPrecisionLines([POINT_LINE, other], []), POINT_BARS, { evidence: POINT_EVIDENCE }
  ), []);
});

test('阻力取实际最高价，支撑取实际最低价，而不是线价或实体价', () => {
  const points = lineTouchPoints(POINT_LINE, POINT_BARS);
  assert.equal(points.find(point => point.kind === 'support').price, 6.08);
  assert.equal(points.find(point => point.date === '2026-08-04').price, 6.22);
  assert.equal(points.find(point => point.date === '2025-11-11').price, null);
});

test('本次阻力点单独突出，必须属于对应主线而不是仅日期相同', () => {
  const series = buildPrecisionPointSeries([POINT_LINE], POINT_BARS, { evidence: POINT_EVIDENCE });
  assert.deepEqual(series.map(item => item.id), ['precision-support', 'precision-anchor']);
  const anchor = series.find(item => item.id === 'precision-anchor');
  assert.deepEqual(anchor.data[0].value, ['2026-08-04', 6.22]);
  assert.equal(anchor.data[0].precisionPoint.confirmed_date, '2026-08-13');
  assert.ok(anchor.symbolSize > series[0].symbolSize);
  assert.ok(buildPrecisionPointSeries([{ ...POINT_LINE, price: 6.37 }], POINT_BARS, { evidence: POINT_EVIDENCE })
    .every(item => item.id !== 'precision-anchor'));
});

test('精准点使用合法样式对象并保留真实坐标，不能用无效 itemStyle 回调', () => {
  const series = buildPrecisionPointSeries([POINT_LINE], POINT_BARS);
  for (const item of series) {
    assert.equal(typeof item.itemStyle, 'object');
    assert.ok(item.itemStyle.borderWidth > 0);
    assert.equal(item.itemStyle.opacity, 1);
    assert.equal(item.tooltip.trigger, 'item');
    assert.equal(item.tooltip.confine, true);
    assert.ok(item.z > 5);
  }
  assert.notEqual(series[0].symbol, series[1].symbol);
  assert.notEqual(series[0].itemStyle.borderColor, series[1].itemStyle.borderColor);
});

test('隐藏精准点只隐藏该图层，空线和缺失价格不能造点', () => {
  assert.deepEqual(buildPrecisionPointSeries([POINT_LINE], POINT_BARS, { showPoints: false }), []);
  assert.deepEqual(buildPrecisionPointSeries([], POINT_BARS), []);
  assert.deepEqual(buildPrecisionPointSeries([POINT_LINE], [{ date: '2026-08-04' }]), []);
  assert.deepEqual(buildPrecisionPointSeries([POINT_LINE], []), []);
});

test('结构化历史点价优先，日期不在图中时不画到别的 K 线上', () => {
  const line = {
    ...POINT_LINE,
    resistance_points: [
      { date: '2025-11-11', price: 6.17, price_basis: 'high' },
      { date: '2026-08-04', price: 6.221234, price_basis: 'high', confirmed_date: '2026-08-13' },
    ],
  };
  const series = buildPrecisionPointSeries([line], POINT_BARS);
  assert.deepEqual(series[0].data.map(point => point.value), [['2026-08-04', 6.221234]]);
});

test('同一个支撑点属于多条线时只画一次，悬停仍列出全部所属线', () => {
  const other = { price: 6.2, support_dates: ['2026-04-07'] };
  const series = buildPrecisionPointSeries([POINT_LINE, other], POINT_BARS);
  const support = series.find(item => item.id === 'precision-support');
  assert.equal(support.data.length, 1);
  assert.deepEqual(support.data[0].precisionPoint.lines, [6.15, 6.2]);
  assert.match(precisionPointTooltip({ data: support.data[0] }), /精准线：6.15、6.2/);
});

test('精准点悬停显示日期、实际点价、所属线和确认日', () => {
  const anchor = buildPrecisionPointSeries([POINT_LINE], POINT_BARS, { evidence: POINT_EVIDENCE })
    .find(item => item.id === 'precision-anchor');
  const tooltip = precisionPointTooltip({ data: anchor.data[0] });
  for (const text of ['本次阻力点', '发生日：2026-08-04', '最高价：6.22', '精准线：6.15', '确认日：2026-08-13']) {
    assert.ok(tooltip.includes(text), tooltip);
  }
  const latest = buildPrecisionPointSeries([POINT_LINE], POINT_BARS, {
    evidence: { ...POINT_EVIDENCE, anchor: { ...POINT_EVIDENCE.anchor, role: 'latest_confirmed' } },
  }).find(item => item.id === 'precision-anchor');
  assert.equal(latest.name, '最近阻力点');
  assert.match(precisionPointTooltip({ data: latest.data[0] }), /最近阻力点/);
});

test('精准点悬停不会把外部字符串直接当 HTML 执行', () => {
  assert.equal(precisionPointTooltip({ data: {} }), '');
  const invalid = { date: '<img src=x onerror=alert(1)>', price: 6.22, lines: [6.15] };
  assert.equal(precisionPointTooltip({ data: { precisionPoint: invalid } }), '');
  const tooltip = precisionPointTooltip({ data: { precisionPoint: {
    ...invalid, date: '2026-08-04', lines: ['<img src=x>'], confirmed_date: '<script>',
  } } });
  assert.doesNotMatch(tooltip, /<img|<script/);
});

test('事件点保留来源、确认时间和整段极值，同一转折两个来源不画两个点', () => {
  const line = {
    price: 6.15, resistance_points: [{
      date: '2026-08-04', price: 6.22, confirmed_date: '2026-08-13',
      origins: ['local_turn', 'resistance_test', '<script>'], event_start_date: '2026-08-01',
      max_high: 6.22, min_low: 5.95,
    }]
  };
  const series = buildPrecisionPointSeries([line], POINT_BARS);
  assert.equal(series.flatMap(item => item.data).length, 1);
  const tooltip = precisionPointTooltip({ data: series[0].data[0] });
  assert.match(tooltip, /事件来源：局部转折、阻力验证/);
  assert.match(tooltip, /整段最高 \/ 最低：6.22 \/ 5.95/);
  assert.match(tooltip, /试探起点：2026-08-01/);
  assert.doesNotMatch(tooltip, /<script>/);
});

test('lineTouchPoints 对空输入返回空数组', () => {
  assert.deepEqual(lineTouchPoints(null), []);
  assert.deepEqual(lineTouchPoints({ price: 1 }), []);
});

// ---- 图上标记 ----
import { buildMarkPoints } from './components/chartLines.js';

const BARS = [
  { date: '2026-01-05', open: 10, close: 10.5, low: 9.8, high: 10.7 },
  { date: '2026-01-06', open: 10.5, close: 10.2, low: 10.0, high: 10.8 }
];
const MARKERS = [
  { date: '2026-01-05', type: 'alert', label: '历史预警', price: 10.5 },
  { date: '2026-01-06', type: 'breakout', label: '突破' }
];

test('关掉历史预警后图上不再有预警标记，突破标记保留', () => {
  const points = buildMarkPoints(BARS, MARKERS, { showAlerts: false });
  assert.deepEqual(points.map((p) => p.name), ['突破']);
});

test('历史预警锚在当日最低价并向下偏移，不压住K线实体', () => {
  const [alert] = buildMarkPoints(BARS, MARKERS, { showAlerts: true });
  assert.equal(alert.name, '历史预警');
  assert.equal(alert.coord[1], 9.8, '应锚定当日最低价而非收盘价');
  assert.ok(alert.symbolOffset[1] > 0, `偏移应把标记推到K线下方，实得 ${alert.symbolOffset}`);
});

test('历史预警标记要小且不带文字，几十个叠在一起才不会糊住图', () => {
  const [alert] = buildMarkPoints(BARS, MARKERS, { showAlerts: true });
  assert.ok(alert.symbolSize <= 10, `标记过大：${alert.symbolSize}`);
  assert.equal(alert.label.show, false);
});

test('突破标记也要小，带文字的大色块会盖住K线', () => {
  const breakout = buildMarkPoints(BARS, MARKERS, { showAlerts: true }).find((p) => p.name === '突破');
  assert.ok(breakout.symbolSize <= 11, `突破标记过大：${breakout.symbolSize}`);
  assert.equal(breakout.label.show, false, '文字标签会形成大色块，改为只在 tooltip 里显示');
});

test('突破标记浮在最高价上方，不与K线实体重叠', () => {
  const breakout = buildMarkPoints(BARS, MARKERS, { showAlerts: true }).find((p) => p.name === '突破');
  assert.equal(breakout.coord[1], 10.8, '应锚定当日最高价');
  assert.ok(breakout.symbolOffset[1] < 0, `偏移应把标记推到K线上方，实得 ${breakout.symbolOffset}`);
});

test('三类标记形状/颜色互不相同，图上能分辨', () => {
  const points = buildMarkPoints(BARS, [
    { date: '2026-01-05', type: 'alert', label: '历史预警' },
    { date: '2026-01-06', type: 'breakout', label: '突破' },
    { date: '2026-01-06', type: 'pullback', label: '回踩' }
  ], { showAlerts: true });
  const colors = points.map((p) => p.itemStyle.color);
  assert.equal(new Set(colors).size, 3, `三类标记应有三种颜色，实得 ${colors}`);
});

test('日期不在K线里的标记直接丢掉', () => {
  const points = buildMarkPoints(BARS, [{ date: '2020-01-01', type: 'alert', label: '历史预警' }], { showAlerts: true });
  assert.deepEqual(points, []);
});

// ---- 精准线在图上的画法 ----
import { buildFormationMarkers, buildKeyLineMarkLines, keyLineColor } from './components/chartLines.js';

const DRAWN = [
  { price: 11.08, point_count: 23, is_primary: true, formation_date: '2024-03-11' },
  { price: 11.46, point_count: 20, is_primary: false, formation_date: '2024-05-02' }
];

test('线上不画文字标签，带底色的文字块会盖住K线', () => {
  for (const item of buildKeyLineMarkLines(DRAWN, [])) {
    assert.equal(item.label.show, false, `${item.name} 仍在图内画标签`);
  }
});

test('每条线仍带名称，供 tooltip 和图例识别', () => {
  const [primary, other] = buildKeyLineMarkLines(DRAWN, []);
  assert.match(primary.name, /主线/);
  assert.match(primary.name, /11\.08/);
  assert.match(primary.name, /23点/);
  assert.doesNotMatch(other.name, /主线/);
});

test('主线画实线且更粗，普通线画虚线', () => {
  const [primary, other] = buildKeyLineMarkLines(DRAWN, []);
  assert.equal(primary.lineStyle.type, 'solid');
  assert.equal(other.lineStyle.type, 'dashed');
  assert.ok(primary.lineStyle.width > other.lineStyle.width);
});

test('点选的线换成高亮色，和图例色块对得上', () => {
  const picked = lineKey(DRAWN[1]);
  const [, other] = buildKeyLineMarkLines(DRAWN, [picked]);
  assert.equal(other.lineStyle.color, '#174ea6');
  assert.equal(keyLineColor(DRAWN[1], 1, [picked]), '#174ea6', '图例必须用同一个取色函数');
});

test('yAxis 取线价，线才画在正确高度', () => {
  assert.equal(buildKeyLineMarkLines(DRAWN, [])[0].yAxis, 11.08);
});

// ---- 线的生效区间 ----
const DATES = ['2012-01-05', '2013-06-11', '2018-02-09', '2024-03-11', '2026-03-19', '2026-08-27'];
const AGED = [{ price: 4.36, point_count: 3, is_primary: true, formation_date: '2026-03-19' }];

test('线整条连上，不因形成日拆断', () => {
  const items = buildKeyLineMarkLines(AGED, [], DATES);
  assert.equal(items.length, 1, `一条线应只产生一个整条条目，实得 ${items.length}`);
  assert.equal(items[0].yAxis, 4.36, '整条线用 yAxis 横贯全图');
});

test('形成日改用线上的小标记表示，不再靠线型区分', () => {
  const [mark] = buildFormationMarkers(AGED, DATES);
  assert.equal(mark.coord[0], '2026-03-19');
  assert.equal(mark.coord[1], 4.36, '标记要落在线价上');
  assert.ok(mark.symbolSize <= 9, `标记过大会挡K线：${mark.symbolSize}`);
  assert.equal(mark.label.show, false, '文字块会挡K线，名称留给 tooltip');
});

test('形成标记带得出是哪条线，供 tooltip 辨认', () => {
  const [mark] = buildFormationMarkers(AGED, DATES);
  assert.match(mark.name, /4\.36/);
  assert.match(mark.name, /形成/);
});

test('形成日不在日期轴内就不画标记，避免飘在图外', () => {
  assert.deepEqual(buildFormationMarkers(AGED, []), []);
  assert.deepEqual(buildFormationMarkers([{ price: 4.36 }], DATES), []);
  assert.deepEqual(buildFormationMarkers(AGED, ['2020-01-01']), []);
});

test('多条线各自出一个形成标记', () => {
  const two = [...AGED, { price: 4.6, point_count: 11, formation_date: '2013-06-11' }];
  assert.equal(buildFormationMarkers(two, DATES).length, 2);
});

// ---- 波峰波谷（含候选点）----
import { buildSwingSeries } from './components/chartLines.js';

const SWING_CHART = {
  peaks: [{ date: '2026-01-05', price: 12.0 }],
  valleys: [{ date: '2026-02-10', price: 9.0 }],
  pending_swing: { date: '2026-03-01', kind: 'peak', price: 13.0 }
};

test('关掉开关时不画任何摆动点', () => {
  assert.deepEqual(buildSwingSeries(SWING_CHART, false), []);
});

test('已确认的峰谷用实心，候选点用空心', () => {
  const series = buildSwingSeries(SWING_CHART, true);
  const byName = Object.fromEntries(series.map((s) => [s.name, s]));

  assert.ok(byName['波峰'].itemStyle.color !== 'transparent', '已确认应是实心');
  assert.equal(byName['候选点'].itemStyle.color, 'transparent', '候选点必须空心');
  assert.ok(byName['候选点'].itemStyle.borderWidth > 0, '空心要有描边才看得见');
});

test('候选波峰朝上、候选波谷朝下，和已确认的方向一致', () => {
  const up = buildSwingSeries(SWING_CHART, true).find((s) => s.name === '候选点');
  const down = buildSwingSeries(
    { ...SWING_CHART, pending_swing: { date: '2026-03-01', kind: 'valley', price: 8.0 } }, true
  ).find((s) => s.name === '候选点');

  assert.equal(up.symbolRotate ?? 0, 0);
  assert.equal(down.symbolRotate, 180);
});

test('没有候选点时不产生空系列', () => {
  const series = buildSwingSeries({ peaks: [], valleys: [], pending_swing: null }, true);
  assert.ok(!series.some((s) => s.name === '候选点'));
});

test('候选点数据带上日期与价格，才能落在正确位置', () => {
  const s = buildSwingSeries(SWING_CHART, true).find((x) => x.name === '候选点');
  assert.deepEqual(s.data, [['2026-03-01', 13.0]]);
});
