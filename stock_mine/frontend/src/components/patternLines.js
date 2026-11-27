// 将后端给出的三角形趋势线端点转成 ECharts 线段。
// 线段用 markLine 的坐标对画，任意两根K线之间都能连，形态互相重叠也不会串线。

export const PATTERN_COLORS = {
  triangleUp: '#c27c0e',
  triangleDown: '#2f7d8c',
  triangle: '#8a8f99'
};

export const TRIANGLE_SERIES_NAME = '三角形整理';

function segment(a, b, lineStyle) {
  return [
    { coord: [a.date, a.price], lineStyle },
    { coord: [b.date, b.price] }
  ];
}

function label(text, date, price, color, below = false) {
  return {
    name: text,
    coord: [date, price],
    value: text,
    symbol: 'rect',
    symbolSize: [text.length * 11 + 8, 16],
    symbolOffset: [0, below ? 14 : -14],
    itemStyle: { color, opacity: 0.92 },
    label: { show: true, color: '#fff', fontSize: 10, formatter: text }
  };
}

function valid(point, dates) {
  return point && dates.has(point.date) && Number.isFinite(point.price);
}

function overlay(lines, points, length) {
  const color = PATTERN_COLORS.triangleUp;
  return {
    id: 'pattern-triangle',
    name: TRIANGLE_SERIES_NAME,
    type: 'line',
    data: Array.from({ length }, () => null),
    showSymbol: false,
    lineStyle: { color, width: 2 },
    itemStyle: { color },
    z: 6,
    emphasis: { disabled: true },
    markLine: { symbol: ['none', 'none'], silent: true, animation: false, label: { show: false }, data: lines },
    markPoint: { silent: true, animation: false, data: points }
  };
}

export function buildPatternSeries(patterns, bars = [], { showTriangles = true } = {}) {
  const items = patterns?.triangles || [];
  if (!showTriangles || !items.length || !bars.length) return [];
  const dates = new Set(bars.map(bar => bar.date));
  const lines = [];
  const points = [];
  items.forEach(item => {
    const [u0, u1] = item.upper || [];
    const [l0, l1] = item.lower || [];
    if (![u0, u1, l0, l1].every(point => valid(point, dates))) return;
    const color = item.direction === 'up' ? PATTERN_COLORS.triangleUp
      : item.direction === 'down' ? PATTERN_COLORS.triangleDown : PATTERN_COLORS.triangle;
    const type = item.status === 'active' ? 'dashed' : 'solid';
    lines.push(segment(u0, u1, { color, width: 1.8, type }), segment(l0, l1, { color, width: 1.8, type }));
    if (item.direction === 'up') points.push(label('△↑', u1.date, u1.price, color));
    else if (item.direction === 'down') points.push(label('△↓', l1.date, l1.price, color, true));
    else if (item.status === 'active') points.push(label('△', u1.date, u1.price, color));
  });
  return lines.length ? [overlay(lines, points, bars.length)] : [];
}

// 最近一个形态的状态，用于图下方的说明行。
export function latestPatternSummary(patterns) {
  const triangles = patterns?.triangles || [];
  const latestTriangle = triangles.reduce((latest, item) => {
    const date = item.breakout_date || item.detected_date || '';
    return !latest || date > latest.date ? { date, label: item.status_label, status: item.status } : latest;
  }, null);
  return { triangleCount: triangles.length, latestTriangle };
}
