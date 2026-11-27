export function buildFlagSeries(pattern, bars = []) {
  if (!pattern || !bars.length) return [];
  const dates = new Map(bars.map((bar, i) => [bar.date, i]));
  function segment(points, field) {
    if (points?.length !== 2) return null;
    const start = dates.get(points[0].date);
    const end = dates.get(points[1].date);
    const a = points[0][field], b = points[1][field];
    if (start === undefined || end === undefined || end <= start || !Number.isFinite(a) || !Number.isFinite(b)) return null;
    return bars.map((_, i) => i < start || i > end ? null : a + (b - a) * (i - start) / (end - start));
  }
  return [
    { name: '旗杆', data: segment(pattern.pole, 'price'), color: '#b36a14', type: 'solid' },
    { name: '旗面上轨', data: segment(pattern.channel, 'upper'), color: '#197b70', type: 'dashed' },
    { name: '旗面下轨', data: segment(pattern.channel, 'lower'), color: '#b34f61', type: 'dashed' }
  ].filter(item => item.data).map(item => ({
    name: item.name, type: 'line', data: item.data, showSymbol: false,
    connectNulls: false, lineStyle: { color: item.color, width: 2, type: item.type },
    itemStyle: { color: item.color }, z: 5, emphasis: { disabled: true }
  }));
}
