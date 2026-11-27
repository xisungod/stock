// K 线图上默认只画少数几根精准线：全部画出来会糊成一片，反而看不出价格贴着哪条走。
// 其余的线在右侧「精准线结构」里点开后再叠加显示。
export const DEFAULT_VISIBLE_LINES = 2;

export function lineKey(line) {
  if (!line) return '';
  if (line.version_id) return line.version_id;
  return `${line.price}|${line.formation_date || ''}`;
}

export function snapshotMatchesBars(line, bars = []) {
  const points = [...(line?.resistance_points || []), ...(line?.support_points || [])];
  const byDate = new Map(bars.map(bar => [bar.date, bar]));
  return points.length > 0 && points.every(point => {
    const bar = byDate.get(point.date);
    const actual = Number(bar?.[point.price_basis || 'high']);
    return bar && Number.isFinite(actual) && Number.isFinite(Number(point.price))
      && Math.abs(actual - Number(point.price)) <= .00101;
  });
}
function distanceTo(price) {
  return (line) => Math.abs(Number(line.price) - price);
}

function byPriceDesc(a, b) {
  return Number(b.price) - Number(a.price);
}

export function defaultVisibleLines(lines, currentPrice, limit = DEFAULT_VISIBLE_LINES) {
  const all = lines || [];
  if (!all.length) return [];
  const slots = Math.max(1, limit);
  // 触发预警的主线一定要画：结果列表里指的就是它，图上找不到会对不上。
  const primary = all.filter((line) => line.is_primary);
  const others = all
    .filter((line) => !line.is_primary)
    .sort((a, b) => distanceTo(currentPrice)(a) - distanceTo(currentPrice)(b));
  return [...primary, ...others].slice(0, slots).sort(byPriceDesc);
}

export function resolveVisibleLines(lines, currentPrice, selectedKeys = [], limit = DEFAULT_VISIBLE_LINES) {
  const base = defaultVisibleLines(lines, currentPrice, limit);
  const shown = new Set(base.map(lineKey));
  const picked = new Set(selectedKeys);
  const extra = (lines || []).filter((line) => picked.has(lineKey(line)) && !shown.has(lineKey(line)));
  return [...base, ...extra].sort(byPriceDesc);
}

export function selectedPrecisionLines(lines, selectedKeys = []) {
  const picked = new Set(selectedKeys || []);
  return (lines || []).filter(line => line && picked.has(lineKey(line)));
}

function pointDate(value) {
  return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value.slice(0, 10))
    ? value.slice(0, 10) : null;
}

function pointPrice(value) {
  const number = value == null || value === '' ? NaN : Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

export function lineTouchPoints(line, bars = []) {
  if (!line) return [];
  const byDate = bars instanceof Map ? bars : new Map(
    (Array.isArray(bars) ? bars : []).filter(Boolean).map(bar => [pointDate(bar.date), bar])
  );
  const points = ['resistance', 'support'].flatMap(kind => {
    const saved = new Map((Array.isArray(line[`${kind}_points`]) ? line[`${kind}_points`] : [])
      .filter(point => pointDate(point?.date)).map(point => [pointDate(point.date), point]));
    const dates = new Set([
      ...(Array.isArray(line[`${kind}_dates`]) ? line[`${kind}_dates`] : []).map(pointDate).filter(Boolean),
      ...saved.keys(),
    ]);
    return [...dates].map(date => {
      const record = saved.get(date);
      const basis = record?.price_basis || line[`${kind}_price_basis`] || (kind === 'resistance' ? 'high' : 'low');
      const bar = byDate.get(date);
      const body = basis === 'body_high' || basis === 'body_low';
      const bodyPrice = body && pointPrice(bar?.open) != null && pointPrice(bar?.close) != null
        ? (basis === 'body_high' ? Math.max : Math.min)(Number(bar.open), Number(bar.close)) : null;
      return {
        date, kind, price: pointPrice(record?.price) ?? (body ? bodyPrice : pointPrice(bar?.[basis])),
        price_basis: basis, confirmed_date: pointDate(record?.confirmed_date),
        line_price: pointPrice(line.price),
        origins: Array.isArray(record?.origins) ? record.origins : [],
        event_start_date: pointDate(record?.event_start_date),
        max_high: pointPrice(record?.max_high), min_low: pointPrice(record?.min_low),
        episode_id: record?.episode_id || null,
        local_confirmed_date: pointDate(record?.local_confirmed_date),
      };
    });
  });
  return points.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
}

export const PRECISION_POINT_COLORS = {
  resistance: '#b26a00', support: '#148a64', anchor: '#174ea6'
};

export function precisionAnchorLabel(anchor) {
  return anchor?.role === 'latest_confirmed' ? '最近阻力点' : '本次阻力点';
}

export function precisionPointTooltip(params) {
  const point = params?.data?.precisionPoint;
  if (!point) return '';
  const price = pointPrice(point.price);
  const date = pointDate(point.date);
  if (price == null || !date) return '';
  const basis = { high: '最高价', low: '最低价', body_high: '实体上沿', body_low: '实体下沿', close: '收盘价' }[point.price_basis] || '点价';
  const title = point.is_anchor ? precisionAnchorLabel(point)
    : point.kind === 'support' ? '支撑精准点' : '阻力精准点';
  const lines = point.lines.map(pointPrice).filter(value => value != null).join('、');
  const confirmed = pointDate(point.confirmed_date);
  const types = { local_turn: '局部转折', resistance_test: '阻力验证', support_test: '支撑验证' };
  const origins = (Array.isArray(point.origins) ? point.origins : []).map(origin => types[origin]).filter(Boolean);
  const start = pointDate(point.event_start_date);
  const high = pointPrice(point.max_high), low = pointPrice(point.min_low);
  const localConfirmed = pointDate(point.local_confirmed_date);
  return `<strong>${title}</strong><br/>发生日：${date}<br/>${basis}：${price}<br/>精准线：${lines}`
    + (confirmed ? `<br/>确认日：${confirmed}` : '')
    + (localConfirmed && localConfirmed !== confirmed ? `<br/>局部转折确认日：${localConfirmed}` : '')
    + (origins.length ? `<br/>事件来源：${origins.join('、')}` : '')
    + (start ? `<br/>试探起点：${start}` : '')
    + (high != null && low != null ? `<br/>整段最高 / 最低：${high} / ${low}` : '');
}

export function buildPrecisionPointSeries(lines, bars = [], { showPoints = true, evidence = null } = {}) {
  if (!showPoints) return [];
  const byDate = new Map((Array.isArray(bars) ? bars : []).filter(Boolean).map(bar => [pointDate(bar.date), bar]));
  const points = new Map();
  for (const line of lines || []) {
    for (const point of lineTouchPoints(line, byDate)) {
      // Old snapshots may have dates but no candles/prices. Never plot the line median as a point.
      if (point.price == null || !byDate.has(point.date)) continue;
      const anchor = point.kind === 'resistance' && point.date === evidence?.anchor?.date
        && point.line_price != null && pointPrice(evidence.linePrice) != null
        && Math.abs(point.line_price - Number(evidence.linePrice)) < 1e-6;
      const key = `${point.date}|${point.kind}|${point.price}`;
      const previous = points.get(key);
      const merged = previous || { ...point, lines: [], is_anchor: false };
      if (point.line_price != null && !merged.lines.includes(point.line_price)) merged.lines.push(point.line_price);
      if (anchor) {
        merged.is_anchor = true;
        merged.role = evidence.anchor.role;
        merged.confirmed_date = evidence.anchor.confirmed_date || point.confirmed_date;
      }
      points.set(key, merged);
    }
  }
  return [
    { kind: 'resistance', name: '阻力精准点', symbol: 'circle', size: 10 },
    { kind: 'support', name: '支撑精准点', symbol: 'diamond', size: 10 },
    { kind: 'anchor', name: precisionAnchorLabel(evidence?.anchor), symbol: 'circle', size: 14 },
  ].flatMap(group => {
    const data = [...points.values()].filter(point => (point.is_anchor ? 'anchor' : point.kind) === group.kind);
    if (!data.length) return [];
    return [{
      id: `precision-${group.kind}`, name: group.name, type: 'scatter',
      data: data.map(point => ({ value: [point.date, point.price], precisionPoint: point })),
      encode: { x: 0, y: 1 }, symbol: group.symbol, symbolSize: group.size,
      itemStyle: { color: '#fff', borderColor: PRECISION_POINT_COLORS[group.kind], borderWidth: group.kind === 'anchor' ? 3 : 2, opacity: 1 },
      tooltip: { trigger: 'item', confine: true, formatter: precisionPointTooltip },
      emphasis: { scale: 1.3 }, z: 7,
    }];
  });
}

const UP_COLOR = '#d94841';

export function buildMarkPoints(bars, markers, { showAlerts = true } = {}) {
  const byDate = new Map();
  (markers || []).forEach((marker) => {
    const list = byDate.get(marker.date) || [];
    list.push(marker);
    byDate.set(marker.date, list);
  });
  return (bars || []).flatMap((bar) => {
    const onThisDay = (byDate.get(bar.date) || []).filter((marker) => marker.type !== 'alert' || showAlerts);
    return onThisDay.map((marker, index) => {
      const alert = marker.type === 'alert';
      const breakout = marker.type === 'breakout';
      // 所有标记都不带文字：带文字就得撑到 20px 以上，形成的色块会盖住K线实体。
      // 名称留给 tooltip。预警贴在最低价下方，突破/回踩浮在最高价上方，
      // 靠形状（▲/●）和颜色区分，互不遮挡。
      return {
        name: marker.label,
        coord: [bar.date, alert ? bar.low : (marker.price ?? bar.high)],
        value: marker.label,
        symbol: alert || breakout ? 'triangle' : 'circle',
        symbolSize: alert ? 7 : breakout ? 10 : 9,
        symbolOffset: alert ? [0, 13] : [index * 9, -10],
        label: { show: false, color: '#fff', fontSize: 9, formatter: '{b}' },
        itemStyle: {
          color: alert ? 'rgba(23,78,166,.7)' : breakout ? UP_COLOR : '#c78112',
          borderColor: '#fff',
          borderWidth: alert ? 0 : 1
        }
      };
    });
  });
}

const PICKED_COLOR = '#174ea6';
const PRIMARY_COLOR = '#b26a00';

export function keyLineColor(line, index, selectedKeys = []) {
  if (selectedKeys.includes(lineKey(line))) return PICKED_COLOR;
  if (line.is_primary) return PRIMARY_COLOR;
  return index % 2 ? '#5176a8' : '#6f7e92';
}

export function keyLineLabel(line) {
  const points = line.point_count === null || line.point_count === undefined
    ? ''
    : ` · ${line.point_count}点`;
  const price = Number(line.price).toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  return `${line.is_primary ? '主线 ' : ''}${price}${points}`;
}

export function buildKeyLineMarkLines(lines, selectedKeys = []) {
  return (lines || []).map((line, index) => {
    const primary = Boolean(line.is_primary);
    const picked = selectedKeys.includes(lineKey(line));
    return {
      yAxis: line.price,
      name: keyLineLabel(line),
      lineStyle: {
        color: keyLineColor(line, index, selectedKeys),
        width: picked ? 2 : primary ? 1.8 : 1.1,
        type: picked || primary ? 'solid' : 'dashed',
        opacity: picked || primary ? 0.95 : 0.78
      },
      // 图内一律不画文字：带底色的标签块正好压在K线上，而图表下方的图例
      // 已经用同色色块列出了每条线的价格与点数，信息并不丢失。
      label: { show: false, formatter: keyLineLabel(line) }
    };
  });
}

// Formation dates come from the chosen engine's frozen evidence, not a chart-side point count.
export function buildFormationMarkers(lines, dates = []) {
  return (lines || []).flatMap((line, index) => {
    if (!line.formation_date || !dates.includes(line.formation_date)) return [];
    return [{
      name: `${keyLineLabel(line)} 形成`,
      coord: [line.formation_date, line.price],
      value: '',
      symbol: 'diamond',
      symbolSize: 8,
      label: { show: false },
      itemStyle: {
        color: '#fff',
        borderColor: keyLineColor(line, index, []),
        borderWidth: 1.6
      }
    }];
  });
}



const PEAK_COLOR = 'rgba(217,72,65,.88)';
const VALLEY_COLOR = 'rgba(20,138,100,.88)';

export function buildSwingSeries(chart, showSwings) {
  if (!showSwings) return [];
  const base = {
    type: 'scatter',
    symbol: 'triangle',
    symbolSize: 7,
    emphasis: { disabled: true },
    z: 5
  };
  const series = [
    {
      ...base,
      name: '波峰',
      data: (chart?.peaks || []).map((p) => [p.date, p.price]),
      symbolOffset: [0, -11],
      itemStyle: { color: PEAK_COLOR }
    },
    {
      ...base,
      name: '波谷',
      data: (chart?.valleys || []).map((p) => [p.date, p.price]),
      symbolRotate: 180,
      symbolOffset: [0, 11],
      itemStyle: { color: VALLEY_COLOR }
    }
  ];

  // 最右侧那个极值要等回撤发生才算数。在此之前它只是候选——画成空心，
  // 免得看图的人把"还没确认"当成"已确认"，回测里那等于用了未来数据。
  const pending = chart?.pending_swing;
  if (pending && pending.date) {
    const isPeak = pending.kind === 'peak';
    series.push({
      ...base,
      name: '候选点',
      data: [[pending.date, pending.price]],
      symbolSize: 9,
      symbolRotate: isPeak ? 0 : 180,
      symbolOffset: [0, isPeak ? -12 : 12],
      itemStyle: {
        color: 'transparent',
        borderColor: isPeak ? PEAK_COLOR : VALLEY_COLOR,
        borderWidth: 1.6
      }
    });
  }
  return series;
}
