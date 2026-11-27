import { matchLabel, patternMatches } from './flagPatterns.js';

function dateString(value) {
  if (typeof value !== 'string') return null;
  const date = value.slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : null;
}

function positiveNumber(value) {
  if (value == null || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function cleanText(values) {
  return [...new Set((Array.isArray(values) ? values : [])
    .filter(value => typeof value === 'string')
    .map(value => value.trim()).filter(Boolean))];
}

export function priceBasisLabel(basis) {
  return { high: '最高价', body_high: '实体上沿', close: '收盘价' }[basis] || '取价未记录';
}

export function formatPointPrice(value) {
  const number = positiveNumber(value);
  return number == null ? '未记录' : number.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
}

function snapshotPrice(bar, basis) {
  if (!bar) return null;
  if (basis === 'high') return positiveNumber(bar.high);
  if (basis === 'close') return positiveNumber(bar.close);
  if (basis === 'body_high') {
    const open = positiveNumber(bar.open);
    const close = positiveNumber(bar.close);
    return open == null || close == null ? null : Math.max(open, close);
  }
  return null;
}

export function resistanceEvidence(row = {}) {
  row = row || {};
  const precision = patternMatches(row).find(match => (match.pattern_type || 'precision_pullback') === 'precision_pullback');
  if (!precision) return null;
  const linePrice = positiveNumber(precision.key_line_price);
  const lines = (Array.isArray(row.key_lines) && row.key_lines.length ? row.key_lines
    : Array.isArray(row.chart?.key_lines) ? row.chart.key_lines : []).filter(line => line && typeof line === 'object');
  const samePrice = line => linePrice != null && positiveNumber(line.price) != null
    && Math.abs(Number(line.price) - linePrice) <= 1e-6;
  const line = lines.find(item => item.is_primary && samePrice(item))
    || lines.find(samePrice) || null;
  const bars = new Map((Array.isArray(row.chart?.bars) ? row.chart.bars : [])
    .filter(bar => bar && typeof bar === 'object').map(bar => [dateString(bar.date), bar]));
  const basis = line?.resistance_price_basis || 'high';
  const completeList = Array.isArray(line?.resistance_points) || Array.isArray(line?.resistance_dates);
  const rawPoints = Array.isArray(line?.resistance_points)
    ? line.resistance_points : (Array.isArray(line?.resistance_dates) ? line.resistance_dates : []).map(date => ({ date }));
  const points = new Map();

  function normalizedPoint(raw) {
    const date = dateString(raw?.date);
    if (!date) return null;
    const pointBasis = raw.price_basis || basis;
    const recorded = positiveNumber(raw.price);
    const fromChart = snapshotPrice(bars.get(date), pointBasis);
    return {
      date, price: recorded ?? fromChart,
      confirmed_date: dateString(raw.confirmed_date),
      price_basis: pointBasis,
      price_source: recorded != null ? raw.price_source || 'recorded' : fromChart != null ? 'snapshot' : 'missing',
    };
  }

  for (const raw of rawPoints) {
    const point = normalizedPoint(raw);
    if (point) points.set(point.date, point);
  }
  // A saved anchor is authoritative. Never substitute the line median for its point price.
  const primaryPrecision = (row.pattern_type || 'precision_pullback') === 'precision_pullback';
  const savedAnchor = precision.resistance_anchor ?? (primaryPrecision ? row.resistance_anchor : null);
  const check = precision.prebreakout_check ?? (primaryPrecision ? row.prebreakout_check : null);
  let rawAnchor = savedAnchor;
  if (!rawAnchor && check?.anchor_date) {
    const known = points.get(dateString(check.anchor_date));
    rawAnchor = {
      date: check.anchor_date, price: check.anchor_price ?? known?.price,
      confirmed_date: check.anchor_confirmed_date ?? known?.confirmed_date,
      price_basis: check.anchor_price_basis || known?.price_basis || basis,
      price_source: positiveNumber(check.anchor_price) != null ? 'recorded' : known?.price_source,
      role: 'before_breakout',
    };
  }
  if (!rawAnchor) {
    const breakout = dateString(precision.breakout_date);
    const eligible = [...points.values()].filter(point => !breakout || (
      point.confirmed_date && point.date < breakout && point.confirmed_date < breakout
    )).sort((a, b) => b.date.localeCompare(a.date));
    if (eligible.length) rawAnchor = { ...eligible[0], role: breakout ? 'before_breakout' : 'latest_confirmed' };
  }
  const normalized = rawAnchor ? normalizedPoint(rawAnchor) : null;
  const anchor = normalized ? { ...normalized, role: rawAnchor.role || 'before_breakout' } : null;
  if (anchor) points.set(anchor.date, { ...points.get(anchor.date), ...anchor });
  const allPoints = [...points.values()].sort((a, b) => b.date.localeCompare(a.date));
  return {
    linePrice, anchor, points: allPoints, basis,
    count: Number.isInteger(line?.resistance_point_count) ? line.resistance_point_count : completeList ? allPoints.length : null,
    missingPrices: allPoints.filter(point => point.price == null).length,
  };
}

export function selectionEvidence(row = {}) {
  row = row || {};
  const primaryType = row.pattern_type || 'precision_pullback';
  const groups = patternMatches(row).map(match => {
    const type = match.pattern_type || 'precision_pullback';
    const primary = type === primaryType;
    const reasons = cleanText(match.reasons ?? (primary ? row.reasons : []));
    const suggestion = typeof match.suggestion === 'string' ? match.suggestion
      : primary && typeof row.suggestion === 'string' ? row.suggestion : '';
    // Counts already have their own columns; keep the recorded trading reasons in the preview.
    const nonCounts = reasons.filter(reason => !reason.startsWith('全历史累计')
      && !reason.startsWith('精准线在本次突破前'));
    const preview = (nonCounts.length ? nonCounts : reasons).slice(0, 2);
    return {
      type, label: matchLabel(match), category: match.category,
      reasons, suggestion, preview: preview.length ? preview : suggestion ? [suggestion] : [],
      risks: cleanText(match.risks ?? (primary ? row.risks : [])),
    };
  });
  const risks = cleanText([
    ...groups.flatMap(group => group.risks), ...cleanText(row.risks),
    ...(['high', 'uncovered'].includes(row.industry_risk_level) || row.industry_stale
      ? [row.industry_risk_message] : []),
  ]).filter(risk => risk !== '技术预警不构成投资建议');
  return {
    groups, risks, resistance: resistanceEvidence(row),
    reasonCount: groups.reduce((count, group) => count + (group.reasons.length || (group.suggestion ? 1 : 0)), 0),
  };
}
