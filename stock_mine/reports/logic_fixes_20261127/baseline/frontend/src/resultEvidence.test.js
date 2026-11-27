import assert from 'node:assert/strict';
import test from 'node:test';
import { formatPointPrice, priceBasisLabel, resistanceEvidence, selectionEvidence } from './resultEvidence.js';

function legacyRow() {
  return {
    stock_code: '600780', stock_name: '通宝能源', category: 'buy_alert',
    key_line_price: 6.15, breakout_date: '2026-08-31',
    prebreakout_check: { anchor_date: '2026-08-04', anchor_confirmed_date: '2026-08-13' },
    key_lines: [{
      price: 6.15, is_primary: true, resistance_point_count: 2,
      resistance_dates: ['2025-11-11', '2026-08-04'], support_dates: ['2026-08-20'],
    }],
    chart: { bars: [{ date: '2026-08-04', open: 6.1, close: 6.21, high: 6.22 }] },
    reasons: ['全历史累计 3 个精准点（阻力 2、支撑 1）', '精准线在本次突破前被独立波峰确认 2 次',
      '最近出现收盘突破，突破量比过滤已关闭', '缩量回踩确认', '收盘站上 MA20'],
  };
}

test('legacy anchor uses its own saved candle high, not the line median or candle body', () => {
  const evidence = resistanceEvidence(legacyRow());
  assert.equal(evidence.anchor.price, 6.22);
  assert.equal(evidence.anchor.price_basis, 'high');
  assert.equal(evidence.anchor.price_source, 'snapshot');
  assert.equal(evidence.anchor.date, '2026-08-04');
  assert.equal(evidence.anchor.confirmed_date, '2026-08-13');
  assert.equal(evidence.anchor.role, 'before_breakout');
  assert.equal(evidence.linePrice, 6.15);
  assert.equal(evidence.count, 2);
});

test('all saved resistance dates remain visible while missing old prices stay missing', () => {
  const evidence = resistanceEvidence(legacyRow());
  assert.deepEqual(evidence.points.map(point => point.date), ['2026-08-04', '2025-11-11']);
  assert.equal(evidence.points[1].price, null);
  assert.equal(evidence.points[1].price_source, 'missing');
  assert.equal(evidence.missingPrices, 1);
  assert.equal(formatPointPrice(evidence.points[1].price), '未记录');
});

test('structured point prices and their recorded price basis take precedence over charts', () => {
  const row = legacyRow();
  row.key_lines[0].resistance_price_basis = 'body_high';
  row.key_lines[0].resistance_points = [
    { date: '2026-08-04', price: 6.215678, price_basis: 'body_high', confirmed_date: '2026-08-13' },
    { date: '2025-11-11', price: 6.17, price_basis: 'body_high', confirmed_date: '2025-11-14' },
  ];
  const evidence = resistanceEvidence(row);
  assert.equal(evidence.anchor.price, 6.215678);
  assert.equal(evidence.anchor.price_source, 'recorded');
  assert.equal(evidence.basis, 'body_high');
  assert.equal(evidence.missingPrices, 0);
  assert.equal(formatPointPrice(evidence.anchor.price), '6.215678');
  assert.equal(priceBasisLabel(evidence.basis), '实体上沿');
});

test('new anchor survives without charts, key lines, or enabled path filters', () => {
  const row = {
    category: 'buy_alert', key_line_price: 6.15,
    resistance_anchor: {
      date: '2026-08-04', price: 6.22, price_basis: 'high',
      confirmed_date: '2026-08-13', role: 'before_breakout',
    },
    prebreakout_check: { status: 'disabled' },
  };
  const evidence = resistanceEvidence(row);
  assert.equal(evidence.anchor.price, 6.22);
  assert.equal(evidence.points.length, 1);
  assert.equal(evidence.count, null, 'one anchor is not the full historical point count');
});

test('fallback anchor must already be confirmed before breakout', () => {
  const row = legacyRow();
  delete row.prebreakout_check;
  assert.equal(resistanceEvidence(row).anchor, null, 'legacy dates alone cannot prove confirmation timing');
  row.key_lines[0].resistance_points = [
    { date: '2026-08-04', price: 6.22, confirmed_date: '2026-08-13' },
    { date: '2026-08-20', price: 6.2, confirmed_date: '2026-08-31' },
    { date: '2026-08-21', price: 6.21, confirmed_date: '2026-09-02' },
  ];
  assert.equal(resistanceEvidence(row).anchor.date, '2026-08-04');
  delete row.breakout_date;
  assert.equal(resistanceEvidence(row).anchor.role, 'latest_confirmed');
  assert.equal(resistanceEvidence(row).anchor.date, '2026-08-21');
});

test('flag-only matches do not invent precision resistance or precision reasons', () => {
  const row = { ...legacyRow(), pattern_type: 'flag_breakout', flag_direction: 'horizontal', reasons: ['旗面缩量后突破上轨'] };
  const evidence = selectionEvidence(row);
  assert.equal(evidence.resistance, null);
  assert.equal(evidence.groups.length, 1);
  assert.equal(evidence.groups[0].label, '水平旗形');
  assert.deepEqual(evidence.groups[0].preview, ['旗面缩量后突破上轨']);
});

test('mixed matches keep precision anchor and pending reasons separate from the primary flag', () => {
  const precision = { ...legacyRow(), pattern_type: 'precision_pullback', category: 'watch' };
  const flag = {
    pattern_type: 'flag_breakout', flag_direction: 'ascending', category: 'buy_alert',
    reasons: ['放量突破旗面上轨'], risks: ['旗面上倾，留意回落'],
  };
  const row = {
    ...legacyRow(), ...flag, key_line_price: null, reasons: ['联合筛选汇总'],
    pattern_matches: [precision, flag],
  };
  const evidence = selectionEvidence(row);
  assert.equal(evidence.resistance.anchor.price, 6.22);
  assert.equal(evidence.resistance.linePrice, 6.15);
  assert.equal(evidence.groups[0].label, '精准线回调（待确认）');
  assert.deepEqual(evidence.groups[1].reasons, ['放量突破旗面上轨']);
  assert.ok(!evidence.groups[0].reasons.includes('联合筛选汇总'));
  assert.deepEqual(evidence.risks, ['旗面上倾，留意回落']);
});

test('previews use recorded trading reasons and full evidence preserves all unique reasons and risks', () => {
  const row = legacyRow();
  row.reasons.push('缩量回踩确认', '', null);
  row.risks = [' 技术预警不构成投资建议 ', '回踩量高于常态', '回踩量高于常态'];
  row.industry_risk_level = 'uncovered';
  row.industry_risk_message = '行业数据暂未覆盖';
  const evidence = selectionEvidence(row);
  assert.deepEqual(evidence.groups[0].preview, ['最近出现收盘突破，突破量比过滤已关闭', '缩量回踩确认']);
  assert.equal(evidence.groups[0].reasons.length, 5);
  assert.equal(evidence.reasonCount, 5);
  assert.deepEqual(evidence.risks, ['回踩量高于常态', '行业数据暂未覆盖']);
});

test('missing reasons can show the recorded suggestion without inventing an explanation', () => {
  assert.deepEqual(selectionEvidence({ category: 'watch', suggestion: '等待回踩' }).groups[0].preview, ['等待回踩']);
  assert.deepEqual(selectionEvidence({ category: 'watch' }).groups[0].preview, []);
  assert.deepEqual(selectionEvidence({ category: 'none' }).groups, []);
});

test('malformed optional historical fields do not break the results table', () => {
  for (const row of [null, undefined, {}, {
    ...legacyRow(), chart: { bars: {}, key_lines: {} }, key_lines: [null, 4],
    risks: {}, reasons: {}, pattern_matches: [null, 7, { category: 'watch', risks: 'not an array' }],
  }]) {
    assert.doesNotThrow(() => selectionEvidence(row));
  }
  const row = legacyRow();
  row.key_lines = [];
  row.chart.key_lines = [{ price: 6.15, resistance_dates: 'not an array' }];
  assert.doesNotThrow(() => selectionEvidence(row));
  assert.equal(formatPointPrice(NaN), '未记录');
  assert.equal(formatPointPrice(0), '未记录');
  assert.equal(priceBasisLabel('unknown'), '取价未记录');
});
