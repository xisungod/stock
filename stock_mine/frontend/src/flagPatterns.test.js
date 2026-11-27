import test from 'node:test';
import assert from 'node:assert/strict';
import { scanPayload, patternLabel, patternMatches, matchLabel, referencePrice } from './flagPatterns.js';
import { buildFlagSeries } from './components/flagLines.js';

test('flag requests clear only inapplicable breakout-age filters', () => {
  const form = { provider: 'auto', strategy_mode: 'flag_breakout', flag_direction: 'ascending',
    min_days_since_breakout: '5', max_days_since_breakout: '7', breakout_vol_mult: '1.8', require_breakout: true, signal_offset: 0 };
  const sent = scanPayload(form);
  assert.equal(sent.min_days_since_breakout, 0);
  assert.equal(sent.max_days_since_breakout, 0);
  assert.equal(sent.flag_direction, 'ascending');
  assert.equal(sent.require_breakout, true);
  assert.equal(sent.breakout_vol_mult, 1.8);
  assert.equal(form.min_days_since_breakout, '5');
  assert.equal(scanPayload({ ...form, strategy_mode: 'precision_pullback' }).min_days_since_breakout, 5);
});

test('combined is the default and each filter stays with its own branch', () => {
  assert.equal(scanPayload({}).strategy_mode, 'combined');
  const sent = scanPayload({ strategy_mode: 'combined', flag_direction: 'ascending', min_days_since_breakout: 5 });
  assert.equal(sent.flag_direction, 'ascending');
  assert.equal(sent.min_days_since_breakout, 5);
});

test('precision volume filtering defaults off while flags still require volume', () => {
  assert.equal(scanPayload({}).require_breakout_volume, false);
  assert.equal(scanPayload({ strategy_mode: 'precision_pullback' }).require_breakout_volume, false);
  const enabled = scanPayload({ strategy_mode: 'precision_pullback', require_breakout_volume: true, breakout_vol_mult: '1.8' });
  assert.equal(enabled.require_breakout_volume, true);
  assert.equal(enabled.breakout_vol_mult, 1.8);
  const flag = scanPayload({ strategy_mode: 'flag_breakout', require_breakout_volume: false, breakout_vol_mult: '1.8' });
  assert.equal(flag.require_breakout_volume, true);
  assert.equal(flag.breakout_vol_mult, 1.8);
});

test('disabled hidden volume input cannot block a precision scan or change its other filters', () => {
  const sent = scanPayload({ strategy_mode: 'precision_pullback', require_breakout_volume: false, breakout_vol_mult: -10 });
  assert.equal(sent.breakout_vol_mult, 1.5);
  assert.equal(sent.require_breakout_volume, false);
  assert.equal(sent.require_pullback, true);
  assert.equal(sent.max_pullback_days, 10);
  assert.equal(sent.prebreakout_reject_above_days, 30);
  assert.equal(sent.max_prebreakout_extension_pct, .5);
});

test('confirmed pullbacks default to day 1 through 10 and support inclusive configurable days', () => {
  const initial = scanPayload({});
  assert.equal(initial.require_pullback, true);
  assert.equal(initial.min_pullback_days, 1);
  assert.equal(initial.max_pullback_days, 10);
  for (const [minimum, maximum] of [[3, 7], [5, 5], [12, 20], [60, 60]]) {
    const sent = scanPayload({ min_pullback_days: String(minimum), max_pullback_days: String(maximum) });
    assert.equal(sent.min_pullback_days, minimum);
    assert.equal(sent.max_pullback_days, maximum);
  }
  assert.equal(scanPayload({ require_pullback: false }).require_pullback, false);
  for (const values of [[0, 10], [3, 2], [1, 61], [1, 0], [1.5, 10], [1, 2.5], ['', 10], [1, ''], ['abc', 10], [1, Infinity]]) {
    assert.throws(() => scanPayload({ min_pullback_days: values[0], max_pullback_days: values[1] }), /回调天数/);
  }
});

test('flag-only ignores the precision pullback gate and hidden invalid day range', () => {
  const form = { strategy_mode: 'flag_breakout', require_pullback: true,
    min_pullback_days: 0, max_pullback_days: 'bad' };
  const sent = scanPayload(form);
  assert.equal(sent.require_pullback, false);
  assert.equal(sent.min_pullback_days, 1);
  assert.equal(sent.max_pullback_days, null);
  assert.equal(form.require_pullback, true);
  assert.equal(form.max_pullback_days, 'bad');
});

test('prebreakout defaults to 50 percent and preserves explicit old values and zero', () => {
  assert.equal(scanPayload({}).max_prebreakout_extension_pct, .5);
  assert.equal(scanPayload({ prebreakout_extension_percent: '50' }).max_prebreakout_extension_pct, .5);
  assert.equal(scanPayload({ prebreakout_extension_percent: '20' }).max_prebreakout_extension_pct, .2);
  assert.equal(scanPayload({ prebreakout_extension_percent: '0' }).max_prebreakout_extension_pct, 0);
  assert.equal(scanPayload({ prebreakout_extension_percent: '' }).max_prebreakout_extension_pct, .5);
  assert.equal(scanPayload({ strategy_mode: 'flag_breakout', prebreakout_extension_percent: 20 }).max_prebreakout_extension_pct, null);
  for (const value of ['abc', -1, 101, Infinity]) {
    assert.throws(() => scanPayload({ prebreakout_extension_percent: value }), /0～100/);
  }
});

test('duration defaults to rejecting at thirty trading days with a two percent close buffer', () => {
  const sent = scanPayload({});
  assert.equal(sent.prebreakout_reject_above_days, 30);
  assert.equal(sent.prebreakout_above_buffer_pct, .02);
  assert.equal(sent.breakout_vol_mult, 1.5);
  assert.equal(sent.max_prebreakout_extension_pct, .5);
  assert.equal(sent.max_pullback_days, 10);
  assert.equal(scanPayload({ prebreakout_reject_above_days: '10' }).prebreakout_reject_above_days, 10);
  assert.equal(scanPayload({ prebreakout_reject_above_days: '30' }).prebreakout_reject_above_days, 30);
  const edited = scanPayload({ prebreakout_reject_above_days: '8', prebreakout_above_buffer_percent: '0' });
  assert.equal(edited.prebreakout_reject_above_days, 8);
  assert.equal(edited.prebreakout_above_buffer_pct, 0);
  assert.equal(scanPayload({ prebreakout_reject_above_days: '' }).prebreakout_reject_above_days, 30);
  assert.equal(scanPayload({ prebreakout_above_buffer_percent: '' }).prebreakout_above_buffer_pct, .02);
  for (const value of [0, -1, 251, 9.5, 'abc', Infinity]) {
    assert.throws(() => scanPayload({ prebreakout_reject_above_days: value }), /1～250/);
  }
  for (const value of [-1, 101, 'abc', Infinity]) {
    assert.throws(() => scanPayload({ prebreakout_above_buffer_percent: value }), /0～100/);
  }
});

test('flag-only ignores both path filters even when their hidden form values are invalid', () => {
  const sent = scanPayload({ strategy_mode: 'flag_breakout', prebreakout_extension_percent: 'abc',
    prebreakout_reject_above_days: 0, prebreakout_above_buffer_percent: -1 });
  assert.equal(sent.max_prebreakout_extension_pct, null);
  assert.equal(sent.prebreakout_reject_above_days, null);
  assert.equal(sent.prebreakout_above_buffer_pct, .02);
});

test('both hits retain both labels and pending is never presented as confirmed', () => {
  const row = { pattern_type: 'precision_pullback', pattern_matches: [
    { pattern_type: 'precision_pullback', category: 'buy_alert' },
    { pattern_type: 'flag_breakout', flag_direction: 'ascending', category: 'watch' }
  ] };
  assert.equal(patternLabel(row), '精准线回调 + 上倾旗形（待确认）');
  assert.equal(patternMatches(row).length, 2);
  assert.equal(patternLabel({ pattern_matches: [] }), '未命中');
  assert.equal(matchLabel({ pattern_type: 'flag_breakout', flag_direction: 'ascending', category: 'watch', data_stale: true }), '上倾旗形（待更新）');
  assert.equal(patternLabel({ ...row, pattern_matches: [{ pattern_type: 'flag_breakout', category: 'none' }] }), '未命中');
});

test('a flag upper boundary is not mislabeled as a precision line', () => {
  const row = { pattern_type: 'flag_breakout', key_line_price: null, flag_pattern: { direction: 'ascending', upper_price: 12 } };
  assert.equal(patternLabel(row), '上倾旗形');
  assert.equal(referencePrice(row), 12);
  assert.equal(patternLabel({}), '精准线回调');
});

test('channel overlays end at the original signal and use trading indices', () => {
  const bars = ['2026-08-03', '2026-08-04', '2026-08-05', '2026-08-06', '2026-08-07'].map(date => ({ date }));
  const pattern = { channel: [{ date: bars[1].date, upper: 12, lower: 11 }, { date: bars[3].date, upper: 13, lower: 12 }] };
  const series = buildFlagSeries(pattern, bars);
  assert.deepEqual(series[0].data, [null, 12, 12.5, 13, null]);
  assert.deepEqual(series[1].data, [null, 11, 11.5, 12, null]);
  assert.equal(series[0].connectNulls, false);
  assert.deepEqual(buildFlagSeries(pattern, bars.slice(3)), []);
});

test('new scans send the settled signal bar and the 2026-11-27 rule defaults', () => {
  const sent = scanPayload({});
  assert.equal(sent.signal_offset, 0);
  assert.equal(sent.pullback_max_normal_volume_ratio, 1);
  assert.equal(sent.max_post_breakout_runup_pct, 0.1);
  assert.equal(sent.max_event_age_bars, 750);
  assert.equal(sent.market_filter, 'risk');
  assert.equal(scanPayload({ market_filter: 'downgrade', signal_offset: '1' }).market_filter, 'downgrade');
  assert.equal(scanPayload({ signal_offset: '1' }).signal_offset, 1);
});
