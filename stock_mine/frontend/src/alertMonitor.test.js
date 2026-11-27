import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { emptyMonitorFilters, monitorPercent, monitorPrice, monitorQuery, validateMonitorFilters } from './alertMonitor.js';

test('monitor filters preserve zero, negative returns and equal day bounds', () => {
  const filters = { ...emptyMonitorFilters, min_days: '2', max_days: '2', min_return: '-10', max_return: '0' };
  assert.equal(validateMonitorFilters(filters), '');
  const query = new URLSearchParams(monitorQuery(filters));
  assert.equal(query.get('max_return'), '0');
  assert.equal(query.get('min_days'), '2');
  assert.equal(query.get('max_days'), '2');
  assert.equal(query.get('min_return'), '-10');
  assert.equal(query.get('selected_from'), null);
});

test('invalid ranges cannot silently send a different filter', () => {
  assert.match(validateMonitorFilters({ ...emptyMonitorFilters, min_days: '3', max_days: '2' }), /下限/);
  assert.match(validateMonitorFilters({ ...emptyMonitorFilters, min_days: '-1' }), /非负/);
  assert.match(validateMonitorFilters({ ...emptyMonitorFilters, min_return: 'NaN' }), /数字/);
  assert.match(validateMonitorFilters({ ...emptyMonitorFilters, selected_from: '2026-09-14', selected_to: '2026-09-11' }), /入选日期/);
});

test('missing and stale returns never render as zero gain', () => {
  assert.equal(monitorPercent(null), '--');
  assert.equal(monitorPercent(undefined), '--');
  assert.equal(monitorPercent(NaN), '--');
  assert.equal(monitorPercent(0), '0.00%');
  assert.equal(monitorPercent(10), '+10.00%');
  assert.equal(monitorPercent(-5), '-5.00%');
  assert.equal(monitorPrice(null), '--');
});

test('monitor query encodes names and includes pagination and dated filters', () => {
  const params = new URLSearchParams(monitorQuery({ ...emptyMonitorFilters, q: '银行 & 测试', selected_from: '2026-09-10', scope: 'all' }, 3));
  assert.equal(params.get('q'), '银行 & 测试');
  assert.equal(params.get('selected_from'), '2026-09-10');
  assert.equal(params.get('scope'), 'all');
  assert.equal(params.get('page'), '3');
});

test('tracking has a separate view, frozen-price columns, filters and persisted history', () => {
  const app = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8');
  const panel = readFileSync(new URL('./components/AlertMonitorPanel.jsx', import.meta.url), 'utf8');
  const api = readFileSync(new URL('./api.js', import.meta.url), 'utf8');
  assert.match(app, /view.*alert-monitor/);
  assert.match(app, /row\.first_scan_id/);
  for (const field of ['baseline_close', 'selected_date', 'trading_days', 'return_pct', 'quote_date']) assert.ok(panel.includes(`row.${field}`));
  assert.match(panel, /api\.refreshAlertMonitor/);
  assert.match(panel, /api\.alertMonitorHistory/);
  assert.match(panel, /min_days/);
  assert.match(panel, /min_return/);
  assert.match(api, /\/api\/alert-monitor\/refresh/);
});
