import assert from 'node:assert/strict';
import test from 'node:test';
import { EXECUTION_MODEL, backtestMetric, backtestReturn, isNetBacktest, entryLabel } from './backtestPresentation.js';

test('net metrics never fall back to old gross wins', () => {
  const summary = { execution_model: EXECUTION_MODEL, t5_positive_rate: 100, net_t5_positive_rate: 0, net_t5_sample_count: 3 };
  assert.equal(backtestMetric(summary).rate, 0);
  assert.equal(backtestMetric(summary).count, 3);
  assert.equal(backtestMetric(summary).label, 'T+5 净胜率');
  assert.equal(backtestMetric({ execution_model: EXECUTION_MODEL, t5_positive_rate: 100 }).rate, undefined);
});

test('legacy reports retain an explicit signal-return label', () => {
  assert.equal(isNetBacktest({ t5_positive_rate: 60 }), false);
  assert.equal(backtestMetric({ t5_positive_rate: 60 }).label, 'T+5 信号后正收益');
  assert.equal(backtestMetric({ t5_positive_rate: 60 }).rate, 60);
});

test('blocked and unfinished trades never display the old profitable return as net', () => {
  assert.equal(backtestReturn({ execution_model: EXECUTION_MODEL, net_return_5: null, forward_return_5: 10 }), null);
  assert.equal(backtestReturn({ execution_model: EXECUTION_MODEL, net_return_5: -2.97, forward_return_5: 5 }), -2.97);
  assert.equal(entryLabel('blocked'), '不可入场');
  assert.equal(entryLabel('pending'), '待入场');
});

test('old saved net reports remain net without pretending to use the new calendar model', () => {
  const old = { execution_model: 'next_open_cost_v2', net_t5_positive_rate: 40, net_return_5: -3, forward_return_5: 5 };
  assert.equal(backtestMetric(old).rate, 40);
  assert.equal(backtestReturn(old), -3);
  assert.equal(old.execution_model, 'next_open_cost_v2');
  assert.equal(isNetBacktest({ execution_model: 'unrecognized' }), false);
});
