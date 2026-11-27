import test from 'node:test';
import assert from 'node:assert/strict';
import { isTerminal, formatProgress, progressPercent } from './scanProgress.js';

test('isTerminal 识别结束状态', () => {
  assert.equal(isTerminal({ status: 'running' }), false);
  assert.equal(isTerminal({ status: 'completed' }), true);
  assert.equal(isTerminal({ status: 'failed' }), true);
  assert.equal(isTerminal({ status: 'cancelled' }), true);
  assert.equal(isTerminal(null), false);
});

test('progress identifies stale-data exclusions within the failure count', () => {
  assert.match(formatProgress({ processed: 5, total: 5, hits: 0, failed: 2, stale: 1 }), /行情未更新 1/);
});

test('formatProgress 生成进度文案', () => {
  const text = formatProgress({
    processed: 320, total: 4997, hits: 8, failed: 12, current_stock: '300750 宁德时代'
  });
  assert.match(text, /320\/4997/);
  assert.match(text, /命中 8/);
  assert.match(text, /失败 12/);
  assert.match(text, /300750/);
});

test('formatProgress 在无当前股票时省略该段', () => {
  const text = formatProgress({ processed: 0, total: 100, hits: 0, failed: 0, current_stock: '' });
  assert.doesNotMatch(text, /当前/);
});

test('progressPercent 计算百分比并处理 total 为 0', () => {
  assert.equal(progressPercent({ processed: 50, total: 200 }), 25);
  assert.equal(progressPercent({ processed: 0, total: 0 }), 0);
  assert.equal(progressPercent({ processed: 300, total: 200 }), 100);
});
