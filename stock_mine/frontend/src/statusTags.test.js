import test from 'node:test';
import assert from 'node:assert/strict';
import { tagTone, sortTags } from './statusTags.js';

test('缩量回踩是利好信号，用正面色', () => {
  assert.equal(tagTone('缩量回踩（第3天）'), 'good');
});

test('放量回踩说明抛压仍重，不能和缩量回踩同色', () => {
  assert.equal(tagTone('放量回踩（第2天）'), 'warn');
  assert.notEqual(tagTone('放量回踩（第2天）'), tagTone('缩量回踩（第2天）'));
});

test('跌破突破位是风险，用警示色', () => {
  assert.equal(tagTone('跌破突破位（第4天）'), 'risk');
});

test('未回踩是中性状态', () => {
  assert.equal(tagTone('未回踩'), 'neutral');
});

test('没见过的标记退化为中性，不能崩', () => {
  assert.equal(tagTone('某个新标记'), 'neutral');
  assert.equal(tagTone(''), 'neutral');
  assert.equal(tagTone(undefined), 'neutral');
});

test('风险标记排在前面，先看见坏消息', () => {
  const sorted = sortTags(['缩量回踩（第1天）', '跌破突破位（第3天）']);
  assert.deepEqual(sorted, ['跌破突破位（第3天）', '缩量回踩（第1天）']);
});

test('同类标记保持原有先后', () => {
  const input = ['缩量回踩（第1天）', '未回踩'];
  assert.deepEqual(sortTags(input), input);
  assert.deepEqual(sortTags([]), []);
  assert.deepEqual(sortTags(undefined), []);
});
