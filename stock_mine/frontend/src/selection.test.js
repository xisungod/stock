import test from 'node:test';
import assert from 'node:assert/strict';
import { toggleCode, toggleAll, isAllSelected, keepExisting } from './selection.js';

const ROWS = [{ stock_code: '000001' }, { stock_code: '600519' }, { stock_code: '300750' }];

test('点一次选中，再点一次取消', () => {
  assert.deepEqual(toggleCode([], '000001'), ['000001']);
  assert.deepEqual(toggleCode(['000001'], '000001'), []);
  assert.deepEqual(toggleCode(['000001'], '600519'), ['000001', '600519']);
});

test('全选把当前列表的代码都放进来，且不丢掉列表外已选的', () => {
  assert.deepEqual(toggleAll(['999999'], ROWS).sort(), ['000001', '300750', '600519', '999999']);
});

test('已全选时再点全选＝清空这批，其余保留', () => {
  const all = ['000001', '600519', '300750', '999999'];
  assert.deepEqual(toggleAll(all, ROWS), ['999999']);
});

test('isAllSelected 只看当前列表这几只', () => {
  assert.equal(isAllSelected(['000001', '600519', '300750'], ROWS), true);
  assert.equal(isAllSelected(['000001', '600519'], ROWS), false);
  assert.equal(isAllSelected([], ROWS), false);
});

test('空列表不算全选，否则表头复选框会显示成勾上', () => {
  assert.equal(isAllSelected([], []), false);
  assert.equal(isAllSelected(['000001'], []), false);
});

test('列表刷新后，已不存在的选中项要清掉，避免对幽灵代码执行批量操作', () => {
  assert.deepEqual(keepExisting(['000001', '999999'], ROWS), ['000001']);
  assert.deepEqual(keepExisting([], ROWS), []);
  assert.deepEqual(keepExisting(['000001'], []), []);
});
