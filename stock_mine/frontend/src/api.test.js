import test from 'node:test';
import assert from 'node:assert/strict';
import { api, errorMessage } from './api.js';

test('HTTPException 的字符串 detail 原样使用', () => {
  const payload = { detail: 'lookback_days 必须为 0（全部历史）或 120～1500' };
  assert.equal(errorMessage(payload, '兜底'), 'lookback_days 必须为 0（全部历史）或 120～1500');
});

test('Pydantic 的数组 detail 要取出 msg，不能整块弹给用户', () => {
  // 扫描填了 min_days_since_breakout=15 时后端的真实返回
  const payload = {
    detail: [
      {
        type: 'less_than_equal',
        loc: ['body', 'min_days_since_breakout'],
        msg: 'Input should be less than or equal to 9',
        input: 15,
        ctx: { le: 9 }
      }
    ]
  };
  const message = errorMessage(payload, '兜底');
  assert.equal(message, 'Input should be less than or equal to 9');
  assert.ok(!message.includes('less_than_equal'), '不能把 type/loc/ctx 这些字段暴露出来');
  assert.ok(!message.includes('['), '不能是 JSON 数组的字面量');
});

test('自定义校验器的中文提示要去掉 Value error 前缀', () => {
  const payload = {
    detail: [
      {
        type: 'value_error',
        loc: ['body'],
        msg: 'Value error, max_days_since_breakout 必须为 0，或不小于 min_days_since_breakout'
      }
    ]
  };
  assert.equal(
    errorMessage(payload, '兜底'),
    'max_days_since_breakout 必须为 0，或不小于 min_days_since_breakout'
  );
});

test('多条校验错误合并显示', () => {
  const payload = { detail: [{ msg: '第一条' }, { msg: '第二条' }] };
  assert.equal(errorMessage(payload, '兜底'), '第一条；第二条');
});

test('detail 缺失或无法解析时退回兜底文案', () => {
  assert.equal(errorMessage({}, '请求失败：500'), '请求失败：500');
  assert.equal(errorMessage(null, '请求失败：500'), '请求失败：500');
  assert.equal(errorMessage({ detail: [] }, '请求失败：422'), '请求失败：422');
  assert.equal(errorMessage({ detail: [{ loc: ['body'] }] }, '请求失败：422'), '请求失败：422');
});

test('request 真的用上了归一化——不是只有纯函数对', () => {
  // 只测 errorMessage 是不够的：把 request 里改回 payload.detail || message，
  // 上面几条照样全绿，而用户看到的仍是原始 JSON。
  const original = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: false,
    status: 422,
    json: async () => ({
      detail: [{ type: 'value_error', loc: ['body'], msg: 'Value error, 天数窗口不合法' }]
    })
  });
  return api
    .scanProgress(1)
    .then(
      () => assert.fail('应当抛错'),
      (error) => {
        assert.equal(error.message, '天数窗口不合法');
        assert.equal(error.status, 422);
      }
    )
    .finally(() => {
      globalThis.fetch = original;
    });
});
