export const EXECUTION_MODEL = 'shared_calendar_daily_proxy_v1';
const NET_MODELS = new Set([EXECUTION_MODEL, 'next_open_cost_v2']);

export function isNetBacktest(summary = {}) {
  return NET_MODELS.has(summary.execution_model);
}

export function backtestMetric(summary = {}, window = 5) {
  const net = isNetBacktest(summary);
  const prefix = `${net ? 'net_' : ''}t${window}`;
  return {
    label: net ? `T+${window} 净胜率` : `T+${window} 信号后正收益`,
    rate: summary[`${prefix}_positive_rate`],
    mean: summary[`${prefix}_average_return`],
    count: summary[`${prefix}_sample_count`] || 0,
    pending: net ? summary[`${prefix}_pending_count`] || 0 : 0,
  };
}

export function backtestReturn(record = {}, window = 5) {
  return NET_MODELS.has(record.execution_model)
    ? record[`net_return_${window}`]
    : record[`forward_return_${window}`];
}

export function entryLabel(status) {
  return { filled: '模拟入场', blocked: '不可入场', pending: '待入场' }[status] || '旧口径';
}
