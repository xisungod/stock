export const emptyMonitorFilters = {
  q: '', scope: 'latest', min_days: '', max_days: '', min_return: '', max_return: '',
  selected_from: '', selected_to: '', data_status: 'all', sort: 'selected_desc'
};

export function monitorQuery(filters, page = 1, pageSize = 50) {
  const result = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  for (const key of Object.keys(emptyMonitorFilters)) {
    const value = filters[key];
    if (value !== '' && value !== null && value !== undefined) result.set(key, String(value));
  }
  return result.toString();
}

export function validateMonitorFilters(filters) {
  for (const [lower, upper, label] of [
    ['min_days', 'max_days', '交易日数'],
    ['min_return', 'max_return', '涨跌幅']
  ]) {
    const a = filters[lower] === '' ? null : Number(filters[lower]);
    const b = filters[upper] === '' ? null : Number(filters[upper]);
    if ((a !== null && !Number.isFinite(a)) || (b !== null && !Number.isFinite(b))) return `${label}必须是有效数字`;
    if (label === '交易日数' && [a, b].some(n => n !== null && (!Number.isInteger(n) || n < 0))) return '交易日数必须是非负整数';
    if (a !== null && b !== null && a > b) return `${label}下限不能大于上限`;
  }
  if (filters.selected_from && filters.selected_to && filters.selected_from > filters.selected_to) return '入选日期下限不能晚于上限';
  return '';
}

export function monitorPercent(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '--';
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
}

export function monitorPrice(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '--';
}

export function monitorTone(value) {
  return typeof value !== 'number' || !Number.isFinite(value) || value === 0 ? '' : value > 0 ? 'monitor-gain' : 'monitor-loss';
}
