const jsonHeaders = { 'Content-Type': 'application/json' };

// scan_id 让后端按"打开它的那次筛选"的参数判定，保证详情与列表结论一致
function _stockQuery(stockName, scanId, precisionEngine, asOf) {
  const params = new URLSearchParams();
  if (stockName) params.set('stock_name', stockName);
  if (scanId) params.set('scan_id', String(scanId));
  if (precisionEngine) params.set('precision_engine', precisionEngine);
  if (asOf) params.set('as_of', asOf);
  const query = params.toString();
  return query ? `?${query}` : '';
}

// FastAPI 的 detail 有两种形状：HTTPException 给字符串，Pydantic 校验失败给数组。
// 直接当字符串用会把 [{"type":"less_than_equal","loc":[...],...}] 整块弹给用户，
// 精心写的中文提示反而埋在 msg 里。
export function errorMessage(payload, fallback) {
  const detail = payload && payload.detail;
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => String((item && item.msg) || '').replace(/^Value error,\s*/, ''))
      .filter(Boolean);
    if (parts.length) return parts.join('；');
  }
  return fallback;
}

async function request(path, options = {}) {
  const { timeoutMs = 30000, ...fetchOptions } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(path, { ...fetchOptions, signal: controller.signal });
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('请求超时，数据源响应过慢');
    throw error;
  } finally {
    clearTimeout(timer);
  }
  if (!response.ok) {
    let message = `请求失败：${response.status}`;
    try {
      const payload = await response.json();
      message = errorMessage(payload, message);
    } catch {
      // Keep fallback message.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function requestWithRetry(path, options, retries = 1) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await request(path, options);
    } catch (error) {
      lastError = error;
      const retryable = error.status === undefined || error.status === 429 || error.status >= 500;
      if (!retryable || attempt === retries) throw error;
      await new Promise((resolve) => setTimeout(resolve, 700 * (attempt + 1)));
    }
  }
  throw lastError;
}

export const api = {
  alertMonitor: (query) => requestWithRetry(`/api/alert-monitor?${query}`, {}, 1),
  alertMonitorStatus: () => request('/api/alert-monitor/status'),
  refreshAlertMonitor: () => request('/api/alert-monitor/refresh', { method: 'POST' }),
  alertMonitorHistory: (entryId) => request(`/api/alert-monitor/${entryId}/history`),
  defaults: () => request('/api/config/defaults'),
  latestScan: () => request('/api/scans/latest'),
  runScan: (payload) =>
    request('/api/scans/run', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  cancelScan: (scanId) => request(`/api/scans/${scanId}/cancel`, { method: 'POST' }),
  scanProgress: (scanId) => request(`/api/scans/${scanId}/progress`),
  activeScan: () => request('/api/scans/active'),
  watchlist: () => request('/api/watchlist'),
  bulkAddWatch: (items) =>
    request('/api/watchlist/bulk', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ items })
    }),
  bulkDeleteWatch: (stockCodes) =>
    request('/api/watchlist/bulk-delete', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ stock_codes: stockCodes })
    }),
  refreshWatchlist: () =>
    request('/api/watchlist/refresh', {
      method: 'POST',
      timeoutMs: 120000
    }),
  addWatch: (payload) =>
    request('/api/watchlist', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  updateWatch: (stockCode, payload) =>
    request(`/api/watchlist/${stockCode}`, {
      method: 'PATCH',
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  removeWatch: (stockCode) =>
    request(`/api/watchlist/${stockCode}`, {
      method: 'DELETE'
    }),
  analyzeStock: (stockCode, stockName) => {
    const params = stockName ? `?stock_name=${encodeURIComponent(stockName)}` : '';
    return request(`/api/stocks/${stockCode}/analysis${params}`);
  },
  stockDetail: (stockCode, stockName, scanId, precisionEngine, asOf) =>
    request(`/api/stocks/${stockCode}/detail${_stockQuery(stockName, scanId, precisionEngine, asOf)}`, { timeoutMs: 45000 }),
  stockComparison: (stockCode, stockName, scanId, asOf) =>
    request(`/api/stocks/${stockCode}/strategy-comparison${_stockQuery(stockName, scanId, null, asOf)}`, { timeoutMs: 60000 }),
  // 首次调用要把 465 个板块指数拉下来建缓存（约 90 秒），之后 24 小时内秒回
  stockRelatedBoard: (stockCode, years = 3) =>
    request(`/api/stocks/${stockCode}/related-board?years=${years}`, { timeoutMs: 180000 }),
  stockBacktest: (stockCode, lookbackDays = 0, scanId = '', precisionEngine, asOf) => {
    const params = new URLSearchParams({ lookback_days: String(lookbackDays) });
    if (scanId) params.set('scan_id', String(scanId));
    if (precisionEngine) params.set('precision_engine', precisionEngine);
    if (asOf) params.set('as_of', asOf);
    return request(`/api/stocks/${stockCode}/backtest?${params.toString()}`, { timeoutMs: 300000 });
  },
  stockCompany: (stockCode) => request(`/api/stocks/${stockCode}/company`, { timeoutMs: 20000 }),
  stockNews: (stockCode) => request(`/api/stocks/${stockCode}/news`, { timeoutMs: 20000 }),
  stockIndustryMarket: (stockCode) => request(`/api/stocks/${stockCode}/industry-market`, { timeoutMs: 20000 }),
  latestBacktest: () => request('/api/backtests/latest'),
  activeBacktest: () => request('/api/backtests/active'),
  runBacktest: (payload) =>
    request('/api/backtests/run', {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  backtestProgress: (runId) => request(`/api/backtests/${runId}/progress`)
};
