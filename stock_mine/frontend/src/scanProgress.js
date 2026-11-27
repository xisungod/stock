export const POLL_INTERVAL_MS = 1500;
export const MAX_POLL_FAILURES = 3;

export function isTerminal(state) {
  if (!state) return false;
  return state.status === 'completed' || state.status === 'failed' || state.status === 'cancelled';
}

export function progressPercent(state) {
  if (!state || !state.total) return 0;
  const pct = Math.round((state.processed / state.total) * 100);
  return Math.max(0, Math.min(100, pct));
}

export function formatProgress(state) {
  if (!state) return '';
  const parts = [
    `已处理 ${state.processed}/${state.total}`,
    `命中 ${state.hits}`,
    `失败 ${state.failed}`
  ];
  if (state.stale) parts.push(`行情未更新 ${state.stale}`);
  if (state.current_stock) {
    parts.push(`当前 ${state.current_stock}`);
  }
  return parts.join(' · ');
}
