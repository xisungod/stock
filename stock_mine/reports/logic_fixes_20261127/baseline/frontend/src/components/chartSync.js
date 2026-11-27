function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function valueIndex(value, dates, fallbackPercent) {
  if (!dates.length) return 0;
  if (typeof value === 'string' && value !== '') {
    const exact = dates.indexOf(value);
    if (exact >= 0) return exact;
    return nearestDateIndex(dates, value);
  }
  if (value != null && value !== '' && Number.isFinite(Number(value))) {
    return clamp(Math.round(Number(value)), 0, dates.length - 1);
  }
  return clamp(Math.round((Number(fallbackPercent || 0) / 100) * (dates.length - 1)), 0, dates.length - 1);
}

export function nearestDateIndex(dates, targetDate) {
  if (!dates.length) return 0;
  if (!targetDate || targetDate <= dates[0]) return 0;
  if (targetDate >= dates[dates.length - 1]) return dates.length - 1;

  let low = 0;
  let high = dates.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (dates[middle] === targetDate) return middle;
    if (dates[middle] < targetDate) low = middle + 1;
    else high = middle - 1;
  }

  const beforeDistance = Math.abs(Date.parse(targetDate) - Date.parse(dates[high]));
  const afterDistance = Math.abs(Date.parse(dates[low]) - Date.parse(targetDate));
  return beforeDistance <= afterDistance ? high : low;
}

export function dateIndexRange(dates, startDate, endDate) {
  if (!dates.length) return { startIndex: 0, endIndex: 0 };

  let startIndex = nearestDateIndex(dates, startDate);
  while (startIndex < dates.length - 1 && dates[startIndex] < startDate) startIndex += 1;

  let endIndex = nearestDateIndex(dates, endDate);
  while (endIndex > 0 && dates[endIndex] > endDate) endIndex -= 1;

  if (endIndex < startIndex) {
    const nearest = nearestDateIndex(dates, startDate || endDate);
    return { startIndex: nearest, endIndex: nearest };
  }
  return { startIndex, endIndex };
}

export function presetDateRange(dates, presetBars) {
  if (!dates.length) return { startDate: '', endDate: '', startIndex: 0, endIndex: 0 };
  const endIndex = dates.length - 1;
  const startIndex = presetBars ? Math.max(0, dates.length - presetBars) : 0;
  return {
    startDate: dates[startIndex],
    endDate: dates[endIndex],
    startIndex,
    endIndex
  };
}

export function zoomDateRange(payload, dates) {
  if (!dates.length) return null;
  const item = payload?.batch?.[0] || payload || {};
  let startIndex = valueIndex(item.startValue, dates, item.start);
  let endIndex = valueIndex(item.endValue, dates, item.end ?? 100);
  if (endIndex < startIndex) [startIndex, endIndex] = [endIndex, startIndex];
  return {
    startDate: dates[startIndex],
    endDate: dates[endIndex],
    startIndex,
    endIndex
  };
}

export function wheelDeltaPixels(deltaY, deltaMode = 0) {
  if (!Number.isFinite(deltaY)) return 0;
  const multiplier = deltaMode === 1 ? 16 : deltaMode === 2 ? 800 : 1;
  return clamp(deltaY * multiplier, -240, 240);
}

export function rightAnchoredWheelRange(dates, current, deltaPixels, minBars = 21) {
  if (!dates.length || !current || !Number.isFinite(deltaPixels) || deltaPixels === 0
      || !Number.isFinite(current.startIndex) || !Number.isFinite(current.endIndex)) return null;
  const endIndex = clamp(Math.round(current.endIndex), 0, dates.length - 1);
  const startIndex = clamp(Math.round(current.startIndex), 0, endIndex);
  const count = endIndex - startIndex + 1;
  const scaled = count * Math.exp(clamp(deltaPixels, -240, 240) * 0.002);
  // Small trackpad deltas must still make progress near the minimum window.
  const change = Math.round(scaled - count) || Math.sign(deltaPixels);
  const minimum = Math.min(Math.max(1, minBars), endIndex + 1);
  const nextCount = clamp(count + change, minimum, endIndex + 1);
  const nextStart = endIndex - nextCount + 1;
  return { startIndex: nextStart, endIndex, startDate: dates[nextStart], endDate: dates[endIndex] };
}

export function pointerDate(event, dates) {
  const axis = event?.axesInfo?.find((item) => item.axisDim === 'x' && item.axisIndex === 0)
    || event?.axesInfo?.find((item) => item.axisDim === 'x');
  if (!axis || !dates.length) return '';
  const value = Array.isArray(axis.value) ? axis.value[0] : axis.value;
  if (typeof value === 'string') return dates.includes(value) ? value : dates[nearestDateIndex(dates, value)];
  if (Number.isFinite(Number(value))) return dates[clamp(Math.round(Number(value)), 0, dates.length - 1)];
  return '';
}
