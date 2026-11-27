export const flagDirectionLabels = {
  descending: '下倾旗形',
  horizontal: '水平旗形',
  ascending: '上倾旗形'
};

export const strategyLabels = {
  combined: '联合筛选',
  precision_pullback: '精准线回调',
  flag_breakout: '旗形突破',
  triangle_breakout: '三角整理'
};

export const precisionEngineLabels = {
  platform_events: '平台验证事件',
  pivot_fixed: '波峰法 · 容差修正',
  legacy: '原版波峰法'
};

export function patternMatches(row) {
  if (!row) return [];
  const matches = Array.isArray(row.pattern_matches) ? row.pattern_matches : [row];
  return [...new Map(matches.filter(item => item && typeof item === 'object' && item.category !== 'none')
    .map(item => [item.pattern_type || 'precision_pullback', item])).values()];
}

export function matchLabel(match) {
  const label = match?.pattern_type === 'flag_breakout'
    ? flagDirectionLabels[match.flag_direction || match.flag_pattern?.direction] || '旗形'
    : match?.pattern_type === 'triangle_breakout'
      ? '三角整理'
      : '精准线回调';
  return `${label}${match?.data_stale ? '（待更新）' : match?.category === 'watch' ? '（待确认）' : ''}`;
}

export function patternLabel(row) {
  return patternMatches(row).map(matchLabel).join(' + ') || '未命中';
}

export function referencePrice(row) {
  return row?.pattern_type === 'flag_breakout' ? row.flag_pattern?.upper_price : row?.key_line_price;
}

export function scanPayload(form) {
  const mode = form.strategy_mode || 'combined';
  const flagOnly = mode === 'flag_breakout';
  const triangleOnly = mode === 'triangle_breakout';
  const withoutPrecision = flagOnly || triangleOnly;
  const requireBreakoutVolume = flagOnly || Boolean(form.require_breakout_volume ?? false);
  const extensionPercent = form.prebreakout_extension_percent;
  const prebreakoutLimit = extensionPercent === '' || extensionPercent == null ? 50 : Number(extensionPercent);
  if (!withoutPrecision && (!Number.isFinite(prebreakoutLimit) || prebreakoutLimit < 0 || prebreakoutLimit > 100)) {
    throw new Error('突破前越线上限必须为0～100%');
  }
  const days = form.prebreakout_reject_above_days;
  const rejectDays = days === '' || days == null ? 30 : Number(days);
  const bufferPercent = form.prebreakout_above_buffer_percent;
  const aboveBuffer = bufferPercent === '' || bufferPercent == null ? 2 : Number(bufferPercent);
  if (!withoutPrecision && (!Number.isInteger(rejectDays) || rejectDays < 1 || rejectDays > 250)) {
    throw new Error('突破前连续站上排除天数必须为1～250的整数');
  }
  if (!withoutPrecision && (!Number.isFinite(aboveBuffer) || aboveBuffer < 0 || aboveBuffer > 100)) {
    throw new Error('有效收线上缓冲必须为0～100%');
  }
  const minPullbackDays = Number(form.min_pullback_days ?? 1);
  const maxPullbackDays = Number(form.max_pullback_days ?? 10);
  if (!withoutPrecision && (
    !Number.isInteger(minPullbackDays) || !Number.isInteger(maxPullbackDays)
    || minPullbackDays < 1 || maxPullbackDays > 60 || minPullbackDays > maxPullbackDays
  )) {
    throw new Error('回调天数须为1～60的整数，且上限不能小于下限');
  }
  return {
    provider: form.provider,
    limit: Math.max(0, Number(form.limit) || 0),
    signal_offset: Number(form.signal_offset ?? 0),
    min_days_since_breakout: withoutPrecision ? 0 : Number(form.min_days_since_breakout) || 0,
    max_days_since_breakout: withoutPrecision ? 0 : Number(form.max_days_since_breakout) || 0,
    require_breakout: Boolean(form.require_breakout),
    require_pullback: !withoutPrecision && Boolean(form.require_pullback ?? true),
    min_pullback_days: withoutPrecision ? 1 : minPullbackDays,
    max_pullback_days: withoutPrecision ? null : maxPullbackDays,
    require_breakout_volume: triangleOnly ? false : requireBreakoutVolume,
    breakout_vol_mult: mode === 'precision_pullback' && !requireBreakoutVolume ? 1.5 : Number(form.breakout_vol_mult) || 1.5,
    strategy_mode: mode,
    precision_engine: form.precision_engine || 'platform_events',
    flag_direction: mode === 'combined' || flagOnly ? form.flag_direction || 'all' : 'all',
    max_prebreakout_extension_pct: withoutPrecision ? null : prebreakoutLimit / 100,
    prebreakout_reject_above_days: withoutPrecision ? null : rejectDays,
    prebreakout_above_buffer_pct: flagOnly ? 0.02 : aboveBuffer / 100,
    // 2026-11-27 规则：缩量相对常态量、冲高回落不算回踩、验证证据时效、大盘环境
    pullback_max_normal_volume_ratio: Number(form.pullback_max_normal_volume_ratio ?? 1),
    max_post_breakout_runup_pct: Number(form.max_post_breakout_runup_pct ?? 0.1),
    max_event_age_bars: Number(form.max_event_age_bars ?? 750),
    market_filter: form.market_filter || 'risk'
  };
}
