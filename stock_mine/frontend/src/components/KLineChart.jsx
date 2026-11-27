import { useEffect, useMemo, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { buildFlagSeries } from './flagLines.js';
import { dateIndexRange, pointerDate, presetDateRange, rightAnchoredWheelRange, wheelDeltaPixels, zoomDateRange } from './chartSync.js';
import {
  DEFAULT_VISIBLE_LINES,
  buildFormationMarkers,
  buildPrecisionPointSeries,
  buildSwingSeries,
  buildKeyLineMarkLines,
  buildMarkPoints,
  keyLineColor,
  keyLineLabel,
  PRECISION_POINT_COLORS,
  precisionAnchorLabel,
  resolveVisibleLines,
  selectedPrecisionLines
} from './chartLines.js';

const RANGE_OPTIONS = [
  { label: '全部', value: 0 },
  { label: '5年', value: 1220 },
  { label: '2年', value: 490 },
  { label: '1年', value: 245 },
  { label: '6月', value: 122 }
];

const upColor = '#d94841';
const downColor = '#148a64';

function values(bars, key) {
  return bars.map((bar) => bar[key] ?? null);
}

function rangeOptionForCount(count, total) {
  if (count >= total - 1) return 0;
  const match = RANGE_OPTIONS.find((option) => option.value && Math.abs(option.value - count) <= 2);
  return match?.value ?? null;
}

export default function KLineChart({
  chart,
  full = false,
  chartId = '',
  linkedRange = null,
  linkedCursor = null,
  selectedLineKeys = [],
  showSwings = false,
  showAlerts = true,
  precisionEvidence = null,
  maxLines = DEFAULT_VISIBLE_LINES,
  onRangeChange,
  onCursorChange
}) {
  const ref = useRef(null);
  const instanceRef = useRef(null);
  const onRangeChangeRef = useRef(onRangeChange);
  const onCursorChangeRef = useRef(onCursorChange);
  const applyingRangeRef = useRef('');
  const applyingCursorRef = useRef('');
  const rangeReleaseTimerRef = useRef(null);
  const cursorReleaseTimerRef = useRef(null);
  const cursorFrameRef = useRef(null);
  const zoomIndexRef = useRef(null);
  const selectedRangeRef = useRef(full ? 490 : 0);
  const [range, setRange] = useState(full ? 490 : 0);
  const [visibleRange, setVisibleRange] = useState({ startDate: '', endDate: '' });
  const bars = chart?.bars || [];
  const dates = useMemo(() => bars.map((bar) => bar.date), [bars]);
  const keyLines = useMemo(() => {
    if (chart?.key_lines?.length) return chart.key_lines;
    if (chart?.key_line === null || chart?.key_line === undefined) return [];
    return [{ price: chart.key_line, point_count: null, is_primary: true }];
  }, [chart]);
  const alertCount = (chart?.markers || []).filter((marker) => marker.type === 'alert' && dates.includes(marker.date)).length;
  const lastClose = bars.length ? Number(bars[bars.length - 1].close) : 0;
  const selectedSignature = (selectedLineKeys || []).join('|');
  // 图上只画离现价最近的几根 + 用户在右侧点开的那几根，其余留在列表里。
  const drawnLines = useMemo(
    () => resolveVisibleLines(keyLines, lastClose, selectedLineKeys, maxLines),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [keyLines, lastClose, selectedSignature, maxLines]
  );
  const pointLines = useMemo(
    () => selectedPrecisionLines(drawnLines, selectedLineKeys),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [drawnLines, selectedSignature]
  );
  const precisionSeries = useMemo(() => buildPrecisionPointSeries(pointLines, bars, {
    evidence: precisionEvidence
  }), [pointLines, bars, precisionEvidence]);
  const precisionPoints = useMemo(() => precisionSeries.flatMap(series => series.data), [precisionSeries]);
  const visiblePointCount = precisionPoints.filter(point => (
    point.value[0] >= visibleRange.startDate && point.value[0] <= visibleRange.endDate
  )).length;

  useEffect(() => {
    onRangeChangeRef.current = onRangeChange;
    onCursorChangeRef.current = onCursorChange;
  }, [onCursorChange, onRangeChange]);

  useEffect(() => {
    if (!ref.current || !bars.length) return undefined;
    const instance = echarts.init(ref.current, null, { renderer: 'canvas' });
    instanceRef.current = instance;
    const saved = zoomIndexRef.current;
    const initialRange = saved && saved.endIndex < dates.length && saved.startIndex >= 0
      ? { ...saved, startDate: dates[saved.startIndex], endDate: dates[saved.endIndex] }
      : presetDateRange(dates, selectedRangeRef.current);
    setVisibleRange({ startDate: initialRange.startDate, endDate: initialRange.endDate });
    const axisCount = full ? 3 : 2;
    const axisIndexes = Array.from({ length: axisCount }, (_, index) => index);
    const markPoints = buildMarkPoints(bars, chart.markers, { showAlerts });
    const swingSeries = buildSwingSeries(chart, showSwings);
    const volume = bars.map((bar) => ({
      value: bar.volume,
      itemStyle: { color: bar.close >= bar.open ? 'rgba(217,72,65,.58)' : 'rgba(20,138,100,.58)' }
    }));
    const macd = bars.map((bar) => ({
      value: bar.macd ?? 0,
      itemStyle: { color: (bar.macd ?? 0) >= 0 ? 'rgba(217,72,65,.72)' : 'rgba(20,138,100,.72)' }
    }));
    const grid = full
      ? [
          { left: 58, right: 54, top: 38, height: '53%' },
          { left: 58, right: 54, top: '63%', height: '11%' },
          { left: 58, right: 54, top: '79%', height: '13%' }
        ]
      : [
          { left: 48, right: 44, top: 24, height: '62%' },
          { left: 48, right: 44, top: '73%', height: '16%' }
        ];
    const xAxis = axisIndexes.map((index) => ({
      type: 'category',
      gridIndex: index,
      data: dates,
      boundaryGap: true,
      axisLine: { lineStyle: { color: '#dfe4eb' } },
      axisTick: { show: false },
      axisLabel: { show: index === axisCount - 1, color: '#7a8595', fontSize: 10 },
      min: 'dataMin',
      max: 'dataMax'
    }));
    const yAxis = axisIndexes.map((index) => ({
      scale: true,
      gridIndex: index,
      position: 'right',
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: '#7a8595',
        fontSize: 10,
        formatter: index === 1 ? (value) => `${(value / 10000).toFixed(0)}万` : undefined
      },
      splitLine: { lineStyle: { color: '#e9edf2', type: 'dashed' } }
    }));
    const movingAverages = full
      ? [
          ['MA5', 'ma5', '#d79a19'],
          ['MA10', 'ma10', '#4573c6'],
          ['MA20', 'ma20', '#9a56a5'],
          ['MA60', 'ma60', '#65707e']
        ].map(([name, key, color]) => ({
          name,
          type: 'line',
          data: values(bars, key),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1.15, color },
          emphasis: { disabled: true }
        }))
      : [];
    const series = [
      {
        name: 'K线',
        type: 'candlestick',
        data: bars.map((bar) => [bar.open, bar.close, bar.low, bar.high]),
        itemStyle: { color: upColor, color0: downColor, borderColor: upColor, borderColor0: downColor },
        markLine: drawnLines.length
          ? {
              symbol: ['none', 'none'],
              silent: true,
              data: buildKeyLineMarkLines(drawnLines, selectedLineKeys)
            }
          : undefined,
        markPoint: { data: [...markPoints, ...buildFormationMarkers(pointLines, dates)] }
      },
      ...movingAverages,
      ...swingSeries,
      ...precisionSeries,
      ...buildFlagSeries(chart?.flag_pattern, bars),
      { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: volume, barMaxWidth: 7 }
    ];
    if (full) {
      series.push(
        { name: 'MACD', type: 'bar', xAxisIndex: 2, yAxisIndex: 2, data: macd, barMaxWidth: 5 },
        { name: 'DIF', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: values(bars, 'dif'), showSymbol: false, lineStyle: { width: 1, color: '#d79a19' } },
        { name: 'DEA', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: values(bars, 'dea'), showSymbol: false, lineStyle: { width: 1, color: '#4573c6' } }
      );
    }
    instance.setOption({
      animation: false,
      backgroundColor: '#fff',
      // Point layers have their own controls and legend outside the plotting area.
      legend: full
        ? {
            type: 'scroll',
            top: 5,
            left: 58,
            right: 54,
            itemWidth: 13,
            itemHeight: 7,
            textStyle: { color: '#667386', fontSize: 11 },
            data: series.filter(item => !item.id?.startsWith('precision-'))
              .map((item) => item.name).filter((name) => !['波峰', '波谷', '候选点'].includes(name))
          }
        : { show: false },
      tooltip: { trigger: 'axis', confine: true, axisPointer: { type: 'cross' }, borderColor: '#d9e0e9', backgroundColor: 'rgba(255,255,255,.97)', textStyle: { color: '#1c2738', fontSize: 12 } },
      axisPointer: { link: [{ xAxisIndex: 'all' }], label: { backgroundColor: '#2c3748' } },
      grid,
      xAxis,
      yAxis,
      dataZoom: [
        { type: 'inside', xAxisIndex: axisIndexes, startValue: initialRange.startIndex, endValue: initialRange.endIndex,
          minValueSpan: Math.min(20, dates.length - 1), zoomOnMouseWheel: false, moveOnMouseWheel: false },
        { type: 'slider', xAxisIndex: axisIndexes, startValue: initialRange.startIndex, endValue: initialRange.endIndex, bottom: 3, height: 20, borderColor: '#dfe4eb', fillerColor: 'rgba(23,78,166,.12)', handleStyle: { color: '#174ea6' } }
      ],
      series
    }, true);

    const handleDataZoom = (event) => {
      let nextRange = zoomDateRange(event, dates);
      if (!nextRange) nextRange = zoomDateRange(instance.getOption()?.dataZoom?.[0], dates);
      if (!nextRange) return;
      setVisibleRange({ startDate: nextRange.startDate, endDate: nextRange.endDate });
      zoomIndexRef.current = { startIndex: nextRange.startIndex, endIndex: nextRange.endIndex };
      const count = nextRange.endIndex - nextRange.startIndex + 1;
      const selected = rangeOptionForCount(count, dates.length);
      selectedRangeRef.current = selected;
      setRange(selected);

      const rangeKey = `${nextRange.startDate}|${nextRange.endDate}`;
      if (applyingRangeRef.current === rangeKey) return;
      onRangeChangeRef.current?.({
        source: chartId,
        startDate: nextRange.startDate,
        endDate: nextRange.endDate,
        presetBars: selected
      });
    };

    const handleAxisPointer = (event) => {
      const date = pointerDate(event, dates);
      if (!date || applyingCursorRef.current === date) return;
      if (cursorFrameRef.current) cancelAnimationFrame(cursorFrameRef.current);
      cursorFrameRef.current = requestAnimationFrame(() => {
        onCursorChangeRef.current?.({ source: chartId, date });
      });
    };

    const element = ref.current;
    let wheelFrame = null;
    let pendingWheelDelta = 0;
    const handleWheel = (event) => {
      if (event.ctrlKey || event.metaKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
      const delta = wheelDeltaPixels(event.deltaY, event.deltaMode);
      if (!delta) return;
      const rect = element.getBoundingClientRect();
      const point = [event.clientX - rect.left, event.clientY - rect.top];
      if (!axisIndexes.some(index => instance.containPixel({ gridIndex: index }, point))) return;
      event.preventDefault();
      event.stopPropagation();
      pendingWheelDelta += delta;
      if (wheelFrame !== null) return;
      wheelFrame = requestAnimationFrame(() => {
        wheelFrame = null;
        const current = zoomDateRange(instance.getOption()?.dataZoom?.[0], dates);
        const next = rightAnchoredWheelRange(dates, current, pendingWheelDelta);
        pendingWheelDelta = 0;
        if (!next || next.startIndex === current.startIndex) return;
        instance.dispatchAction({ type: 'dataZoom', startValue: next.startIndex, endValue: next.endIndex });
      });
    };

    instance.on('datazoom', handleDataZoom);
    instance.on('updateAxisPointer', handleAxisPointer);
    // ZRender consumes bubbling wheel events before they reach the chart container.
    element.addEventListener('wheel', handleWheel, { passive: false, capture: true });
    const observer = new ResizeObserver(() => {
      instance.resize();
      instance.dispatchAction({ type: 'hideTip' });
    });
    observer.observe(ref.current);
    return () => {
      if (cursorFrameRef.current) cancelAnimationFrame(cursorFrameRef.current);
      if (rangeReleaseTimerRef.current) clearTimeout(rangeReleaseTimerRef.current);
      if (cursorReleaseTimerRef.current) clearTimeout(cursorReleaseTimerRef.current);
      if (wheelFrame !== null) cancelAnimationFrame(wheelFrame);
      element.removeEventListener('wheel', handleWheel, true);
      observer.disconnect();
      instance.off('datazoom', handleDataZoom);
      instance.off('updateAxisPointer', handleAxisPointer);
      instance.dispose();
      if (instanceRef.current === instance) instanceRef.current = null;
    };
    // selectedLineKeys 是数组，默认值 [] 每次渲染都是新引用；放进依赖会让本 effect
    // 每渲染一次就重建一次图表，而 effect 里的 setVisibleRange 又触发渲染 —— 死循环。
    // 依赖用它的字符串签名 selectedSignature。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bars, chart, chartId, dates, drawnLines, full, keyLines, maxLines, selectedSignature, showAlerts, showSwings, pointLines, precisionSeries]);

  useEffect(() => {
    const instance = instanceRef.current;
    if (!instance || !linkedRange || linkedRange.source === chartId || !dates.length) return;
    const { startIndex, endIndex } = dateIndexRange(dates, linkedRange.startDate, linkedRange.endDate);
    const targetStart = dates[startIndex];
    const targetEnd = dates[endIndex];
    applyingRangeRef.current = `${targetStart}|${targetEnd}`;
    setVisibleRange({ startDate: targetStart, endDate: targetEnd });
    const selected = rangeOptionForCount(endIndex - startIndex + 1, dates.length);
    selectedRangeRef.current = selected;
    setRange(selected);
    instance.dispatchAction({ type: 'dataZoom', startValue: startIndex, endValue: endIndex });
    if (rangeReleaseTimerRef.current) clearTimeout(rangeReleaseTimerRef.current);
    rangeReleaseTimerRef.current = setTimeout(() => {
      applyingRangeRef.current = '';
    }, 120);
  }, [chartId, dates, linkedRange]);

  useEffect(() => {
    const instance = instanceRef.current;
    if (!instance || !linkedCursor?.date || linkedCursor.source === chartId || !dates.length) return;
    const index = dateIndexRange(dates, linkedCursor.date, linkedCursor.date).startIndex;
    const date = dates[index];
    applyingCursorRef.current = date;
    instance.dispatchAction({ type: 'updateAxisPointer', xAxisIndex: 0, value: date });
    instance.dispatchAction({ type: 'showTip', seriesIndex: 0, dataIndex: index });
    if (cursorReleaseTimerRef.current) clearTimeout(cursorReleaseTimerRef.current);
    cursorReleaseTimerRef.current = setTimeout(() => {
      applyingCursorRef.current = '';
    }, 120);
  }, [chartId, dates, linkedCursor]);

  function selectRange(presetBars) {
    const instance = instanceRef.current;
    if (!instance || !dates.length) return;
    selectedRangeRef.current = presetBars;
    setRange(presetBars);
    const nextRange = presetDateRange(dates, presetBars);
    instance.dispatchAction({ type: 'dataZoom', startValue: nextRange.startIndex, endValue: nextRange.endIndex });
  }

  if (!bars.length) return <div className="empty-chart">暂无 K 线数据</div>;
  return (
    <div
      className={full ? 'chart-shell chart-shell-full' : 'chart-shell'}
      data-chart-id={chartId || undefined}
      data-range-start={visibleRange.startDate || undefined}
      data-range-end={visibleRange.endDate || undefined}
      data-alert-count={alertCount}
      data-key-line-count={drawnLines.length}
      data-total-key-lines={keyLines.length}
      data-precision-point-count={precisionPoints.length}
      data-visible-precision-point-count={visiblePointCount}
      data-flag-direction={chart?.flag_pattern?.direction || undefined}
    >
      {full && (
        <div className="chart-ranges" aria-label="K线时间范围">
          {RANGE_OPTIONS.map((option) => (
            <button key={option.label} className={range === option.value ? 'active' : ''} onClick={() => selectRange(option.value)}>
              {option.label}
            </button>
          ))}
        </div>
      )}
      <div className={full ? 'kline-chart kline-chart-full' : 'kline-chart'} ref={ref} />
      {full && pointLines.length > 0 && (
        <div className="chart-precision-legend" aria-label="精准点图例">
          <span><i className="precision-point-key resistance" style={{ borderColor: PRECISION_POINT_COLORS.resistance }} />阻力精准点</span>
          <span><i className="precision-point-key support" style={{ borderColor: PRECISION_POINT_COLORS.support }} />支撑精准点</span>
          {precisionSeries.some(series => series.id === 'precision-anchor') && <span>
            <i className="precision-point-key anchor" style={{ borderColor: PRECISION_POINT_COLORS.anchor }} />
            {precisionAnchorLabel(precisionEvidence?.anchor)}
          </span>}
          <span className="precision-point-count">当前区间 {visiblePointCount} 点 · 已绘制 {precisionPoints.length} 点</span>
        </div>
      )}
      {full && drawnLines.length > 0 && (
        <div className="chart-key-line-legend" aria-label="精准线点数">
          {drawnLines.map((line, index) => (
            <span key={`${line.price}-${index}`}>
              <i style={{ backgroundColor: keyLineColor(line, index, selectedLineKeys) }} />
              {keyLineLabel(line)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
