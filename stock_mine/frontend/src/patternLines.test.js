import test from 'node:test';
import assert from 'node:assert/strict';
import { buildPatternSeries, latestPatternSummary, TRIANGLE_SERIES_NAME } from './components/patternLines.js';

const bars = ['2024-01-01', '2024-01-10', '2024-01-20', '2024-02-01', '2024-02-10', '2024-02-20', '2024-03-01']
  .map(date => ({ date }));

const patterns = {
  w_bottoms: [{
    points: [
      { date: '2024-01-01', price: 12, role: '前高' }, { date: '2024-01-10', price: 10, role: '第一底' },
      { date: '2024-01-20', price: 11.2, role: '颈线' }, { date: '2024-02-01', price: 10.1, role: '第二底' },
      { date: '2024-02-10', price: 11.4, role: '突破' }
    ],
    neckline: { price: 11.2, start_date: '2024-01-20', end_date: '2024-02-10' },
    confirmed_date: '2024-02-01', breakout_date: '2024-02-10', status: 'breakout', status_label: '已突破颈线'
  }],
  triangles: [{
    upper: [{ date: '2024-01-10', price: 12 }, { date: '2024-03-01', price: 11.3 }],
    lower: [{ date: '2024-01-10', price: 9.7 }, { date: '2024-03-01', price: 10.4 }],
    detected_date: '2024-02-20', breakout_date: '2024-03-01', direction: 'up', status: 'up', status_label: '向上突破'
  }]
};

test('W bottoms in older chart payloads never produce a visible series', () => {
  assert.deepEqual(buildPatternSeries({ w_bottoms: patterns.w_bottoms }, bars), []);
  assert.deepEqual(buildPatternSeries(patterns, bars).map(series => series.id), ['pattern-triangle']);
});

test('triangles draw both trendlines and mark the breakout side', () => {
  const [triangle] = buildPatternSeries(patterns, bars);
  assert.equal(triangle.name, TRIANGLE_SERIES_NAME);
  assert.equal(triangle.markLine.data.length, 2);
  assert.deepEqual(triangle.markLine.data[0].map(end => end.coord), [['2024-01-10', 12], ['2024-03-01', 11.3]]);
  assert.deepEqual(triangle.markPoint.data.map(point => point.value), ['△↑']);
  const down = { triangles: [{ ...patterns.triangles[0], direction: 'down', status: 'down' }] };
  assert.deepEqual(buildPatternSeries(down, bars)[0].markPoint.data.map(point => point.value), ['△↓']);
  const active = { triangles: [{ ...patterns.triangles[0], direction: null, status: 'active', breakout_date: null }] };
  const [forming] = buildPatternSeries(active, bars);
  assert.equal(forming.markLine.data[0][0].lineStyle.type, 'dashed', 'a triangle still forming is drawn dashed');
});

test('toggles, missing payloads and off-chart dates draw nothing', () => {
  assert.deepEqual(buildPatternSeries(patterns, bars, { showTriangles: false }), []);
  assert.deepEqual(buildPatternSeries(undefined, bars), []);
  assert.deepEqual(buildPatternSeries(patterns, []), []);
  const shifted = { triangles: [{ ...patterns.triangles[0], upper: [{ date: '2023-01-01', price: 1 }, patterns.triangles[0].upper[1]] }] };
  assert.deepEqual(buildPatternSeries(shifted, bars), []);
});

test('the legend summary reports counts and the latest status', () => {
  assert.deepEqual(latestPatternSummary(patterns), {
    triangleCount: 1,
    latestTriangle: { date: '2024-03-01', label: '向上突破', status: 'up' }
  });
  assert.deepEqual(latestPatternSummary({ w_bottoms: patterns.w_bottoms }), { triangleCount: 0, latestTriangle: null });
  assert.deepEqual(latestPatternSummary(null), { triangleCount: 0, latestTriangle: null });
});
