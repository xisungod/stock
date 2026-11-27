import assert from 'node:assert/strict';
import test from 'node:test';
import { dateIndexRange, nearestDateIndex, pointerDate, presetDateRange, rightAnchoredWheelRange, wheelDeltaPixels, zoomDateRange } from './components/chartSync.js';

const dates = ['2026-08-10', '2026-08-11', '2026-08-13', '2026-08-14', '2026-08-17'];

test('maps a shared calendar range onto available trading dates', () => {
  assert.deepEqual(dateIndexRange(dates, '2026-08-12', '2026-08-16'), { startIndex: 2, endIndex: 3 });
});

test('finds the nearest date when a stock was suspended', () => {
  assert.equal(nearestDateIndex(dates, '2026-08-12'), 1);
  assert.equal(nearestDateIndex(dates, '2026-08-15'), 3);
});

test('converts presets and data zoom percentages to dates', () => {
  assert.deepEqual(presetDateRange(dates, 3), {
    startDate: '2026-08-13',
    endDate: '2026-08-17',
    startIndex: 2,
    endIndex: 4
  });
  assert.deepEqual(zoomDateRange({ start: 25, end: 75 }, dates), {
    startDate: '2026-08-11',
    endDate: '2026-08-14',
    startIndex: 1,
    endIndex: 3
  });
});

test('resolves an axis pointer index to its trading date', () => {
  assert.equal(pointerDate({ axesInfo: [{ axisDim: 'x', axisIndex: 0, value: 2 }] }, dates), '2026-08-13');
});

test('null zoom value endpoints use the percentage range instead of index zero', () => {
  assert.deepEqual(zoomDateRange({ startValue: null, endValue: null, start: 25, end: 75 }, dates), {
    startIndex: 1, endIndex: 3, startDate: dates[1], endDate: dates[3],
  });
});

const longDates = Array.from({ length: 1000 }, (_, index) => `session-${index}`);

test('wheel zoom changes only the left boundary, including in a historical window', () => {
  for (const endIndex of [600, 999]) {
    const current = { startIndex: endIndex - 199, endIndex };
    const closer = rightAnchoredWheelRange(longDates, current, -120);
    assert.equal(closer.endIndex, endIndex);
    assert.equal(closer.endDate, longDates[endIndex]);
    assert.ok(closer.startIndex > current.startIndex);
    const wider = rightAnchoredWheelRange(longDates, closer, 120);
    assert.equal(wider.endIndex, endIndex);
    assert.ok(wider.startIndex < closer.startIndex);
  }
});

test('repeated wheel input respects minimum and full-history limits without moving the right edge', () => {
  let range = { startIndex: 400, endIndex: 600 };
  for (let i = 0; i < 100; i++) range = rightAnchoredWheelRange(longDates, range, -240);
  assert.equal(range.startIndex, 580);
  assert.equal(range.endIndex, 600);
  for (let i = 0; i < 100; i++) range = rightAnchoredWheelRange(longDates, range, 240);
  assert.equal(range.startIndex, 0);
  assert.equal(range.endIndex, 600);
});

test('short histories and tiny trackpad deltas cannot shift the right anchor', () => {
  const short = rightAnchoredWheelRange(dates, { startIndex: 0, endIndex: 3 }, -120);
  assert.deepEqual(short, { startIndex: 0, endIndex: 3, startDate: dates[0], endDate: dates[3] });
  const small = rightAnchoredWheelRange(longDates, { startIndex: 80, endIndex: 100 }, 1);
  assert.equal(small.startIndex, 79);
  assert.equal(small.endIndex, 100);
  assert.equal(rightAnchoredWheelRange([], null, -120), null);
  assert.equal(rightAnchoredWheelRange(dates, {}, -120), null);
});

test('wheel deltas normalize pixel, line and page modes and ignore invalid input', () => {
  assert.equal(wheelDeltaPixels(3, 1), 48);
  assert.equal(wheelDeltaPixels(-1, 2), -240);
  assert.equal(wheelDeltaPixels(120), 120);
  assert.equal(wheelDeltaPixels(Infinity), 0);
});
