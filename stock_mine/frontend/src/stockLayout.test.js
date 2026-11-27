import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('./StockDetailPage.jsx', import.meta.url), 'utf8');
const css = readFileSync(new URL('./styles.css', import.meta.url), 'utf8');

test('stock identity and decision precede the chart and secondary research views', () => {
  const identity = page.indexOf('className="stock-identity"');
  const decision = page.indexOf('aria-label="关键结论"');
  const chart = page.indexOf('id="price-chart"');
  const tabs = page.indexOf('aria-label="个股研究视图"');
  assert.ok(identity > 0 && identity < decision && decision < chart && chart < tabs);
});

test('the chart is not stretched to the height of a long adjacent parameter panel', () => {
  const rule = css.match(/\.stock-analysis-grid\s*\{([^}]+)\}/)?.[1];
  assert.match(rule, /display:\s*block/);
  assert.doesNotMatch(rule, /align-items:\s*stretch/);
  assert.match(page, /<details className="strategy-parameters">/);
  assert.doesNotMatch(page, /<details className="strategy-parameters" open/);
});

test('empty sector data gets a compact state instead of a full-height chart', () => {
  assert.match(page, /detail\.industry_market\?\.chart\?\.bars\?\.length > 0 \? <KLineChart/);
  assert.match(page, /related\.building \|\| !related\.chart\?\.bars\?\.length/);
  assert.match(page, /function SectionState/);
});

test('research tabs are keyboard navigable and historical analysis exposes only historical research', () => {
  assert.match(page, /role="tablist"/);
  assert.match(page, /role="tab" aria-selected=\{activeResearchTab === id\}/);
  assert.match(page, /ArrowRight: 1, ArrowLeft: -1/);
  assert.match(page, /historicalView \? 'backtest' : researchTab/);
  assert.match(page, /activeResearchTab === 'company'/);
  assert.match(page, /activeResearchTab === 'news'/);
});

test('the chart line picker controls the same selected-point state as the evidence list', () => {
  assert.match(page, /aria-label="图表精准线" value=\{selectedLineKeys\[0\] \|\| ''\}/);
  assert.match(page, /setSelectedLineKeys\(event\.target\.value \? \[event\.target\.value\] : \[\]\)/);
  assert.match(page, /document\.getElementById\('price-chart'\)\?\.scrollIntoView/);
});
