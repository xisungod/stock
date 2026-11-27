// 突破后的状态标记怎么显示。标记文案由后端产出（如「缩量回踩（第3天）」），
// 这里只决定配色与先后，避免前后端各写一套判断而走样。
const TONES = [
  { match: '跌破', tone: 'risk' },
  { match: '放量回踩', tone: 'warn' },
  { match: '缩量回踩', tone: 'good' }
];

const ORDER = { risk: 0, warn: 1, good: 2, neutral: 3 };

export function tagTone(tag) {
  const text = String(tag || '');
  return TONES.find((item) => text.includes(item.match))?.tone || 'neutral';
}

export function sortTags(tags) {
  // 稳定排序：同类保持后端给的时间先后，只把风险类提到前面
  return [...(tags || [])].sort((a, b) => ORDER[tagTone(a)] - ORDER[tagTone(b)]);
}
