// 表格多选的纯逻辑。抽出来是因为"全选"要处理跨列表的已选项，
// 以及列表刷新后清理幽灵选中项——这两处最容易写错。
const codesOf = (rows) => (rows || []).map((row) => row.stock_code);

export function toggleCode(selected, code) {
  const current = selected || [];
  return current.includes(code) ? current.filter((item) => item !== code) : [...current, code];
}

export function isAllSelected(selected, rows) {
  const codes = codesOf(rows);
  if (!codes.length) return false;
  const current = new Set(selected || []);
  return codes.every((code) => current.has(code));
}

export function toggleAll(selected, rows) {
  const codes = codesOf(rows);
  const current = selected || [];
  if (isAllSelected(current, rows)) {
    const drop = new Set(codes);
    return current.filter((code) => !drop.has(code));
  }
  return [...new Set([...current, ...codes])];
}

export function keepExisting(selected, rows) {
  const codes = new Set(codesOf(rows));
  return (selected || []).filter((code) => codes.has(code));
}
