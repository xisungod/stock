import { ExternalLink } from 'lucide-react';
import { formatPointPrice, priceBasisLabel } from '../resultEvidence.js';

export function ResistanceSummary({ evidence }) {
  if (!evidence) return <span className="evidence-muted" title="旗形不使用精准线阻力点">不适用</span>;
  const { anchor, count } = evidence;
  if (!anchor) return <span className="evidence-muted">本次点位未记录</span>;
  return <div className="resistance-summary">
    <strong>{formatPointPrice(anchor.price)} <small>{priceBasisLabel(anchor.price_basis)}</small></strong>
    <time dateTime={anchor.date}>{anchor.date}</time>
    <small>{anchor.role === 'latest_confirmed' ? '最近阻力点' : '突破前阻力点'}{count != null ? ` · 共${count}点` : ''}</small>
  </div>;
}

export function ReasonPreview({ groups }) {
  if (!groups.length) return <span className="evidence-muted">选股理由未记录</span>;
  return <div className="reason-preview">
    {groups.map(group => <div className="reason-preview-group" key={group.type}>
      {groups.length > 1 && <strong>{group.label}</strong>}
      {group.preview.length
        ? <ul>{group.preview.map(reason => <li key={reason}>{reason}</li>)}</ul>
        : <span className="evidence-muted">该形态理由未记录</span>}
    </div>)}
  </div>;
}

export function ExpandedEvidence({ row, evidence, id, onOpen }) {
  const { groups, resistance, risks } = evidence;
  return <div id={id} className={`result-evidence ${resistance ? 'has-resistance' : ''}`}
    role="region" aria-label={`${row.stock_name}的选股依据`}>
    <section className="result-reason-details">
      <div className="result-evidence-heading">
        <h3>{row.stock_name} · 选股理由</h3>
        <button type="button" className="icon-button" title="打开个股完整详情"
          aria-label={`打开${row.stock_name}完整详情`} onClick={() => onOpen(row)}>
          <ExternalLink size={15} />
        </button>
      </div>
      {groups.map(group => <section className="evidence-reason-group" key={group.type}>
        <h4>{group.label}</h4>
        {group.reasons.length
          ? <ul>{group.reasons.map(reason => <li key={reason}>{reason}</li>)}</ul>
          : <p className="evidence-muted">{group.suggestion || '未记录具体理由'}</p>}
      </section>)}
      {!groups.length && <p className="evidence-muted">未记录形态理由</p>}
      <h4 className="evidence-risk-heading">风险与缺失项</h4>
      {risks.length
        ? <ul className="evidence-risks">{risks.map(risk => <li key={risk}>{risk}</li>)}</ul>
        : <p className="evidence-muted">未记录额外风险提示</p>}
    </section>
    {resistance && <section className="result-resistance-details">
      <div className="result-evidence-heading">
        <h3>主线阻力点{resistance.count != null ? ` · ${resistance.count}个` : ''}</h3>
      </div>
      <p className="evidence-line-meta">精准线 {formatPointPrice(resistance.linePrice)} · {priceBasisLabel(resistance.basis)}口径</p>
      {resistance.points.length
        ? <div className="resistance-point-table-wrap">
            <table className="resistance-point-table" aria-label={`${row.stock_name}主线阻力点`}>
              <colgroup><col style={{ width: '28%' }} /><col style={{ width: '20%' }} /><col style={{ width: '28%' }} /><col style={{ width: '24%' }} /></colgroup>
              <thead><tr><th>发生日</th><th>点价</th><th>确认日</th><th>用途</th></tr></thead>
              <tbody>{resistance.points.map(point => {
                const anchor = point.date === resistance.anchor?.date;
                return <tr key={point.date} className={anchor ? 'anchor-point-row' : ''}>
                  <td><time dateTime={point.date}>{point.date}</time></td>
                  <td title={priceBasisLabel(point.price_basis)}>{formatPointPrice(point.price)}</td>
                  <td>{point.confirmed_date || '未记录'}</td>
                  <td>{anchor ? resistance.anchor.role === 'latest_confirmed' ? '最近阻力点' : '本次阻力点' : '构线点'}</td>
                </tr>;
              })}</tbody>
            </table>
          </div>
        : <p className="evidence-muted">该批次未记录阻力点明细</p>}
      {resistance.missingPrices > 0 && <p className="evidence-data-note">历史批次有{resistance.missingPrices}个点价未记录。</p>}
    </section>}
  </div>;
}
