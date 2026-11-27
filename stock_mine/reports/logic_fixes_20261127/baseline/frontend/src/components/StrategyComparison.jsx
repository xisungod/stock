import { ArrowUpRight } from 'lucide-react';
import { precisionEngineLabels } from '../flagPatterns.js';
import { formatPointPrice } from '../resultEvidence.js';
import { lineTouchPoints } from './chartLines.js';

export default function StrategyComparison({ comparison, onSelect }) {
  return <section className="section-panel comparison-panel">
    <div className="section-heading"><h2>精准线引擎对照</h2><span>{comparison.analysis_as_of || '同一信号日'} · 不含行业修正</span></div>
    <div className="table-wrap comparison-table-wrap">
      <table><thead><tr><th>引擎</th><th>状态</th><th>参考线</th><th>验证点</th><th>判定依据</th><th>明细</th></tr></thead>
        <tbody>{comparison.results.map(item => {
          const result = item.result;
          const line = result?.line_snapshot || result?.key_lines?.find(line => line.is_primary);
          const points = lineTouchPoints(line, result?.chart?.bars || []);
          return <tr key={item.engine}>
            <td><button type="button" className="comparison-engine-button" onClick={() => onSelect(item.engine)}
              title={`查看${precisionEngineLabels[item.engine]}`}><ArrowUpRight size={14} />{precisionEngineLabels[item.engine]}</button></td>
            <td>{result?.category === 'buy_alert' ? '买入预警' : result?.category === 'watch' ? '观察' : '未入选'}</td>
            <td>{formatPointPrice(result?.key_line_price)}</td>
            <td>{line ? `${line.point_count} · 阻力${line.resistance_point_count} / 支撑${line.support_point_count}` : '-'}</td>
            <td className="comparison-reason">{result?.suggestion || item.message}</td>
            <td><details><summary>{points.length} 点</summary>
              <ul className="comparison-points">{points.map(point => <li key={`${point.kind}-${point.date}`}>
                {point.date} · {point.kind === 'resistance' ? '阻力' : '支撑'} {formatPointPrice(point.price)}
                <small>确认 {point.confirmed_date || '未记录'}</small>
                {point.origins?.length > 0 && <small>{point.origins.map(origin => ({
                  local_turn: '局部转折', resistance_test: '阻力验证', support_test: '支撑验证'
                })[origin]).filter(Boolean).join(' / ')}</small>}
              </li>)}</ul>
            </details></td>
          </tr>;
        })}</tbody>
      </table>
    </div>
  </section>;
}
