export default function FlagBreakdown({ rows = [] }) {
  if (!rows.length) return null;
  const percent = (value) => value == null ? '-' : `${Number(value).toFixed(2)}%`;
  return <div className="flag-breakdown">
    <h3>旗形分型复盘 <small>样本内统计</small></h3>
    <div className="table-wrap">
      <table>
        <thead><tr><th>旗面</th><th>预警</th><th>T+5样本</th><th>T+5净胜率</th><th>T+5平均净收益</th></tr></thead>
        <tbody>{rows.map(row => <tr key={row.direction}>
          <td>{row.label}</td><td>{row.signal_count}</td><td>{row.net_t5_sample_count || 0}</td>
          <td>{percent(row.net_t5_positive_rate)}</td><td>{percent(row.net_t5_average_return)}</td>
        </tr>)}</tbody>
      </table>
    </div>
  </div>;
}
