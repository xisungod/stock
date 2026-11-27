import { Flag, LineChart } from 'lucide-react';
import { matchLabel, patternMatches } from '../flagPatterns.js';

export default function PatternTags({ row, activeType, onSelect }) {
  const matches = patternMatches(row);
  return <div className="pattern-tags" aria-label="命中形态">
    {matches.map(match => {
      const type = match.pattern_type || 'precision_pullback';
      const className = `pattern-tag ${type === 'flag_breakout' ? 'pattern-flag' : 'pattern-precision'}${match.category === 'watch' ? ' pattern-pending' : ''}`;
      const title = match.suggestion || (match.category === 'watch' ? '观察形态，尚未确认买入预警' : '满足该形态的预警条件');
      return onSelect
        ? <button key={type} type="button" className={className} title={title} aria-pressed={activeType === type} onClick={() => onSelect(type)}>
            {type === 'flag_breakout' ? <Flag size={13} /> : <LineChart size={13} />}{matchLabel(match)}
          </button>
        : <span key={type} className={className} title={title}>{matchLabel(match)}</span>;
    })}
  </div>;
}
