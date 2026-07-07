import { describe, expect, it } from 'vitest';
import { buildResultCsv } from '../src/ui/export_csv.js';

describe('buildResultCsv', () => {
  it('emits a header and one row per valid IS day with compounding cumulative', () => {
    const csv = buildResultCsv({
      is_timeseries: { dates: ['2020-01-02', '2020-01-03'], daily_returns: [0.1, 0.1] },
    });
    const lines = csv.split('\n');
    expect(lines[0]).toBe('date,window,daily_return,cumulative_return');
    expect(lines).toHaveLength(3);

    const r1 = lines[1].split(',');
    expect([r1[0], r1[1]]).toEqual(['2020-01-02', 'IS']);
    expect(Number(r1[2])).toBeCloseTo(0.1, 10);
    expect(Number(r1[3])).toBeCloseTo(0.1, 10); // cum after 1 day

    const r2 = lines[2].split(',');
    expect(Number(r2[3])).toBeCloseTo(0.21, 10); // (1.1 × 1.1) − 1
  });

  it('appends the OOS window and carries the equity curve across it', () => {
    const csv = buildResultCsv({
      is_timeseries: { dates: ['2020-01-02'], daily_returns: [0.0] },
      oos_timeseries: { dates: ['2020-06-01'], daily_returns: [0.05] },
    });
    const lines = csv.split('\n');
    expect(lines[1].split(',').slice(0, 2)).toEqual(['2020-01-02', 'IS']);
    const oos = lines[2].split(',');
    expect(oos.slice(0, 2)).toEqual(['2020-06-01', 'OOS']);
    expect(Number(oos[3])).toBeCloseTo(0.05, 10);
  });

  it('skips null/non-finite returns and falls back to the legacy timeseries key', () => {
    const csv = buildResultCsv({
      timeseries: { dates: ['a', 'b', 'c'], daily_returns: [0.01, null, 0.02] },
    });
    const lines = csv.split('\n');
    expect(lines).toHaveLength(3); // header + 2 valid rows (null skipped)
    expect(lines[1].split(',').slice(0, 3)).toEqual(['a', 'IS', '0.01']);
    expect(lines[2].split(',').slice(0, 3)).toEqual(['c', 'IS', '0.02']);
  });

  it('returns just the header when there is no series', () => {
    expect(buildResultCsv({})).toBe('date,window,daily_return,cumulative_return');
  });
});
