import { beforeEach, describe, expect, it } from 'vitest';
import { createBrainValidator, parseReturnsCsv } from '../src/components/brain_validator.js';

describe('parseReturnsCsv', () => {
  it('parses date,value rows and skips header + blanks + junk', () => {
    const csv = [
      'date,pnl',
      '2020-01-01,0.01',
      '2020-01-02,-0.005',
      '',
      '2020-01-03,notanum',
      '2020-01-06,0.02',
    ].join('\n');
    const out = parseReturnsCsv(csv);
    expect(out.dates).toEqual(['2020-01-01', '2020-01-02', '2020-01-06']);
    expect(out.returns).toEqual([0.01, -0.005, 0.02]);
  });

  it('handles semicolon + tab delimiters and empty input', () => {
    expect(parseReturnsCsv('2020-01-01;0.03').dates).toEqual(['2020-01-01']);
    expect(parseReturnsCsv('2020-01-01\t0.04').returns).toEqual([0.04]);
    expect(parseReturnsCsv('')).toEqual({ dates: [], returns: [] });
  });
});

describe('createBrainValidator', () => {
  let container;
  let bv;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    bv = createBrainValidator(container);
  });

  it('starts hidden with the validate button disabled', () => {
    expect(container.style.display).toBe('none');
    expect(container.querySelector('[data-role="run"]').disabled).toBe(true);
  });

  it('reveals itself once a local backtest result is set', () => {
    bv.setLocalResult({ dates: ['2020-01-01', '2020-01-02'], daily_returns: [0.01, -0.02] });
    expect(container.style.display).not.toBe('none');
  });

  it('ignores an empty or invalid local result', () => {
    bv.setLocalResult({ dates: [], daily_returns: [] });
    expect(container.style.display).toBe('none');
    bv.setLocalResult(null);
    expect(container.style.display).toBe('none');
  });
});
