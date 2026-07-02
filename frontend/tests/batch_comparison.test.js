import { describe, it, expect, beforeEach } from 'vitest';
import { createBatchComparison } from '../src/components/batch_comparison.js';

function mkPayload() {
  return {
    n_alphas: 3,
    n_ok: 2,
    elapsed_sec: 1.23,
    results: [
      {
        id: 'a',
        expression: 'rank(close)',
        metrics: {
          sharpe: 0.5, ic: 0.01, ic_tstat: 1.0, annual_return: 0.1,
          max_drawdown: -0.2, avg_turnover: 0.3, fitness: 0.4, win_rate: 0.52,
        },
      },
      {
        id: 'b',
        expression: 'rank(volume)',
        metrics: {
          sharpe: 1.5, ic: 0.03, ic_tstat: 2.0, annual_return: 0.2,
          max_drawdown: -0.1, avg_turnover: 0.2, fitness: 0.9, win_rate: 0.55,
        },
      },
      { id: 'c', expression: 'bad(', error: 'parse error: unexpected end' },
    ],
    correlation_matrix: {
      labels: ['a', 'b'],
      matrix: [[1.0, 0.8], [0.8, 1.0]],
      redundant_pairs: [{ a: 'a', b: 'b', rho: 0.8 }],
    },
  };
}

describe('batch_comparison', () => {
  let el;
  beforeEach(() => {
    el = document.createElement('div');
    document.body.appendChild(el);
  });

  it('renders one row per alpha, default-sorted by Sharpe desc, error row last', () => {
    const c = createBatchComparison(el);
    c.render(mkPayload());
    const rows = el.querySelectorAll('.batch-table tbody tr');
    expect(rows.length).toBe(3);
    const ids = [...rows].map((r) => r.dataset.id);
    expect(ids[0]).toBe('b'); // sharpe 1.5 > 0.5
    expect(ids[ids.length - 1]).toBe('c'); // errored → bottom
    expect(el.querySelector('.batch-row-error')).toBeTruthy();
    expect(el.textContent).toMatch(/3 alphas · 2 ok/);
  });

  it('toggles to ascending when the Sharpe header is clicked', () => {
    const c = createBatchComparison(el);
    c.render(mkPayload());
    const sharpeTh = [...el.querySelectorAll('.batch-sort')].find((t) => t.dataset.key === 'sharpe');
    sharpeTh.click();
    const ids = [...el.querySelectorAll('.batch-table tbody tr')].map((r) => r.dataset.id);
    expect(ids[0]).toBe('a'); // asc: 0.5 first
    expect(ids[ids.length - 1]).toBe('c'); // errors still last
  });

  it('renders the correlation heatmap and a redundant-pair note', () => {
    const c = createBatchComparison(el);
    c.render(mkPayload());
    expect(el.querySelector('.corr-table')).toBeTruthy();
    expect(el.textContent).toMatch(/Redundant/i);
  });

  it('fires onSelectAlpha with the row id on click of a successful row', () => {
    const c = createBatchComparison(el);
    let picked = null;
    c.setOnSelectAlpha((id) => { picked = id; });
    c.render(mkPayload());
    el.querySelector('.batch-row:not(.batch-row-error)').click();
    expect(picked).toBe('b'); // top row after default sort
  });

  it('uses the provided display names', () => {
    const c = createBatchComparison(el);
    c.render(mkPayload(), { names: { a: 'Momentum', b: 'Volume tilt' } });
    expect(el.textContent).toMatch(/Momentum/);
    expect(el.textContent).toMatch(/Volume tilt/);
  });
});
