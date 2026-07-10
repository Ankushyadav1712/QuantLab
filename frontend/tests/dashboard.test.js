import { beforeEach, describe, expect, it } from 'vitest';
import { createDashboard } from '../src/components/dashboard.js';

// Longs concentrated in the top deciles, shorts in the bottom → a clear
// large-cap tilt (decile_tilt = long_avg_decile − short_avg_decile > 0).
const CAP_DIST = {
  n_buckets: 10,
  long_per_bucket: [0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2, 0.23],
  short_per_bucket: [0.23, 0.2, 0.15, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02],
  long_avg_decile: 6.8,
  short_avg_decile: 3.2,
  decile_tilt: 3.6,
  n_days: 1000,
  is_approximation: false,
};

describe('dashboard size-tilt chart', () => {
  let container;
  let dash;
  let sizeTiltEl;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    dash = createDashboard(container);
    sizeTiltEl = container.querySelector('[data-role="size-tilt"]');
  });

  it('renders when market_cap_distribution is present', () => {
    dash.setMetrics({ market_cap_distribution: CAP_DIST });
    expect(sizeTiltEl.style.display).not.toBe('none');
    expect(sizeTiltEl.textContent).toContain('Size Tilt');
    // 10 deciles × 2 bars (long + short) = 20 <rect> bars.
    expect(sizeTiltEl.querySelectorAll('rect')).toHaveLength(20);
    // decile_tilt 3.6 > 0 → large-cap tilt verdict.
    expect(sizeTiltEl.textContent).toContain('Large-cap tilt');
  });

  it('calls a size-neutral book flat', () => {
    const flat = new Array(10).fill(0.1);
    dash.setMetrics({
      market_cap_distribution: { ...CAP_DIST, long_per_bucket: flat, short_per_bucket: flat, decile_tilt: 0 },
    });
    expect(sizeTiltEl.textContent).toContain('Size-neutral');
  });

  it('flags the close-price approximation', () => {
    dash.setMetrics({ market_cap_distribution: { ...CAP_DIST, is_approximation: true } });
    expect(sizeTiltEl.textContent).toContain('proxied via close');
  });

  it('hides the chart when the distribution is absent', () => {
    dash.setMetrics({ sharpe: 1.2 });
    expect(sizeTiltEl.style.display).toBe('none');
    expect(sizeTiltEl.innerHTML).toBe('');
  });

  it('self-hides when the bucket arrays are malformed', () => {
    dash.setMetrics({
      market_cap_distribution: { n_buckets: 10, long_per_bucket: [0.5], short_per_bucket: [] },
    });
    expect(sizeTiltEl.style.display).toBe('none');
  });
});

describe('dashboard Brain-style yearly table', () => {
  let container;
  let dash;
  let yearlyEl;

  const YEAR_ROW = {
    year: 2023,
    sharpe: 1.24,
    annual_return: 0.05,
    n_days: 251,
    turnover_frac: 0.8198,
    fitness_wq: 0.38,
    annual_return_arith: 0.0752,
    max_drawdown: -0.0646,
    margin_bps: 1.83,
    long_count: 478,
    short_count: 477,
  };

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    dash = createDashboard(container);
    yearlyEl = container.querySelector('[data-role="yearly-sharpe"]');
  });

  it('renders one row per year plus an All footer, in Brain units', () => {
    dash.setMetrics({
      yearly_returns: [YEAR_ROW, { ...YEAR_ROW, year: 2024, margin_bps: -5.2 }],
      yearly_total: { ...YEAR_ROW, n_days: 502 },
    });
    const table = yearlyEl.querySelector('.yearly-table');
    expect(table).not.toBeNull();
    // header + 2 year rows + All row
    expect(table.querySelectorAll('tbody tr')).toHaveLength(3);
    const text = table.textContent;
    expect(text).toContain('81.98%');   // turnover_frac as %
    expect(text).toContain('7.52%');    // returns on half-book as %
    expect(text).toContain('6.46%');    // drawdown shown positive, Brain-style
    expect(text).toContain('1.83 bps'); // margin
    expect(text).toContain('All');
  });

  it('skips the table (keeps the bars) for old payloads without Brain keys', () => {
    dash.setMetrics({
      yearly_returns: [{ year: 2023, sharpe: 1.0, annual_return: 0.05, n_days: 251 }],
    });
    expect(yearlyEl.querySelector('.yearly-table')).toBeNull();
    expect(yearlyEl.querySelectorAll('.yearly-col').length).toBe(1);
  });

  it('renders missing cells as em-dash', () => {
    dash.setMetrics({
      yearly_returns: [{ ...YEAR_ROW, margin_bps: null, long_count: null }],
      yearly_total: null,
    });
    const table = yearlyEl.querySelector('.yearly-table');
    expect(table.textContent).toContain('—');
  });
});
