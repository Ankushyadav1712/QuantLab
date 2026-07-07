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
