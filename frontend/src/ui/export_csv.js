// Build a CSV of a backtest's daily return series for download. Concatenates
// the in-sample window with the out-of-sample window (if present) into one
// continuous equity curve, tagging each row's window. Pure + testable — the
// download plumbing lives in main.js.

export function buildResultCsv(resp) {
  const rows = [['date', 'window', 'daily_return', 'cumulative_return']];
  let equity = 1; // compounding growth factor, carried across IS → OOS

  const addSeries = (ts, window) => {
    if (!ts || !Array.isArray(ts.dates) || !Array.isArray(ts.daily_returns)) return;
    const n = Math.min(ts.dates.length, ts.daily_returns.length);
    for (let i = 0; i < n; i++) {
      const raw = ts.daily_returns[i];
      if (raw == null) continue; // null/undefined = missing day (backend NaN → null)
      const r = Number(raw);
      if (!Number.isFinite(r)) continue;
      equity *= 1 + r;
      rows.push([ts.dates[i], window, r, equity - 1]);
    }
  };

  // New shape: is_timeseries (+ optional oos_timeseries). Legacy saved alphas
  // used a flat `timeseries` key — fall back to it so old results export too.
  addSeries(resp?.is_timeseries || resp?.timeseries, 'IS');
  addSeries(resp?.oos_timeseries, 'OOS');

  return rows.map((row) => row.join(',')).join('\n');
}
