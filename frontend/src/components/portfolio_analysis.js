// Portfolio analysis pane — currently shows the Pareto frontier of saved
// alphas on the (Turnover, Sharpe) plane.  Diversification curve (Tier 3
// step 3) will land here too.  Designed so the consumer wires a refresh()
// call after any save/delete on the sidebar's alpha list.

import { api } from '../api.js';

// Layout constants tuned for a ~480px-wide right-rail pane; CSS scales
// them via viewBox so we don't recompute on resize.
const PAD_LEFT = 50;     // axis label clearance (sharpe ticks live here)
const PAD_RIGHT = 16;
const PAD_TOP = 24;
const PAD_BOT = 38;      // x-axis label + tick clearance
const W = 480;
const H = 260;

export function createPortfolioAnalysis(container) {
  container.classList.add('portfolio-pane');
  container.innerHTML = `
    <div class="pa-header">
      <div class="pa-title">Portfolio Analysis</div>
      <div class="pa-subtitle">Pareto frontier across your saved alphas — Sharpe vs Turnover</div>
    </div>
    <div data-role="pa-empty" class="pa-empty" style="display:none;">
      No saved alphas yet — save a few from the Run results and they'll show up here.
    </div>
    <div data-role="pa-stats" class="pa-stats"></div>
    <div data-role="pa-chart" class="pa-chart-wrap"></div>
    <div data-role="pa-legend" class="pa-legend">
      <span class="pa-legend-item"><span class="pa-dot pa-dot-pareto"></span> on the frontier</span>
      <span class="pa-legend-item"><span class="pa-dot pa-dot-dominated"></span> dominated</span>
      <span class="pa-legend-item pa-legend-axis">x = avg daily turnover ($) · y = Sharpe (IS)</span>
    </div>

    <div class="pa-section-divider"></div>

    <div class="pa-header">
      <div class="pa-title">Diversification curve</div>
      <div class="pa-subtitle">Ensemble Sharpe as a function of portfolio size — flat = redundant alphas, rising = real diversification</div>
    </div>
    <div data-role="pa-div-empty" class="pa-empty" style="display:none;">
      Save 2+ alphas with backtest results to see the diversification curve.
    </div>
    <div data-role="pa-div-stats" class="pa-stats"></div>
    <div data-role="pa-div-chart" class="pa-chart-wrap"></div>
    <div class="pa-legend">
      <span class="pa-legend-item"><span class="pa-dot pa-dot-line"></span> median Sharpe</span>
      <span class="pa-legend-item"><span class="pa-dot pa-dot-band"></span> IQR (25–75th pct)</span>
      <span class="pa-legend-item pa-legend-axis">x = portfolio size · y = ensemble Sharpe (IS)</span>
    </div>
  `;

  const emptyEl = container.querySelector('[data-role="pa-empty"]');
  const statsEl = container.querySelector('[data-role="pa-stats"]');
  const chartEl = container.querySelector('[data-role="pa-chart"]');
  const divEmptyEl = container.querySelector('[data-role="pa-div-empty"]');
  const divStatsEl = container.querySelector('[data-role="pa-div-stats"]');
  const divChartEl = container.querySelector('[data-role="pa-div-chart"]');

  let onSelectAlpha = null;

  function setOnSelectAlpha(cb) {
    onSelectAlpha = cb;
  }

  async function refresh() {
    // Run both fetches in parallel — they're independent.
    const [paretoPayload, divPayload] = await Promise.all([
      api.getParetoAlphas().catch((e) => ({ _error: e.message })),
      api.getDiversificationCurve(20).catch((e) => ({ _error: e.message })),
    ]);
    if (paretoPayload._error) {
      statsEl.textContent = `Failed to load: ${paretoPayload._error}`;
    } else {
      render(paretoPayload);
    }
    if (divPayload._error) {
      divStatsEl.textContent = `Failed to load: ${divPayload._error}`;
    } else {
      renderDiversification(divPayload);
    }
  }

  function render({ alphas, n_total, n_pareto, n_dominated }) {
    if (!alphas || alphas.length === 0) {
      emptyEl.style.display = '';
      statsEl.innerHTML = '';
      chartEl.innerHTML = '';
      return;
    }
    emptyEl.style.display = 'none';

    statsEl.innerHTML = `
      <span><b>${n_total}</b> saved</span>
      <span class="pa-stat-sep">·</span>
      <span class="pa-good"><b>${n_pareto}</b> on frontier</span>
      <span class="pa-stat-sep">·</span>
      <span class="pa-muted"><b>${n_dominated}</b> dominated</span>
    `;

    // Filter to alphas with valid axes — render only those.  Negative-Sharpe
    // and missing-data alphas are flagged in the response but we still want
    // to show them; we just clip negatives to the y-axis floor so they
    // don't blow out the chart.
    const valid = alphas.filter(
      (a) => a.sharpe != null && !Number.isNaN(a.sharpe)
            && a.turnover != null && !Number.isNaN(a.turnover) && a.turnover > 0
    );
    if (valid.length === 0) {
      chartEl.innerHTML = '<div class="pa-empty">All saved alphas have missing or invalid metrics.</div>';
      return;
    }

    // Log scale on x (turnover spans 4+ orders of magnitude in practice).
    // Linear y (Sharpe sits in a small range).
    const xVals = valid.map((a) => a.turnover);
    const yVals = valid.map((a) => a.sharpe);
    const xMin = Math.max(1, Math.min(...xVals)) * 0.7;
    const xMax = Math.max(...xVals) * 1.3;
    const yMin = Math.min(Math.min(...yVals), 0) - 0.2;   // include zero line
    const yMax = Math.max(Math.max(...yVals), 0.5) + 0.2;
    const logXMin = Math.log10(xMin);
    const logXMax = Math.log10(xMax);
    const plotW = W - PAD_LEFT - PAD_RIGHT;
    const plotH = H - PAD_TOP - PAD_BOT;

    const scaleX = (v) => PAD_LEFT + ((Math.log10(v) - logXMin) / (logXMax - logXMin)) * plotW;
    const scaleY = (v) => PAD_TOP + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Frontier line: connect the Pareto-marked points sorted by turnover.
    const frontier = valid
      .filter((a) => a.is_pareto)
      .slice()
      .sort((a, b) => a.turnover - b.turnover);
    const frontierPath = frontier.length >= 2
      ? frontier.map((a, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(a.turnover).toFixed(1)} ${scaleY(a.sharpe).toFixed(1)}`).join(' ')
      : '';

    // Y=0 reference line
    const y0 = scaleY(0);

    // X-axis tick positions at decade boundaries (10K, 100K, 1M, 10M, 100M)
    const decadeStart = Math.ceil(logXMin);
    const decadeEnd = Math.floor(logXMax);
    const xTicks = [];
    for (let p = decadeStart; p <= decadeEnd; p++) {
      const v = Math.pow(10, p);
      xTicks.push({ v, x: scaleX(v), label: fmtMoney(v) });
    }

    // Y ticks at 0.5 increments
    const yTicks = [];
    for (let y = Math.ceil(yMin / 0.5) * 0.5; y <= yMax; y += 0.5) {
      yTicks.push({ v: y, y: scaleY(y), label: y.toFixed(1) });
    }

    const dots = valid.map((a) => {
      const x = scaleX(a.turnover);
      const y = scaleY(Math.max(a.sharpe, yMin));   // clip negatives to floor
      const cls = a.is_pareto ? 'pa-dot-pareto' : 'pa-dot-dominated';
      const tooltip = `${a.name} · Sharpe ${a.sharpe.toFixed(2)} · Turnover ${fmtMoney(a.turnover)}` +
                      (a.is_pareto ? ' · Pareto-optimal' : ' · dominated');
      return `<circle class="pa-dot-svg ${cls}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5"
                       data-id="${a.id}"><title>${tooltip}</title></circle>`;
    }).join('');

    chartEl.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" class="pa-svg" xmlns="http://www.w3.org/2000/svg">
        <!-- axes -->
        <line class="pa-axis" x1="${PAD_LEFT}" y1="${PAD_TOP}" x2="${PAD_LEFT}" y2="${H - PAD_BOT}"></line>
        <line class="pa-axis" x1="${PAD_LEFT}" y1="${H - PAD_BOT}" x2="${W - PAD_RIGHT}" y2="${H - PAD_BOT}"></line>
        <!-- y=0 reference -->
        <line class="pa-zero" x1="${PAD_LEFT}" y1="${y0.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${y0.toFixed(1)}"></line>
        <!-- y ticks + labels -->
        ${yTicks.map((t) => `
          <line class="pa-grid" x1="${PAD_LEFT}" y1="${t.y.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${t.y.toFixed(1)}"></line>
          <text class="pa-tick-label" x="${PAD_LEFT - 4}" y="${(t.y + 3).toFixed(1)}" text-anchor="end">${t.label}</text>
        `).join('')}
        <!-- x ticks + labels -->
        ${xTicks.map((t) => `
          <line class="pa-tick" x1="${t.x.toFixed(1)}" y1="${H - PAD_BOT}" x2="${t.x.toFixed(1)}" y2="${H - PAD_BOT + 4}"></line>
          <text class="pa-tick-label" x="${t.x.toFixed(1)}" y="${H - PAD_BOT + 14}" text-anchor="middle">${t.label}</text>
        `).join('')}
        <!-- frontier line -->
        ${frontierPath ? `<path class="pa-frontier" d="${frontierPath}" fill="none"></path>` : ''}
        <!-- alpha dots -->
        ${dots}
      </svg>
    `;

    // Wire click-to-load
    chartEl.querySelectorAll('.pa-dot-svg').forEach((el) => {
      el.addEventListener('click', () => {
        const id = Number(el.dataset.id);
        if (onSelectAlpha && id) onSelectAlpha(id);
      });
    });
  }

  function renderDiversification({ curve, n_alphas_with_pnl, n_alphas_total }) {
    // Need at least 2 points to draw a meaningful curve
    if (!curve || curve.length < 2) {
      divEmptyEl.style.display = '';
      divStatsEl.innerHTML = '';
      divChartEl.innerHTML = '';
      return;
    }
    divEmptyEl.style.display = 'none';

    // Headline: where does the curve plateau?  Look for the smallest n
    // where the next step adds < 5% to median Sharpe.
    let plateauN = curve[curve.length - 1].n;
    for (let i = 0; i < curve.length - 1; i++) {
      const cur = curve[i].median_sharpe;
      const nxt = curve[i + 1].median_sharpe;
      if (cur > 0 && (nxt - cur) / Math.abs(cur) < 0.05) {
        plateauN = curve[i].n;
        break;
      }
    }
    const peak = curve[curve.length - 1].median_sharpe;

    divStatsEl.innerHTML = `
      <span><b>${n_alphas_with_pnl}/${n_alphas_total}</b> alphas with PnL</span>
      <span class="pa-stat-sep">·</span>
      <span>Peak Sharpe @ n=${curve[curve.length - 1].n}: <b>${peak.toFixed(2)}</b></span>
      <span class="pa-stat-sep">·</span>
      <span class="pa-muted">plateau ≈ n=${plateauN}</span>
    `;

    // Drawing geometry — mirror the Pareto chart's viewBox so the two
    // panels feel visually consistent.
    const xs = curve.map((r) => r.n);
    const allValues = curve.flatMap((r) => [r.q1, r.q3, r.median_sharpe, r.min, r.max]);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(Math.min(...allValues), 0) - 0.2;
    const yMax = Math.max(...allValues) + 0.2;
    const plotW = W - PAD_LEFT - PAD_RIGHT;
    const plotH = H - PAD_TOP - PAD_BOT;

    // Linear x-axis on portfolio size (small range, log scale not useful)
    const scaleX = (v) => PAD_LEFT + ((v - xMin) / Math.max(1, xMax - xMin)) * plotW;
    const scaleY = (v) => PAD_TOP + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    const y0 = scaleY(0);

    // Build IQR band as a polygon: forward along q3, back along q1
    const upper = curve.map((r) => `${scaleX(r.n).toFixed(1)},${scaleY(r.q3).toFixed(1)}`);
    const lower = curve.slice().reverse().map((r) => `${scaleX(r.n).toFixed(1)},${scaleY(r.q1).toFixed(1)}`);
    const bandPoints = upper.concat(lower).join(' ');

    // Median line
    const medianPath = curve
      .map((r, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(r.n).toFixed(1)} ${scaleY(r.median_sharpe).toFixed(1)}`)
      .join(' ');

    // Dots at each median + tooltip
    const dots = curve.map((r) => {
      const x = scaleX(r.n);
      const y = scaleY(r.median_sharpe);
      const tip = `n=${r.n} · median ${r.median_sharpe.toFixed(2)} · IQR ${r.q1.toFixed(2)}–${r.q3.toFixed(2)} · ${r.n_samples} samples`;
      return `<circle class="pa-div-dot" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3"><title>${tip}</title></circle>`;
    }).join('');

    // X ticks at every n
    const xTicks = curve.map((r) => ({
      x: scaleX(r.n),
      label: String(r.n),
    }));
    // Y ticks at 0.5 increments
    const yTicks = [];
    for (let y = Math.ceil(yMin / 0.5) * 0.5; y <= yMax; y += 0.5) {
      yTicks.push({ y: scaleY(y), label: y.toFixed(1) });
    }

    divChartEl.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" class="pa-svg" xmlns="http://www.w3.org/2000/svg">
        <line class="pa-axis" x1="${PAD_LEFT}" y1="${PAD_TOP}" x2="${PAD_LEFT}" y2="${H - PAD_BOT}"></line>
        <line class="pa-axis" x1="${PAD_LEFT}" y1="${H - PAD_BOT}" x2="${W - PAD_RIGHT}" y2="${H - PAD_BOT}"></line>
        <line class="pa-zero" x1="${PAD_LEFT}" y1="${y0.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${y0.toFixed(1)}"></line>
        ${yTicks.map((t) => `
          <line class="pa-grid" x1="${PAD_LEFT}" y1="${t.y.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${t.y.toFixed(1)}"></line>
          <text class="pa-tick-label" x="${PAD_LEFT - 4}" y="${(t.y + 3).toFixed(1)}" text-anchor="end">${t.label}</text>
        `).join('')}
        ${xTicks.map((t) => `
          <line class="pa-tick" x1="${t.x.toFixed(1)}" y1="${H - PAD_BOT}" x2="${t.x.toFixed(1)}" y2="${H - PAD_BOT + 4}"></line>
          <text class="pa-tick-label" x="${t.x.toFixed(1)}" y="${H - PAD_BOT + 14}" text-anchor="middle">${t.label}</text>
        `).join('')}
        <polygon class="pa-div-band" points="${bandPoints}"></polygon>
        <path class="pa-div-line" d="${medianPath}" fill="none"></path>
        ${dots}
      </svg>
    `;
  }

  function fmtMoney(v) {
    if (v >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B';
    if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K';
    return '$' + Math.round(v);
  }

  return { refresh, setOnSelectAlpha };
}
