// Side-by-side comparison view — overlay 2-4 alphas on the same charts.
// Uses inline SVG paths (lightweight; no external chart lib for this view).
// Each alpha is color-coded; legend at top, metric table at bottom.

const COLORS = ['var(--accent-blue)', '#ff9b29', 'var(--accent-green)', '#ff6bd1'];

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function fmtNum(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return (v * 100).toFixed(2) + '%';
}

// Build an SVG <path d="..."> string from a series of [x_idx, y_value] points
// rescaled to fit the given viewBox.  Skips NaN / null cells (line breaks).
function pathFromSeries(values, width, height, yMin, yMax) {
  if (!values || values.length === 0) return '';
  const xStep = width / Math.max(1, values.length - 1);
  const yRange = yMax - yMin;
  if (yRange === 0) return ''; // flat line; skip
  let d = '';
  let lastWasGap = true;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v == null || Number.isNaN(v)) {
      lastWasGap = true;
      continue;
    }
    const x = i * xStep;
    const y = height - ((v - yMin) / yRange) * height;
    d += (lastWasGap ? 'M ' : 'L ') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
    lastWasGap = false;
  }
  return d.trim();
}

// Compute (yMin, yMax) across all series passed in — so all overlays share
// the same y-axis and visual differences are honest.
function combinedYRange(seriesList, opts = {}) {
  const { padFrac = 0.05 } = opts;
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of seriesList) {
    if (!s) continue;
    for (const v of s) {
      if (v == null || Number.isNaN(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }
  const pad = (hi - lo) * padFrac;
  return [lo - pad, hi + pad];
}

export function createComparison(container) {
  container.classList.add('comparison-section');
  container.innerHTML = ''; // empty by default; .render() populates

  function clear() {
    container.innerHTML = '';
  }

  function render(payload) {
    const alphas = payload?.alphas || [];
    if (alphas.length === 0) {
      clear();
      return;
    }

    // Per-alpha legend + color assignment, skipping cells with errors.
    const valid = alphas.filter((a) => !a.error);
    const errored = alphas.filter((a) => a.error);

    const legend = alphas.map((a, i) => {
      const tone = a.error ? 'errored' : '';
      return `
        <div class="cmp-legend-item ${tone}">
          <span class="cmp-swatch" style="background:${COLORS[i % COLORS.length]}"></span>
          <span class="cmp-legend-label">${a.label}</span>
          <code class="cmp-legend-expr" title="${escapeHtml(a.expression)}">${escapeHtml(a.expression)}</code>
          ${a.error ? `<span class="cmp-error-badge">error</span>` : ''}
        </div>
      `;
    }).join('');

    // ---------- Charts: equity, drawdown, rolling Sharpe ----------
    const W = 720, H = 160;
    const charts = [
      {
        title: 'Cumulative PnL',
        getValues: (a) => a.timeseries?.cumulative_pnl,
        zeroLine: true,
      },
      {
        title: 'Drawdown',
        getValues: (a) => a.timeseries?.drawdown,
        zeroLine: true,
      },
      {
        title: 'Rolling Sharpe (63d)',
        getValues: (a) => a.timeseries?.rolling_sharpe,
        zeroLine: true,
      },
    ];

    const chartHtml = charts.map((c) => {
      const seriesList = valid.map((a) => c.getValues(a));
      const [yMin, yMax] = combinedYRange(seriesList);
      const paths = valid.map((a, i) => {
        const idx = alphas.indexOf(a);
        const color = COLORS[idx % COLORS.length];
        const d = pathFromSeries(c.getValues(a), W, H, yMin, yMax);
        return `<path d="${d}" stroke="${color}" stroke-width="1.4" fill="none" />`;
      }).join('');
      // Zero baseline if it's inside the y-range
      let baseline = '';
      if (c.zeroLine && yMin <= 0 && yMax >= 0) {
        const yZero = H - ((0 - yMin) / (yMax - yMin)) * H;
        baseline = `<line x1="0" y1="${yZero.toFixed(1)}" x2="${W}" y2="${yZero.toFixed(1)}" stroke="rgba(230, 237, 243, 0.18)" stroke-dasharray="3 3" />`;
      }
      return `
        <div class="cmp-chart">
          <div class="cmp-chart-title">${c.title}</div>
          <svg class="cmp-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
            ${baseline}
            ${paths}
          </svg>
        </div>
      `;
    }).join('');

    // ---------- Metric table ----------
    const headers = ['', ...alphas.map((a) => a.label)];
    const rows = [
      ['Sharpe', (a) => fmtNum(a.metrics?.sharpe), 'is-num'],
      ['Annual Return', (a) => fmtPct(a.metrics?.annual_return), 'is-num'],
      ['Max Drawdown', (a) => fmtPct(a.metrics?.max_drawdown), 'is-num neg'],
      ['Annual Vol', (a) => fmtPct(a.metrics?.annual_vol), 'is-num'],
      ['Sortino', (a) => fmtNum(a.metrics?.sortino_ratio), 'is-num'],
      ['Win Rate', (a) => fmtPct(a.metrics?.win_rate), 'is-num'],
      ['Avg Turnover', (a) => '$' + Math.round(a.metrics?.avg_turnover ?? 0).toLocaleString(), 'is-num'],
      ['Fitness', (a) => fmtNum(a.metrics?.fitness, 3), 'is-num'],
    ];

    const headRow = `<tr>${headers.map((h, i) => {
      const color = i === 0 ? '' : `style="color:${COLORS[(i - 1) % COLORS.length]}"`;
      return `<th ${color}>${escapeHtml(h)}</th>`;
    }).join('')}</tr>`;

    const bodyRows = rows.map(([label, getter, cls]) => {
      const cells = alphas.map((a) => {
        if (a.error) return `<td class="${cls} cmp-cell-error">—</td>`;
        return `<td class="${cls}">${getter(a)}</td>`;
      }).join('');
      return `<tr><th class="cmp-row-label">${label}</th>${cells}</tr>`;
    }).join('');

    const errorBlock = errored.length === 0 ? '' : `
      <div class="cmp-errors">
        ${errored.map((a) => `
          <div class="cmp-error-row">
            <strong>${a.label}</strong> · <code>${escapeHtml(a.expression)}</code><br/>
            <span class="cmp-error-msg">${escapeHtml(a.error)}</span>
          </div>
        `).join('')}
      </div>
    `;

    container.innerHTML = `
      <div class="cmp-header">
        <div>
          <div class="cmp-title">Side-by-side comparison</div>
          <div class="cmp-subtitle">${valid.length} alpha${valid.length === 1 ? '' : 's'} overlaid · in-sample only</div>
        </div>
        <div class="cmp-legend">${legend}</div>
      </div>

      <div class="cmp-charts">${chartHtml}</div>

      <table class="cmp-table">
        <thead>${headRow}</thead>
        <tbody>${bodyRows}</tbody>
      </table>

      ${errorBlock}

      <div class="cmp-explainer">
        Compare runs each expression independently with the same settings,
        IS-only (no OOS / walk-forward / factor decomp).  Pick a winner and
        run it through the regular Run Backtest for full validation.
      </div>
    `;
  }

  return { render, clear };
}
