// Printable tearsheet — single-button HTML report for a backtest result.
//
// Layout-only: no PDF generator, no extra dependencies.  Builds a
// hidden-overlay <div>, fills it with a print-friendly snapshot of the
// current backtest's expression + metrics + key panels, then calls
// window.print().  The browser's "Save as PDF" dialog is the universal
// PDF engine — no jsPDF, no Puppeteer, no server-side rendering.
//
// The print stylesheet (in index.css) hides the rest of the app so only
// the tearsheet is printed.

function fmtNum(v, decimals = 3) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(decimals);
}
function fmtPct(v, decimals = 2) {
  if (v == null || Number.isNaN(v)) return '—';
  return (v * 100).toFixed(decimals) + '%';
}
function fmtMoney(v) {
  if (v == null || Number.isNaN(v)) return '—';
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(2) + 'B';
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(1) + 'K';
  return '$' + Math.round(v);
}

function renderMetricCard(label, value, sub = '') {
  return `
    <div class="ts-metric">
      <div class="ts-metric-label">${label}</div>
      <div class="ts-metric-value">${value}</div>
      ${sub ? `<div class="ts-metric-sub">${sub}</div>` : ''}
    </div>
  `;
}

function renderSignalQuality(m) {
  if (m.ic == null) return '';
  const decay = m.alpha_decay || {};
  const ddDur = m.drawdown_durations || {};
  return `
    <section class="ts-section">
      <h3>Signal Quality</h3>
      <div class="ts-grid">
        ${renderMetricCard('IC (Rank)', fmtNum(m.ic, 4), `over ${m.ic_n_days || 0}d`)}
        ${renderMetricCard('ICIR', fmtNum(m.icir, 2), m.ic_tstat != null ? `t=${fmtNum(m.ic_tstat, 2)}` : '')}
        ${renderMetricCard('IC % Positive', fmtPct(m.ic_pct_positive, 1))}
        ${renderMetricCard('Rank Stability', fmtNum(m.rank_stability, 3))}
        ${renderMetricCard('Fitness (WQ)', fmtNum(m.fitness_wq, 3))}
        ${renderMetricCard('Tail Ratio', fmtNum(m.tail_ratio, 2))}
        ${renderMetricCard('Positive Months', fmtPct(m.positive_months_pct, 1))}
        ${renderMetricCard('Max DD Days', ddDur.max_dd_days ?? '—',
          ddDur.avg_dd_days != null ? `avg ${fmtNum(ddDur.avg_dd_days, 1)}d` : '')}
        ${decay.half_life_days != null
          ? renderMetricCard('Alpha Half-life', `${fmtNum(decay.half_life_days, 1)}d`,
              decay.r_squared != null ? `R²=${fmtNum(decay.r_squared, 2)}` : '')
          : ''}
      </div>
    </section>
  `;
}

function renderSectorExposure(m) {
  const se = m.sector_exposure;
  if (!se || !se.by_sector) return '';
  const entries = Object.entries(se.by_sector)
    .filter(([, v]) => v && Number.isFinite(v.avg_net))
    .sort((a, b) => b[1].avg_net - a[1].avg_net);
  if (!entries.length) return '';
  const maxAbs = Math.max(0.01, ...entries.map(([, v]) => Math.abs(v.avg_net)));
  const rows = entries.map(([name, v]) => {
    const widthPct = (Math.abs(v.avg_net) / maxAbs) * 100;
    const tone = v.avg_net >= 0 ? 'pos' : 'neg';
    return `
      <div class="ts-exp-row">
        <div class="ts-exp-label">${name}</div>
        <div class="ts-exp-bar-track">
          <div class="ts-exp-bar ts-exp-bar-${tone}" style="width:${widthPct}%;"></div>
        </div>
        <div class="ts-exp-val ts-exp-val-${tone}">${(v.avg_net * 100).toFixed(2)}%</div>
      </div>
    `;
  }).join('');
  const h = se.headline || {};
  const headline = h.max_long_sector
    ? `<div class="ts-note">Hidden tilt: long <b>${h.max_long_sector}</b> ${(h.max_long_exposure * 100).toFixed(1)}%${h.max_short_sector ? ` / short <b>${h.max_short_sector}</b> ${(Math.abs(h.max_short_exposure) * 100).toFixed(1)}%` : ''}</div>`
    : '';
  return `
    <section class="ts-section">
      <h3>Sector Exposure</h3>
      ${headline}
      <div class="ts-exp-rows">${rows}</div>
    </section>
  `;
}

function renderStress(m) {
  const stress = m.stress_test;
  if (!Array.isArray(stress) || !stress.length) return '';
  const rows = stress.map((r) => {
    const sev = r.sharpe == null ? 'warn'
      : r.sharpe >= 0.5 ? 'good'
      : r.sharpe >= -0.5 ? 'warn' : 'bad';
    return `
      <tr class="ts-stress-row ts-stress-${sev}">
        <td>${r.label}</td>
        <td class="ts-stress-period">${r.start} → ${r.end} · ${r.n_days}d</td>
        <td class="ts-stress-num ts-stress-${sev}">${fmtNum(r.sharpe, 2)}</td>
        <td class="ts-stress-num">${fmtPct(r.total_return, 2)}</td>
        <td class="ts-stress-num">${fmtPct(r.max_drawdown, 2)}</td>
        <td class="ts-stress-num">${fmtPct(r.hit_rate, 0)}</td>
      </tr>
    `;
  }).join('');
  return `
    <section class="ts-section">
      <h3>Crisis Performance</h3>
      <table class="ts-stress-table">
        <thead>
          <tr>
            <th>Regime</th><th>Period</th><th>Sharpe</th><th>Return</th><th>Max DD</th><th>Hit %</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

function renderFactorDecomp(decomp) {
  if (!decomp || !decomp.coefficients) return '';
  const coefs = decomp.coefficients;
  const rows = Object.entries(coefs).map(([name, c]) => `
    <tr>
      <td>${name}</td>
      <td class="ts-num">${fmtNum(c.coef, 4)}</td>
      <td class="ts-num">${c.tstat != null ? fmtNum(c.tstat, 2) : '—'}</td>
    </tr>
  `).join('');
  return `
    <section class="ts-section">
      <h3>Fama-French 5-factor decomposition</h3>
      <div class="ts-note">
        Residual α: <b>${fmtNum(decomp.alpha, 4)}</b>
        ${decomp.alpha_tstat != null ? ` (t=${fmtNum(decomp.alpha_tstat, 2)})` : ''}
        · R²: <b>${fmtNum(decomp.r_squared, 3)}</b>
      </div>
      <table class="ts-factor-table">
        <thead><tr><th>Factor</th><th>Loading</th><th>t-stat</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

export function openTearsheet(response) {
  if (!response) return;
  const m = response.is_metrics || {};
  const expression = response.expression || '(unknown)';
  const settings = response.settings || {};
  const universeStr = settings.universe_id || 'default';
  const periodStr = `${m.start_date || '?'} → ${m.end_date || '?'}`;
  const generatedAt = new Date().toISOString().slice(0, 19).replace('T', ' ');

  // Build (or reuse) the tearsheet container as a sibling of <main>, not
  // inside it — so the print stylesheet can hide everything else cleanly.
  let host = document.getElementById('tearsheet');
  if (!host) {
    host = document.createElement('div');
    host.id = 'tearsheet';
    document.body.appendChild(host);
  }

  host.innerHTML = `
    <div class="ts-page">
      <header class="ts-header">
        <div class="ts-brand">QuantLab · Backtest Tearsheet</div>
        <div class="ts-meta">Generated ${generatedAt}</div>
      </header>
      <div class="ts-alpha-box">
        <div class="ts-alpha-label">Alpha expression</div>
        <pre class="ts-alpha-expr">${escapeHtml(expression)}</pre>
        <div class="ts-context">
          Universe: <b>${escapeHtml(universeStr)}</b>
          · Period: <b>${periodStr}</b>
          · Neutralization: <b>${escapeHtml(settings.neutralization || 'market')}</b>
          · Booksize: <b>${fmtMoney(settings.booksize || 20_000_000)}</b>
          · TCost: <b>${settings.transaction_cost_bps ?? 5} bps</b>
        </div>
      </div>

      <section class="ts-section">
        <h3>Headline metrics</h3>
        <div class="ts-grid ts-grid-headline">
          ${renderMetricCard('Sharpe', fmtNum(m.sharpe, 3))}
          ${renderMetricCard('Annual Return', fmtPct(m.annual_return))}
          ${renderMetricCard('Max Drawdown', fmtPct(m.max_drawdown))}
          ${renderMetricCard('Calmar', fmtNum(m.calmar_ratio, 2))}
          ${renderMetricCard('Sortino', fmtNum(m.sortino_ratio, 2))}
          ${renderMetricCard('Turnover', fmtMoney(m.avg_turnover), '/day')}
          ${renderMetricCard('Win Rate', fmtPct(m.win_rate, 1))}
          ${renderMetricCard('Fitness', fmtNum(m.fitness, 3))}
        </div>
      </section>

      ${renderSignalQuality(m)}
      ${renderSectorExposure(m)}
      ${renderStress(m)}
      ${renderFactorDecomp(response.factor_decomposition)}

      <footer class="ts-footer">
        QuantLab tearsheet — print via your browser's Save-as-PDF.
        Headline metrics are in-sample; OOS performance and walk-forward
        analysis available in the dashboard.
      </footer>
    </div>
  `;

  document.body.classList.add('tearsheet-active');
  // Two RAFs so layout + fonts settle before the print dialog opens.
  // Otherwise Chrome sometimes prints an empty page or mis-paginates.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      window.print();
      // After the dialog closes, restore the normal app layout.
      const cleanup = () => {
        document.body.classList.remove('tearsheet-active');
        window.removeEventListener('afterprint', cleanup);
      };
      window.addEventListener('afterprint', cleanup);
    });
  });
}

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
