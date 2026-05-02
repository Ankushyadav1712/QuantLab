// Dashboard — primary 6-card grid (IS metrics) + IS/OOS comparison panel.
// The 6 cards always reflect the in-sample window; the comparison section
// shows the OOS half side-by-side with a decay/overfitting verdict.

const METRICS = [
  {
    key: 'sharpe',
    label: 'Sharpe',
    format: (v) => v.toFixed(2),
    classify: (v) => (v >= 1 ? 'good' : v <= 0 ? 'bad' : ''),
    tooltip: 'Annualized Sharpe ratio: mean(daily_returns) / std(daily_returns) × √252. >1 is solid, <0 means the alpha lost money.',
  },
  {
    key: 'annual_return',
    label: 'Annual Return',
    format: (v) => (v * 100).toFixed(2) + '%',
    classify: (v) => (v > 0 ? 'good' : 'bad'),
    tooltip: 'Compound annual growth rate: (1 + total_return)^(252/n_days) − 1.',
  },
  {
    key: 'max_drawdown',
    label: 'Max Drawdown',
    format: (v) => (v * 100).toFixed(2) + '%',
    classify: (v) => (v > -0.1 ? 'good' : v < -0.3 ? 'bad' : ''),
    tooltip: 'Worst peak-to-trough decline of the equity curve. Smaller magnitude is better.',
  },
  {
    key: 'avg_turnover',
    label: 'Turnover',
    format: (v) => '$' + Math.round(v).toLocaleString(),
    classify: () => '',
    tooltip: 'Average dollar value traded per day (|Δposition|.sum). Drives transaction costs.',
  },
  {
    key: 'fitness',
    label: 'Fitness',
    format: (v) => v.toFixed(3),
    classify: (v) => (v > 0.5 ? 'good' : v < 0 ? 'bad' : ''),
    tooltip: 'BRAIN-style composite: sharpe × √|annual_return| × (1 − fractional_turnover).',
  },
  {
    key: 'win_rate',
    label: 'Win Rate',
    format: (v) => (v * 100).toFixed(1) + '%',
    classify: (v) => (v > 0.55 ? 'good' : v < 0.45 ? 'bad' : ''),
    tooltip: 'Fraction of days where daily PnL > 0.',
  },
];

const SEVERITY_CLASS = {
  robust: 'good',
  moderate: 'warn',
  high: 'warn',
  severe: 'bad',
};

const SEVERITY_ICON = {
  robust: '✅',
  moderate: '⚠️',
  high: '⚠️',
  severe: '❌',
};

function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return (v * 100).toFixed(2) + '%';
}
function fmtNum(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(2);
}
function fmtPeriod(p) {
  if (!p || (!p.start && !p.end)) return '';
  return `${p.start ?? '?'} → ${p.end ?? '?'}`;
}

export function createDashboard(container) {
  container.classList.add('dashboard-wrap');
  container.innerHTML = `
    <div data-role="data-quality" class="data-quality-banner" style="display:none;"></div>
    <div data-role="metrics-grid" class="dashboard"></div>
    <div data-role="is-oos" class="is-oos-section glass" style="display:none;"></div>
    <div data-role="yearly-sharpe" class="yearly-section glass" style="display:none;"></div>
    <div data-role="walk-forward" class="wf-section glass" style="display:none;"></div>
    <div data-role="factor-decomp" class="factor-section glass" style="display:none;"></div>
  `;

  const dataQualityEl = container.querySelector('[data-role="data-quality"]');
  const metricsGrid = container.querySelector('[data-role="metrics-grid"]');
  const isOosEl = container.querySelector('[data-role="is-oos"]');
  const yearlyEl = container.querySelector('[data-role="yearly-sharpe"]');
  const wfEl = container.querySelector('[data-role="walk-forward"]');
  const factorEl = container.querySelector('[data-role="factor-decomp"]');

  // Build the 6 metric cards
  const cards = {};
  for (const m of METRICS) {
    const card = document.createElement('div');
    card.className = 'metric-card glass';
    card.dataset.key = m.key;
    card.innerHTML = `
      <div class="label">${m.label}</div>
      <div class="value">—</div>
    `;
    metricsGrid.appendChild(card);
    cards[m.key] = card;
  }

  // Tooltip element (single shared instance)
  const tip = document.createElement('div');
  tip.className = 'tooltip';
  tip.style.display = 'none';
  document.body.appendChild(tip);

  for (const m of METRICS) {
    const card = cards[m.key];
    card.addEventListener('mouseenter', (e) => {
      tip.textContent = m.tooltip;
      tip.style.display = 'block';
      positionTip(e);
    });
    card.addEventListener('mousemove', positionTip);
    card.addEventListener('mouseleave', () => {
      tip.style.display = 'none';
    });
  }

  function positionTip(e) {
    const margin = 12;
    let x = e.clientX + margin;
    let y = e.clientY + margin;
    if (x + 300 > window.innerWidth) x = e.clientX - 300;
    if (y + 80 > window.innerHeight) y = e.clientY - 80;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }

  function animateNumber(el, target, formatter, durationMs = 700) {
    if (target == null || Number.isNaN(target)) {
      el.textContent = '—';
      return;
    }
    const start = performance.now();
    function step(now) {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = target * eased;
      el.textContent = formatter(v);
      if (t < 1) requestAnimationFrame(step);
      else el.textContent = formatter(target);
    }
    requestAnimationFrame(step);
  }

  function setMetrics(metrics, opts = {}) {
    for (const m of METRICS) {
      const card = cards[m.key];
      const valueEl = card.querySelector('.value');
      const v = metrics ? metrics[m.key] : null;
      valueEl.classList.remove('good', 'bad');
      if (v == null || Number.isNaN(v)) {
        valueEl.textContent = '—';
        continue;
      }
      const cls = m.classify(v);
      if (cls) valueEl.classList.add(cls);
      animateNumber(valueEl, v, m.format);
    }

    const oos = opts.oos_metrics;
    const analysis = opts.overfitting_analysis;
    if (metrics && oos && analysis) {
      renderIsOos(metrics, oos, analysis);
      isOosEl.style.display = '';
    } else {
      isOosEl.style.display = 'none';
      isOosEl.innerHTML = '';
    }

    const quality = opts.data_quality;
    if (metrics && quality && quality.notes && quality.notes.length) {
      renderDataQuality(quality);
      dataQualityEl.style.display = '';
    } else {
      dataQualityEl.style.display = 'none';
      dataQualityEl.innerHTML = '';
    }

    const decomp = opts.factor_decomposition;
    if (metrics && decomp) {
      renderFactorDecomp(decomp);
      factorEl.style.display = '';
    } else {
      factorEl.style.display = 'none';
      factorEl.innerHTML = '';
    }

    const yearly = metrics?.yearly_returns || [];
    if (yearly.length) {
      renderYearlyShape(yearly);
      yearlyEl.style.display = '';
    } else {
      yearlyEl.style.display = 'none';
      yearlyEl.innerHTML = '';
    }

    const wf = opts.walk_forward;
    if (wf && wf.length) {
      renderWalkForward(wf);
      wfEl.style.display = '';
    } else {
      wfEl.style.display = 'none';
      wfEl.innerHTML = '';
    }
  }

  function renderYearlyShape(yearly) {
    // Bidirectional bar chart: bars grow up for positive Sharpe, down for negative.
    const maxAbs = Math.max(0.5, ...yearly.map((y) => Math.abs(y.sharpe ?? 0)));
    const bars = yearly
      .map((y) => {
        const sh = y.sharpe ?? 0;
        const isPos = sh >= 0;
        const heightPct = (Math.abs(sh) / maxAbs) * 50; // ±50% of column
        const tone = isPos ? 'pos' : 'neg';
        const tooltip = `${y.year}: Sharpe ${sh.toFixed(2)} · Σret ${(y.annual_return * 100).toFixed(1)}% · ${y.n_days}d`;
        return `
          <div class="yearly-col" title="${tooltip}">
            <div class="yearly-bar-stack">
              ${isPos
                ? `<div class="yearly-bar ${tone}" style="height:${heightPct}%;"></div><div class="yearly-bar-spacer"></div>`
                : `<div class="yearly-bar-spacer"></div><div class="yearly-bar ${tone}" style="height:${heightPct}%;"></div>`}
            </div>
            <div class="yearly-axis"></div>
            <div class="yearly-label">${y.year}</div>
            <div class="yearly-value ${tone}">${sh.toFixed(2)}</div>
          </div>
        `;
      })
      .join('');
    yearlyEl.innerHTML = `
      <div class="yearly-header">
        <div class="yearly-title">Sharpe by year</div>
        <div class="yearly-subtitle">Regime-fragility check — a strong overall Sharpe can hide a losing year</div>
      </div>
      <div class="yearly-grid">${bars}</div>
    `;
  }

  function renderWalkForward(windows) {
    const ts = windows.map((w) => w.test_sharpe ?? 0);
    const maxAbs = Math.max(0.5, ...ts.map((v) => Math.abs(v)));
    const positiveCount = ts.filter((v) => v > 0).length;
    const negativeCount = ts.filter((v) => v < 0).length;
    const meanTest = ts.reduce((a, b) => a + b, 0) / ts.length;
    const minTest = Math.min(...ts);
    const maxTest = Math.max(...ts);

    const bars = windows
      .map((w) => {
        const sh = w.test_sharpe ?? 0;
        const isPos = sh >= 0;
        const heightPct = (Math.abs(sh) / maxAbs) * 48;
        const tone = isPos ? 'pos' : 'neg';
        const tooltip = `${w.test_start} → ${w.test_end}\nTrain Sharpe: ${(w.train_sharpe ?? 0).toFixed(2)}\nTest Sharpe:  ${sh.toFixed(2)}`;
        return `
          <div class="wf-col" title="${tooltip}">
            <div class="wf-bar-stack">
              ${isPos
                ? `<div class="wf-bar ${tone}" style="height:${heightPct}%;"></div><div class="wf-bar-spacer"></div>`
                : `<div class="wf-bar-spacer"></div><div class="wf-bar ${tone}" style="height:${heightPct}%;"></div>`}
            </div>
            <div class="wf-axis"></div>
          </div>
        `;
      })
      .join('');

    const startLabel = windows[0]?.test_start || '';
    const endLabel = windows[windows.length - 1]?.test_end || '';

    wfEl.innerHTML = `
      <div class="wf-header">
        <div>
          <div class="wf-title">Walk-forward — out-of-sample Sharpe per rolling window</div>
          <div class="wf-subtitle">${windows.length} windows · ${startLabel} → ${endLabel}</div>
        </div>
        <div class="wf-stats">
          <span><span class="wf-stat-label">Mean OOS</span> <strong>${meanTest.toFixed(2)}</strong></span>
          <span><span class="wf-stat-label">Min</span> <strong class="${minTest < 0 ? 'neg' : 'pos'}">${minTest.toFixed(2)}</strong></span>
          <span><span class="wf-stat-label">Max</span> <strong class="pos">${maxTest.toFixed(2)}</strong></span>
          <span><span class="wf-stat-label">Positive</span> <strong>${positiveCount}/${windows.length}</strong></span>
        </div>
      </div>
      <div class="wf-grid">${bars}</div>
      <div class="wf-explainer">
        Each bar is the OOS Sharpe from training on a 252-day window then testing on the next 63 days.
        A signal that genuinely generalizes shows mostly positive bars with similar height.
        Highly variable bars (large negatives or extremes) suggest the alpha is regime-dependent
        rather than persistent.
      </div>
    `;
  }

  function renderFactorDecomp(d) {
    const sigBadge = d.alpha_significant
      ? '<span class="factor-sig good">significant (|t|>2)</span>'
      : '<span class="factor-sig warn">not significant (|t|≤2)</span>';

    const alphaPct = d.alpha_annualized != null
      ? `${(d.alpha_annualized * 100).toFixed(2)}%`
      : '—';
    const alphaT = d.alpha_t_stat != null ? d.alpha_t_stat.toFixed(2) : '—';
    const alphaTone =
      d.alpha_annualized != null && d.alpha_significant
        ? d.alpha_annualized > 0
          ? 'good'
          : 'bad'
        : '';

    const r2 = d.r_squared != null ? (d.r_squared * 100).toFixed(1) + '%' : '—';
    const factorShare = d.factor_share != null
      ? (d.factor_share * 100).toFixed(1) + '%'
      : '—';

    // Bar chart of factor loadings (β values)
    const loadings = d.loadings || {};
    const order = ['market', 'size', 'value', 'profitability', 'investment'];
    const maxAbsBeta = Math.max(
      0.1,
      ...order.map((k) => Math.abs(loadings[k]?.beta ?? 0))
    );

    const factorRows = order
      .map((key) => {
        const item = loadings[key] || {};
        const beta = item.beta;
        const t = item.t_stat;
        const sig = t != null && Math.abs(t) > 2;
        const pct = beta == null ? 0 : (beta / maxAbsBeta) * 50; // ±50% of bar width
        const tone = beta == null ? '' : beta >= 0 ? 'pos' : 'neg';
        const widthPct = Math.abs(pct);
        const left = beta == null || beta >= 0 ? 50 : 50 - widthPct;
        const niceLabel = key.charAt(0).toUpperCase() + key.slice(1);
        return `
          <div class="factor-row">
            <div class="factor-name">${niceLabel}</div>
            <div class="factor-bar">
              <div class="factor-bar-axis"></div>
              <div class="factor-bar-fill ${tone}"
                   style="left:${left}%; width:${widthPct}%;"></div>
            </div>
            <div class="factor-beta">
              ${beta == null ? '—' : (beta >= 0 ? '+' : '') + beta.toFixed(3)}
            </div>
            <div class="factor-tstat ${sig ? 'sig' : ''}">
              ${t == null ? '' : `t=${t.toFixed(2)}`}
            </div>
          </div>
        `;
      })
      .join('');

    factorEl.innerHTML = `
      <div class="factor-header">
        <div class="factor-title">Fama-French 5-factor decomposition</div>
        <div class="factor-period">${d.period?.start ?? '—'} → ${d.period?.end ?? '—'}  ·  ${d.sample_size} obs</div>
      </div>

      <div class="factor-summary">
        <div class="factor-stat">
          <div class="factor-stat-label">Pure alpha (annualized)</div>
          <div class="factor-stat-value ${alphaTone}">${alphaPct}</div>
          <div class="factor-stat-sub">t = ${alphaT} · ${sigBadge}</div>
        </div>
        <div class="factor-stat">
          <div class="factor-stat-label">Variance explained by factors</div>
          <div class="factor-stat-value">${factorShare}</div>
          <div class="factor-stat-sub">R² = ${r2}</div>
        </div>
      </div>

      <div class="factor-bars">${factorRows}</div>

      <div class="factor-explainer">
        <strong>What this means:</strong> the headline Sharpe is partly explained by
        exposure to known risk premia. <em>Pure alpha</em> is the residual return after
        regressing out market, size, value, profitability, and investment factors —
        i.e. the part of your return that <em>isn't</em> just standard factor exposure.
        Statistically significant pure alpha (|t|&nbsp;&gt;&nbsp;2) is the only number
        most quant firms accept as real edge.
      </div>
    `;
  }

  function renderDataQuality(q) {
    const inflation =
      q.expected_sharpe_inflation == null
        ? ''
        : ` <span class="data-quality-pill">Sharpe ↑ ~${(+q.expected_sharpe_inflation).toFixed(2)}</span>`;
    const notes = (q.notes || [])
      .map((n) => `<li>${escapeHtml(n)}</li>`)
      .join('');
    dataQualityEl.innerHTML = `
      <details class="data-quality-details">
        <summary>
          <span class="data-quality-icon">ℹ️</span>
          <strong>Data caveat — survivorship bias.</strong>
          ${inflation}
          <span class="data-quality-toggle">Show details</span>
        </summary>
        <ul class="data-quality-notes">${notes}</ul>
      </details>
    `;
  }

  function escapeHtml(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function renderIsOos(isM, oosM, analysis) {
    const isSharpe = isM.sharpe ?? 0;
    const oosSharpe = oosM.sharpe ?? 0;
    // OOS-color heuristic from spec: green if OOS > 0.6×IS, yellow if > 0.3×IS, red otherwise
    let oosTone = 'bad';
    if (isSharpe > 0) {
      if (oosSharpe >= isSharpe * 0.6) oosTone = 'good';
      else if (oosSharpe >= isSharpe * 0.3) oosTone = 'warn';
    } else if (oosSharpe >= 0) {
      oosTone = 'good';
    }

    const severity = analysis.severity || 'moderate';
    const badgeClass = SEVERITY_CLASS[severity] || 'warn';
    const icon = SEVERITY_ICON[severity] || '⚠️';
    const decayPct = analysis.sharpe_decay == null
      ? '—'
      : `${(analysis.sharpe_decay * 100).toFixed(1)}%`;

    const warningBanner = analysis.overfitting_flag
      ? `<div class="oos-warning">⚠️ This alpha shows signs of overfitting. OOS performance dropped significantly.</div>`
      : '';

    isOosEl.innerHTML = `
      <div class="is-oos-header">
        <div>
          <div class="is-oos-side-label">In-Sample</div>
          <div class="is-oos-period">${fmtPeriod(analysis.is_period)}</div>
        </div>
        <div class="is-oos-arrow">→</div>
        <div>
          <div class="is-oos-side-label">Out-of-Sample</div>
          <div class="is-oos-period">${fmtPeriod(analysis.oos_period)}</div>
        </div>
        <div class="is-oos-verdict">
          <span class="oos-badge ${badgeClass}">${icon} ${analysis.overfitting_label || ''}</span>
        </div>
      </div>

      <div class="is-oos-grid">
        <div class="is-oos-side">
          <div class="is-oos-row"><span>Sharpe</span><span class="is-oos-val">${fmtNum(isM.sharpe)}</span></div>
          <div class="is-oos-row"><span>Return</span><span class="is-oos-val">${fmtPct(isM.annual_return)}</span></div>
          <div class="is-oos-row"><span>MDD</span><span class="is-oos-val">${fmtPct(isM.max_drawdown)}</span></div>
        </div>
        <div class="is-oos-arrow-mid">→</div>
        <div class="is-oos-side">
          <div class="is-oos-row"><span>Sharpe</span><span class="is-oos-val ${oosTone}">${fmtNum(oosM.sharpe)}</span></div>
          <div class="is-oos-row"><span>Return</span><span class="is-oos-val">${fmtPct(oosM.annual_return)}</span></div>
          <div class="is-oos-row"><span>MDD</span><span class="is-oos-val">${fmtPct(oosM.max_drawdown)}</span></div>
        </div>
      </div>

      <div class="is-oos-decay">Sharpe decay: <strong>${decayPct}</strong></div>
      ${warningBanner}
    `;
  }

  function clear() {
    setMetrics(null);
  }

  return { setMetrics, clear };
}
