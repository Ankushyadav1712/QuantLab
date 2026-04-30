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
    <div data-role="metrics-grid" class="dashboard"></div>
    <div data-role="is-oos" class="is-oos-section glass" style="display:none;"></div>
  `;

  const metricsGrid = container.querySelector('[data-role="metrics-grid"]');
  const isOosEl = container.querySelector('[data-role="is-oos"]');

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
