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
    <div data-role="signal-quality" class="signal-quality-section glass" style="display:none;"></div>
    <div data-role="exposure" class="exposure-section glass" style="display:none;"></div>
    <div data-role="stress-test" class="stress-section glass" style="display:none;"></div>
    <div data-role="deflated" class="deflated-section glass" style="display:none;"></div>
    <div data-role="is-oos" class="is-oos-section glass" style="display:none;"></div>
    <div data-role="yearly-sharpe" class="yearly-section glass" style="display:none;"></div>
    <div data-role="walk-forward" class="wf-section glass" style="display:none;"></div>
    <div data-role="factor-decomp" class="factor-section glass" style="display:none;"></div>
  `;

  const dataQualityEl = container.querySelector('[data-role="data-quality"]');
  const metricsGrid = container.querySelector('[data-role="metrics-grid"]');
  const signalQualityEl = container.querySelector('[data-role="signal-quality"]');
  const exposureEl = container.querySelector('[data-role="exposure"]');
  const stressEl = container.querySelector('[data-role="stress-test"]');
  const deflatedEl = container.querySelector('[data-role="deflated"]');
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

    // Signal Quality panel: IC / ICIR / alpha decay / rank stability / etc.
    // Hidden if metrics is missing OR if IC didn't compute (older saved
    // results predate the signal_matrix field on BacktestResult).
    const decompForSQ = opts.factor_decomposition || null;
    if (metrics && metrics.ic != null) {
      renderSignalQuality(metrics, decompForSQ);
      signalQualityEl.style.display = '';
    } else {
      signalQualityEl.style.display = 'none';
      signalQualityEl.innerHTML = '';
    }

    // Tier 2: sector exposure panel (hidden if no GICS or empty)
    const sectorExp = metrics?.sector_exposure;
    if (sectorExp && sectorExp.by_sector && Object.keys(sectorExp.by_sector).length) {
      renderExposure(sectorExp);
      exposureEl.style.display = '';
    } else {
      exposureEl.style.display = 'none';
      exposureEl.innerHTML = '';
    }

    // Tier 2: crisis-window stress test
    const stress = metrics?.stress_test;
    if (Array.isArray(stress) && stress.length) {
      renderStress(stress);
      stressEl.style.display = '';
    } else {
      stressEl.style.display = 'none';
      stressEl.innerHTML = '';
    }

    const deflated = metrics?.deflated_sharpe;
    if (metrics && deflated) {
      renderDeflated(metrics, deflated);
      deflatedEl.style.display = '';
    } else {
      deflatedEl.style.display = 'none';
      deflatedEl.innerHTML = '';
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

  function renderSignalQuality(metrics, decomp) {
    // Signal Quality panel — the Tier-1 research metrics that turn a
    // backtest from "did it make money?" into "is this signal real?".
    // Cards grouped: IC quality (top row) + alpha decay (middle, with mini
    // chart) + payoff shape & DD duration (bottom).
    const ic = metrics.ic ?? 0;
    const icir = metrics.icir;
    const icT = metrics.ic_tstat;
    const icPctPos = metrics.ic_pct_positive;
    const icDays = metrics.ic_n_days ?? 0;
    const rankStab = metrics.rank_stability;
    const tailRatio = metrics.tail_ratio;
    const posMonths = metrics.positive_months_pct;
    const fitnessWq = metrics.fitness_wq;
    const decay = metrics.alpha_decay || {};
    const ddDur = metrics.drawdown_durations || {};

    const icClass = ic > 0.03 ? 'good' : ic < 0 ? 'bad' : '';
    const icirClass = icir != null && icir > 0.5 ? 'good' : icir != null && icir < 0 ? 'bad' : '';
    const tailClass = tailRatio != null && tailRatio > 1 ? 'good' : tailRatio != null && tailRatio < 0.7 ? 'bad' : '';
    const posMonthsClass = posMonths != null && posMonths > 0.6 ? 'good' : posMonths != null && posMonths < 0.4 ? 'bad' : '';
    const fitnessWqClass = fitnessWq != null && fitnessWq > 0.5 ? 'good' : fitnessWq != null && fitnessWq < 0 ? 'bad' : '';

    // Inline IC decay mini-chart (SVG bars at horizons 1,2,3,5,10,21)
    const horizonIcs = decay.ic_by_horizon || {};
    const horizons = Object.keys(horizonIcs).map((h) => Number(h)).sort((a, b) => a - b);
    let decaySvg = '<div class="sq-decay-empty">Decay fit unavailable (signal noise dominates)</div>';
    if (horizons.length >= 2) {
      const values = horizons.map((h) => horizonIcs[h] ?? 0);
      const maxAbs = Math.max(0.01, ...values.map((v) => Math.abs(v)));
      const W = 260;
      const H = 80;
      const barW = (W - 20) / horizons.length;
      const bars = horizons.map((h, i) => {
        const v = values[i] ?? 0;
        const heightPx = (Math.abs(v) / maxAbs) * (H / 2 - 8);
        const x = 10 + i * barW + 4;
        const w = barW - 8;
        const y = v >= 0 ? H / 2 - heightPx : H / 2;
        const color = v >= 0 ? 'var(--accent-cyan, #4dd0e1)' : 'var(--accent-rose, #f06292)';
        return `<rect x="${x}" y="${y}" width="${w}" height="${heightPx}" fill="${color}"></rect>
                <text x="${x + w / 2}" y="${H - 2}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.6">${h}d</text>`;
      }).join('');
      decaySvg = `
        <svg viewBox="0 0 ${W} ${H}" class="sq-decay-svg">
          <line x1="10" y1="${H / 2}" x2="${W - 4}" y2="${H / 2}" stroke="currentColor" stroke-opacity="0.2"></line>
          ${bars}
        </svg>
      `;
    }

    const halfLife = decay.half_life_days;
    const halfLifeText = halfLife != null
      ? `${halfLife.toFixed(1)} days`
      : '—';
    const halfLifeNote = halfLife == null
      ? 'IC pattern not fit by exponential decay'
      : halfLife < 5 ? 'Fast alpha — rebalance often'
      : halfLife < 21 ? 'Medium horizon'
      : 'Slow alpha — low turnover OK';

    signalQualityEl.innerHTML = `
      <div class="sq-header">
        <div class="sq-title">Signal Quality</div>
        <div class="sq-subtitle">Cross-sectional prediction lens — does the ranking actually predict next-day returns?</div>
      </div>
      <div class="sq-grid">
        <div class="metric-card glass" title="Mean daily rank correlation between signal and next-day return. >0.03 is the industry pass bar.">
          <div class="label">IC (Rank)</div>
          <div class="value ${icClass}">${ic.toFixed(4)}</div>
          <div class="sub">over ${icDays} days</div>
        </div>
        <div class="metric-card glass" title="ICIR = mean(IC) / std(IC) × √252. Stability of the predictive power. >0.5 is solid.">
          <div class="label">ICIR</div>
          <div class="value ${icirClass}">${icir != null ? icir.toFixed(2) : '—'}</div>
          <div class="sub">${icT != null ? `t=${icT.toFixed(2)}` : ''}</div>
        </div>
        <div class="metric-card glass" title="Share of days with positive IC. Complement to ICIR — high % positive + reasonable mean = persistent edge.">
          <div class="label">IC % Positive</div>
          <div class="value">${icPctPos != null ? (icPctPos * 100).toFixed(1) + '%' : '—'}</div>
        </div>
        <div class="metric-card glass" title="Day-over-day Spearman of cross-sectional ranks. 1.0 = stable ranking, 0 = reshuffles randomly.">
          <div class="label">Rank Stability</div>
          <div class="value">${rankStab != null ? rankStab.toFixed(3) : '—'}</div>
        </div>
        <div class="metric-card glass" title="WorldQuant fitness: sign × √(|return|/max(turnover,0.125)) × Sharpe. Brain's composite signal-quality score.">
          <div class="label">Fitness (WQ)</div>
          <div class="value ${fitnessWqClass}">${fitnessWq != null ? fitnessWq.toFixed(3) : '—'}</div>
        </div>
        <div class="metric-card glass" title="|95th-pct return| / |5th-pct return|. >1 means asymmetric gains, <1 means crash-tail risk.">
          <div class="label">Tail Ratio</div>
          <div class="value ${tailClass}">${tailRatio != null ? tailRatio.toFixed(2) : '—'}</div>
        </div>
        <div class="metric-card glass" title="Share of calendar months with net-positive return. More robust to autocorrelation than headline Sharpe.">
          <div class="label">Positive Months</div>
          <div class="value ${posMonthsClass}">${posMonths != null ? (posMonths * 100).toFixed(1) + '%' : '—'}</div>
        </div>
        <div class="metric-card glass" title="Longest underwater stretch. Depth (Max DD) without duration hides whether recovery is days or months.">
          <div class="label">Max DD Days</div>
          <div class="value">${ddDur.max_dd_days ?? '—'}</div>
          <div class="sub">avg ${ddDur.avg_dd_days != null ? ddDur.avg_dd_days.toFixed(1) : '—'}d</div>
        </div>
        ${renderSizeCard(metrics)}
        ${renderBetaCard(decomp)}
      </div>
      <div class="sq-decay-panel">
        <div class="sq-decay-header">
          <div class="sq-decay-title">Alpha decay across horizons</div>
          <div class="sq-decay-halflife">
            <span class="sq-half-label">Half-life</span>
            <span class="sq-half-value">${halfLifeText}</span>
            <span class="sq-half-note">${halfLifeNote}</span>
          </div>
        </div>
        ${decaySvg}
        <div class="sq-decay-axis-label">Forward horizon (trading days)</div>
      </div>
    `;
  }

  function renderSizeCard(metrics) {
    // Tier 2: Size factor exposure card.  Hidden cleanly if the metric is
    // unavailable; flag if it used the close-price approximation.
    const se = metrics?.size_exposure;
    if (!se || se.size_corr == null) return '';
    const v = se.size_corr;
    const cls = Math.abs(v) < 0.2 ? 'good' : Math.abs(v) < 0.5 ? '' : 'bad';
    const approxNote = se.is_approximation
      ? '<div class="sub" style="opacity:.6">via close (proxy)</div>'
      : '';
    const tip = 'Daily Pearson corr(weights, log(market_cap)). >0.3 means systematic mega-cap bet; <-0.3 means small-cap tilt. Near 0 = size-neutral.';
    return `
      <div class="metric-card glass" title="${tip}">
        <div class="label">Size Exposure</div>
        <div class="value ${cls}">${v.toFixed(3)}</div>
        ${approxNote}
      </div>
    `;
  }

  function renderBetaCard(decomp) {
    // Tier 2: surface the FF5 Mkt-RF coefficient as a standalone card.
    // Hidden if the regression didn't run or didn't include Mkt-RF.
    const c = decomp?.coefficients?.['Mkt-RF'];
    if (!c || c.coef == null) return '';
    const b = c.coef;
    const t = c.tstat;
    const cls = Math.abs(b) < 0.2 ? 'good' : Math.abs(b) < 0.5 ? '' : 'bad';
    const tip = 'Beta vs market (Fama-French Mkt-RF factor). For a "market-neutral" alpha this should be ~0; |beta| > 0.3 means significant market exposure.';
    return `
      <div class="metric-card glass" title="${tip}">
        <div class="label">Market Beta</div>
        <div class="value ${cls}">${b.toFixed(3)}</div>
        <div class="sub">${t != null ? 't=' + t.toFixed(2) : ''}</div>
      </div>
    `;
  }

  function renderExposure(sectorExp) {
    // Horizontal bar chart of per-sector net exposure, sorted by magnitude.
    // Green = long, red = short.  Headline up top calls out the biggest tilt.
    const entries = Object.entries(sectorExp.by_sector || {})
      .filter(([, v]) => v && Number.isFinite(v.avg_net))
      .sort((a, b) => b[1].avg_net - a[1].avg_net);

    if (!entries.length) {
      exposureEl.innerHTML = '';
      return;
    }

    const maxAbs = Math.max(0.01, ...entries.map(([, v]) => Math.abs(v.avg_net)));
    const rows = entries.map(([name, v]) => {
      const net = v.avg_net;
      const widthPct = (Math.abs(net) / maxAbs) * 100;
      const tone = net >= 0 ? 'pos' : 'neg';
      const label = name.length > 28 ? name.slice(0, 27) + '…' : name;
      // Each row: sector label on the left, bar in the middle, signed value
      // on the right.  Layout uses a 3-col grid so labels align across rows.
      return `
        <div class="exp-row" title="${name} · n=${v.n_tickers} · gross ${(v.avg_gross * 100).toFixed(2)}%">
          <div class="exp-label">${label}</div>
          <div class="exp-bar-track">
            <div class="exp-bar exp-bar-${tone}" style="width:${widthPct}%;"></div>
          </div>
          <div class="exp-value exp-value-${tone}">${(net * 100).toFixed(2)}%</div>
        </div>
      `;
    }).join('');

    const h = sectorExp.headline || {};
    const headlineText = h.max_long_sector
      ? `Hidden tilt: long <b>${h.max_long_sector}</b> ${(h.max_long_exposure * 100).toFixed(1)}%` +
        (h.max_short_sector ? ` / short <b>${h.max_short_sector}</b> ${(Math.abs(h.max_short_exposure) * 100).toFixed(1)}%` : '')
      : 'No directional sector tilt';

    exposureEl.innerHTML = `
      <div class="exp-header">
        <div class="exp-title">Sector Exposure</div>
        <div class="exp-subtitle">Net long/short per GICS sector — exposes hidden thematic bets</div>
      </div>
      <div class="exp-headline">${headlineText}</div>
      <div class="exp-rows">${rows}</div>
    `;
  }

  function renderStress(regimes) {
    // Crisis-window performance table.  One row per regime that overlapped
    // the backtest's date range.  Sharpe colored by severity.
    const rows = regimes.map((r) => {
      const sharpe = r.sharpe;
      const sev = sharpe == null ? 'warn'
        : sharpe >= 0.5 ? 'good'
        : sharpe >= -0.5 ? 'warn'
        : 'bad';
      const sharpeStr = sharpe == null ? '—' : sharpe.toFixed(2);
      const totalStr = r.total_return != null ? (r.total_return * 100).toFixed(2) + '%' : '—';
      const ddStr = r.max_drawdown != null ? (r.max_drawdown * 100).toFixed(2) + '%' : '—';
      const hitStr = r.hit_rate != null ? (r.hit_rate * 100).toFixed(0) + '%' : '—';
      return `
        <div class="stress-row stress-sev-${sev}">
          <div class="stress-label">
            <div class="stress-name">${r.label}</div>
            <div class="stress-period">${r.start} → ${r.end} · ${r.n_days}d</div>
          </div>
          <div class="stress-metric"><span class="stress-key">Sharpe</span><span class="stress-val stress-val-${sev}">${sharpeStr}</span></div>
          <div class="stress-metric"><span class="stress-key">Return</span><span class="stress-val">${totalStr}</span></div>
          <div class="stress-metric"><span class="stress-key">Max DD</span><span class="stress-val">${ddStr}</span></div>
          <div class="stress-metric"><span class="stress-key">Hit %</span><span class="stress-val">${hitStr}</span></div>
        </div>
      `;
    }).join('');

    stressEl.innerHTML = `
      <div class="stress-header">
        <div class="stress-title">Crisis Performance</div>
        <div class="stress-subtitle">How the alpha behaved in well-known stress windows — a single overall Sharpe averages these regimes together.</div>
      </div>
      <div class="stress-rows">${rows}</div>
    `;
  }

  function renderDeflated(metrics, d) {
    // d shape: { deflated_sharpe_annualized, p_value, sharpe_threshold_annualized,
    //            n_trials, n_obs, is_significant }
    const headlineSharpe = metrics.sharpe ?? 0;
    const deflatedSharpe = d.deflated_sharpe_annualized ?? 0;
    const threshold = d.sharpe_threshold_annualized ?? 0;
    const pValue = d.p_value ?? 0;
    const n = d.n_trials ?? 1;

    // Verdict — three buckets matching the spec.
    let verdict = 'Likely noise';
    let verdictClass = 'bad';
    let verdictIcon = '❌';
    if (pValue >= 0.95) {
      verdict = 'Statistically real edge';
      verdictClass = 'good';
      verdictIcon = '✅';
    } else if (pValue >= 0.80) {
      verdict = 'Suggestive but not conclusive';
      verdictClass = 'warn';
      verdictIcon = '⚠️';
    }

    const deflatedTone =
      deflatedSharpe > 0.3 ? 'good' : deflatedSharpe < 0 ? 'bad' : 'warn';

    deflatedEl.innerHTML = `
      <div class="deflated-header">
        <div>
          <div class="deflated-title">Deflated Sharpe Ratio</div>
          <div class="deflated-subtitle">
            Adjusted for selection bias from ${n} trial${n === 1 ? '' : 's'} this session
          </div>
        </div>
        <span class="oos-badge ${verdictClass}">${verdictIcon} ${verdict}</span>
      </div>

      <div class="deflated-grid">
        <div class="deflated-stat">
          <div class="deflated-stat-label">Headline Sharpe</div>
          <div class="deflated-stat-value">${fmtNum(headlineSharpe)}</div>
        </div>
        <div class="deflated-stat">
          <div class="deflated-stat-label">Threshold (luck)</div>
          <div class="deflated-stat-value">${fmtNum(threshold)}</div>
          <div class="deflated-stat-sub">expected max from ${n} trials</div>
        </div>
        <div class="deflated-stat">
          <div class="deflated-stat-label">Deflated Sharpe</div>
          <div class="deflated-stat-value ${deflatedTone}">${fmtNum(deflatedSharpe)}</div>
          <div class="deflated-stat-sub">headline − threshold</div>
        </div>
        <div class="deflated-stat">
          <div class="deflated-stat-label">P(true SR &gt; 0)</div>
          <div class="deflated-stat-value">${(pValue * 100).toFixed(1)}%</div>
          <div class="deflated-stat-sub">${d.n_obs ?? 0} obs</div>
        </div>
      </div>

      <div class="deflated-explainer">
        With ${n} trial${n === 1 ? '' : 's'}, the best one is biased upward — the
        threshold is the Sharpe you'd expect to see <em>by chance</em> from ${n}
        random strategies. The deflated Sharpe subtracts that bias. P-value is
        the probability that the true Sharpe is positive after correcting for
        selection, return skew, and fat tails (Bailey &amp; López de Prado, 2014).
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
