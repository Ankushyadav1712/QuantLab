// Dashboard — 6 metric cards with animated counters and tooltips.

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

export function createDashboard(container) {
  container.classList.add('dashboard');
  container.innerHTML = '';

  const cards = {};
  for (const m of METRICS) {
    const card = document.createElement('div');
    card.className = 'metric-card glass';
    card.dataset.key = m.key;
    card.innerHTML = `
      <div class="label">${m.label}</div>
      <div class="value">—</div>
    `;
    container.appendChild(card);
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

  function setMetrics(metrics) {
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
  }

  function clear() {
    setMetrics(null);
  }

  return { setMetrics, clear };
}
