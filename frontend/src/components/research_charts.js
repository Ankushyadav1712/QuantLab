// Tier 4 research-grade chart bundle.  All pure SVG — no dependencies.
// Four panels stacked vertically:
//   1) IC time series with rolling-21d MA + ±2σ z-score bands
//   2) IC decay across horizons (full-size; mirrors the Signal Quality mini)
//   3) Factor quintile bars — does the signal predict monotonically?
//   4) Risk decomposition — FF5 factor_share + per-factor loadings
//
// Wires into a single host element; consumer calls .render(metrics, decomp).

const W = 540;
const H_TS = 200;    // IC time series — taller, more data points
const H_BARS = 180;  // decay / quintile / risk charts
const PAD_LEFT = 50;
const PAD_RIGHT = 16;
const PAD_TOP = 18;
const PAD_BOT = 36;

export function createResearchCharts(container) {
  container.classList.add('research-charts-wrap');
  container.innerHTML = `
    <div class="rc-header">
      <div class="rc-title">Research Charts</div>
      <div class="rc-subtitle">IC time series · alpha decay · quintile returns · risk decomposition</div>
    </div>
    <div data-role="rc-empty" class="rc-empty" style="display:none;">
      Run a backtest with at least 50 days of valid IC data to populate these panels.
    </div>
    <div class="rc-grid">
      <div data-role="rc-ic-ts" class="rc-panel"></div>
      <div data-role="rc-ic-decay" class="rc-panel"></div>
      <div data-role="rc-quintiles" class="rc-panel"></div>
      <div data-role="rc-risk" class="rc-panel"></div>
    </div>
  `;
  const emptyEl = container.querySelector('[data-role="rc-empty"]');
  const tsEl = container.querySelector('[data-role="rc-ic-ts"]');
  const decayEl = container.querySelector('[data-role="rc-ic-decay"]');
  const qEl = container.querySelector('[data-role="rc-quintiles"]');
  const riskEl = container.querySelector('[data-role="rc-risk"]');

  function render(metrics, decomp) {
    // If we have no IC data at all (e.g. legacy saved alpha), hide
    // everything — the panel is meaningless without it.
    const hasIc = metrics && metrics.ic_series && Array.isArray(metrics.ic_series.ic)
                  && metrics.ic_series.ic.length >= 5;
    const hasDecay = metrics && metrics.alpha_decay
                      && metrics.alpha_decay.ic_by_horizon
                      && Object.keys(metrics.alpha_decay.ic_by_horizon).length > 0;
    const hasQuint = metrics && Array.isArray(metrics.quintile_returns)
                      && metrics.quintile_returns.length >= 2;
    const hasRisk = decomp && decomp.factor_share != null;

    if (!hasIc && !hasDecay && !hasQuint && !hasRisk) {
      emptyEl.style.display = '';
      tsEl.innerHTML = decayEl.innerHTML = qEl.innerHTML = riskEl.innerHTML = '';
      container.classList.add('rc-collapsed');
      return;
    }
    emptyEl.style.display = 'none';
    container.classList.remove('rc-collapsed');

    tsEl.innerHTML = hasIc ? renderICTimeSeries(metrics.ic_series) : panelEmpty('IC time series', 'Needs IC data.');
    decayEl.innerHTML = hasDecay ? renderICDecay(metrics.alpha_decay) : panelEmpty('Alpha decay', 'Decay fit unavailable.');
    qEl.innerHTML = hasQuint ? renderQuintiles(metrics.quintile_returns) : panelEmpty('Factor quintile returns', 'Need at least 5 tickers for quintile bucketing.');
    riskEl.innerHTML = hasRisk ? renderRiskDecomp(decomp) : panelEmpty('Risk decomposition', 'Run Fama-French decomposition to populate.');
  }

  return { render };
}

function panelEmpty(title, msg) {
  return `<div class="rc-panel-head">${title}</div>
          <div class="rc-panel-empty">${msg}</div>`;
}

// ---------- IC time series ----------

function renderICTimeSeries({ dates, ic }) {
  // Clip the chart length to avoid pathologically wide SVGs.
  // Display the most recent ~1000 days, downsample if longer.
  const MAX_POINTS = 1000;
  let xs = dates, ys = ic;
  if (xs.length > MAX_POINTS) {
    const step = Math.ceil(xs.length / MAX_POINTS);
    xs = xs.filter((_, i) => i % step === 0);
    ys = ic.filter((_, i) => i % step === 0);
  }
  // Rolling 21-day MA + global mean/std for ±2σ bands.  We compute these
  // here rather than on the backend so callers can change the window
  // without a roundtrip.
  const WINDOW = 21;
  const ma = ys.map((_, i) => {
    if (i < WINDOW - 1) return null;
    let s = 0;
    for (let k = 0; k < WINDOW; k++) s += ys[i - k];
    return s / WINDOW;
  });
  const validIc = ys.filter((v) => v != null && !Number.isNaN(v));
  const mean = validIc.reduce((a, b) => a + b, 0) / Math.max(1, validIc.length);
  const variance = validIc.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, validIc.length - 1);
  const std = Math.sqrt(variance);
  const upper2 = mean + 2 * std;
  const lower2 = mean - 2 * std;

  // Plotting geometry
  const yMin = Math.min(lower2, ...ys) - 0.02;
  const yMax = Math.max(upper2, ...ys) + 0.02;
  const plotW = W - PAD_LEFT - PAD_RIGHT;
  const plotH = H_TS - PAD_TOP - PAD_BOT;
  const scaleX = (i) => PAD_LEFT + (i / Math.max(1, ys.length - 1)) * plotW;
  const scaleY = (v) => PAD_TOP + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  // Raw IC line (thin)
  const rawPath = ys.map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(1)} ${scaleY(v).toFixed(1)}`).join(' ');
  // Rolling MA line (thick), only where defined
  const maPathSegs = [];
  let started = false;
  ma.forEach((v, i) => {
    if (v == null) { started = false; return; }
    maPathSegs.push(`${started ? 'L' : 'M'} ${scaleX(i).toFixed(1)} ${scaleY(v).toFixed(1)}`);
    started = true;
  });

  // X-axis tick labels: first / middle / last date
  const tickIdx = [0, Math.floor(ys.length / 2), ys.length - 1];
  const xTicks = tickIdx.map((i) => ({
    x: scaleX(i),
    label: (xs[i] || '').slice(0, 7),  // YYYY-MM
  }));
  // Y-axis ticks at every 0.05 IC unit
  const yTicks = [];
  const step = pickYStep(yMin, yMax);
  for (let y = Math.ceil(yMin / step) * step; y <= yMax; y += step) {
    yTicks.push({ y: scaleY(y), label: y.toFixed(2) });
  }
  const zeroY = scaleY(0);
  const upperY = scaleY(upper2);
  const lowerY = scaleY(lower2);
  const meanY = scaleY(mean);

  return `
    <div class="rc-panel-head">IC time series <span class="rc-panel-sub">daily · 21d MA · ±2σ bands</span></div>
    <svg viewBox="0 0 ${W} ${H_TS}" class="rc-svg" xmlns="http://www.w3.org/2000/svg">
      <!-- ±2σ shaded band -->
      <rect class="rc-ts-band" x="${PAD_LEFT}" y="${upperY.toFixed(1)}"
            width="${plotW}" height="${(lowerY - upperY).toFixed(1)}"></rect>
      <!-- zero reference -->
      <line class="rc-ts-zero" x1="${PAD_LEFT}" y1="${zeroY.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${zeroY.toFixed(1)}"></line>
      <!-- mean reference -->
      <line class="rc-ts-mean" x1="${PAD_LEFT}" y1="${meanY.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${meanY.toFixed(1)}"></line>
      <!-- y grid + labels -->
      ${yTicks.map((t) => `
        <line class="rc-grid" x1="${PAD_LEFT}" y1="${t.y.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${t.y.toFixed(1)}"></line>
        <text class="rc-tick-label" x="${PAD_LEFT - 4}" y="${(t.y + 3).toFixed(1)}" text-anchor="end">${t.label}</text>
      `).join('')}
      <!-- axes -->
      <line class="rc-axis" x1="${PAD_LEFT}" y1="${PAD_TOP}" x2="${PAD_LEFT}" y2="${H_TS - PAD_BOT}"></line>
      <line class="rc-axis" x1="${PAD_LEFT}" y1="${H_TS - PAD_BOT}" x2="${W - PAD_RIGHT}" y2="${H_TS - PAD_BOT}"></line>
      <!-- x ticks -->
      ${xTicks.map((t) => `
        <text class="rc-tick-label" x="${t.x.toFixed(1)}" y="${H_TS - PAD_BOT + 14}" text-anchor="middle">${t.label}</text>
      `).join('')}
      <!-- raw IC line -->
      <path class="rc-ts-raw" d="${rawPath}"></path>
      <!-- rolling MA -->
      <path class="rc-ts-ma" d="${maPathSegs.join(' ')}"></path>
    </svg>
    <div class="rc-legend">
      <span><span class="rc-legend-line rc-ts-raw-bar"></span> daily IC</span>
      <span><span class="rc-legend-line rc-ts-ma-bar"></span> 21d MA</span>
      <span><span class="rc-legend-line rc-ts-band-bar"></span> ±2σ band (μ=${mean.toFixed(3)}, σ=${std.toFixed(3)})</span>
    </div>
  `;
}

function pickYStep(yMin, yMax) {
  const range = yMax - yMin;
  if (range < 0.05) return 0.01;
  if (range < 0.2) return 0.05;
  if (range < 1) return 0.1;
  return 0.25;
}

// ---------- IC decay (full-size) ----------

function renderICDecay(decay) {
  const horizons = Object.keys(decay.ic_by_horizon || {}).map(Number).sort((a, b) => a - b);
  if (horizons.length === 0) {
    return panelEmpty('Alpha decay across horizons', 'No horizons reported.');
  }
  const values = horizons.map((h) => decay.ic_by_horizon[h] ?? 0);
  const maxAbs = Math.max(0.01, ...values.map((v) => Math.abs(v)));
  const halfLife = decay.half_life_days;
  const r2 = decay.r_squared;

  const plotW = W - PAD_LEFT - PAD_RIGHT;
  const plotH = H_BARS - PAD_TOP - PAD_BOT;
  const barW = (plotW - 20) / horizons.length;
  const zeroY = PAD_TOP + plotH / 2;
  const scaleY = (v) => zeroY - (v / maxAbs) * (plotH / 2 - 8);

  const bars = horizons.map((h, i) => {
    const v = values[i];
    const x = PAD_LEFT + 10 + i * barW + 6;
    const w = barW - 12;
    const y = v >= 0 ? scaleY(v) : zeroY;
    const height = Math.abs(scaleY(v) - zeroY);
    const tone = v >= 0 ? 'pos' : 'neg';
    return `
      <rect class="rc-bar-${tone}" x="${x.toFixed(1)}" y="${y.toFixed(1)}"
            width="${w.toFixed(1)}" height="${height.toFixed(1)}"></rect>
      <text class="rc-bar-value" x="${(x + w / 2).toFixed(1)}" y="${(v >= 0 ? y - 4 : y + height + 12).toFixed(1)}"
            text-anchor="middle">${v.toFixed(3)}</text>
      <text class="rc-tick-label" x="${(x + w / 2).toFixed(1)}" y="${(H_BARS - PAD_BOT + 14).toFixed(1)}"
            text-anchor="middle">${h}d</text>
    `;
  }).join('');

  // Optional exponential decay overlay if half-life is fit
  let overlayPath = '';
  if (halfLife != null && halfLife > 0 && values[0] != null) {
    const tau = halfLife / Math.LN2;
    const sign = values[0] >= 0 ? 1 : -1;
    const amp = Math.abs(values[0]);
    const pts = [];
    const xs = horizons[horizons.length - 1] + 1;
    for (let h = horizons[0]; h <= xs; h += 0.25) {
      const v = sign * amp * Math.exp(-h / tau);
      const idx = horizons.findIndex((hh) => hh >= h);
      // Interpolate x position based on the discrete bar positions
      const fracIdx = h <= horizons[0] ? 0 :
                       h >= horizons[horizons.length - 1] ? horizons.length - 1 :
                       horizons.length - 1 - (horizons.length - 1 - idx) - (horizons[idx] - h) / Math.max(1e-9, horizons[idx] - (horizons[idx - 1] ?? horizons[idx]));
      const x = PAD_LEFT + 10 + (fracIdx + 0.5) * barW;
      const y = scaleY(v);
      pts.push(`${pts.length === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`);
    }
    overlayPath = `<path class="rc-decay-fit" d="${pts.join(' ')}"></path>`;
  }

  const halfLifeText = halfLife != null ? `${halfLife.toFixed(1)}d` : '—';
  const r2Text = r2 != null ? `R²=${r2.toFixed(2)}` : '';
  const interp = halfLife == null ? 'No clean exponential fit' :
                  halfLife < 5 ? 'Fast alpha — rebalance daily' :
                  halfLife < 21 ? 'Medium horizon — weekly rebalance acceptable' :
                  'Slow alpha — low-turnover OK';

  return `
    <div class="rc-panel-head">Alpha decay across horizons <span class="rc-panel-sub">half-life ${halfLifeText} · ${r2Text} · ${interp}</span></div>
    <svg viewBox="0 0 ${W} ${H_BARS}" class="rc-svg" xmlns="http://www.w3.org/2000/svg">
      <line class="rc-axis" x1="${PAD_LEFT}" y1="${PAD_TOP}" x2="${PAD_LEFT}" y2="${H_BARS - PAD_BOT}"></line>
      <line class="rc-zero-strong" x1="${PAD_LEFT}" y1="${zeroY.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${zeroY.toFixed(1)}"></line>
      ${bars}
      ${overlayPath}
    </svg>
  `;
}

// ---------- Factor quintile bars ----------

function renderQuintiles(quintiles) {
  // Bars showing per-quintile mean forward return.  Monotonic Q1→Q5 = good.
  const xs = quintiles.map((q) => q.quantile);
  const ys = quintiles.map((q) => q.mean_return ?? 0);
  const maxAbs = Math.max(...ys.map(Math.abs), 0.0001);
  const plotW = W - PAD_LEFT - PAD_RIGHT;
  const plotH = H_BARS - PAD_TOP - PAD_BOT;
  const barW = (plotW - 20) / xs.length;
  const zeroY = PAD_TOP + plotH / 2;
  const scaleY = (v) => zeroY - (v / maxAbs) * (plotH / 2 - 12);

  const bars = quintiles.map((q, i) => {
    const v = q.mean_return ?? 0;
    const x = PAD_LEFT + 10 + i * barW + 6;
    const w = barW - 12;
    const y = v >= 0 ? scaleY(v) : zeroY;
    const height = Math.abs(scaleY(v) - zeroY);
    const tone = v >= 0 ? 'pos' : 'neg';
    return `
      <rect class="rc-bar-${tone}" x="${x.toFixed(1)}" y="${y.toFixed(1)}"
            width="${w.toFixed(1)}" height="${height.toFixed(1)}"></rect>
      <text class="rc-bar-value" x="${(x + w / 2).toFixed(1)}" y="${(v >= 0 ? y - 4 : y + height + 12).toFixed(1)}"
            text-anchor="middle">${(v * 10000).toFixed(1)}bp</text>
      <text class="rc-tick-label" x="${(x + w / 2).toFixed(1)}" y="${(H_BARS - PAD_BOT + 14).toFixed(1)}"
            text-anchor="middle">Q${q.quantile}</text>
    `;
  }).join('');

  // Monotonic check — Q1<Q2<…<Qn for an ideal alpha
  let monotonic = true;
  for (let i = 1; i < ys.length; i++) {
    if (ys[i] < ys[i - 1]) { monotonic = false; break; }
  }
  const spread = (ys[ys.length - 1] - ys[0]) * 10000;
  const verdict = monotonic ? `Monotonic ✓ · Q${xs[xs.length - 1]}−Q1 = ${spread.toFixed(1)}bp/day`
                            : `Non-monotonic — signal may only capture tails`;

  return `
    <div class="rc-panel-head">Factor quintile returns <span class="rc-panel-sub">bps/day · ${verdict}</span></div>
    <svg viewBox="0 0 ${W} ${H_BARS}" class="rc-svg" xmlns="http://www.w3.org/2000/svg">
      <line class="rc-axis" x1="${PAD_LEFT}" y1="${PAD_TOP}" x2="${PAD_LEFT}" y2="${H_BARS - PAD_BOT}"></line>
      <line class="rc-zero-strong" x1="${PAD_LEFT}" y1="${zeroY.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${zeroY.toFixed(1)}"></line>
      ${bars}
    </svg>
  `;
}

// ---------- Risk decomposition ----------

function renderRiskDecomp(decomp) {
  // Top row: stacked bar showing factor_share vs residual α-share.
  // Bottom row: per-factor loadings with t-stats.
  const factorShare = Math.max(0, Math.min(1, decomp.factor_share ?? 0));
  const residualShare = 1 - factorShare;
  const loadings = decomp.loadings || {};
  const factorNames = Object.keys(loadings);

  // Stacked bar (horizontal)
  const stackW = W - PAD_LEFT - PAD_RIGHT;
  const stackY = PAD_TOP;
  const stackH = 24;
  const facWidth = stackW * factorShare;
  const resWidth = stackW * residualShare;

  // Per-factor loadings panel (below the stacked bar)
  const loadingsY = stackY + stackH + 28;
  const loadingsH = H_BARS - PAD_TOP - PAD_BOT - stackH - 28;
  const maxBetaAbs = Math.max(0.001, ...factorNames.map((n) => Math.abs(loadings[n].beta ?? 0)));
  const factorW = stackW / Math.max(1, factorNames.length);
  const zeroY = loadingsY + loadingsH / 2;
  const scaleY = (v) => zeroY - (v / maxBetaAbs) * (loadingsH / 2 - 14);

  const factorBars = factorNames.map((name, i) => {
    const b = loadings[name].beta ?? 0;
    const t = loadings[name].t_stat;
    const x = PAD_LEFT + i * factorW + factorW * 0.15;
    const w = factorW * 0.7;
    const y = b >= 0 ? scaleY(b) : zeroY;
    const height = Math.abs(scaleY(b) - zeroY);
    const tone = b >= 0 ? 'pos' : 'neg';
    const significant = t != null && Math.abs(t) > 2;
    const star = significant ? '*' : '';
    return `
      <rect class="rc-bar-${tone}" x="${x.toFixed(1)}" y="${y.toFixed(1)}"
            width="${w.toFixed(1)}" height="${height.toFixed(1)}"></rect>
      <text class="rc-bar-value" x="${(x + w / 2).toFixed(1)}" y="${(b >= 0 ? y - 4 : y + height + 12).toFixed(1)}"
            text-anchor="middle">${b.toFixed(3)}${star}</text>
      <text class="rc-tick-label" x="${(x + w / 2).toFixed(1)}"
            y="${(loadingsY + loadingsH + 14).toFixed(1)}" text-anchor="middle">${name}</text>
    `;
  }).join('');

  const alpha = decomp.alpha_annualized;
  const alphaT = decomp.alpha_t_stat;
  const r2 = decomp.r_squared;
  const alphaText = alpha != null ? `${(alpha * 100).toFixed(2)}%` : '—';
  const alphaSig = alphaT != null && Math.abs(alphaT) > 2 ? ' (significant)' : '';

  return `
    <div class="rc-panel-head">Risk decomposition <span class="rc-panel-sub">
      α=${alphaText}${alphaSig} · R²=${(r2 ?? 0).toFixed(3)} · * = |t|>2
    </span></div>
    <svg viewBox="0 0 ${W} ${H_BARS}" class="rc-svg" xmlns="http://www.w3.org/2000/svg">
      <!-- Variance share stacked bar -->
      <rect class="rc-stack-factors" x="${PAD_LEFT}" y="${stackY}"
            width="${facWidth.toFixed(1)}" height="${stackH}"></rect>
      <rect class="rc-stack-residual" x="${(PAD_LEFT + facWidth).toFixed(1)}" y="${stackY}"
            width="${resWidth.toFixed(1)}" height="${stackH}"></rect>
      <text class="rc-stack-label" x="${(PAD_LEFT + facWidth / 2).toFixed(1)}" y="${(stackY + stackH / 2 + 4).toFixed(1)}" text-anchor="middle">
        FF5 factors ${(factorShare * 100).toFixed(1)}%
      </text>
      <text class="rc-stack-label" x="${(PAD_LEFT + facWidth + resWidth / 2).toFixed(1)}" y="${(stackY + stackH / 2 + 4).toFixed(1)}" text-anchor="middle">
        Residual α ${(residualShare * 100).toFixed(1)}%
      </text>
      <!-- Per-factor loadings axis -->
      <line class="rc-zero-strong" x1="${PAD_LEFT}" y1="${zeroY.toFixed(1)}" x2="${W - PAD_RIGHT}" y2="${zeroY.toFixed(1)}"></line>
      ${factorBars}
    </svg>
  `;
}
