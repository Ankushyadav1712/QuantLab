// Charts panel — lightweight-charts (loaded from CDN as window.LightweightCharts) +
// canvas/grid for histogram and monthly heatmap.

const CHART_THEME = {
  layout: {
    background: { type: 'solid', color: 'rgba(0,0,0,0)' },
    textColor: '#8b949e',
    fontFamily: 'JetBrains Mono, monospace',
  },
  grid: {
    vertLines: { color: 'rgba(48,54,61,0.4)' },
    horzLines: { color: 'rgba(48,54,61,0.4)' },
  },
  rightPriceScale: { borderColor: '#30363d' },
  timeScale: { borderColor: '#30363d', timeVisible: false, secondsVisible: false },
  crosshair: { mode: 0 },
};

export function createCharts(container) {
  container.classList.add('charts');
  container.innerHTML = `
    <div class="chart-card glass">
      <h4>Cumulative PnL</h4>
      <div class="chart-host" data-chart="equity"></div>
    </div>
    <div class="chart-row two">
      <div class="chart-card glass">
        <h4>Drawdown</h4>
        <div class="chart-host" data-chart="drawdown"></div>
      </div>
      <div class="chart-card glass">
        <h4>Rolling Sharpe (63d)</h4>
        <div class="chart-host" data-chart="sharpe"></div>
      </div>
    </div>
    <div class="chart-row two">
      <div class="chart-card glass">
        <h4>Daily Return Distribution</h4>
        <canvas data-chart="histogram" style="width:100%;height:220px;"></canvas>
      </div>
      <div class="chart-card glass">
        <h4>Monthly Returns</h4>
        <div class="heatmap-grid" data-chart="heatmap"></div>
      </div>
    </div>
  `;

  const hosts = {
    equity: container.querySelector('[data-chart="equity"]'),
    drawdown: container.querySelector('[data-chart="drawdown"]'),
    sharpe: container.querySelector('[data-chart="sharpe"]'),
    histogram: container.querySelector('[data-chart="histogram"]'),
    heatmap: container.querySelector('[data-chart="heatmap"]'),
  };

  const charts = {};

  function ensureChart(host, key, seriesType, seriesOpts) {
    if (charts[key]) return charts[key];
    const lc = window.LightweightCharts;
    if (!lc) {
      host.textContent = 'Lightweight-charts failed to load.';
      return null;
    }
    const chart = lc.createChart(host, {
      ...CHART_THEME,
      width: host.clientWidth,
      height: host.clientHeight,
    });
    let series;
    if (seriesType === 'area') series = chart.addAreaSeries(seriesOpts);
    else if (seriesType === 'line') series = chart.addLineSeries(seriesOpts);
    else if (seriesType === 'baseline') series = chart.addBaselineSeries(seriesOpts);
    charts[key] = { chart, series };

    new ResizeObserver(() => {
      chart.applyOptions({ width: host.clientWidth, height: host.clientHeight });
    }).observe(host);

    return charts[key];
  }

  function destroyChart(key) {
    if (charts[key]) {
      try { charts[key].chart.remove(); } catch (_) {}
      delete charts[key];
    }
  }

  function pairData(dates, values) {
    const out = [];
    for (let i = 0; i < dates.length; i++) {
      const v = values[i];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      out.push({ time: dates[i], value: Number(v) });
    }
    return out;
  }

  function renderEquityCurve(dates, pnl) {
    destroyChart('equity');
    const c = ensureChart(hosts.equity, 'equity', 'area', {
      lineColor: '#00d4ff',
      topColor: 'rgba(0,212,255,0.4)',
      bottomColor: 'rgba(0,212,255,0.02)',
      lineWidth: 2,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    });
    if (!c) return;
    c.series.setData(pairData(dates, pnl));
    c.chart.timeScale().fitContent();
  }

  function renderDrawdown(dates, dd) {
    destroyChart('drawdown');
    const c = ensureChart(hosts.drawdown, 'drawdown', 'area', {
      lineColor: '#ff6b6b',
      topColor: 'rgba(255,107,107,0.05)',
      bottomColor: 'rgba(255,107,107,0.5)',
      lineWidth: 1,
      priceFormat: { type: 'percent', precision: 2 },
    });
    if (!c) return;
    const pct = dd.map((v) => (v == null ? null : v * 100));
    c.series.setData(pairData(dates, pct));
    c.chart.timeScale().fitContent();
  }

  function renderRollingSharpe(dates, sharpe) {
    destroyChart('sharpe');
    const c = ensureChart(hosts.sharpe, 'sharpe', 'baseline', {
      baseValue: { type: 'price', price: 0 },
      topLineColor: '#39d353',
      topFillColor1: 'rgba(57,211,83,0.3)',
      topFillColor2: 'rgba(57,211,83,0.02)',
      bottomLineColor: '#ff6b6b',
      bottomFillColor1: 'rgba(255,107,107,0.02)',
      bottomFillColor2: 'rgba(255,107,107,0.3)',
      lineWidth: 2,
    });
    if (!c) return;
    c.series.setData(pairData(dates, sharpe));
    c.chart.timeScale().fitContent();
  }

  function renderReturnsHistogram(returns) {
    const canvas = hosts.histogram;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssW, cssH);

    const clean = returns.filter((v) => v != null && !Number.isNaN(v)).map(Number);
    if (clean.length === 0) return;

    const binPct = 0.5; // 0.5%
    const binSize = binPct / 100;
    const minR = Math.min(...clean);
    const maxR = Math.max(...clean);
    const lo = Math.floor(minR / binSize) * binSize;
    const hi = Math.ceil(maxR / binSize) * binSize;
    const nBins = Math.max(1, Math.round((hi - lo) / binSize));
    const bins = new Array(nBins).fill(0);
    for (const r of clean) {
      const i = Math.min(nBins - 1, Math.max(0, Math.floor((r - lo) / binSize)));
      bins[i] += 1;
    }
    const maxCount = Math.max(...bins);

    const padL = 8, padR = 8, padT = 8, padB = 24;
    const plotW = cssW - padL - padR;
    const plotH = cssH - padT - padB;
    const bw = plotW / nBins;

    // y-axis baseline
    ctx.strokeStyle = '#30363d';
    ctx.beginPath();
    ctx.moveTo(padL, cssH - padB);
    ctx.lineTo(cssW - padR, cssH - padB);
    ctx.stroke();

    for (let i = 0; i < nBins; i++) {
      const binStart = lo + i * binSize;
      const binEnd = binStart + binSize;
      const isNeg = binEnd <= 0;
      const isPos = binStart >= 0;
      const h = (bins[i] / maxCount) * plotH;
      const x = padL + i * bw;
      const y = cssH - padB - h;
      ctx.fillStyle = isNeg ? 'rgba(255,107,107,0.7)'
                      : isPos ? 'rgba(57,211,83,0.7)'
                      : 'rgba(139,148,158,0.5)';
      ctx.fillRect(x, y, Math.max(1, bw - 1), h);
    }

    // zero line
    if (lo < 0 && hi > 0) {
      const xZero = padL + ((-lo) / (hi - lo)) * plotW;
      ctx.strokeStyle = 'rgba(230,237,243,0.5)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xZero, padT);
      ctx.lineTo(xZero, cssH - padB);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // x-axis labels (lo, 0, hi)
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText((lo * 100).toFixed(1) + '%', padL, cssH - padB + 6);
    ctx.textAlign = 'right';
    ctx.fillText((hi * 100).toFixed(1) + '%', cssW - padR, cssH - padB + 6);
  }

  function colorForReturn(r, scale = 0.05) {
    const clamped = Math.max(-scale, Math.min(scale, r));
    const intensity = Math.abs(clamped) / scale; // 0..1
    if (clamped < 0)
      return `rgba(255,107,107,${0.15 + 0.7 * intensity})`;
    return `rgba(57,211,83,${0.15 + 0.7 * intensity})`;
  }

  function renderMonthlyHeatmap(monthlyReturns) {
    const host = hosts.heatmap;
    host.innerHTML = '';
    if (!monthlyReturns || monthlyReturns.length === 0) {
      host.innerHTML = '<div class="placeholder">No data</div>';
      return;
    }
    const byYear = {};
    for (const [y, m, v] of monthlyReturns) {
      if (!byYear[y]) byYear[y] = {};
      byYear[y][m] = v;
    }
    const years = Object.keys(byYear).map(Number).sort((a, b) => a - b);

    // header row: months
    const header = document.createElement('div');
    header.className = 'heatmap-row';
    header.appendChild(makeCell('', 'year-label'));
    ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'].forEach((m) => {
      header.appendChild(makeCell(m, 'year-label'));
    });
    host.appendChild(header);

    for (const y of years) {
      const row = document.createElement('div');
      row.className = 'heatmap-row';
      row.appendChild(makeCell(String(y), 'year-label'));
      for (let m = 1; m <= 12; m++) {
        const v = byYear[y][m];
        const cell = makeCell(v == null ? '' : (v * 100).toFixed(1));
        if (v != null) {
          cell.style.background = colorForReturn(v);
          cell.title = `${y}-${String(m).padStart(2, '0')}: ${(v * 100).toFixed(2)}%`;
        }
        row.appendChild(cell);
      }
      host.appendChild(row);
    }
  }

  function makeCell(text, extra = '') {
    const d = document.createElement('div');
    d.className = 'heatmap-cell ' + extra;
    d.textContent = text;
    return d;
  }

  function clear() {
    Object.keys(charts).forEach(destroyChart);
    hosts.heatmap.innerHTML = '';
    const ctx = hosts.histogram.getContext('2d');
    ctx.clearRect(0, 0, hosts.histogram.width, hosts.histogram.height);
  }

  function setData(timeseries, monthlyReturns) {
    if (!timeseries) {
      clear();
      return;
    }
    renderEquityCurve(timeseries.dates, timeseries.cumulative_pnl);
    renderDrawdown(timeseries.dates, timeseries.drawdown);
    renderRollingSharpe(timeseries.dates, timeseries.rolling_sharpe);
    renderReturnsHistogram(timeseries.daily_returns);
    renderMonthlyHeatmap(monthlyReturns);
  }

  return { setData, clear };
}
