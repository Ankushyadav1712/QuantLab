// Charts panel — lightweight-charts (loaded from CDN as window.LightweightCharts)
// for time-series; Canvas/CSS-grid for histogram and heatmap.
//
// Equity / drawdown / rolling-Sharpe accept BOTH IS and OOS timeseries; each
// half is rendered as its own series with a distinct color, and a vertical
// dashed line marks the IS/OOS boundary date.

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

  // Each chart is recreated from scratch on every setData call (simpler than
  // diffing series).  charts[key] holds {chart, series:[]} so destroy works.
  const charts = {};

  function pairData(dates, values) {
    const out = [];
    for (let i = 0; i < dates.length; i++) {
      const v = values[i];
      if (v === null || v === undefined || Number.isNaN(v)) continue;
      out.push({ time: dates[i], value: Number(v) });
    }
    return out;
  }

  function destroyChart(key) {
    if (!charts[key]) return;
    try { charts[key].chart.remove(); } catch (_) {}
    if (charts[key].overlay) charts[key].overlay.remove();
    delete charts[key];
  }

  function makeChart(host, key) {
    destroyChart(key);
    const lc = window.LightweightCharts;
    if (!lc) {
      host.textContent = 'Lightweight-charts failed to load.';
      return null;
    }
    host.style.position = 'relative';
    const chart = lc.createChart(host, {
      ...CHART_THEME,
      width: host.clientWidth,
      height: host.clientHeight,
    });
    const entry = { chart, series: [], host, key };
    charts[key] = entry;
    new ResizeObserver(() => {
      chart.applyOptions({ width: host.clientWidth, height: host.clientHeight });
      placeBoundary(entry);
      placeRegionLabels(entry);
    }).observe(host);
    return entry;
  }

  // Build a vertical dashed-white line overlay at boundaryTime, plus the
  // "In-Sample" / "Out-of-Sample" text labels.  Re-positioned on resize and
  // when the visible time range changes.
  function ensureOverlay(entry, boundaryTime) {
    if (!entry) return;
    if (!entry.overlay) {
      const el = document.createElement('div');
      el.className = 'chart-overlay';
      el.innerHTML = `
        <div class="boundary-line" data-role="boundary"></div>
        <div class="region-label is" data-role="is-label">In-Sample</div>
        <div class="region-label oos" data-role="oos-label">Out-of-Sample</div>
      `;
      entry.host.appendChild(el);
      entry.overlay = el;
    }
    entry.boundaryTime = boundaryTime;
    placeBoundary(entry);
    placeRegionLabels(entry);
    entry.chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      placeBoundary(entry);
      placeRegionLabels(entry);
    });
  }

  function placeBoundary(entry) {
    if (!entry || !entry.overlay || !entry.boundaryTime) return;
    const line = entry.overlay.querySelector('[data-role="boundary"]');
    const x = entry.chart.timeScale().timeToCoordinate(entry.boundaryTime);
    if (x == null) {
      line.style.display = 'none';
    } else {
      line.style.display = '';
      line.style.left = `${x}px`;
    }
  }

  function placeRegionLabels(entry) {
    if (!entry || !entry.overlay || !entry.boundaryTime) return;
    const isLabel = entry.overlay.querySelector('[data-role="is-label"]');
    const oosLabel = entry.overlay.querySelector('[data-role="oos-label"]');
    const x = entry.chart.timeScale().timeToCoordinate(entry.boundaryTime);
    if (x == null) return;
    const w = entry.host.clientWidth;
    isLabel.style.left = `${Math.max(8, x / 2 - 35)}px`;
    oosLabel.style.left = `${Math.min(w - 110, x + (w - x) / 2 - 50)}px`;
  }

  // ---------- Equity / drawdown / Sharpe ----------

  function renderEquityCurve(isTs, oosTs) {
    const entry = makeChart(hosts.equity, 'equity');
    if (!entry) return;

    const isData = pairData(isTs.dates, isTs.cumulative_pnl);
    let oosData = [];
    let boundaryTime = null;

    if (oosTs && oosTs.dates && oosTs.dates.length) {
      // Stitch OOS continuously onto the IS equity curve: shift every OOS
      // point up by the final IS cumulative PnL.
      const isFinal = isData.length ? isData[isData.length - 1].value : 0;
      oosData = pairData(oosTs.dates, oosTs.cumulative_pnl).map((p) => ({
        time: p.time, value: p.value + isFinal,
      }));
      boundaryTime = oosTs.dates[0];
    }

    const isSeries = entry.chart.addLineSeries({
      color: '#00d4ff', lineWidth: 2,
      priceFormat: { type: 'price', precision: 0, minMove: 1 },
    });
    isSeries.setData(isData);
    entry.series.push(isSeries);

    if (oosData.length) {
      const oosSeries = entry.chart.addLineSeries({
        color: '#39d353', lineWidth: 2,
        priceFormat: { type: 'price', precision: 0, minMove: 1 },
      });
      oosSeries.setData(oosData);
      entry.series.push(oosSeries);
      ensureOverlay(entry, boundaryTime);
    }

    entry.chart.timeScale().fitContent();
  }

  function renderDrawdown(isTs, oosTs) {
    const entry = makeChart(hosts.drawdown, 'drawdown');
    if (!entry) return;

    const toPct = (xs) => (xs || []).map((v) => (v == null ? null : v * 100));

    const isSeries = entry.chart.addLineSeries({
      color: '#00d4ff', lineWidth: 1,
      priceFormat: { type: 'percent', precision: 2 },
    });
    isSeries.setData(pairData(isTs.dates, toPct(isTs.drawdown)));
    entry.series.push(isSeries);

    if (oosTs && oosTs.dates && oosTs.dates.length) {
      const oosSeries = entry.chart.addLineSeries({
        color: '#ff6b6b', lineWidth: 1,
        priceFormat: { type: 'percent', precision: 2 },
      });
      oosSeries.setData(pairData(oosTs.dates, toPct(oosTs.drawdown)));
      entry.series.push(oosSeries);
      ensureOverlay(entry, oosTs.dates[0]);
    }

    entry.chart.timeScale().fitContent();
  }

  function renderRollingSharpe(isTs, oosTs) {
    const entry = makeChart(hosts.sharpe, 'sharpe');
    if (!entry) return;

    // Use a baseline series for IS so we keep the green/red split around 0.
    const isSeries = entry.chart.addBaselineSeries({
      baseValue: { type: 'price', price: 0 },
      topLineColor: '#39d353',
      topFillColor1: 'rgba(57,211,83,0.3)',
      topFillColor2: 'rgba(57,211,83,0.02)',
      bottomLineColor: '#ff6b6b',
      bottomFillColor1: 'rgba(255,107,107,0.02)',
      bottomFillColor2: 'rgba(255,107,107,0.3)',
      lineWidth: 2,
    });
    isSeries.setData(pairData(isTs.dates, isTs.rolling_sharpe));
    entry.series.push(isSeries);

    if (oosTs && oosTs.dates && oosTs.dates.length) {
      const oosSeries = entry.chart.addLineSeries({
        color: '#a78bfa', lineWidth: 2,
      });
      oosSeries.setData(pairData(oosTs.dates, oosTs.rolling_sharpe));
      entry.series.push(oosSeries);
      ensureOverlay(entry, oosTs.dates[0]);
    }

    entry.chart.timeScale().fitContent();
  }

  // ---------- Histogram (combined IS + OOS) ----------

  function renderReturnsHistogram(isTs, oosTs) {
    const canvas = hosts.histogram;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssW, cssH);

    const all = [...(isTs.daily_returns || []), ...((oosTs && oosTs.daily_returns) || [])];
    const clean = all.filter((v) => v != null && !Number.isNaN(v)).map(Number);
    if (clean.length === 0) return;

    const binPct = 0.5;
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

    ctx.fillStyle = '#8b949e';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText((lo * 100).toFixed(1) + '%', padL, cssH - padB + 6);
    ctx.textAlign = 'right';
    ctx.fillText((hi * 100).toFixed(1) + '%', cssW - padR, cssH - padB + 6);
  }

  // ---------- Monthly heatmap ----------

  function colorForReturn(r, scale = 0.05) {
    const clamped = Math.max(-scale, Math.min(scale, r));
    const intensity = Math.abs(clamped) / scale;
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

  function setData(isTs, monthlyReturns, opts = {}) {
    if (!isTs) { clear(); return; }
    const oosTs = opts.oos_timeseries || null;
    renderEquityCurve(isTs, oosTs);
    renderDrawdown(isTs, oosTs);
    renderRollingSharpe(isTs, oosTs);
    renderReturnsHistogram(isTs, oosTs);
    renderMonthlyHeatmap(monthlyReturns);
  }

  return { setData, clear };
}
