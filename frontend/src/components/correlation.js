// Correlation matrix heatmap.

export function createCorrelation(container) {
  container.classList.add('chart-card', 'glass');
  container.style.display = 'none';
  container.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <h4 style="margin:0;">Alpha Correlations</h4>
      <button type="button" class="ghost" data-role="hide">close</button>
    </div>
    <div data-role="body"></div>
  `;
  const body = container.querySelector('[data-role="body"]');
  container.querySelector('[data-role="hide"]').addEventListener('click', hide);

  function colorFor(v) {
    if (v == null) return 'var(--bg-tertiary)';
    const clamped = Math.max(-1, Math.min(1, v));
    const intensity = Math.abs(clamped);
    if (clamped < 0) return `rgba(255,107,107,${0.15 + 0.7 * intensity})`;
    return `rgba(0,212,255,${0.15 + 0.7 * intensity})`;
  }

  function show(payload) {
    const labels = payload?.tickers || [];
    const matrix = payload?.matrix || [];
    if (labels.length === 0) {
      body.innerHTML = '<div class="placeholder">No correlation data.</div>';
      container.style.display = '';
      return;
    }
    const tbl = document.createElement('table');
    tbl.className = 'corr-table';
    const header = document.createElement('tr');
    header.appendChild(document.createElement('th'));
    labels.forEach((l) => {
      const th = document.createElement('th');
      th.textContent = truncate(l, 16);
      th.title = l;
      header.appendChild(th);
    });
    tbl.appendChild(header);

    matrix.forEach((row, i) => {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.textContent = truncate(labels[i], 16);
      th.title = labels[i];
      tr.appendChild(th);
      row.forEach((v) => {
        const td = document.createElement('td');
        td.textContent = v == null ? '—' : v.toFixed(2);
        td.style.background = colorFor(v);
        tr.appendChild(td);
      });
      tbl.appendChild(tr);
    });

    body.innerHTML = '';
    body.appendChild(tbl);
    container.style.display = '';
  }

  function hide() {
    container.style.display = 'none';
  }

  function truncate(s, n) {
    s = String(s ?? '');
    return s.length > n ? s.slice(0, n) + '…' : s;
  }

  return { show, hide };
}
