// Multi-alpha batch comparison — a sortable ranked metrics table plus a
// return-correlation heatmap for N alphas run through /api/batch_simulate.
// Click a row to load that alpha's full backtest. IS-only, like /api/compare,
// but scaled to many alphas and ranked.

const METRIC_COLS = [
  { key: 'sharpe', label: 'Sharpe', fmt: (v) => fmtNum(v, 2) },
  { key: 'ic', label: 'IC', fmt: (v) => fmtNum(v, 3) },
  { key: 'ic_tstat', label: 'IC t', fmt: (v) => fmtNum(v, 2) },
  { key: 'annual_return', label: 'Ann. Ret', fmt: fmtPct },
  { key: 'max_drawdown', label: 'Max DD', fmt: fmtPct, neg: true },
  { key: 'avg_turnover', label: 'Turnover', fmt: (v) => fmtNum(v, 2) },
  { key: 'fitness', label: 'Fitness', fmt: (v) => fmtNum(v, 3) },
  { key: 'win_rate', label: 'Win', fmt: fmtPct },
];

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtNum(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return (v * 100).toFixed(2) + '%';
}

// Heatmap cell color — red for negative, cyan for positive (same scheme as
// components/correlation.js so the two views read identically).
function colorFor(v) {
  if (v == null) return 'var(--bg-tertiary)';
  const clamped = Math.max(-1, Math.min(1, v));
  const intensity = Math.abs(clamped);
  if (clamped < 0) return `rgba(255,107,107,${0.15 + 0.7 * intensity})`;
  return `rgba(0,212,255,${0.15 + 0.7 * intensity})`;
}

function truncate(s, n) {
  s = String(s ?? '');
  return s.length > n ? s.slice(0, n) + '…' : s;
}

export function createBatchComparison(container) {
  container.classList.add('batch-section');
  container.style.display = 'none';
  container.innerHTML = '';

  let payload = null;
  let names = {}; // id -> display name (optional)
  let sortKey = 'sharpe';
  let sortDir = -1; // -1 = descending, +1 = ascending
  const callbacks = { onSelectAlpha: null };

  function label(id) {
    return names[String(id)] || String(id);
  }

  function clear() {
    container.style.display = 'none';
    container.innerHTML = '';
    payload = null;
  }

  function render(p, opts = {}) {
    payload = p;
    names = opts.names || {};
    sortKey = 'sharpe';
    sortDir = -1;
    draw();
    container.style.display = '';
  }

  // Transient progress line shown while a streaming batch is running; replaced
  // by the ranked table once render() is called with the final payload.
  function setStatus(text) {
    container.style.display = '';
    container.innerHTML = `<div class="placeholder">${escapeHtml(text)}</div>`;
  }

  // Sort a copy of the results by the active column. Errored alphas (no
  // metrics) always sink to the bottom regardless of direction.
  function sortedResults() {
    const rows = [...(payload?.results || [])];
    rows.sort((a, b) => {
      const aOk = a.metrics != null;
      const bOk = b.metrics != null;
      if (aOk !== bOk) return aOk ? -1 : 1;
      if (!aOk) return 0;
      const av = a.metrics[sortKey];
      const bv = b.metrics[sortKey];
      const an = av == null || Number.isNaN(av) ? -Infinity : av;
      const bn = bv == null || Number.isNaN(bv) ? -Infinity : bv;
      return (an - bn) * sortDir;
    });
    return rows;
  }

  function draw() {
    if (!payload || !(payload.results || []).length) {
      container.innerHTML = '<div class="placeholder">No batch results.</div>';
      return;
    }
    const results = sortedResults();
    const nTotal = payload.n_alphas ?? results.length;
    const nOk = payload.n_ok ?? results.filter((r) => r.metrics).length;
    const elapsed = payload.elapsed_sec != null ? ` · ${payload.elapsed_sec}s` : '';

    const headCells = ['Alpha', ...METRIC_COLS.map((c) => c.label)]
      .map((h, i) => {
        if (i === 0) return `<th style="text-align:left">${h}</th>`;
        const col = METRIC_COLS[i - 1];
        const arrow = sortKey === col.key ? (sortDir === -1 ? ' ▼' : ' ▲') : '';
        return `<th class="batch-sort" data-key="${col.key}" style="cursor:pointer" title="Sort by ${col.label}">${h}${arrow}</th>`;
      })
      .join('');

    const bodyRows = results
      .map((r) => {
        const nm = escapeHtml(label(r.id));
        const exprTitle = escapeHtml(r.expression || '');
        const exprLine = `<div class="batch-expr" title="${exprTitle}" style="font-size:11px;color:var(--text-secondary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:220px;">${escapeHtml(truncate(r.expression || '', 44))}</div>`;
        if (r.metrics == null) {
          return `<tr class="batch-row batch-row-error" data-id="${escapeHtml(String(r.id))}">
            <td style="text-align:left"><span class="batch-name">${nm}</span> <span class="cmp-error-badge">error</span>${exprLine}</td>
            <td colspan="${METRIC_COLS.length}" class="batch-err-msg" style="color:var(--accent-red,#ff6b6b);font-size:12px;">${escapeHtml(r.error || 'failed')}</td>
          </tr>`;
        }
        const cells = METRIC_COLS.map((c) => {
          const cls = c.neg ? 'is-num neg' : 'is-num';
          return `<td class="${cls}">${c.fmt(r.metrics[c.key])}</td>`;
        }).join('');
        return `<tr class="batch-row" data-id="${escapeHtml(String(r.id))}" style="cursor:pointer" title="Click to load ${nm}">
          <td style="text-align:left"><span class="batch-name">${nm}</span>${exprLine}</td>
          ${cells}
        </tr>`;
      })
      .join('');

    const corr = payload.correlation_matrix || {};
    const corrLabels = corr.labels || [];
    const corrMatrix = corr.matrix || [];
    let corrHtml = '';
    if (corrLabels.length >= 2) {
      const headerRow = `<tr><th></th>${corrLabels
        .map((l) => `<th title="${escapeHtml(label(l))}">${escapeHtml(truncate(label(l), 12))}</th>`)
        .join('')}</tr>`;
      const matrixRows = corrMatrix
        .map((row, i) => {
          const cells = row
            .map((v) => `<td style="background:${colorFor(v)}">${v == null ? '—' : v.toFixed(2)}</td>`)
            .join('');
          return `<tr><th title="${escapeHtml(label(corrLabels[i]))}">${escapeHtml(truncate(label(corrLabels[i]), 12))}</th>${cells}</tr>`;
        })
        .join('');
      const redundant = corr.redundant_pairs || [];
      const redNote = redundant.length
        ? `<div class="batch-redundant" style="font-size:12px;color:var(--accent-amber,#e8a838);margin-top:6px;">⚠ Redundant (|ρ| ≥ 0.7): ${redundant
            .map((p) => `${escapeHtml(label(p.a))}↔${escapeHtml(label(p.b))} (${Number(p.rho).toFixed(2)})`)
            .join(', ')}</div>`
        : `<div class="batch-redundant ok" style="font-size:12px;color:var(--text-secondary);margin-top:6px;">No highly-correlated pairs — all |ρ| &lt; 0.7.</div>`;
      corrHtml = `<div class="batch-corr" style="margin-top:16px;"><div class="cmp-chart-title">Return correlation</div><table class="corr-table">${headerRow}${matrixRows}</table>${redNote}</div>`;
    }

    container.innerHTML = `
      <div class="cmp-header">
        <div>
          <div class="cmp-title">Batch comparison</div>
          <div class="cmp-subtitle">${nTotal} alpha${nTotal === 1 ? '' : 's'} · ${nOk} ok · in-sample only${elapsed}</div>
        </div>
      </div>
      <table class="cmp-table batch-table">
        <thead><tr>${headCells}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>
      ${corrHtml}
      <div class="cmp-explainer">
        Ranked in-sample metrics for each alpha, run in parallel. Click a row to
        load its full backtest. Correlation is of daily returns — pairs with
        |ρ| ≥ 0.7 add little diversification.
      </div>
    `;

    container.querySelectorAll('.batch-sort').forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.dataset.key;
        if (sortKey === key) sortDir = -sortDir;
        else {
          sortKey = key;
          sortDir = -1;
        }
        draw();
      });
    });

    container.querySelectorAll('.batch-row:not(.batch-row-error)').forEach((tr) => {
      tr.addEventListener('click', () => {
        if (callbacks.onSelectAlpha) callbacks.onSelectAlpha(tr.dataset.id);
      });
    });
  }

  return {
    render,
    clear,
    setStatus,
    setOnSelectAlpha: (cb) => {
      callbacks.onSelectAlpha = cb;
    },
  };
}
