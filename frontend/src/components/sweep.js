// Sweep results — heatmap (1D bar table / 2D color grid) + sortable table.
// 1 dimension → table with horizontal Sharpe bars
// 2 dimensions → 2D heatmap, color = Sharpe
// 3+ dimensions → sortable table only

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function fmtNum(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return (v * 100).toFixed(1) + '%';
}

// Sharpe → diverging red/green color, clipped at ±2 for visual saturation.
// Uses the dashboard's existing accent palette so the heatmap reads
// consistently with the rest of the UI.
function sharpeColor(s) {
  if (s == null || Number.isNaN(s)) return 'rgba(110, 118, 129, 0.18)';
  const clipped = Math.max(-2, Math.min(2, s));
  if (clipped >= 0) {
    const a = 0.15 + (clipped / 2) * 0.55;
    return `rgba(57, 211, 83, ${a.toFixed(3)})`;
  }
  const a = 0.15 + (Math.abs(clipped) / 2) * 0.55;
  return `rgba(255, 107, 107, ${a.toFixed(3)})`;
}

export function createSweepResults(container) {
  container.classList.add('sweep-section');
  container.innerHTML = '';

  function clear() { container.innerHTML = ''; }

  function render(payload) {
    const dims = payload.dimensions || [];
    const cells = payload.cells || [];

    const validSharpes = cells.map((c) => c.sharpe).filter((v) => v != null);
    const best = validSharpes.length ? Math.max(...validSharpes) : null;
    const worst = validSharpes.length ? Math.min(...validSharpes) : null;
    const mean = validSharpes.length
      ? validSharpes.reduce((a, b) => a + b, 0) / validSharpes.length
      : null;

    const header = `
      <div class="sweep-header">
        <div>
          <div class="sweep-title">Parameter sweep — ${cells.length} backtests</div>
          <div class="sweep-subtitle">${escapeHtml(payload.expression)}</div>
        </div>
        <div class="sweep-stats">
          <span><span class="sweep-stat-label">Best Sharpe</span> <strong class="pos">${fmtNum(best)}</strong></span>
          <span><span class="sweep-stat-label">Worst</span> <strong class="${worst != null && worst < 0 ? 'neg' : 'pos'}">${fmtNum(worst)}</strong></span>
          <span><span class="sweep-stat-label">Mean</span> <strong>${fmtNum(mean)}</strong></span>
        </div>
      </div>
    `;

    let body;
    if (dims.length === 1) body = render1D(dims[0], cells);
    else if (dims.length === 2) body = render2D(dims[0], dims[1], cells);
    else body = renderTable(dims, cells);

    const tableSection = dims.length <= 2 ? `
      <details class="sweep-table-details">
        <summary>Show all ${cells.length} as a sortable table</summary>
        ${renderTable(dims, cells)}
      </details>
    ` : '';

    container.innerHTML = header + body + tableSection + `
      <div class="sweep-explainer">
        Each cell is an in-sample backtest with all other settings held constant.
        Pick a winning cell, paste the resolved expression into the editor, and
        run a full IS/OOS backtest on it for proper validation.
      </div>
    `;
  }

  function render1D(dim, cells) {
    const rows = cells.map((c) => {
      const v = c.params[dim.token];
      const sh = c.sharpe;
      const errCell = c.error
        ? `<td colspan="3" class="sweep-cell-error">${escapeHtml(c.error)}</td>`
        : `
          <td class="sweep-cell-bar">
            <div class="sweep-bar" style="background:${sharpeColor(sh)};">
              <span class="sweep-bar-num">${fmtNum(sh)}</span>
            </div>
          </td>
          <td class="sweep-cell-num">${fmtPct(c.annual_return)}</td>
          <td class="sweep-cell-num neg">${fmtPct(c.max_drawdown)}</td>
        `;
      return `
        <tr>
          <td class="sweep-cell-label">${escapeHtml(dim.token)} = <strong>${v}</strong></td>
          ${errCell}
        </tr>
      `;
    }).join('');
    return `
      <table class="sweep-1d">
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Sharpe</th>
            <th>Ann. Return</th>
            <th>Max DD</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function render2D(rowDim, colDim, cells) {
    const lookup = new Map();
    for (const c of cells) {
      lookup.set(`${c.params[rowDim.token]}__${c.params[colDim.token]}`, c);
    }

    const head = `
      <tr>
        <th class="sweep-corner">${escapeHtml(rowDim.token)} \\ ${escapeHtml(colDim.token)}</th>
        ${colDim.values.map((v) => `<th>${v}</th>`).join('')}
      </tr>
    `;

    const body = rowDim.values.map((rv) => {
      const row = colDim.values.map((cv) => {
        const c = lookup.get(`${rv}__${cv}`);
        const sh = c?.sharpe;
        const tooltip = c?.error
          ? `${c.expression}\n${c.error}`
          : c
            ? `${c.expression}\nSharpe ${fmtNum(sh)} · Return ${fmtPct(c.annual_return)} · MDD ${fmtPct(c.max_drawdown)}`
            : 'no data';
        return `<td class="sweep-heat-cell" style="background:${sharpeColor(sh)};" title="${escapeHtml(tooltip)}">${fmtNum(sh)}</td>`;
      }).join('');
      return `<tr><th class="sweep-row-label">${rv}</th>${row}</tr>`;
    }).join('');

    return `
      <div class="sweep-2d-wrap">
        <table class="sweep-2d">
          <thead>${head}</thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    `;
  }

  function renderTable(dims, cells) {
    // Sort by Sharpe descending so the winner is at the top
    const sorted = [...cells].sort(
      (a, b) => (b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity)
    );
    const dimHeaders = dims.map((d) => `<th>${escapeHtml(d.token)}</th>`).join('');
    const rows = sorted.map((c, idx) => {
      const dimCells = dims
        .map((d) => `<td class="sweep-cell-num">${c.params[d.token]}</td>`)
        .join('');
      const errorCell = c.error
        ? `<td colspan="4" class="sweep-cell-error">${escapeHtml(c.error)}</td>`
        : `
          <td class="sweep-cell-num"><strong>${fmtNum(c.sharpe)}</strong></td>
          <td class="sweep-cell-num">${fmtPct(c.annual_return)}</td>
          <td class="sweep-cell-num neg">${fmtPct(c.max_drawdown)}</td>
          <td class="sweep-cell-num">${fmtNum(c.fitness, 3)}</td>
        `;
      return `
        <tr class="${idx === 0 && !c.error ? 'sweep-best-row' : ''}">
          <td class="sweep-rank">${idx + 1}</td>
          ${dimCells}
          ${errorCell}
        </tr>
      `;
    }).join('');
    return `
      <table class="sweep-table">
        <thead>
          <tr>
            <th>#</th>
            ${dimHeaders}
            <th>Sharpe</th>
            <th>Ann. Return</th>
            <th>Max DD</th>
            <th>Fitness</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  return { render, clear };
}
