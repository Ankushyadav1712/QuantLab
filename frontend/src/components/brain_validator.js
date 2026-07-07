// Brain Validator — upload an external daily returns/PnL CSV (e.g. a WorldQuant
// Brain export) and check how faithfully the local backtester reproduces it:
// day-by-day return correlation + annualised-Sharpe gap against a tolerance.
// Answers "is my local sandbox a trustworthy proxy for the competition server?"

import { api } from '../api.js';
import { toast } from '../ui/toast.js';

// Parse a two-column CSV into { dates, returns }. Date in the first column,
// value (daily return OR daily PnL — both scale-invariant) in the second.
// Skips a header row, blank lines, and unparseable rows. Accepts comma,
// semicolon, or tab delimiters.
export function parseReturnsCsv(text) {
  const dates = [];
  const returns = [];
  for (const raw of String(text ?? '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const parts = line.split(/[,;\t]/).map((p) => p.trim());
    if (parts.length < 2) continue;
    const d = parts[0];
    const v = Number(parts[1]);
    if (!d || !Number.isFinite(v)) continue; // header row / junk
    dates.push(d);
    returns.push(v);
  }
  return { dates, returns };
}

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function createBrainValidator(container) {
  container.classList.add('brain-validator-section', 'glass');
  container.style.display = 'none'; // hidden until a local backtest exists
  container.innerHTML = `
    <div class="exp-header">
      <div class="exp-title">Brain Validator</div>
      <div class="exp-subtitle">Correlate your last local backtest with an external daily returns/PnL series (e.g. a WorldQuant Brain export) to confirm the local sandbox is a faithful proxy.</div>
    </div>
    <div class="bv-controls" style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:10px 0;">
      <input type="file" data-role="csv" accept=".csv,.txt" />
      <label style="font-size:12px; color:var(--text-secondary,#8b949e);">Sharpe tolerance
        <input type="number" data-role="tol" value="3" min="0" max="100" step="0.5" style="width:56px; margin-left:4px;" /> %
      </label>
      <button type="button" data-role="run" disabled>Validate</button>
    </div>
    <div class="bv-hint" style="font-size:11px; color:var(--text-secondary,#8b949e);">CSV: two columns — date, daily return or PnL (a header row is fine).</div>
    <div data-role="result"></div>
  `;

  const fileInput = container.querySelector('[data-role="csv"]');
  const tolInput = container.querySelector('[data-role="tol"]');
  const runBtn = container.querySelector('[data-role="run"]');
  const resultEl = container.querySelector('[data-role="result"]');

  let localSeries = null; // { dates, returns } from the last backtest

  function updateRunState() {
    runBtn.disabled = !(localSeries && fileInput.files && fileInput.files.length);
  }

  // Called by main.js after every backtest with the IS timeseries block.
  function setLocalResult(isTs) {
    if (
      isTs &&
      Array.isArray(isTs.dates) &&
      Array.isArray(isTs.daily_returns) &&
      isTs.dates.length
    ) {
      localSeries = { dates: isTs.dates, returns: isTs.daily_returns };
      container.style.display = '';
      updateRunState();
    }
  }

  fileInput.addEventListener('change', updateRunState);

  runBtn.addEventListener('click', async () => {
    if (!localSeries) {
      toast('Run a backtest first.', 'warning');
      return;
    }
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      toast('Choose a CSV file first.', 'warning');
      return;
    }
    let external;
    try {
      external = parseReturnsCsv(await file.text());
    } catch (e) {
      toast(e.message, 'error', { title: 'CSV parse failed' });
      return;
    }
    if (!external.dates.length) {
      toast('No (date, value) rows found in the CSV.', 'warning');
      return;
    }
    const tol = Number(tolInput.value) || 3.0;
    runBtn.disabled = true;
    const label = runBtn.textContent;
    runBtn.textContent = 'Validating…';
    try {
      renderResult(await api.validateCorrelation(localSeries, external, tol));
    } catch (e) {
      toast(e.message, 'error', { title: 'Validation failed' });
    } finally {
      runBtn.textContent = label;
      updateRunState();
    }
  });

  function renderResult(res) {
    if (!res || !res.ok) {
      resultEl.innerHTML = `<div class="bv-error" style="color:var(--text-secondary,#8b949e); font-size:13px; margin-top:8px;">${escapeHtml((res && res.error) || 'Could not compare the two series.')}</div>`;
      return;
    }
    const color = res.verdict === 'pass' ? '#3fb950' : res.verdict === 'fail' ? '#f85149' : '#d29922';
    const label = res.verdict === 'pass' ? 'PASS' : res.verdict === 'fail' ? 'FAIL' : 'REVIEW';
    const num = (x, d = 2) => (x == null ? '—' : Number(x).toFixed(d));
    const pct = (x) => (x == null ? '—' : Number(x).toFixed(1) + '%');
    const tolBadge =
      res.within_tolerance == null
        ? '<span style="opacity:.6">n/a</span>'
        : res.within_tolerance
          ? '<span style="color:#3fb950">within tolerance</span>'
          : '<span style="color:#f85149">out of tolerance</span>';
    resultEl.innerHTML = `
      <div class="bv-verdict" style="display:flex; align-items:center; gap:10px; margin:12px 0 6px;">
        <span style="font-weight:700; letter-spacing:.05em; padding:2px 10px; border-radius:6px; color:#fff; background:${color};">${label}</span>
        <span style="font-size:12px; color:var(--text-secondary,#8b949e);">${escapeHtml(res.verdict_detail || '')}</span>
      </div>
      <div class="bv-grid" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin-top:8px;">
        <div class="metric-card glass"><div class="label">Return correlation</div><div class="value">${num(res.correlation, 3)}</div><div class="sub">${escapeHtml(res.correlation_quality || '')} · ${res.n_overlap} days</div></div>
        <div class="metric-card glass"><div class="label">Local Sharpe</div><div class="value">${num(res.local_sharpe)}</div></div>
        <div class="metric-card glass"><div class="label">External Sharpe</div><div class="value">${num(res.external_sharpe)}</div></div>
        <div class="metric-card glass"><div class="label">Sharpe gap</div><div class="value">${pct(res.sharpe_diff_pct)}</div><div class="sub">${tolBadge} (±${num(res.sharpe_tolerance_pct, 1)}%)</div></div>
      </div>
    `;
  }

  return { setLocalResult };
}
