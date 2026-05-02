// Editor — textarea with overlaid syntax highlighting, debounced validation,
// quick-insert buttons, settings panel, and a Run Backtest button.

import { api } from '../api.js';

const OPERATOR_NAMES = new Set([
  'ts_mean', 'ts_std', 'ts_min', 'ts_max', 'ts_sum', 'ts_rank',
  'delta', 'delay', 'decay_linear', 'ts_corr', 'ts_cov',
  'rank', 'zscore', 'demean', 'scale', 'normalize',
  'abs', 'log', 'sign', 'power', 'max', 'min', 'if_else',
]);
// Stays in sync with backend FIELDS in main.py (kept hardcoded so syntax
// highlighting works on first paint without an extra fetch). 'range' is the
// user-facing alias for the canonical 'range_'.
const FIELD_NAMES = new Set([
  'open', 'high', 'low', 'close', 'volume', 'returns', 'vwap',
  'median_price', 'weighted_close', 'range_', 'range', 'body',
  'upper_shadow', 'lower_shadow', 'gap',
  'log_returns', 'abs_returns', 'intraday_return',
  'overnight_return', 'signed_volume',
  'dollar_volume', 'adv20', 'volume_ratio', 'amihud',
  'true_range', 'atr', 'realized_vol', 'skewness', 'kurtosis',
  'momentum_5', 'momentum_20', 'close_to_high_252', 'high_low_ratio',
]);

const QUICK = [
  { label: 'rank', insert: 'rank()' },
  { label: 'delta', insert: 'delta(close, 5)' },
  { label: 'ts_mean', insert: 'ts_mean(close, 20)' },
  { label: 'ts_std', insert: 'ts_std(returns, 20)' },
  { label: 'ts_corr', insert: 'ts_corr(close, volume, 10)' },
  { label: 'zscore', insert: 'zscore()' },
];

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function highlight(text) {
  // Escape, then replace tokens with spans. Don't break on partial-word matches.
  const escaped = escapeHtml(text);
  return escaped.replace(
    /(\d+\.\d+|\d+)|([A-Za-z_][A-Za-z0-9_]*)/g,
    (match, num, ident) => {
      if (num) return `<span class="hl-num">${match}</span>`;
      if (ident) {
        if (OPERATOR_NAMES.has(ident)) return `<span class="hl-op">${match}</span>`;
        if (FIELD_NAMES.has(ident)) return `<span class="hl-field">${match}</span>`;
      }
      return match;
    }
  );
}

export function createEditor(container, { initialExpression = '-rank(delta(close, 5)) * ts_std(returns, 20)' } = {}) {
  container.classList.add('editor-section', 'glass');
  container.innerHTML = `
    <div class="editor-wrap">
      <pre class="editor-highlight" aria-hidden="true"></pre>
      <textarea class="editor-input" spellcheck="false" autocapitalize="off" autocomplete="off"></textarea>
    </div>
    <div class="editor-status" data-role="status"></div>

    <div class="editor-quickrow" data-role="quick"></div>

    <div class="editor-actions">
      <button type="button" class="primary" data-role="run">Run Backtest</button>
      <button type="button" class="settings-toggle" data-role="settings-toggle">
        <svg class="settings-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="3"/>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
        <span>Settings</span>
        <span class="settings-modified-pill" data-role="modified-pill" hidden></span>
        <svg class="settings-caret" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>
      <button type="button" data-role="save" class="save-action" disabled>Save Alpha</button>
    </div>

    <div class="settings-panel" data-role="settings">
      <!-- Period -->
      <div class="settings-block" data-block="period">
        <div class="settings-block-head">
          <span class="settings-block-label">Period</span>
          <div class="settings-presets" data-role="date-presets">
            <button type="button" data-preset="full">Full</button>
            <button type="button" data-preset="5y">5y</button>
            <button type="button" data-preset="3y">3y</button>
            <button type="button" data-preset="1y">1y</button>
          </div>
        </div>
        <div class="settings-period-row">
          <input type="date" data-setting="start_date" value="2019-01-01" />
          <span class="settings-arrow">→</span>
          <input type="date" data-setting="end_date" value="2024-12-31" />
        </div>
      </div>

      <!-- Strategy: neutralization (segmented) + decay (slider) -->
      <div class="settings-grid">
        <div class="settings-block" data-block="neutralization">
          <div class="settings-block-label">Neutralization</div>
          <div class="settings-segmented" data-setting="neutralization" data-value="market">
            <button type="button" data-value="none">None</button>
            <button type="button" data-value="market" class="active">Market</button>
            <button type="button" data-value="sector">Sector</button>
          </div>
        </div>

        <div class="settings-block" data-block="decay">
          <div class="settings-block-label">
            Decay window <span class="settings-value" data-role="decay-value">off</span>
          </div>
          <input type="range" class="settings-slider" min="0" max="40" step="1" value="0" data-setting="decay" />
        </div>

        <div class="settings-block" data-block="booksize">
          <div class="settings-block-label">
            Book size <span class="settings-value" data-role="booksize-value">$20M</span>
          </div>
          <input type="range" class="settings-slider" min="1" max="100" step="1" value="20" data-setting="booksize_millions" />
        </div>

        <div class="settings-block" data-block="truncation">
          <div class="settings-block-label">
            Truncation cap <span class="settings-value" data-role="truncation-value">5.0%</span>
          </div>
          <input type="range" class="settings-slider" min="1" max="20" step="0.5" value="5" data-setting="truncation_pct" />
        </div>

        <div class="settings-block" data-block="cost" style="grid-column: span 2;">
          <div class="settings-block-label">
            Transaction cost <span class="settings-value" data-role="cost-value">5.0 bps</span>
          </div>
          <input type="range" class="settings-slider" min="0" max="20" step="0.5" value="5" data-setting="transaction_cost_bps" />
        </div>
      </div>

      <!-- Cost model -->
      <div class="settings-block" data-block="cost-model">
        <div class="settings-block-label">Cost model</div>
        <div class="settings-segmented" data-setting="cost_model" data-value="flat">
          <button type="button" data-value="flat" class="active">Flat (bps × turnover)</button>
          <button type="button" data-value="sqrt_impact">√-impact (Almgren–Chriss)</button>
        </div>
      </div>

      <!-- Validation block -->
      <div class="settings-block" data-block="validation">
        <label class="settings-toggle-row">
          <input type="checkbox" data-setting="run_oos" checked />
          <span>Run out-of-sample validation (70 / 30 split)</span>
        </label>
        <div class="settings-oos-bar" data-role="oos-bar">
          <div class="settings-oos-bar-is"></div>
          <div class="settings-oos-bar-oos"></div>
        </div>
        <label class="settings-toggle-row" style="margin-top:6px;">
          <input type="checkbox" data-setting="run_walk_forward" />
          <span>Run walk-forward (rolling 252/63 windows · slower)</span>
        </label>
        <label class="settings-toggle-row" style="margin-top:6px;">
          <input type="checkbox" data-setting="t1_execution" />
          <span>T+1 execution lag (signal at close → trade next open)</span>
        </label>
        <label class="settings-toggle-row" style="margin-top:6px;">
          <input type="checkbox" data-setting="point_in_time_universe" />
          <span>Point-in-time universe (gate names by index inclusion date)</span>
        </label>
      </div>

      <div class="settings-footer">
        <button type="button" class="ghost" data-role="reset">Reset to defaults</button>
      </div>
    </div>
  `;

  const textarea = container.querySelector('.editor-input');
  const highlightEl = container.querySelector('.editor-highlight');
  const statusEl = container.querySelector('[data-role="status"]');
  const quickRow = container.querySelector('[data-role="quick"]');
  const settingsToggle = container.querySelector('[data-role="settings-toggle"]');
  const settingsPanel = container.querySelector('[data-role="settings"]');
  const modifiedPill = container.querySelector('[data-role="modified-pill"]');
  const runBtn = container.querySelector('[data-role="run"]');
  const saveBtn = container.querySelector('[data-role="save"]');

  // Settings refs
  const startInput = container.querySelector('[data-setting="start_date"]');
  const endInput = container.querySelector('[data-setting="end_date"]');
  const neutSegmented = container.querySelector('[data-setting="neutralization"]');
  const decaySlider = container.querySelector('[data-setting="decay"]');
  const booksizeSlider = container.querySelector('[data-setting="booksize_millions"]');
  const truncationSlider = container.querySelector('[data-setting="truncation_pct"]');
  const costSlider = container.querySelector('[data-setting="transaction_cost_bps"]');
  const costModelSegmented = container.querySelector('[data-setting="cost_model"]');
  const oosCheckbox = container.querySelector('[data-setting="run_oos"]');
  const wfCheckbox = container.querySelector('[data-setting="run_walk_forward"]');
  const t1Checkbox = container.querySelector('[data-setting="t1_execution"]');
  const pitCheckbox = container.querySelector('[data-setting="point_in_time_universe"]');
  const oosBar = container.querySelector('[data-role="oos-bar"]');

  const decayValueEl = container.querySelector('[data-role="decay-value"]');
  const booksizeValueEl = container.querySelector('[data-role="booksize-value"]');
  const truncationValueEl = container.querySelector('[data-role="truncation-value"]');
  const costValueEl = container.querySelector('[data-role="cost-value"]');

  const DEFAULTS = {
    start_date: '2019-01-01',
    end_date: '2024-12-31',
    neutralization: 'market',
    decay: 0,
    booksize_millions: 20,
    truncation_pct: 5,
    transaction_cost_bps: 5,
    run_oos: true,
    cost_model: 'flat',
    run_walk_forward: false,
    t1_execution: false,
    point_in_time_universe: false,
  };

  textarea.value = initialExpression;
  syncHighlight();

  function syncHighlight() {
    let text = textarea.value;
    if (text.endsWith('\n')) text += ' ';
    highlightEl.innerHTML = highlight(text);
  }

  textarea.addEventListener('input', () => {
    syncHighlight();
    debouncedValidate();
  });
  textarea.addEventListener('scroll', () => {
    highlightEl.scrollTop = textarea.scrollTop;
    highlightEl.scrollLeft = textarea.scrollLeft;
  });

  // Quick insert buttons
  for (const item of QUICK) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = item.label;
    b.addEventListener('click', () => insertAtCursor(item.insert));
    quickRow.appendChild(b);
  }

  function insertAtCursor(text) {
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const before = textarea.value.slice(0, start);
    const after = textarea.value.slice(end);
    textarea.value = before + text + after;
    const caret = before.length + text.length;
    textarea.focus();
    textarea.setSelectionRange(caret, caret);
    syncHighlight();
    debouncedValidate();
  }

  // Settings panel toggle (animated). The button keeps its own .open state so
  // CSS can animate the caret + active-state styling independently.
  settingsToggle.addEventListener('click', () => {
    const open = settingsPanel.classList.toggle('open');
    settingsToggle.classList.toggle('open', open);
    settingsToggle.setAttribute('aria-expanded', String(open));
  });
  settingsToggle.setAttribute('aria-expanded', 'false');

  // ---------- "N modified" indicator ----------

  function countModified() {
    let n = 0;
    if (startInput.value !== DEFAULTS.start_date) n++;
    if (endInput.value !== DEFAULTS.end_date) n++;
    if (neutSegmented.dataset.value !== DEFAULTS.neutralization) n++;
    if (Number(decaySlider.value) !== DEFAULTS.decay) n++;
    if (Number(booksizeSlider.value) !== DEFAULTS.booksize_millions) n++;
    if (Number(truncationSlider.value) !== DEFAULTS.truncation_pct) n++;
    if (Number(costSlider.value) !== DEFAULTS.transaction_cost_bps) n++;
    if (oosCheckbox.checked !== DEFAULTS.run_oos) n++;
    if (costModelSegmented.dataset.value !== DEFAULTS.cost_model) n++;
    if (wfCheckbox.checked !== DEFAULTS.run_walk_forward) n++;
    if (t1Checkbox.checked !== DEFAULTS.t1_execution) n++;
    if (pitCheckbox.checked !== DEFAULTS.point_in_time_universe) n++;
    return n;
  }
  function updateModifiedPill() {
    const n = countModified();
    if (n === 0) {
      modifiedPill.hidden = true;
      modifiedPill.textContent = '';
    } else {
      modifiedPill.hidden = false;
      modifiedPill.textContent = `${n} modified`;
    }
  }
  // Re-count on any input change, including segmented + reset.  Use bubbling
  // listeners on the panel so we don't have to wire up every control twice.
  settingsPanel.addEventListener('input', updateModifiedPill);
  settingsPanel.addEventListener('change', updateModifiedPill);
  settingsPanel.addEventListener('click', (e) => {
    // Segmented + presets + reset don't fire 'input', so re-check on click too
    if (e.target.closest('button')) {
      requestAnimationFrame(updateModifiedPill);
    }
  });
  updateModifiedPill();

  // ---------- Slider live-display helpers ----------

  function updateSliderFill(slider) {
    const min = Number(slider.min);
    const max = Number(slider.max);
    const v = Number(slider.value);
    const pct = ((v - min) / (max - min)) * 100;
    slider.style.setProperty('--fill', `${pct}%`);
  }

  function fmtMoney(m) {
    if (m >= 1000) return `$${(m / 1000).toFixed(1)}B`;
    return `$${m}M`;
  }

  function syncDecayLabel() {
    const v = Number(decaySlider.value);
    decayValueEl.textContent = v === 0 ? 'off' : `${v} days`;
    updateSliderFill(decaySlider);
  }
  function syncBooksizeLabel() {
    booksizeValueEl.textContent = fmtMoney(Number(booksizeSlider.value));
    updateSliderFill(booksizeSlider);
  }
  function syncTruncationLabel() {
    truncationValueEl.textContent = `${Number(truncationSlider.value).toFixed(1)}%`;
    updateSliderFill(truncationSlider);
  }
  function syncCostLabel() {
    costValueEl.textContent = `${Number(costSlider.value).toFixed(1)} bps`;
    updateSliderFill(costSlider);
  }
  function syncOosBar() {
    const enabled = oosCheckbox.checked;
    oosBar.classList.toggle('disabled', !enabled);
  }

  decaySlider.addEventListener('input', syncDecayLabel);
  booksizeSlider.addEventListener('input', syncBooksizeLabel);
  truncationSlider.addEventListener('input', syncTruncationLabel);
  costSlider.addEventListener('input', syncCostLabel);
  oosCheckbox.addEventListener('change', syncOosBar);

  // Initial paint of slider fills + labels
  syncDecayLabel();
  syncBooksizeLabel();
  syncTruncationLabel();
  syncCostLabel();
  syncOosBar();

  // ---------- Segmented controls (neutralization, cost model) ----------

  function wireSegmented(el) {
    el.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-value]');
      if (!btn) return;
      el.querySelectorAll('button').forEach((b) =>
        b.classList.toggle('active', b === btn)
      );
      el.dataset.value = btn.dataset.value;
    });
  }
  wireSegmented(neutSegmented);
  wireSegmented(costModelSegmented);

  // ---------- Date presets ----------

  const datePresets = container.querySelector('[data-role="date-presets"]');
  datePresets.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-preset]');
    if (!btn) return;
    const preset = btn.dataset.preset;
    const end = '2024-12-31'; // dataset's hard end
    let start;
    if (preset === 'full') start = '2019-01-01';
    else if (preset === '5y') start = '2020-01-01';
    else if (preset === '3y') start = '2022-01-01';
    else if (preset === '1y') start = '2024-01-01';
    if (start) {
      startInput.value = start;
      endInput.value = end;
      datePresets.querySelectorAll('button').forEach((b) =>
        b.classList.toggle('active', b === btn)
      );
    }
  });

  // ---------- Reset to defaults ----------

  const resetBtn = container.querySelector('[data-role="reset"]');
  resetBtn.addEventListener('click', () => {
    startInput.value = DEFAULTS.start_date;
    endInput.value = DEFAULTS.end_date;
    decaySlider.value = DEFAULTS.decay;
    booksizeSlider.value = DEFAULTS.booksize_millions;
    truncationSlider.value = DEFAULTS.truncation_pct;
    costSlider.value = DEFAULTS.transaction_cost_bps;
    oosCheckbox.checked = DEFAULTS.run_oos;
    wfCheckbox.checked = DEFAULTS.run_walk_forward;
    t1Checkbox.checked = DEFAULTS.t1_execution;
    pitCheckbox.checked = DEFAULTS.point_in_time_universe;

    // Segmented: reset to defaults
    neutSegmented.querySelectorAll('button').forEach((b) =>
      b.classList.toggle('active', b.dataset.value === DEFAULTS.neutralization)
    );
    neutSegmented.dataset.value = DEFAULTS.neutralization;
    costModelSegmented.querySelectorAll('button').forEach((b) =>
      b.classList.toggle('active', b.dataset.value === DEFAULTS.cost_model)
    );
    costModelSegmented.dataset.value = DEFAULTS.cost_model;

    // Clear preset highlights
    datePresets.querySelectorAll('button').forEach((b) => b.classList.remove('active'));

    syncDecayLabel();
    syncBooksizeLabel();
    syncTruncationLabel();
    syncCostLabel();
    syncOosBar();
  });

  // Validation (debounced 500ms)
  let validateTimer = null;
  let validateAbort = null;
  function debouncedValidate() {
    if (validateTimer) clearTimeout(validateTimer);
    validateTimer = setTimeout(runValidate, 500);
  }
  async function runValidate() {
    const expr = textarea.value.trim();
    if (!expr) {
      setStatus('idle', '', []);
      return;
    }
    setStatus('checking', 'checking…', []);
    if (validateAbort) validateAbort.abort();
    validateAbort = new AbortController();
    try {
      const res = await api.validateExpression(expr);
      const diags = res.diagnostics || [];
      if (res.valid) {
        const warns = diags.filter((d) => d.severity === 'warning');
        if (warns.length > 0) {
          setStatus('warn', `⚠ ${warns.length} warning${warns.length > 1 ? 's' : ''}`, diags);
        } else {
          setStatus('valid', '✓ valid expression', []);
        }
      } else {
        setStatus('invalid', '✗ ' + (res.error || 'invalid'), diags);
      }
    } catch (e) {
      setStatus('invalid', '✗ ' + e.message, []);
    }
  }

  function setStatus(kind, text, diagnostics) {
    statusEl.classList.remove('valid', 'invalid', 'warn');
    if (kind === 'valid') statusEl.classList.add('valid');
    else if (kind === 'invalid') statusEl.classList.add('invalid');
    else if (kind === 'warn') statusEl.classList.add('warn');

    if (!diagnostics || diagnostics.length === 0) {
      statusEl.textContent = text;
      return;
    }
    // Render diagnostics as a list under the headline
    statusEl.innerHTML = `
      <div class="status-headline">${escapeHtml(text)}</div>
      <ul class="status-diagnostics">
        ${diagnostics
          .map((d) => `
            <li class="diag-${d.severity}">
              <span class="diag-tag">${d.severity}</span>
              <span class="diag-op code">${escapeHtml(d.op || '')}</span>
              ${escapeHtml(d.message || '')}
            </li>
          `)
          .join('')}
      </ul>
    `;
  }

  function escapeHtml(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // Run button
  let onRun = null;
  let onSave = null;
  runBtn.addEventListener('click', async () => {
    if (!onRun) return;
    setRunning(true);
    try {
      await onRun();
    } finally {
      setRunning(false);
    }
  });
  saveBtn.addEventListener('click', () => {
    if (onSave) onSave();
  });

  function setRunning(running) {
    runBtn.disabled = running;
    runBtn.innerHTML = running
      ? '<span class="spinner"></span> Running…'
      : 'Run Backtest';
  }

  function getExpression() {
    return textarea.value.trim();
  }
  function setExpression(text) {
    textarea.value = text;
    syncHighlight();
    debouncedValidate();
  }
  function getSettings() {
    return {
      start_date: startInput.value,
      end_date: endInput.value,
      neutralization: neutSegmented.dataset.value,
      decay: Number(decaySlider.value),
      booksize: Number(booksizeSlider.value) * 1_000_000,
      truncation: Number(truncationSlider.value) / 100,
      transaction_cost_bps: Number(costSlider.value),
      run_oos: oosCheckbox.checked,
      cost_model: costModelSegmented.dataset.value,
      run_walk_forward: wfCheckbox.checked,
      execution_lag_days: t1Checkbox.checked ? 2 : 1,
      point_in_time_universe: pitCheckbox.checked,
    };
  }
  function setSaveEnabled(enabled) {
    saveBtn.disabled = !enabled;
  }

  // Initial validate
  debouncedValidate();

  return {
    getExpression,
    setExpression,
    getSettings,
    setOnRun: (cb) => { onRun = cb; },
    setOnSave: (cb) => { onSave = cb; },
    setSaveEnabled,
    setRunning,
  };
}
