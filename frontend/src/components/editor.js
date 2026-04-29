// Editor — textarea with overlaid syntax highlighting, debounced validation,
// quick-insert buttons, settings panel, and a Run Backtest button.

import { api } from '../api.js';

const OPERATOR_NAMES = new Set([
  'ts_mean', 'ts_std', 'ts_min', 'ts_max', 'ts_sum', 'ts_rank',
  'delta', 'delay', 'decay_linear', 'ts_corr', 'ts_cov',
  'rank', 'zscore', 'demean', 'scale', 'normalize',
  'abs', 'log', 'sign', 'power', 'max', 'min', 'if_else',
]);
const FIELD_NAMES = new Set(['open', 'high', 'low', 'close', 'volume', 'returns', 'vwap']);

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

    <button type="button" class="settings-toggle" data-role="settings-toggle">
      <span data-role="settings-caret">▸</span> Settings
    </button>
    <div class="settings-panel" data-role="settings">
      <div class="settings-row">
        <label>Start date</label>
        <input type="date" data-setting="start_date" value="2019-01-01" />
      </div>
      <div class="settings-row">
        <label>End date</label>
        <input type="date" data-setting="end_date" value="2024-12-31" />
      </div>
      <div class="settings-row">
        <label>Neutralization</label>
        <select data-setting="neutralization">
          <option value="none">none</option>
          <option value="market" selected>market</option>
          <option value="sector">sector</option>
        </select>
      </div>
      <div class="settings-row">
        <label>Tx cost (bps): <span data-role="cost-value">5</span></label>
        <input type="range" min="0" max="20" step="0.5" value="5" data-setting="transaction_cost_bps" />
      </div>
    </div>

    <div class="editor-actions">
      <button type="button" class="primary" data-role="run">Run Backtest</button>
      <button type="button" data-role="save" class="save-action" disabled>Save Alpha</button>
    </div>
  `;

  const textarea = container.querySelector('.editor-input');
  const highlightEl = container.querySelector('.editor-highlight');
  const statusEl = container.querySelector('[data-role="status"]');
  const quickRow = container.querySelector('[data-role="quick"]');
  const settingsToggle = container.querySelector('[data-role="settings-toggle"]');
  const settingsCaret = container.querySelector('[data-role="settings-caret"]');
  const settingsPanel = container.querySelector('[data-role="settings"]');
  const costValue = container.querySelector('[data-role="cost-value"]');
  const costSlider = container.querySelector('[data-setting="transaction_cost_bps"]');
  const runBtn = container.querySelector('[data-role="run"]');
  const saveBtn = container.querySelector('[data-role="save"]');

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

  // Settings panel toggle
  settingsToggle.addEventListener('click', () => {
    const open = settingsPanel.classList.toggle('open');
    settingsCaret.textContent = open ? '▾' : '▸';
  });
  costSlider.addEventListener('input', () => {
    costValue.textContent = costSlider.value;
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
      setStatus('idle', '');
      return;
    }
    setStatus('checking', 'checking…');
    if (validateAbort) validateAbort.abort();
    validateAbort = new AbortController();
    try {
      const res = await api.validateExpression(expr);
      if (res.valid) setStatus('valid', '✓ valid expression');
      else setStatus('invalid', '✗ ' + (res.error || 'invalid'));
    } catch (e) {
      setStatus('invalid', '✗ ' + e.message);
    }
  }

  function setStatus(kind, text) {
    statusEl.classList.remove('valid', 'invalid');
    if (kind === 'valid') statusEl.classList.add('valid');
    if (kind === 'invalid') statusEl.classList.add('invalid');
    statusEl.textContent = text;
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
    const inputs = container.querySelectorAll('[data-setting]');
    const out = {};
    inputs.forEach((el) => {
      const key = el.dataset.setting;
      let v = el.value;
      if (el.type === 'range' || el.type === 'number') v = Number(v);
      out[key] = v;
    });
    return out;
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
