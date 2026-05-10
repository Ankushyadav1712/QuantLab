// Sidebar — saved alphas list with checkboxes, weight inputs, blend + correlate
// actions, delete buttons.

import { api } from '../api.js';
import { confirmDialog, toast } from '../ui/toast.js';

const TRUNC = 40;

export function createSidebar(container) {
  container.classList.add('sidebar');
  container.innerHTML = `
    <h3>Saved Alphas</h3>
    <div data-role="list" style="display:flex; flex-direction:column; gap:8px;">
      <div class="placeholder">No saved alphas yet.</div>
    </div>
    <div class="sidebar-actions">
      <button type="button" data-role="blend" disabled>Blend Selected</button>
      <button type="button" data-role="compare" disabled>Compare Selected</button>
      <button type="button" data-role="correlate" disabled>Show Correlations</button>
    </div>
  `;

  const listEl = container.querySelector('[data-role="list"]');
  const blendBtn = container.querySelector('[data-role="blend"]');
  const compareBtn = container.querySelector('[data-role="compare"]');
  const corrBtn = container.querySelector('[data-role="correlate"]');

  let alphas = [];
  let selected = new Set();
  let weights = {}; // id -> number
  const callbacks = {
    onLoad: null, onDelete: null, onBlend: null,
    onCompare: null, onCorrelate: null,
  };

  blendBtn.addEventListener('click', () => {
    if (callbacks.onBlend) callbacks.onBlend(getSelectedItems());
  });
  compareBtn.addEventListener('click', () => {
    if (callbacks.onCompare) callbacks.onCompare(getSelectedItems());
  });
  corrBtn.addEventListener('click', () => {
    if (callbacks.onCorrelate) callbacks.onCorrelate([...selected]);
  });

  function updateActionState() {
    const n = selected.size;
    blendBtn.disabled = n < 1;
    // /api/compare requires 2-4 expressions; the button reflects that range
    compareBtn.disabled = n < 2 || n > 4;
    corrBtn.disabled = n < 2;
  }

  function getSelectedItems() {
    return alphas
      .filter((a) => selected.has(a.id))
      .map((a) => ({
        id: a.id,
        expression: a.expression,
        weight: Number(weights[a.id] ?? 1),
        name: a.name,
      }));
  }

  async function refresh() {
    try {
      alphas = await api.listAlphas();
    } catch (e) {
      listEl.innerHTML = `<div class="placeholder">Failed to load: ${e.message}</div>`;
      return;
    }
    render();
  }

  function render() {
    listEl.innerHTML = '';
    if (!alphas || alphas.length === 0) {
      listEl.innerHTML = '<div class="placeholder">No saved alphas yet.</div>';
      updateActionState();
      return;
    }
    for (const a of alphas) {
      listEl.appendChild(renderItem(a));
    }
    updateActionState();
  }

  function renderItem(a) {
    const item = document.createElement('div');
    item.className = 'alpha-item';
    item.dataset.id = a.id;

    const sharpe = a.sharpe;
    const sharpeStr = sharpe == null ? '—' : sharpe.toFixed(2);
    const sharpeCls = sharpe == null ? '' : sharpe >= 0.5 ? 'good' : sharpe < 0 ? 'bad' : '';
    const exprText = (a.expression || '').length > TRUNC
      ? a.expression.slice(0, TRUNC) + '…'
      : a.expression;

    const checked = selected.has(a.id);
    const w = weights[a.id] ?? 1;

    item.innerHTML = `
      <div class="alpha-item-head">
        <input type="checkbox" data-role="check" ${checked ? 'checked' : ''} />
        <span class="alpha-item-name" title="${escapeHtml(a.name)}">${escapeHtml(a.name)}</span>
        <span class="sharpe-badge ${sharpeCls}">${sharpeStr}</span>
      </div>
      <div class="alpha-item-expr code" title="${escapeHtml(a.expression)}">${escapeHtml(exprText)}</div>
      <div class="alpha-item-row2">
        <label style="font-size:11px; color:var(--text-secondary);">w</label>
        <input type="number" class="alpha-item-weight" step="0.1" value="${w}" data-role="weight" />
        <div class="alpha-item-actions">
          <button type="button" class="ghost" data-role="load">load</button>
          <button type="button" class="ghost danger" data-role="delete">×</button>
        </div>
      </div>
    `;

    const check = item.querySelector('[data-role="check"]');
    const weightInput = item.querySelector('[data-role="weight"]');
    const loadBtn = item.querySelector('[data-role="load"]');
    const delBtn = item.querySelector('[data-role="delete"]');

    check.addEventListener('change', () => {
      if (check.checked) selected.add(a.id);
      else selected.delete(a.id);
      updateActionState();
    });
    weightInput.addEventListener('input', () => {
      weights[a.id] = Number(weightInput.value);
    });
    // Suppress checkbox/weight click-bubble triggering load
    [check, weightInput].forEach((el) => {
      el.addEventListener('click', (e) => e.stopPropagation());
    });
    loadBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (callbacks.onLoad) callbacks.onLoad(a.id);
    });
    delBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const ok = await confirmDialog({
        title: `Delete "${a.name}"?`,
        message: 'This removes the saved alpha and its backtest history. This cannot be undone.',
        confirmLabel: 'Delete',
        cancelLabel: 'Cancel',
        danger: true,
      });
      if (!ok) return;
      try {
        await api.deleteAlpha(a.id);
        selected.delete(a.id);
        delete weights[a.id];
        await refresh();
        if (callbacks.onDelete) callbacks.onDelete(a.id);
        toast(`Deleted "${a.name}"`, 'success', { duration: 2500 });
      } catch (err) {
        toast(err.message, 'error', { title: 'Delete failed' });
      }
    });

    // Click anywhere else on the card => load
    item.addEventListener('click', () => {
      if (callbacks.onLoad) callbacks.onLoad(a.id);
    });

    return item;
  }

  function escapeHtml(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  return {
    refresh,
    setOnLoad: (cb) => { callbacks.onLoad = cb; },
    setOnDelete: (cb) => { callbacks.onDelete = cb; },
    setOnBlend: (cb) => { callbacks.onBlend = cb; },
    setOnCompare: (cb) => { callbacks.onCompare = cb; },
    setOnCorrelate: (cb) => { callbacks.onCorrelate = cb; },
  };
}
