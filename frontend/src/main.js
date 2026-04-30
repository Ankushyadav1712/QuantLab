// QuantLab — app entry point.

import './styles/index.css';
import { api } from './api.js';
import { createEditor } from './components/editor.js';
import { createDashboard } from './components/dashboard.js';
import { createCharts } from './components/charts.js';
import { createSidebar } from './components/sidebar.js';
import { createCorrelation } from './components/correlation.js';

// ---------- Layout ----------

const root = document.getElementById('app');
root.innerHTML = `
  <div class="app">
    <header class="header">
      <h1><span class="logo-dot"></span> QuantLab</h1>
      <div class="header-actions">
        <button type="button" id="help-btn" class="icon-btn" title="Show welcome guide" aria-label="Show welcome guide">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
          </svg>
        </button>
        <button type="button" id="op-docs-btn">Operator Docs</button>
      </div>
    </header>
    <aside id="sidebar"></aside>
    <main class="main">
      <section id="editor"></section>
      <section id="dashboard"></section>
      <section id="charts"></section>
      <section id="correlation"></section>
    </main>
  </div>
`;

const editor = createEditor(document.getElementById('editor'));
const dashboard = createDashboard(document.getElementById('dashboard'));
const charts = createCharts(document.getElementById('charts'));
const sidebar = createSidebar(document.getElementById('sidebar'));
const correlation = createCorrelation(document.getElementById('correlation'));

// ---------- State ----------

let lastResponse = null;

// ---------- Run / save ----------

editor.setOnRun(async () => {
  const expression = editor.getExpression();
  if (!expression) {
    alert('Enter an expression first.');
    return;
  }
  const settings = editor.getSettings();
  try {
    const resp = await api.simulate(expression, settings);
    renderResponse(resp);
    editor.setSaveEnabled(true);
  } catch (e) {
    alert('Simulation failed: ' + e.message);
  }
});

editor.setOnSave(() => {
  if (!lastResponse) return;
  openSaveModal(lastResponse);
});

// ---------- Sidebar wiring ----------

sidebar.setOnLoad(async (id) => {
  try {
    const record = await api.getAlpha(id);
    editor.setExpression(record.expression);
    if (record.result) {
      renderResponse(record.result);
      editor.setSaveEnabled(false);
    }
  } catch (e) {
    alert('Load failed: ' + e.message);
  }
});

sidebar.setOnBlend(async (items) => {
  if (!items || items.length === 0) return;
  const settings = editor.getSettings();
  try {
    const resp = await api.multiBlend(
      items.map(({ expression, weight }) => ({ expression, weight })),
      settings
    );
    renderResponse(resp);
    editor.setExpression(
      items.map((i) => `${i.weight.toFixed(2)} * (${i.expression})`).join(' + ')
    );
    editor.setSaveEnabled(true);
  } catch (e) {
    alert('Multi-blend failed: ' + e.message);
  }
});

sidebar.setOnCorrelate(async (ids) => {
  if (!ids || ids.length < 2) return;
  try {
    const data = await api.getCorrelations(ids);
    correlation.show(data);
    document.getElementById('correlation').scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    alert('Correlations failed: ' + e.message);
  }
});

sidebar.refresh();

// ---------- First-visit welcome ----------

const WELCOME_KEY = 'quantlab.welcomed';
const SAMPLE_EXPRESSION = 'decay_linear(rank(momentum_20), 20)';

document.getElementById('help-btn').addEventListener('click', () => openWelcomeModal({ force: true }));

// Show on first visit (after a short delay so the rest of the UI paints first)
if (!localStorage.getItem(WELCOME_KEY)) {
  setTimeout(() => openWelcomeModal({ force: false }), 300);
}

function openWelcomeModal({ force }) {
  showModal(
    'Welcome to QuantLab',
    `
      <div class="welcome">
        <p class="welcome-tagline">A browser-based platform for quantitative alpha research on US&nbsp;equities.</p>

        <ol class="welcome-steps">
          <li>
            <span class="welcome-num">1</span>
            <div>
              <strong>Write a signal</strong> in the expression editor —
              try <code>rank(delta(close, 5))</code> or
              <code>decay_linear(rank(momentum_20), 20)</code>.
              <span class="welcome-hint">23 operators · 32 fields · click <em>Operator Docs</em> for the full reference.</span>
            </div>
          </li>
          <li>
            <span class="welcome-num">2</span>
            <div>
              <strong>Tweak the run.</strong> Open the <em>Settings</em> drawer to set
              date range, neutralization (none / market / sector), book size,
              transaction cost, and decay.
            </div>
          </li>
          <li>
            <span class="welcome-num">3</span>
            <div>
              <strong>Click <em>Run Backtest</em>.</strong>
              You'll get Sharpe, drawdown, fitness, monthly heatmap, and PnL
              curves in under a second.
            </div>
          </li>
          <li>
            <span class="welcome-num">4</span>
            <div>
              <strong>Trust but verify.</strong> Every backtest is automatically
              split <strong>70&nbsp;% in-sample / 30&nbsp;% out-of-sample</strong>
              so you can tell if the alpha generalizes — or just curve-fits the
              first three quarters of your data.
            </div>
          </li>
          <li>
            <span class="welcome-num">5</span>
            <div>
              <strong>Save what works</strong> in the sidebar — then blend
              alphas or compare correlations to find uncorrelated edge.
            </div>
          </li>
        </ol>

        <div class="welcome-tip">
          <strong>New to alpha research?</strong> Click
          <em>Try a sample alpha</em> below — it loads a working long-only
          momentum signal that delivers Sharpe&nbsp;~&nbsp;1.0 over 2019–2024.
        </div>
      </div>
    `,
    [
      {
        label: force ? 'Close' : "Skip — I'll explore",
        action: () => {
          localStorage.setItem(WELCOME_KEY, '1');
          closeModal();
        },
      },
      {
        label: 'Try a sample alpha →',
        primary: true,
        action: async () => {
          localStorage.setItem(WELCOME_KEY, '1');
          closeModal();
          await runSampleAlpha();
        },
      },
    ]
  );
}

async function runSampleAlpha() {
  // Set the editor to a known-good expression, then drive the Run button so
  // the spinner / disabled state behave exactly like a real click.
  editor.setExpression(SAMPLE_EXPRESSION);

  // Flip neutralization to "none" — the long-only setting that makes momentum
  // actually clear costs on this universe.  We mutate the segmented control
  // directly because there's no programmatic API for it.
  const neutSeg = document.querySelector('[data-setting="neutralization"]');
  if (neutSeg) {
    neutSeg.querySelectorAll('button').forEach((b) =>
      b.classList.toggle('active', b.dataset.value === 'none')
    );
    neutSeg.dataset.value = 'none';
    // Bubble an input event so editor.js's modified-count listener re-runs.
    neutSeg.dispatchEvent(new Event('input', { bubbles: true }));
  }

  document.querySelector('[data-role="run"]')?.click();
}

// ---------- Operator Docs modal ----------

document.getElementById('op-docs-btn').addEventListener('click', openOperatorDocs);

let cachedOperators = null;

const CATEGORY_LABELS = {
  price: 'Price (OHLCV + derived)',
  price_structure: 'Price structure (candle / shadows / gap)',
  return_variants: 'Return variants',
  volume_liquidity: 'Volume & liquidity',
  volatility_risk: 'Volatility & risk',
  momentum_relative: 'Momentum & relative',
};

const OP_CATEGORIES = [
  { name: 'Time-series', match: (op) => op.name.startsWith('ts_') || ['delta', 'delay', 'decay_linear'].includes(op.name) },
  { name: 'Cross-sectional', match: (op) => ['rank', 'zscore', 'demean', 'scale', 'normalize'].includes(op.name) },
  { name: 'Arithmetic / element-wise', match: (op) => ['abs', 'log', 'sign', 'power', 'max', 'min', 'if_else'].includes(op.name) },
];

function groupOperators(ops) {
  const groups = OP_CATEGORIES.map((c) => ({ name: c.name, items: [] }));
  const seen = new Set();
  for (const op of ops) {
    for (let i = 0; i < OP_CATEGORIES.length; i++) {
      if (OP_CATEGORIES[i].match(op)) { groups[i].items.push(op); seen.add(op.name); break; }
    }
  }
  const leftover = ops.filter((o) => !seen.has(o.name));
  if (leftover.length) groups.push({ name: 'Other', items: leftover });
  return groups.filter((g) => g.items.length);
}

function groupFields(fields) {
  const order = ['price', 'price_structure', 'return_variants', 'volume_liquidity', 'volatility_risk', 'momentum_relative'];
  const map = new Map();
  for (const f of fields) {
    const key = f.category || 'other';
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(f);
  }
  const sortedKeys = [...map.keys()].sort((a, b) => {
    const ai = order.indexOf(a), bi = order.indexOf(b);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });
  return sortedKeys.map((k) => ({ name: CATEGORY_LABELS[k] || k, items: map.get(k) }));
}

function escapeAttr(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function openOperatorDocs() {
  let payload = cachedOperators;
  if (!payload) {
    try {
      payload = await api.getOperators();
      cachedOperators = payload;
    } catch (e) {
      alert('Could not load operators: ' + e.message);
      return;
    }
  }

  const operators = payload.operators || [];
  // Prefer the rich `fields` array; fall back to the flat list if the backend is older.
  const fields = (payload.fields && payload.fields.length)
    ? payload.fields
    : (payload.data_fields || []).map((name) => ({ name, category: 'price', description: '' }));

  const opsHtml = groupOperators(operators).map((g) => `
    <div class="docs-section">
      <div class="docs-section-title">${g.name}</div>
      <div class="operator-list">
        ${g.items.map((op) => `
          <div class="operator-item">
            <span class="name">${op.name}</span><span class="args">${op.args}</span>
            <span class="desc">${op.description}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');

  const fieldsHtml = groupFields(fields).map((g) => `
    <div class="docs-section">
      <div class="docs-section-title">${g.name}</div>
      <div class="operator-list">
        ${g.items.map((f) => `
          <div class="operator-item" title="${escapeAttr(f.description)}">
            <span class="name" style="color: var(--accent-green);">${f.name}</span>
            ${f.description ? `<span class="desc">${f.description}</span>` : ''}
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');

  showModal(
    'Operators & Data Fields',
    `
      <div class="docs-tabs">
        <button type="button" class="docs-tab active" data-tab="operators">Operators (${operators.length})</button>
        <button type="button" class="docs-tab" data-tab="fields">Data fields (${fields.length})</button>
      </div>
      <div class="docs-pane" data-pane="operators">${opsHtml}</div>
      <div class="docs-pane" data-pane="fields" style="display:none;">${fieldsHtml}</div>
    `,
    [{ label: 'Close', primary: true, action: closeModal }]
  );

  const tabsEl = document.querySelector('.docs-tabs');
  if (tabsEl) {
    tabsEl.addEventListener('click', (e) => {
      const btn = e.target.closest('.docs-tab');
      if (!btn) return;
      tabsEl.querySelectorAll('.docs-tab').forEach((t) => t.classList.toggle('active', t === btn));
      const target = btn.dataset.tab;
      document.querySelectorAll('.docs-pane').forEach((p) => {
        p.style.display = p.dataset.pane === target ? '' : 'none';
      });
    });
  }
}

// ---------- Save modal ----------

function openSaveModal(response) {
  showModal(
    'Save Alpha',
    `
      <div class="row">
        <label>Name</label>
        <input type="text" data-role="name" placeholder="e.g. mom_5d_vol_scaled" />
      </div>
      <div class="row">
        <label>Notes</label>
        <textarea data-role="notes" rows="3" placeholder="Optional"></textarea>
      </div>
      <div class="row" style="font-size:12px; color:var(--text-secondary);">
        Will be saved with the latest backtest results.
      </div>
    `,
    [
      { label: 'Cancel', action: closeModal },
      {
        label: 'Save',
        primary: true,
        action: async (modal) => {
          const name = modal.querySelector('[data-role="name"]').value.trim();
          const notes = modal.querySelector('[data-role="notes"]').value.trim();
          if (!name) {
            alert('Name required.');
            return;
          }
          try {
            await api.saveAlpha(
              name,
              response.expression,
              notes,
              response.settings || {}
            );
            closeModal();
            await sidebar.refresh();
            editor.setSaveEnabled(false);
          } catch (e) {
            alert('Save failed: ' + e.message);
          }
        },
      },
    ]
  );
}

// ---------- Modal helper ----------

let activeModal = null;
function showModal(title, bodyHtml, actions = []) {
  closeModal();
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.innerHTML = `
    <h2>${title}</h2>
    <div data-role="modal-body">${bodyHtml}</div>
    <div class="modal-actions" data-role="modal-actions"></div>
  `;
  overlay.appendChild(modal);
  const actionsBar = modal.querySelector('[data-role="modal-actions"]');
  for (const a of actions) {
    const b = document.createElement('button');
    b.type = 'button';
    if (a.primary) b.className = 'primary';
    b.textContent = a.label;
    b.addEventListener('click', () => a.action(modal));
    actionsBar.appendChild(b);
  }
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeModal();
  });
  document.body.appendChild(overlay);
  activeModal = overlay;
}
function closeModal() {
  if (activeModal) {
    activeModal.remove();
    activeModal = null;
  }
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

// ---------- Render simulate response ----------

function renderResponse(resp) {
  lastResponse = resp;
  // New shape carries IS/OOS pairs.  Old saved-alpha records (pre-OOS) used
  // flat `metrics`/`timeseries` keys — fall back to those so loading legacy
  // alphas from the sidebar still renders the dashboard.
  const isMetrics = resp.is_metrics || resp.metrics || null;
  const isTs = resp.is_timeseries || resp.timeseries || null;
  dashboard.setMetrics(isMetrics, {
    oos_metrics: resp.oos_metrics || null,
    overfitting_analysis: resp.overfitting_analysis || null,
    data_quality: resp.data_quality || null,
    factor_decomposition: resp.factor_decomposition || null,
  });
  charts.setData(isTs, resp.monthly_returns, {
    oos_timeseries: resp.oos_timeseries || null,
  });
}
