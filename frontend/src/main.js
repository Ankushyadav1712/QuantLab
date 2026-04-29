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
      <div>
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

// ---------- Operator Docs modal ----------

document.getElementById('op-docs-btn').addEventListener('click', openOperatorDocs);

let cachedOperators = null;
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
  showModal(
    'Operators & Data Fields',
    `
      <div style="margin-bottom:12px;">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-secondary); margin-bottom:6px;">Data fields</div>
        <div style="font-family:'JetBrains Mono', monospace; font-size:12px;">
          ${(payload.data_fields || [])
            .map(f => `<span style="color:var(--accent-green); margin-right:10px;">${f}</span>`)
            .join('')}
        </div>
      </div>
      <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-secondary); margin-bottom:6px;">Operators</div>
      <div class="operator-list">
        ${payload.operators
          .map(op => `
            <div class="operator-item">
              <span class="name">${op.name}</span><span class="args">${op.args}</span>
              <span class="desc">${op.description}</span>
            </div>
          `)
          .join('')}
      </div>
    `,
    [{ label: 'Close', primary: true, action: closeModal }]
  );
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
  dashboard.setMetrics(resp.metrics);
  charts.setData(resp.timeseries, resp.monthly_returns);
}
