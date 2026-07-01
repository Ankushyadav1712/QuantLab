// QuantLab — app entry point.

import './styles/index.css';
import { api } from './api.js';
import { createEditor } from './components/editor.js';
import { createDashboard } from './components/dashboard.js';
import { createCharts } from './components/charts.js';
import { createPortfolioAnalysis } from './components/portfolio_analysis.js';
import { createResearchCharts } from './components/research_charts.js';
import { createSidebar } from './components/sidebar.js';
import { openTearsheet } from './components/tearsheet.js';
import { createComparison } from './components/comparison.js';
import { createCorrelation } from './components/correlation.js';
import { createSweepResults } from './components/sweep.js';
import { toast } from './ui/toast.js';

// ---------- Layout ----------

const root = document.getElementById('app');
root.innerHTML = `
  <div class="app">
    <header class="header">
      <button type="button" id="sidebar-toggle" class="icon-btn sidebar-toggle" aria-label="Toggle sidebar">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="3" y1="6" x2="21" y2="6"></line>
          <line x1="3" y1="12" x2="21" y2="12"></line>
          <line x1="3" y1="18" x2="21" y2="18"></line>
        </svg>
      </button>
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
        <button type="button" id="export-tearsheet-btn" title="Print or save the current backtest as a PDF tearsheet" disabled>Export PDF</button>
      </div>
    </header>
    <div id="sidebar-scrim" class="sidebar-scrim" aria-hidden="true"></div>
    <aside id="sidebar"></aside>
    <main class="main">
      <section id="editor"></section>
      <section id="sweep"></section>
      <section id="dashboard"></section>
      <section id="charts"></section>
      <section id="comparison"></section>
      <section id="correlation"></section>
      <section id="research-charts"></section>
      <section id="portfolio-analysis"></section>
    </main>
  </div>
`;

const editor = createEditor(document.getElementById('editor'));
const dashboard = createDashboard(document.getElementById('dashboard'));
const charts = createCharts(document.getElementById('charts'));
const sidebar = createSidebar(document.getElementById('sidebar'));
const correlation = createCorrelation(document.getElementById('correlation'));
const comparison = createComparison(document.getElementById('comparison'));
const sweep = createSweepResults(document.getElementById('sweep'));
const portfolioAnalysis = createPortfolioAnalysis(
  document.getElementById('portfolio-analysis')
);
const researchCharts = createResearchCharts(document.getElementById('research-charts'));

// ---------- State ----------

let lastResponse = null;

// Session trial tracking — feeds into the Deflated Sharpe Ratio.  We count the
// number of *distinct* expressions the user has run, since re-running the same
// expression doesn't add a new selection-bias trial.  Persisted in localStorage
// so the count survives reloads but resets if the user clears site data.
const TRIALS_KEY = 'quantlab.tried_expressions';
function loadTriedExpressions() {
  try {
    const raw = localStorage.getItem(TRIALS_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw);
    return new Set(Array.isArray(arr) ? arr : []);
  } catch (_) { return new Set(); }
}
function recordExpressionTrial(expression) {
  const tried = loadTriedExpressions();
  tried.add(expression);
  try { localStorage.setItem(TRIALS_KEY, JSON.stringify([...tried])); } catch (_) {}
  return tried.size;
}

// ---------- Run / save ----------

editor.setOnRun(async () => {
  const expression = editor.getExpression();
  if (!expression) {
    toast('Enter an expression first.', 'warning');
    return;
  }
  const settings = editor.getSettings();

  // Sweep mode short-circuits the regular simulate path: run the parameter
  // grid and render the heatmap, but don't touch the dashboard / charts /
  // save button (those reflect a single backtest).
  if (editor.getIsSweep()) {
    try {
      const resp = await api.sweep(expression, settings, 50);
      sweep.render(resp);
      document.getElementById('sweep').scrollIntoView({ behavior: 'smooth' });
      toast(`Sweep complete · ${resp.n_combinations} backtests`, 'success', { duration: 2500 });
    } catch (e) {
      toast(e.message, 'error', { title: 'Sweep failed', duration: 8000 });
    }
    return;
  }

  const n_trials = recordExpressionTrial(expression);
  try {
    const resp = await api.simulate(expression, settings, n_trials);
    renderResponse(resp);
    editor.setSaveEnabled(true);
    sweep.clear();
    toast('Backtest complete', 'success', { duration: 2500 });
  } catch (e) {
    toast(e.message, 'error', { title: 'Simulation failed', duration: 8000 });
  }
});

editor.setOnSave(() => {
  if (!lastResponse) return;
  openSaveModal(lastResponse);
});

editor.setOnLoadExample(() => {
  openExamplePicker();
});

// ---------- Example picker modal ----------
// Lazy-loads /api/examples on first open and caches in-module.  Renders a
// grouped list (one section per category) with name + description + tags.
// Selecting an example pastes the expression into the editor and applies
// the recommended settings.
let _exampleCache = null;
async function openExamplePicker() {
  let examples = _exampleCache;
  if (examples == null) {
    try {
      const resp = await api.getExamples();
      examples = resp.examples || [];
      _exampleCache = examples;
    } catch (e) {
      toast(e.message, 'error', { title: 'Could not load examples' });
      return;
    }
  }

  // Group by category, preserving first-appearance order
  const byCat = new Map();
  for (const ex of examples) {
    if (!byCat.has(ex.category)) byCat.set(ex.category, []);
    byCat.get(ex.category).push(ex);
  }

  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  const modal = document.createElement('div');
  modal.className = 'modal modal-examples glass';

  const sections = [...byCat.entries()].map(([cat, items]) => {
    const cards = items.map((ex) => `
      <div class="example-card" data-example-id="${ex.id}" tabindex="0">
        <div class="example-card-head">
          <strong class="example-name">${escapeHtml(ex.name)}</strong>
          <code class="example-expr">${escapeHtml(ex.expression)}</code>
        </div>
        <p class="example-desc">${escapeHtml(ex.description)}</p>
        <div class="example-tags">
          ${(ex.teaches || []).map((t) => `<span class="example-tag">${escapeHtml(t)}</span>`).join('')}
        </div>
      </div>
    `).join('');
    return `
      <div class="example-section">
        <div class="example-cat">${escapeHtml(cat)}</div>
        <div class="example-grid">${cards}</div>
      </div>
    `;
  }).join('');

  modal.innerHTML = `
    <div class="modal-head">
      <h2>Load an example alpha</h2>
      <button type="button" class="modal-close" aria-label="Close">×</button>
    </div>
    <p class="modal-sub">
      Click an example to paste it into the editor with the recommended settings.
      You can edit before running.
    </p>
    <div class="example-sections">${sections}</div>
  `;
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  const close = () => overlay.remove();
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  modal.querySelector('.modal-close').addEventListener('click', close);
  document.addEventListener('keydown', function onKey(e) {
    if (e.key === 'Escape') {
      document.removeEventListener('keydown', onKey);
      close();
    }
  });

  // Click handler delegated on the cards
  modal.querySelectorAll('.example-card').forEach((card) => {
    const apply = () => {
      const id = card.dataset.exampleId;
      const ex = examples.find((e) => e.id === id);
      if (!ex) return;
      editor.setExpression(ex.expression);
      editor.applySettings(ex.recommended_settings || {});
      toast(`Loaded: ${ex.name}`, 'success', { duration: 2500 });
      close();
    };
    card.addEventListener('click', apply);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        apply();
      }
    });
  });
}

function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

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
    toast(e.message, 'error', { title: 'Load failed' });
  }
});

sidebar.setOnBlend(async (items) => {
  if (!items || items.length === 0) return;

  // Ask the user which weight method to use.  "equal" preserves the existing
  // user-supplied weights; everything else ignores them and uses the optimizer.
  const choice = await openWeightMethodPicker();
  if (!choice) return;  // user cancelled

  const settings = editor.getSettings();
  try {
    const resp = await api.multiBlend(
      items.map(({ expression, weight }) => ({ expression, weight })),
      settings,
      choice.method,
      choice.target_vol,
    );
    renderResponse(resp);
    // Use the *computed* weights from the response so the editor's preview
    // matches what was actually run.
    const computed = (resp.settings?.alphas) || items.map((i) => ({ expression: i.expression, weight: i.weight }));
    editor.setExpression(
      computed.map((i) => `${i.weight.toFixed(3)} * (${i.expression})`).join(' + ')
    );
    editor.setSaveEnabled(true);
    const methodLabel = resp.settings?.weight_method || choice.method;
    toast(`Blended ${items.length} alphas (${methodLabel})`, 'success', { duration: 2500 });
  } catch (e) {
    toast(e.message, 'error', { title: 'Multi-blend failed' });
  }
});

// Small modal for picking the weighting method before running multi-blend.
// Returns {method, target_vol} or null if cancelled.
function openWeightMethodPicker() {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    const modal = document.createElement('div');
    modal.className = 'modal modal-confirm glass';
    modal.innerHTML = `
      <h2>Blend method</h2>
      <p class="confirm-message">How should we weight the selected alphas?</p>
      <div class="weight-method-options">
        <label><input type="radio" name="wm" value="equal" checked /> <strong>Equal</strong> — use the weights from the sidebar inputs</label>
        <label><input type="radio" name="wm" value="inverse_variance" /> <strong>Inverse-variance</strong> — robust, ignores correlations</label>
        <label><input type="radio" name="wm" value="mv_optimal" /> <strong>Mean-variance optimal</strong> — closed-form Σ⁻¹μ; may produce shorts</label>
        <label><input type="radio" name="wm" value="risk_parity" /> <strong>Risk parity</strong> — equal vol contribution per alpha</label>
      </div>
      <label class="weight-target-vol" style="display:none;">
        Target annualized vol (optional, mv_optimal only):
        <input type="number" step="0.01" min="0" placeholder="e.g. 0.15" data-role="target-vol" />
      </label>
      <div class="modal-actions">
        <button type="button" data-role="cancel">Cancel</button>
        <button type="button" data-role="confirm" class="primary">Run blend</button>
      </div>
    `;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    const targetVolBlock = modal.querySelector('.weight-target-vol');
    const targetVolInput = modal.querySelector('[data-role="target-vol"]');
    modal.querySelectorAll('input[name="wm"]').forEach((r) => {
      r.addEventListener('change', () => {
        const v = modal.querySelector('input[name="wm"]:checked')?.value;
        targetVolBlock.style.display = v === 'mv_optimal' ? '' : 'none';
      });
    });

    const close = (value) => {
      overlay.remove();
      document.removeEventListener('keydown', onKey);
      resolve(value);
    };
    const onKey = (e) => { if (e.key === 'Escape') close(null); };
    document.addEventListener('keydown', onKey);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) close(null); });
    modal.querySelector('[data-role="cancel"]').addEventListener('click', () => close(null));
    modal.querySelector('[data-role="confirm"]').addEventListener('click', () => {
      const method = modal.querySelector('input[name="wm"]:checked')?.value || 'equal';
      const tvRaw = targetVolInput.value.trim();
      const target_vol = method === 'mv_optimal' && tvRaw !== '' ? Number(tvRaw) : null;
      close({ method, target_vol });
    });
  });
}

sidebar.setOnCompare(async (items) => {
  if (!items || items.length < 2 || items.length > 4) return;
  const settings = editor.getSettings();
  try {
    const resp = await api.compare(
      items.map((i) => i.expression),
      settings,
    );
    comparison.render(resp);
    document.getElementById('comparison').scrollIntoView({ behavior: 'smooth' });
    toast(`Compared ${items.length} alphas`, 'success', { duration: 2500 });
  } catch (e) {
    toast(e.message, 'error', { title: 'Compare failed' });
  }
});

sidebar.setOnCorrelate(async (ids) => {
  if (!ids || ids.length < 2) return;
  try {
    const data = await api.getCorrelations(ids);
    correlation.show(data);
    document.getElementById('correlation').scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    toast(e.message, 'error', { title: 'Correlations failed' });
  }
});

sidebar.refresh();
portfolioAnalysis.refresh();
// Refresh the Pareto chart whenever the saved-alphas list mutates.
sidebar.setOnDelete(() => portfolioAnalysis.refresh());

// Selecting an alpha on the Pareto chart loads it into the editor — same
// shortcut as clicking it in the sidebar list.
portfolioAnalysis.setOnSelectAlpha(async (id) => {
  try {
    const alpha = await api.getAlpha(id);
    if (alpha?.expression) {
      editor.setExpression(alpha.expression);
      toast(`Loaded '${alpha.name}' into editor`, 'success', { duration: 2000 });
    }
  } catch (e) {
    toast(e.message, 'error');
  }
});

// ---------- Mobile sidebar drawer ----------
// Below ~720px the sidebar is off-canvas; this hamburger + scrim toggle it.
const sidebarEl = document.getElementById('sidebar');
const sidebarToggleBtn = document.getElementById('sidebar-toggle');
const sidebarScrim = document.getElementById('sidebar-scrim');

function openSidebar() {
  sidebarEl.classList.add('open');
  sidebarScrim.classList.add('open');
  sidebarToggleBtn.setAttribute('aria-expanded', 'true');
}
function closeSidebar() {
  sidebarEl.classList.remove('open');
  sidebarScrim.classList.remove('open');
  sidebarToggleBtn.setAttribute('aria-expanded', 'false');
}
sidebarToggleBtn.addEventListener('click', () => {
  if (sidebarEl.classList.contains('open')) closeSidebar();
  else openSidebar();
});
sidebarScrim.addEventListener('click', closeSidebar);
sidebarToggleBtn.setAttribute('aria-expanded', 'false');

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

// ---------- CLI guide modal ----------
//
// A walkthrough for the `alphatest` CLI — what to type, what to expect, in
// what order.  Visible in every build (dev + production) now that the repo
// is public; visitors can clone + install per Step 0 in the modal and then
// follow the rest of the steps.  Previously dev-only because the repo was
// private and a Vercel visitor would have hit a 404 on `git clone`.
{
  const headerActions = document.querySelector('.header-actions');
  const exportBtn = document.getElementById('export-tearsheet-btn');
  if (headerActions && exportBtn) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.id = 'cli-guide-btn';
    btn.title = 'alphatest CLI — clone the repo and run backtests from a terminal';
    btn.textContent = 'CLI Guide';
    btn.addEventListener('click', openCliGuide);
    headerActions.insertBefore(btn, exportBtn);
  }
}

function openCliGuide() {
  // Code blocks rendered as <pre> so multi-line shell input stays formatted;
  // ESC and overlay-click close as with every other modal.
  showModal(
    'alphatest CLI — smoke-test guide',
    `
      <div class="cli-guide">
        <p class="cli-guide-lead">
          The same engine this web app uses, exposed as a four-subcommand CLI
          (<code>run</code>, <code>shuffle</code>, <code>list</code>,
          <code>verify</code>). Useful for CI checks, cron jobs, and quick
          one-off backtests without a running server.
        </p>

        <div class="cli-guide-step">
          <span class="cli-guide-num">0</span>
          <div>
            <strong>Clone the repo + install Python deps</strong> — one-time
            setup. The CLI runs locally against the same yfinance cache the
            web app uses; the first <code>./alphatest run</code> takes ~30 s
            to populate the parquet cache, then ~1–2 s thereafter.
            <pre class="cli-guide-cmd">git clone https://github.com/Ankushyadav1712/QuantLab.git
cd QuantLab
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt</pre>
            <span class="cli-guide-hint">Requires Python 3.11+ and git. The
              <code>./alphatest</code> wrapper auto-finds the venv —
              no need to activate it manually.</span>
          </div>
        </div>

        <div class="cli-guide-step">
          <span class="cli-guide-num">1</span>
          <div>
            <strong>Open a terminal at the repo root.</strong>
            <pre class="cli-guide-cmd">cd QuantLab</pre>
          </div>
        </div>

        <div class="cli-guide-step">
          <span class="cli-guide-num">2</span>
          <div>
            <strong>Top-level help</strong> — proves the wrapper resolves Python
            + dispatcher loads.
            <pre class="cli-guide-cmd">./alphatest --help</pre>
            <span class="cli-guide-hint">Expect: lists <code>run</code>,
              <code>shuffle</code>, <code>list</code>, <code>verify</code>.</span>
          </div>
        </div>

        <div class="cli-guide-step">
          <span class="cli-guide-num">3</span>
          <div>
            <strong>List saved alphas</strong> — proves the SQLite reader
            (instant, no engine).
            <pre class="cli-guide-cmd">./alphatest list --limit 5
./alphatest list --order sharpe --limit 10</pre>
            <span class="cli-guide-hint">Expect: a fixed-width table; the
              Code/Data columns are <code>—</code> for alphas saved before
              the provenance feature shipped.</span>
          </div>
        </div>

        <div class="cli-guide-step">
          <span class="cli-guide-num">4</span>
          <div>
            <strong>Run a backtest</strong> — proves the engine path (~5–10 s
            on a warm cache, ~30 s cold).
            <pre class="cli-guide-cmd">./alphatest run "rank(close) - rank(open)"
./alphatest run "rank(close)" --neutralization sector --oos</pre>
            <span class="cli-guide-hint">Same Sharpe / annual return / drawdown
              numbers the web UI shows for the identical expression.</span>
          </div>
        </div>

        <div class="cli-guide-step">
          <span class="cli-guide-num">5</span>
          <div>
            <strong>Verify a saved alpha</strong> — re-runs and diffs the three
            signatures.  Pick any id from step&nbsp;3.
            <pre class="cli-guide-cmd">./alphatest verify 84</pre>
            <span class="cli-guide-hint">Diagnostic prints whether the headline
              reproduced and, if not, names the likely cause (code edit, data
              refresh, or legacy alpha with no stored signatures).</span>
          </div>
        </div>

        <div class="cli-guide-step">
          <span class="cli-guide-num">6</span>
          <div>
            <strong>Exit codes</strong> — what makes the CLI useful in a CI
            pipeline.
            <pre class="cli-guide-cmd">./alphatest run "rank(close)" > /dev/null;            echo "ok:   $?"   # → 0
./alphatest run "totally_undefined_field" > /dev/null; echo "err:  $?"   # → 1
./alphatest verify 999999 > /dev/null;                 echo "miss: $?"   # → 2</pre>
          </div>
        </div>

        <div class="cli-guide-step cli-guide-optional">
          <span class="cli-guide-num">7</span>
          <div>
            <strong>(Optional) Shuffle leakage test</strong> — 1–3 min for a
            50-shuffle run; only invoke when you actually want a leakage
            verdict.
            <pre class="cli-guide-cmd">./alphatest shuffle "rank(close)" --iters 25</pre>
            <span class="cli-guide-hint">Exits 0 only when the verdict is
              <code>real-signal</code>; <code>rank(close)</code> is a known
              noise baseline so expect non-zero.</span>
          </div>
        </div>

        <div class="cli-guide-tip">
          <strong>Run the CLI tests</strong> from the same terminal:
          <pre class="cli-guide-cmd">backend/.venv/bin/python -m pytest backend/tests/test_cli.py -v</pre>
          Expect 19 passing in ~1 s.  The full backend suite
          (<code>backend/tests/</code>) takes 7–8 min.
        </div>
      </div>
    `,
    [{ label: 'Close', primary: true, action: closeModal }]
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

document.getElementById('export-tearsheet-btn').addEventListener('click', () => {
  if (!lastResponse) {
    toast('Run a backtest first, then export.', 'warning');
    return;
  }
  openTearsheet(lastResponse);
});

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
      toast(e.message, 'error', { title: 'Could not load operators' });
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
            toast('Name is required.', 'warning');
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
            portfolioAnalysis.refresh();
            editor.setSaveEnabled(false);
            toast(`Saved as "${name}"`, 'success');
          } catch (e) {
            toast(e.message, 'error', { title: 'Save failed' });
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
  // Enable the Export button now that a backtest result is available.
  const exportBtn = document.getElementById('export-tearsheet-btn');
  if (exportBtn) exportBtn.disabled = false;
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
    walk_forward: resp.walk_forward || null,
  });
  charts.setData(isTs, resp.monthly_returns, {
    oos_timeseries: resp.oos_timeseries || null,
  });
  // Tier 4 research charts (IC time series + decay + quintile + risk decomp)
  researchCharts.render(isMetrics, resp.factor_decomposition || null);
}
