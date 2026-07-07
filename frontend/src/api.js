// `VITE_API_URL` is inlined at build time (Vite reads VITE_* env vars).
// Local dev defaults to localhost:8000; Docker / production builds override.
export const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Optional bearer token for write endpoints (POST /api/alphas, DELETE …).
// Resolution order: localStorage override → Vite env var → none.  If the
// backend has QUANTLAB_API_TOKEN set, write endpoints respond 401 unless this
// matches.  Read endpoints work without a token.
const TOKEN_STORAGE_KEY = 'quantlab.api_token';
function getApiToken() {
  try {
    const stored = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (stored) return stored;
  } catch (_) { /* no localStorage (private mode) */ }
  return import.meta.env.VITE_API_TOKEN || '';
}
export function setApiToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_STORAGE_KEY, token);
    else localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch (_) {}
}

async function request(method, path, body, { auth = false } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (auth) {
    const token = getApiToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }
  const opts = { method, headers };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(BASE_URL + path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || JSON.stringify(j);
    } catch (_) {}
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  simulate: (expression, settings = {}, n_trials = 1) =>
    request('POST', '/api/simulate', { expression, settings, n_trials }),
  validateExpression: (expression) =>
    request('POST', '/api/validate', { expression }),
  saveAlpha: (name, expression, notes = '', settings = {}, tags = []) =>
    request('POST', '/api/alphas', { name, expression, notes, settings, tags }, { auth: true }),
  listAlphas: (tag = null) =>
    request('GET', tag ? `/api/alphas?tag=${encodeURIComponent(tag)}` : '/api/alphas'),
  getAlpha: (id) => request('GET', `/api/alphas/${id}`),
  getAlphaVersions: (id) => request('GET', `/api/alphas/${id}/versions`),
  rollbackAlpha: (id, version) =>
    request('POST', `/api/alphas/${id}/rollback/${version}`, undefined, { auth: true }),
  deleteAlpha: (id) => request('DELETE', `/api/alphas/${id}`, undefined, { auth: true }),
  multiBlend: (alphas, settings = {}, weight_method = 'equal', target_vol = null, orthogonalize = false) =>
    request('POST', '/api/alphas/multi-blend', {
      alphas, settings, weight_method, target_vol, orthogonalize,
    }),
  compare: (expressions, settings = {}) =>
    request('POST', '/api/compare', { expressions, settings }),
  batchSimulate: (alphas, settings = {}) =>
    request('POST', '/api/batch_simulate', { alphas, settings }),
  sweep: (expression, settings = {}, max_combinations = 50) =>
    request('POST', '/api/sweep', { expression, settings, max_combinations }),
  getCorrelations: (ids) =>
    request('POST', '/api/alphas/correlations', { alpha_ids: ids }),
  validateCorrelation: (local, external, sharpe_tolerance_pct = 3.0) =>
    request('POST', '/api/validate_correlation', { local, external, sharpe_tolerance_pct }),
  getParetoAlphas: () => request('GET', '/api/alphas/pareto'),
  getDiversificationCurve: (samples = 20) =>
    request('GET', `/api/alphas/diversification_curve?samples=${samples}`),
  getOperators: () => request('GET', '/api/operators'),
  getLoadingStatus: () => request('GET', '/api/loading_status'),
  getUniverse: () => request('GET', '/api/universe'),
  getUniverses: () => request('GET', '/api/universes'),
  getExamples: () => request('GET', '/api/examples'),
};
