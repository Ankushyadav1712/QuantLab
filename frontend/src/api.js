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
  saveAlpha: (name, expression, notes = '', settings = {}) =>
    request('POST', '/api/alphas', { name, expression, notes, settings }, { auth: true }),
  listAlphas: () => request('GET', '/api/alphas'),
  getAlpha: (id) => request('GET', `/api/alphas/${id}`),
  deleteAlpha: (id) => request('DELETE', `/api/alphas/${id}`, undefined, { auth: true }),
  multiBlend: (alphas, settings = {}) =>
    request('POST', '/api/alphas/multi-blend', { alphas, settings }),
  getCorrelations: (ids) =>
    request('POST', '/api/alphas/correlations', { alpha_ids: ids }),
  getOperators: () => request('GET', '/api/operators'),
  getUniverse: () => request('GET', '/api/universe'),
};
