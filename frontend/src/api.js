export const BASE_URL = 'http://localhost:8000';

async function request(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
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
  simulate: (expression, settings = {}) =>
    request('POST', '/api/simulate', { expression, settings }),
  validateExpression: (expression) =>
    request('POST', '/api/validate', { expression }),
  saveAlpha: (name, expression, notes = '', settings = {}) =>
    request('POST', '/api/alphas', { name, expression, notes, settings }),
  listAlphas: () => request('GET', '/api/alphas'),
  getAlpha: (id) => request('GET', `/api/alphas/${id}`),
  deleteAlpha: (id) => request('DELETE', `/api/alphas/${id}`),
  multiBlend: (alphas, settings = {}) =>
    request('POST', '/api/alphas/multi-blend', { alphas, settings }),
  getCorrelations: (ids) =>
    request('POST', '/api/alphas/correlations', { alpha_ids: ids }),
  getOperators: () => request('GET', '/api/operators'),
  getUniverse: () => request('GET', '/api/universe'),
};
