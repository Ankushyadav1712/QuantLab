import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api, BASE_URL, setApiToken } from '../src/api.js';

describe('api', () => {
  let fetchMock;

  beforeEach(() => {
    fetchMock = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ ok: true }),
      })
    );
    globalThis.fetch = fetchMock;
    // setApiToken(null) clears the token storage we touch, no need for full clear()
    setApiToken(null);
  });

  afterEach(() => {
    setApiToken(null);
  });

  it('hits BASE_URL with the right method + path', async () => {
    await api.listAlphas();
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/api/alphas`);
    expect(opts.method).toBe('GET');
    expect(opts.headers['Content-Type']).toBe('application/json');
  });

  it('serializes body as JSON for POST', async () => {
    await api.simulate('rank(close)', { neutralization: 'market' });
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/api/simulate`);
    expect(opts.method).toBe('POST');
    expect(JSON.parse(opts.body)).toEqual({
      expression: 'rank(close)',
      settings: { neutralization: 'market' },
      n_trials: 1,
    });
  });

  it('does NOT send Authorization on read endpoints, even when token set', async () => {
    setApiToken('secret-xyz');
    await api.listAlphas();
    const [, opts] = fetchMock.mock.calls[0];
    expect(opts.headers.Authorization).toBeUndefined();
  });

  it('sends Bearer token on write endpoints when token is set', async () => {
    setApiToken('secret-xyz');
    await api.saveAlpha('test', 'rank(close)', '');
    const [, opts] = fetchMock.mock.calls[0];
    expect(opts.headers.Authorization).toBe('Bearer secret-xyz');
  });

  it('throws with descriptive message when API returns non-ok', async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      json: () => Promise.resolve({ detail: 'token required' }),
    });
    await expect(api.saveAlpha('x', 'rank(close)')).rejects.toThrow(
      /HTTP 401: token required/
    );
  });

  it('multiBlend sends weight_method + orthogonalize in the body', async () => {
    await api.multiBlend([{ expression: 'rank(close)', weight: 1 }], {}, 'ic_weighted', null, true);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/api/alphas/multi-blend`);
    expect(opts.method).toBe('POST');
    const body = JSON.parse(opts.body);
    expect(body.weight_method).toBe('ic_weighted');
    expect(body.orthogonalize).toBe(true);
  });

  it('multiBlend defaults to equal weighting and orthogonalize=false', async () => {
    await api.multiBlend([{ expression: 'rank(close)', weight: 1 }]);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.weight_method).toBe('equal');
    expect(body.orthogonalize).toBe(false);
  });

  it('saveAlpha includes tags in the body', async () => {
    setApiToken('secret-xyz');
    await api.saveAlpha('m', 'rank(close)', 'notes', {}, ['momentum', 'wip']);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.tags).toEqual(['momentum', 'wip']);
  });

  it('listAlphas appends the tag query param when given', async () => {
    await api.listAlphas('momentum');
    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE_URL}/api/alphas?tag=momentum`);
  });

  it('getLoadingStatus is a GET to /api/loading_status with no auth', async () => {
    setApiToken('secret-xyz');
    await api.getLoadingStatus();
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/api/loading_status`);
    expect(opts.method).toBe('GET');
    expect(opts.headers.Authorization).toBeUndefined();
  });

  it('getAlphaVersions + rollbackAlpha hit the right paths', async () => {
    await api.getAlphaVersions(5);
    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE_URL}/api/alphas/5/versions`);
    await api.rollbackAlpha(5, 2);
    const [url, opts] = fetchMock.mock.calls[1];
    expect(url).toBe(`${BASE_URL}/api/alphas/5/rollback/2`);
    expect(opts.method).toBe('POST');
  });

  it('validateCorrelation posts local + external + tolerance', async () => {
    const local = { dates: ['2020-01-01'], returns: [0.01] };
    const external = { dates: ['2020-01-01'], returns: [0.02] };
    await api.validateCorrelation(local, external, 5);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/api/validate_correlation`);
    expect(opts.method).toBe('POST');
    const body = JSON.parse(opts.body);
    expect(body.local).toEqual(local);
    expect(body.external).toEqual(external);
    expect(body.sharpe_tolerance_pct).toBe(5);
  });

  it('validateCorrelation defaults the tolerance to 3', async () => {
    await api.validateCorrelation({ dates: [], returns: [] }, { dates: [], returns: [] });
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.sharpe_tolerance_pct).toBe(3);
  });

  it('getBrainPreset hits /api/presets/brain', async () => {
    await api.getBrainPreset();
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/api/presets/brain`);
    expect(opts.method).toBe('GET');
  });
});
