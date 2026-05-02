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
});
