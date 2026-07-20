import { beforeEach, describe, expect, it, vi } from 'vitest';
import { runSweepJob, BASE_URL } from '../src/api.js';

// Drive runSweepJob against a scripted fetch: POST /api/jobs/sweep returns a
// job_id, then GET /api/jobs/{id} returns a sequence of poll snapshots.
function scriptFetch({ submit, polls }) {
  let pollIdx = 0;
  return vi.fn((url, opts) => {
    const isSubmit = opts?.method === 'POST' && url.endsWith('/api/jobs/sweep');
    if (isSubmit) return Promise.resolve(submit);
    // GET /api/jobs/{id}
    const snap = polls[Math.min(pollIdx++, polls.length - 1)];
    return Promise.resolve(snap);
  });
}

const ok = (body) => ({ ok: true, json: () => Promise.resolve(body) });
const httpErr = (status) => ({
  ok: false,
  status,
  statusText: 'err',
  json: () => Promise.resolve({ detail: 'nope' }),
});

// Resolve when the job flow terminates (onDone or onError).
function drive(opts) {
  return new Promise((resolve) => {
    const handle = runSweepJob('rank(close)*{1..3}', {}, 50, {
      intervalMs: 2,
      onProgress: opts.onProgress,
      onDone: (r) => resolve({ kind: 'done', result: r, handle }),
      onError: (m, e) => resolve({ kind: 'error', msg: m, err: e, handle }),
    });
    opts.captureHandle?.(handle);
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('runSweepJob', () => {
  it('submits, polls through running, resolves onDone with the result', async () => {
    globalThis.fetch = scriptFetch({
      submit: ok({ job_id: 'abc123', status: 'queued' }),
      polls: [
        ok({ status: 'running', progress: { done: 1, total: 3 }, result: null }),
        ok({ status: 'running', progress: { done: 2, total: 3 }, result: null }),
        ok({ status: 'done', progress: { done: 3, total: 3 }, result: { n_combinations: 3, cells: [] } }),
      ],
    });
    const progress = [];
    const res = await drive({ onProgress: (p) => progress.push(p) });
    expect(res.kind).toBe('done');
    expect(res.result.n_combinations).toBe(3);
    expect(progress.at(-1)).toEqual({ done: 3, total: 3 });
  });

  it('resolves onError when the job reports error status', async () => {
    globalThis.fetch = scriptFetch({
      submit: ok({ job_id: 'abc123', status: 'queued' }),
      polls: [ok({ status: 'error', error: 'bad expression', progress: { done: 0, total: 0 } })],
    });
    const res = await drive({});
    expect(res.kind).toBe('error');
    expect(res.msg).toBe('bad expression');
  });

  it('surfaces an HTTP 404 on submit so callers can fall back to the sync path', async () => {
    globalThis.fetch = vi.fn(() => Promise.resolve(httpErr(404)));
    const res = await drive({});
    expect(res.kind).toBe('error');
    expect(res.msg).toMatch(/HTTP 404/); // main.js switches to api.sweep on this
  });

  it('cancel() stops polling — onDone never fires afterward', async () => {
    let calls = 0;
    globalThis.fetch = vi.fn((url, opts) => {
      calls++;
      if (opts?.method === 'POST') return Promise.resolve(ok({ job_id: 'x', status: 'queued' }));
      return Promise.resolve(ok({ status: 'running', progress: { done: 0, total: 3 } }));
    });
    let done = false;
    const handle = runSweepJob('e', {}, 50, {
      intervalMs: 2,
      onDone: () => { done = true; },
      onError: () => {},
    });
    // Let submit + first poll happen, then cancel.
    await new Promise((r) => setTimeout(r, 15));
    handle.cancel();
    const callsAtCancel = calls;
    await new Promise((r) => setTimeout(r, 20));
    expect(done).toBe(false);
    expect(calls).toBe(callsAtCancel); // no further polls after cancel
  });
});
