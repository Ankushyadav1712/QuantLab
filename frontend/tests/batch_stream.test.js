import { beforeEach, describe, expect, it, vi } from 'vitest';
import { openBatchStream } from '../src/api.js';

// Minimal WebSocket stand-in (jsdom has none). Tests drive it via emit().
class MockWS {
  constructor(url) {
    this.url = url;
    this.sent = [];
    this.closed = false;
    this.listeners = {};
    MockWS.instances.push(this);
  }
  addEventListener(type, fn) {
    (this.listeners[type] ||= []).push(fn);
  }
  send(data) {
    this.sent.push(data);
  }
  close() {
    this.closed = true;
  }
  emit(type, ev) {
    (this.listeners[type] || []).forEach((fn) => fn(ev));
  }
}
MockWS.instances = [];

function msg(obj) {
  return { data: JSON.stringify(obj) };
}

beforeEach(() => {
  MockWS.instances = [];
  globalThis.WebSocket = MockWS;
});

describe('openBatchStream', () => {
  it('connects to /ws/batch, sends the request on open, streams result → complete', () => {
    const onResult = vi.fn();
    const onComplete = vi.fn();
    const onError = vi.fn();
    openBatchStream(
      [{ id: 'a', expression: 'rank(close)' }],
      { neutralization: 'market' },
      { onResult, onComplete, onError }
    );
    const ws = MockWS.instances[0];
    expect(ws.url).toMatch(/^ws:\/\/.*\/ws\/batch$/);

    ws.emit('open');
    expect(JSON.parse(ws.sent[0])).toEqual({
      alphas: [{ id: 'a', expression: 'rank(close)' }],
      settings: { neutralization: 'market' },
    });

    ws.emit('message', msg({ type: 'result', index: 0, total: 1, row: { id: 'a' } }));
    expect(onResult).toHaveBeenCalledTimes(1);

    ws.emit('message', msg({ type: 'complete', n_alphas: 1, n_ok: 1, correlation_matrix: {} }));
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
    expect(ws.closed).toBe(true);
  });

  it('fires onError once on a server error message (and not again on close)', () => {
    const onError = vi.fn();
    openBatchStream([], {}, { onError });
    const ws = MockWS.instances[0];
    ws.emit('message', msg({ type: 'error', detail: 'No alphas provided' }));
    ws.emit('close'); // must not double-fire
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith('No alphas provided');
  });

  it('fires onError on a connection error (the fallback trigger)', () => {
    const onError = vi.fn();
    openBatchStream([], {}, { onError });
    MockWS.instances[0].emit('error');
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it('does not fire onError after complete even if a close follows', () => {
    const onComplete = vi.fn();
    const onError = vi.fn();
    openBatchStream([], {}, { onComplete, onError });
    const ws = MockWS.instances[0];
    ws.emit('message', msg({ type: 'complete', n_alphas: 0, n_ok: 0 }));
    ws.emit('close');
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
  });
});
