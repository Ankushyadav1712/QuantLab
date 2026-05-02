// Vitest setup — runs before every test file.
//
// jsdom doesn't provide window.LightweightCharts (loaded via CDN <script> tag
// in production), so any component that calls into it would throw under test.
// We stub a minimal surface.  Tests that need real chart rendering can override.

const noopSeries = {
  setData: () => {},
  setMarkers: () => {},
  applyOptions: () => {},
};
const noopChart = {
  addLineSeries: () => noopSeries,
  addAreaSeries: () => noopSeries,
  addBaselineSeries: () => noopSeries,
  addHistogramSeries: () => noopSeries,
  applyOptions: () => {},
  remove: () => {},
  timeScale: () => ({
    fitContent: () => {},
    timeToCoordinate: () => 0,
    subscribeVisibleTimeRangeChange: () => {},
  }),
};

// Feature-detect — vitest's jsdom env provides window + localStorage; don't
// stomp on it.  Just attach the stub to whichever global object exists.
const target = (typeof window !== 'undefined') ? window : globalThis;
target.LightweightCharts = {
  createChart: () => noopChart,
};

// ResizeObserver isn't in jsdom either.
if (typeof ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

// jsdom in this Vitest version doesn't always provide a working Storage —
// install a simple Map-backed polyfill so api.js's localStorage round-trips
// reliably in tests.
function makeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
    clear: () => { map.clear(); },
    key: (i) => Array.from(map.keys())[i] ?? null,
    get length() { return map.size; },
  };
}
const storage = makeStorage();
target.localStorage = storage;
globalThis.localStorage = storage;
