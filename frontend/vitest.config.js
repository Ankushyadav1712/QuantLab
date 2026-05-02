import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: false,
    include: ['tests/**/*.{test,spec}.js'],
    // Stub the lightweight-charts CDN global that components import.
    setupFiles: ['./tests/setup.js'],
  },
});
