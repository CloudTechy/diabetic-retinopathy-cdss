import { defineConfig } from 'vitest/config';

// Unit tests for the pure helpers (src/utils/*.test.ts). No DOM: the
// screens are exercised against the running stack by the capture scripts.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
