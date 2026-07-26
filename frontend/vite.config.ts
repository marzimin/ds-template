/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// The dev server proxies backend paths so the browser sees a single origin.
// That keeps request URLs relative in the app code, which is also what makes the
// production build work when one host serves both.
//
// /docs and /openapi.json are proxied alongside /api because the footer links to
// the API's interactive documentation. Without them the dev server answers with
// the SPA's own index.html and the link silently reloads the app instead.
const API_TARGET = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000';
const BACKEND_PATHS = ['/api', '/docs', '/redoc', '/openapi.json'];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      BACKEND_PATHS.map((path) => [path, { target: API_TARGET, changeOrigin: true }]),
    ),
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
