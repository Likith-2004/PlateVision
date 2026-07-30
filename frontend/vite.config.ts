import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The build lands in app/web/, which app/main.py serves. That directory is
// generated output -- edit frontend/src, then `npm run build`.
//
// `npm run dev` proxies the API routes through to a locally running uvicorn so
// hot reload works against the real backend rather than mocks.
const API_ROUTES = ['/detect', '/health', '/media', '/docs', '/openapi.json']

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../app/web',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_ROUTES.map((route) => [
        route,
        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      ]),
    ),
  },
})
