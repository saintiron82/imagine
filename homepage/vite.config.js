import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 홈페이지(계정의 집) — Cloudflare Pages 배포. SPA fallback 은 public/_redirects.
export default defineConfig({
  plugins: [react()],
  server: { port: 9280, strictPort: true },
})
