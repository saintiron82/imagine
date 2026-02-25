import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 9274,
    strictPort: true,
  },
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version || '0.6.0'),
    __BUILD_ID__: JSON.stringify('20260225_06'),
  },
})
