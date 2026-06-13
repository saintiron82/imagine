import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 개발 시 백엔드 연동: same-origin /api 요청을 로컬 서버(8000)로 프록시한다.
// → api/client.js 의 getServerUrl()='' 이면 요청이 /api/... 로 나가고 프록시가
//   :8000 으로 넘긴다. CORS(기본 허용 origin 에 v2 dev 포트 없음)를 우회.
// 환경변수 IMAGINE_API_TARGET 로 대상 서버 변경 가능.
const API_TARGET = process.env.IMAGINE_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 9275,
    strictPort: true,
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: true },
    },
  },
})
