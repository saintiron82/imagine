/**
 * 접속 계층 — 외부 접속(터널/릴레이) + LAN 상태.
 *   GET /api/v1/server/connection-info (admin) → { connect_modes[] }
 *     mode: direct_local | direct_lan | manual_external | relay_session
 *     각: { mode, url, reachable_scope, security, available }
 *
 * 사용자 결정: 외부 접속 = Cloudflare Tunnel(주소 비공유). 백엔드는 이를
 * relay_session 모드로 모델링 — 여기선 "외부 접속"으로 표시한다.
 * 실제 터널 구동/Firestore external_url 설정은 인프라(별도).
 */
import { useQuery } from '@tanstack/react-query'
import { apiClient } from './client'

const DEMO = {
  connect_modes: [
    { mode: 'direct_local', url: 'http://localhost:8000', available: true },
    { mode: 'direct_lan', url: 'http://192.168.0.5:8000', available: true },
    { mode: 'relay_session', url: 'ourteam.imagine.app', available: true },
  ],
}

export function useConnectionInfo() {
  const q = useQuery({ queryKey: ['connection-info'], queryFn: () => apiClient.get('/api/v1/server/connection-info') })
  const connected = !q.isError && q.data
  const data = connected ? q.data : DEMO
  const modes = data.connect_modes || []
  const byMode = (m) => modes.find(x => x.mode === m) || null
  return {
    isDemo: !connected,
    external: byMode('relay_session'),   // 외부 접속(Cloudflare 터널/릴레이)
    lan: byMode('direct_lan'),
    local: byMode('direct_local'),
  }
}
