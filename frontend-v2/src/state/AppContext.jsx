import { createContext, useContext, useEffect } from 'react'
import { setUseLocalBackend } from '../services/bridge'
import { useAuth } from './AuthContext'

/**
 * 앱 전역 상태 — 역할/서버는 실제 인증(AuthContext)에서 파생한다(데모 토글 없음).
 * 원칙: 일반 사용자(user)는 검색+설정만 본다. 운영자(admin)는 전체.
 */
const AppContext = createContext(null)

export function AppProvider({ children }) {
  const { isOperator, connected, serverName } = useAuth()

  // v2 는 항상 HTTP 모드 — bridge 는 서버 API(apiClient)를 탄다(IPC 아님).
  useEffect(() => { setUseLocalBackend(false) }, [])

  const value = {
    isOperator,
    server: { name: serverName || '서버', online: connected },
  }
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  return useContext(AppContext)
}
