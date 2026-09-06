import { createContext, useContext } from 'react'
import { useAuth } from './AuthContext'

/**
 * 앱 전역 상태 — 역할/서버는 실제 인증(AuthContext)에서 파생한다(데모 토글 없음).
 * 원칙: 일반 사용자(user)는 검색+설정만 본다. 운영자(admin)는 전체.
 */
const AppContext = createContext(null)

export function AppProvider({ children }) {
  const { isOperator, connected, serverName } = useAuth()

  const value = {
    isOperator,
    server: { name: serverName || '', online: connected },   // 빈 값이면 표시 측에서 i18n 폴백
  }
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  return useContext(AppContext)
}
