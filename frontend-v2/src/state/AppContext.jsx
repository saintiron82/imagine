import { createContext, useContext, useState } from 'react'

/**
 * 앱 전역 상태 — 역할(운영자/사용자)과 서버 연결 상태.
 * 실제 인증이 붙기 전까지는 데모 토글로 역할을 전환한다.
 * 원칙: 일반 사용자는 검색+설정(나/이 컴퓨터)만 본다.
 */
const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [role, setRole] = useState('operator') // 'operator' | 'user'
  const [server] = useState({
    name: '우리팀 라이브러리',
    online: true,
    external: true, // Cloudflare 터널 연결 상태
  })

  const value = {
    role,
    isOperator: role === 'operator',
    toggleRole: () => setRole(r => (r === 'operator' ? 'user' : 'operator')),
    server,
  }
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  return useContext(AppContext)
}
