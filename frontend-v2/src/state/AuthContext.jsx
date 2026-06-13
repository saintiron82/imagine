import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { getAccessToken, setServerUrl, clearTokens } from '../api/client'
import { firebaseConnect, getMe } from '../api/auth'
import { lookupGroup } from '../api/firebase'

// Firebase SDK 는 무겁다 → 동적 import 로 별도 청크 분리(초기 로드 경량화).
// firebaseAuth 가 firebaseApp(SDK init)을 끌어오므로 이 모듈만 lazy 하면 충분.
const fb = () => import('../api/firebaseAuth')

/**
 * 인증(2층): Firebase 신원 + 서버 JWT(/auth/connect).
 * 원칙: 소프트 게이트 — 로그인하면 실데이터, 안 하면 데모로 둘러보기 유지.
 * (만료 시 하드 입장 차단은 백엔드 정책 — IMGV2-9 백엔드 파트로 분리)
 *
 * 흐름: Firebase 로그인 → 서버 이름+비번 입력 → lookupGroup(서버 URL 해석)
 *       → getIdToken → firebaseConnect(idToken, 비번) → JWT 저장 → connected.
 */
const AuthContext = createContext(null)
const SERVER_NAME_KEY = 'imagine-v2-server-name'

export function AuthProvider({ children }) {
  const [firebaseUser, setFirebaseUser] = useState(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [connected, setConnected] = useState(!!getAccessToken())
  const [serverName, setServerName] = useState(() => { try { return localStorage.getItem(SERVER_NAME_KEY) || '' } catch { return '' } })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // Firebase 인증 상태 관찰 (동적 로드 — 오프라인이면 user=null 로 resolve)
  useEffect(() => {
    let unsub = () => {}
    let alive = true
    fb().then(m => {
      if (!alive) return
      try { unsub = m.onAuthStateChanged(u => { setFirebaseUser(u); setAuthLoading(false) }) }
      catch { setAuthLoading(false) }
    }).catch(() => setAuthLoading(false))
    return () => { alive = false; unsub() }
  }, [])

  // 토큰이 이미 있으면 connected 로 보고 me 로 검증(실패 시 해제)
  useEffect(() => {
    if (!getAccessToken()) return
    getMe().then(() => setConnected(true)).catch(() => { clearTokens(); setConnected(false) })
  }, [])

  const signInEmail = useCallback(async (email, pw) => (await fb()).signIn(email, pw), [])
  const signUpEmail = useCallback(async (email, pw, name) => (await fb()).signUp(email, pw, name), [])
  const signInGoogle = useCallback(async () => (await fb()).signInWithGoogle(), [])

  /** 서버 연결: 그룹 조회 → URL 설정 → Firebase idToken → /auth/connect */
  const connectToServer = useCallback(async (groupName, serverPassword, directUrl) => {
    setBusy(true); setError('')
    try {
      let url = directUrl
      if (!url) {
        const group = await lookupGroup(groupName)
        if (!group?.url) throw new Error('그 이름의 서버를 찾을 수 없습니다')
        url = group.url
      }
      setServerUrl(url)
      const idToken = await (await fb()).getIdToken()
      if (!idToken) throw new Error('먼저 로그인하세요')
      await firebaseConnect(idToken, serverPassword) // 성공 시 토큰 저장
      try { localStorage.setItem(SERVER_NAME_KEY, groupName) } catch {}
      setServerName(groupName)
      setConnected(true)
      return { ok: true }
    } catch (e) {
      setError(e.message || '연결 실패')
      return { ok: false, error: e.message }
    } finally {
      setBusy(false)
    }
  }, [])

  const disconnect = useCallback(() => { clearTokens(); setConnected(false) }, [])
  const signOutAll = useCallback(async () => {
    disconnect()
    try { await (await fb()).signOut() } catch {}
  }, [disconnect])

  const value = {
    firebaseUser, authLoading, connected, serverName, busy, error,
    signInEmail, signUpEmail, signInGoogle, connectToServer, disconnect, signOutAll,
  }
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  return useContext(AuthContext)
}
