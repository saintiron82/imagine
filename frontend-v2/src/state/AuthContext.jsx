import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { getAccessToken, setServerUrl, setTokens, clearTokens } from '../api/client'
import { firebaseConnect, getMe, initServer } from '../api/auth'
import { lookupGroup } from '../api/firebase'

// Firebase SDK 는 무겁다 → 동적 import 로 별도 청크 분리(초기 로드 경량화).
// firebaseAuth 가 firebaseApp(SDK init)을 끌어오므로 이 모듈만 lazy 하면 충분.
const fb = () => import('../api/firebaseAuth')

/**
 * 인증(2층): Firebase 신원 + 서버 JWT(/auth/connect).
 * 원칙: **하드 게이트** — 로그인(서버 접속)하지 않으면 앱의 어떤 화면도 보이지
 * 않는다. 미인증이면 무조건 로그인 화면(/start)으로. 데모로 둘러보기 없음.
 *
 * 흐름: Firebase 로그인 → 서버 이름+비번 입력 → lookupGroup(서버 URL 해석)
 *       → getIdToken → firebaseConnect(idToken, 비번) → JWT 저장 → connected.
 */
const AuthContext = createContext(null)
const SERVER_NAME_KEY = 'imagine-v2-server-name'

export function AuthProvider({ children }) {
  const [firebaseUser, setFirebaseUser] = useState(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [connected, setConnected] = useState(false)
  const [role, setRole] = useState('')          // 인증된 사용자의 실제 역할(admin/user)
  // checking: 저장된 토큰을 검증하는 동안 true(게이트가 깜빡이지 않게). 토큰 없으면 즉시 false.
  const [checking, setChecking] = useState(!!getAccessToken())
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

  // 저장된 토큰을 me 로 검증 → connected 확정. 토큰 없거나 실패면 미인증(게이트가 로그인으로).
  useEffect(() => {
    if (!getAccessToken()) { setChecking(false); return }
    getMe()
      .then((me) => { setConnected(true); setRole(me?.role || '') })
      .catch(() => { clearTokens(); setConnected(false) })
      .finally(() => setChecking(false))
  }, [])

  const signInEmail = useCallback(async (email, pw) => (await fb()).signIn(email, pw), [])
  const signUpEmail = useCallback(async (email, pw, name) => (await fb()).signUp(email, pw, name), [])
  const signInGoogle = useCallback(async () => (await fb()).signInWithGoogle(), [])

  /** 서버 연결: 그룹 조회 → URL 설정 → Firebase idToken → /auth/connect.
   * directUrl 없으면 ① Firestore 레지스트리(lookupGroup) ② 실패 시 로컬 백엔드
   * (same-origin /api 가 health 200 — dev/데스크톱) 순으로 해석. '' = same-origin. */
  const connectToServer = useCallback(async (groupName, serverPassword, directUrl) => {
    setBusy(true); setError('')
    try {
      let url = directUrl
      if (url == null) {
        let resolved = null
        try { const g = await lookupGroup(groupName); if (g?.url) resolved = g.url } catch { /* registry miss */ }
        if (resolved == null) {
          try { const r = await fetch('/api/v1/health'); if (r.ok) resolved = '' } catch { /* no local backend */ }
        }
        if (resolved == null) throw new Error('그 이름의 서버를 찾을 수 없습니다')
        url = resolved
      }
      setServerUrl(url)
      const idToken = await (await fb()).getIdToken()
      if (!idToken) throw new Error('먼저 로그인하세요')
      await firebaseConnect(idToken, serverPassword) // 성공 시 토큰 저장
      try { const me = await getMe(); setRole(me?.role || '') } catch { /* role best-effort */ }
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

  /** 이 컴퓨터를 서버로 만들기 — 로컬 백엔드를 /server/init 으로 초기화(생성자=운영자).
   * localhost(프록시 경유) 에서만 허용. 이미 초기화돼 있으면 409 → 접속을 안내. */
  const createServer = useCallback(async (groupName, serverPassword) => {
    setBusy(true); setError('')
    try {
      setServerUrl('') // 로컬 백엔드(same-origin /api)
      const me = firebaseUser
      const idToken = await (await fb()).getIdToken().catch(() => null)
      if (!idToken || !me) throw new Error('먼저 로그인하세요')
      const data = await initServer('', {
        group_name: groupName,
        server_password: serverPassword,
        admin_username: (me.email ? me.email.split('@')[0] : (me.displayName || 'admin')).slice(0, 50),
        admin_password: (crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`).slice(0, 40),
        firebase_uid: me.uid,
        firebase_email: me.email || '',
      })
      if (data?.access_token) setTokens(data.access_token, data.refresh_token)
      try { localStorage.setItem(SERVER_NAME_KEY, groupName) } catch {}
      setServerName(groupName); setRole('admin'); setConnected(true)
      return { ok: true }
    } catch (e) {
      const msg = /already initialized|409/.test(e.message)
        ? '이미 이 컴퓨터에 서버가 있습니다 — 접속을 사용하세요'
        : (e.message || '서버 생성 실패')
      setError(msg)
      return { ok: false, error: msg }
    } finally {
      setBusy(false)
    }
  }, [firebaseUser])

  const disconnect = useCallback(() => { clearTokens(); setConnected(false); setRole('') }, [])
  const signOutAll = useCallback(async () => {
    disconnect()
    try { await (await fb()).signOut() } catch {}
  }, [disconnect])

  const value = {
    firebaseUser, authLoading, connected, checking, role, serverName, busy, error,
    isOperator: role === 'admin',
    signInEmail, signUpEmail, signInGoogle, connectToServer, createServer, disconnect, signOutAll,
  }
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  return useContext(AuthContext)
}
