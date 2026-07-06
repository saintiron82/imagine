import { useState } from 'react'
import { Outlet, NavLink, Navigate, useNavigate, useLocation } from 'react-router-dom'
import { useApp } from '../state/AppContext'
import { useAuth } from '../state/AuthContext'
import { useLicenseStatus } from '../api/members'
import AddFlow from '../flows/AddFlow'
import UpdateNotification from './UpdateNotification'

/**
 * 앱 셸 — 상단 내비게이션 + 전역 [+ 추가] + 서버 상태 칩.
 * 하드 인증 게이트: 미인증이면 어떤 화면도 렌더하지 않고 /start(로그인)로 보낸다.
 * 역할 게이팅: 일반 사용자에겐 폴더/분석/관리/+추가가 존재하지 않는 앱처럼 보인다.
 */
export default function AppShell() {
  const { isOperator, server } = useApp()
  const { connected, checking, firebaseUser, serverName, signOutAll } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [addOpen, setAddOpen] = useState(false)

  const tab = ({ isActive }) => (isActive ? 'active' : '')

  // 하드 게이트: 저장된 세션 검증 중에는 빈 화면, 미인증이면 로그인으로 강제 이동.
  if (checking) {
    return <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--faint)', fontSize: 13 }}>인증 확인 중…</div>
  }
  if (!connected) {
    return <Navigate to="/start" replace />
  }
  // 역할 게이트: 운영자 전용 경로는 URL 직접 접근도 차단한다(내비 숨김만으로는 부족).
  const operatorOnly = ['/folders', '/analysis', '/admin']
  if (!isOperator && operatorOnly.some(p => location.pathname.startsWith(p))) {
    return <Navigate to="/search" replace />
  }

  return (
    <>
      <header className="topbar">
        <div className="logo"><span className="dot" />Imagine</div>
        <nav className="nav">
          <NavLink to="/search" className={tab}>검색</NavLink>
          {isOperator && <NavLink to="/folders" className={tab}>폴더</NavLink>}
          {isOperator && <NavLink to="/analysis" className={tab}>분석</NavLink>}
        </nav>
        <div className="spacer" />
        {isOperator && (
          <button className="btn-add" onClick={() => setAddOpen(true)}>＋ 추가</button>
        )}
        <div className="right-nav">
          {isOperator && <NavLink to="/admin" className={tab}>관리</NavLink>}
          <NavLink to="/settings" className={tab}>설정</NavLink>
        </div>
        {isOperator && server.online && (
          <button className="srv-chip" onClick={() => navigate('/admin')}>
            <span className="sd" />{serverName || '서버'} 온라인
          </button>
        )}
        <button className="role-chip" title={firebaseUser?.email || ''} onClick={signOutAll}>
          {firebaseUser?.email ? firebaseUser.email.split('@')[0] : (serverName || '로그인됨')} · <b>로그아웃</b>
        </button>
      </header>
      {isOperator && <ExpiryBanner />}
      <main>
        <Outlet />
      </main>
      {addOpen && <AddFlow onClose={() => setAddOpen(false)} />}
      <UpdateNotification />
    </>
  )
}

/**
 * 구독/만료 배너 (IMGV2-22) — 운영자에게만. 만료/임박 시 상단 고정 안내.
 * 갱신은 홈페이지('계정의 집')에서 — 배포 URL 이 코드에 고정돼 있지 않아
 * 죽은 링크 대신 경로를 텍스트로 안내한다(URL 확정 시 링크화).
 */
function ExpiryBanner() {
  const { state, daysLeft, expiresAt } = useLicenseStatus()
  const [dismissed, setDismissed] = useState(false)
  if (state === 'ok' || dismissed) return null

  const expired = state === 'expired'
  const when = expiresAt ? new Date(expiresAt).toLocaleDateString('ko-KR') : ''
  const bg = expired ? 'rgba(248,113,113,.12)' : 'rgba(251,191,36,.12)'
  const bd = expired ? 'rgba(248,113,113,.4)' : 'rgba(251,191,36,.4)'
  const fg = expired ? 'var(--red)' : '#fbbf24'
  const msg = expired
    ? `라이선스가 만료되었습니다${when ? ` (${when})` : ''} — 갱신 전까지 새 분석/멤버 추가가 제한됩니다.`
    : `라이선스 만료 ${daysLeft}일 전${when ? ` (${when})` : ''} — 중단 없이 쓰려면 미리 갱신하세요.`

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 18px', fontSize: 12.5, background: bg, borderBottom: `1px solid ${bd}`, color: fg }}>
      <span>{expired ? '⛔' : '⚠'}</span>
      <span style={{ color: 'var(--text)' }}>{msg}</span>
      <span style={{ marginLeft: 'auto', color: 'var(--faint)', fontSize: 11 }}>갱신: 홈페이지 &gt; 내 라이선스</span>
      <button onClick={() => setDismissed(true)} style={{ background: 'none', border: 'none', color: 'var(--faint)', cursor: 'pointer', fontSize: 14, lineHeight: 1 }}>×</button>
    </div>
  )
}
