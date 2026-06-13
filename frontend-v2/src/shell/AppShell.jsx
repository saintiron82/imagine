import { useState } from 'react'
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useApp } from '../state/AppContext'
import { useAuth } from '../state/AuthContext'
import AddFlow from '../flows/AddFlow'

/**
 * 앱 셸 — 상단 내비게이션 + 전역 [+ 추가] + 서버 상태 칩.
 * 역할 게이팅: 일반 사용자에겐 폴더/분석/관리/+추가가 존재하지 않는 앱처럼 보인다.
 */
export default function AppShell() {
  const { isOperator, toggleRole, role, server } = useApp()
  const { connected, firebaseUser, serverName, signOutAll } = useAuth()
  const navigate = useNavigate()
  const [addOpen, setAddOpen] = useState(false)

  const tab = ({ isActive }) => (isActive ? 'active' : '')

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
            <span className="sd" />{connected ? (serverName || '서버') : '서버'} 온라인{server.external ? ' · 외부 접속 연결됨' : ''}
          </button>
        )}
        {connected ? (
          <button className="role-chip" title={firebaseUser?.email || ''} onClick={signOutAll}>
            {firebaseUser?.email ? firebaseUser.email.split('@')[0] : '로그인됨'} · <b>로그아웃</b>
          </button>
        ) : (
          <button className="role-chip" onClick={() => navigate('/start')}>로그인</button>
        )}
        <button className="role-chip" onClick={toggleRole}>
          데모 · 역할: <b>{role === 'operator' ? '운영자' : '일반 사용자'}</b>
        </button>
      </header>
      <main>
        <Outlet />
      </main>
      {addOpen && <AddFlow onClose={() => setAddOpen(false)} />}
    </>
  )
}
