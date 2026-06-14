import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../state/AuthContext'

/**
 * 시작 화면 — ① 로그인(Firebase) ② 서버 접속(/auth/connect).
 * 하드 게이트: 로그인·접속하지 않으면 앱의 어떤 화면도 못 본다.
 * 서버 생성/구매/초대 발송은 홈페이지(계정의 집) — 앱은 접속만 한다.
 */
export default function StartScreen() {
  const navigate = useNavigate()
  const { firebaseUser, authLoading, connected, serverName, busy, error,
    signInEmail, signInGoogle, connectToServer, createServer } = useAuth()

  const [step, setStep] = useState(1)
  const [email, setEmail] = useState('')
  const [pw, setPw] = useState('')
  const [localErr, setLocalErr] = useState('')
  const [srvName, setSrvName] = useState('')   // 원격 서버 이름(다른 팀 서버 접속용)
  const [srvPw, setSrvPw] = useState('')
  const [localInfo, setLocalInfo] = useState(null) // 이 머신 서버 상태: null=로딩 | {initialized, group_name}
  const [view2, setView2] = useState('auto')   // step2: 'auto'(이 머신) | 'remote'(다른 서버)

  useEffect(() => {
    if (connected) navigate('/search')
    else if (firebaseUser && step === 1) setStep(2)
  }, [firebaseUser, connected]) // eslint-disable-line

  // step2 진입 시 이 컴퓨터의 서버 상태 조회(같은-origin → 프록시 → 로컬 백엔드)
  useEffect(() => {
    if (step !== 2 || localInfo) return
    fetch('/api/v1/server/info')
      .then(r => (r.ok ? r.json() : null))
      .then(d => setLocalInfo(d && typeof d.initialized === 'boolean' ? d : { initialized: false }))
      .catch(() => setLocalInfo({ initialized: false }))
  }, [step, localInfo])

  const doEmail = async () => {
    setLocalErr('')
    try { await signInEmail(email, pw); setStep(2) }
    catch (e) { setLocalErr(e.code === 'auth/invalid-credential' ? '이메일 또는 비밀번호가 올바르지 않습니다' : (e.message || '로그인 실패')) }
  }
  const doGoogle = async () => {
    setLocalErr('')
    try { await signInGoogle(); setStep(2) }
    catch (e) { setLocalErr(e.message || 'Google 로그인 실패') }
  }
  const doConnectLocal = async () => {   // 이 머신의 서버 — 이름 고정, 로컬 백엔드 직결
    const r = await connectToServer(localInfo.group_name, srvPw, '')
    if (r.ok) navigate('/search')
  }
  const doConnectRemote = async () => {  // 다른 팀 서버 — 이름으로 조회(lookupGroup)
    const r = await connectToServer(srvName.trim(), srvPw)
    if (r.ok) navigate('/search')
  }
  const doCreate = async () => {
    const r = await createServer(srvName.trim(), srvPw)
    if (r.ok) navigate('/search')
  }

  return (
    <section id="scr-start" className="screen active scr-center" style={{ height: '100vh' }}>
      {step === 1 && (
        <div className="start-card">
          <div className="lg"><span className="dot" />Imagine</div>
          <div className="tag">내 에셋을 자연어로 찾는 검색</div>
          <button className="start-opt" style={{ justifyContent: 'center', gap: 8 }} disabled={authLoading} onClick={doGoogle}>
            <span style={{ fontWeight: 700 }}>G</span><span className="t">Google로 계속하기</span>
          </button>
          <div className="start-div">또는 이메일로</div>
          <input placeholder="이메일" value={email} onChange={e => setEmail(e.target.value)} />
          <input type="password" placeholder="비밀번호" value={pw} onChange={e => setPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doEmail()} />
          {localErr && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{localErr}</div>}
          <button className="pri-w" disabled={authLoading || !email || !pw} onClick={doEmail}>로그인</button>
          <div style={{ fontSize: 10.5, color: 'var(--faint)', textAlign: 'center', marginTop: 8 }}>
            계정이 없거나 새 서버가 필요한가요? <a href="https://imagine.app" target="_blank" rel="noreferrer" style={{ color: '#93c5fd' }}>imagine.app</a> 에서 — 가입·구매·초대 수락
          </div>
        </div>
      )}

      {step === 2 && localInfo === null && (
        <div className="start-card"><div className="tag" style={{ padding: '20px 0' }}>이 컴퓨터의 서버 확인 중…</div></div>
      )}

      {/* 이 머신에 서버가 이미 있음 → 접속(이름 고정, 비번만) */}
      {step === 2 && view2 === 'auto' && localInfo?.initialized && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>이 컴퓨터의 서버에 접속</div>
          <div className="tag">🖥️ <b style={{ color: 'var(--text)' }}>{localInfo.group_name}</b> · {firebaseUser?.email || ''}</div>
          <input type="password" placeholder="서버 비밀번호" value={srvPw} onChange={e => setSrvPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doConnectLocal()} autoFocus />
          {error && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{error}</div>}
          <button className="pri-w" disabled={busy || !srvPw} onClick={doConnectLocal}>{busy ? '접속 중…' : '접속'}</button>
          <div style={{ textAlign: 'center', marginTop: 12 }}>
            <span style={{ fontSize: 10.5, color: 'var(--faint)', cursor: 'pointer' }} onClick={() => { setView2('remote') }}>다른 팀 서버에 접속 →</span>
          </div>
        </div>
      )}

      {/* 이 머신에 서버 없음 → 만들기 */}
      {step === 2 && view2 === 'auto' && localInfo && !localInfo.initialized && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>이 컴퓨터를 서버로 만들기</div>
          <div className="tag">이 PC가 팀의 라이브러리 서버가 됩니다 — 로그인한 계정이 운영자</div>
          <input placeholder="서버 이름 — 팀원이 접속할 이름" value={srvName} onChange={e => setSrvName(e.target.value)} />
          <input type="password" placeholder="서버 관리 비밀번호" value={srvPw} onChange={e => setSrvPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doCreate()} />
          {error && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{error}</div>}
          <button className="pri-w" disabled={busy || !srvName.trim() || !srvPw} onClick={doCreate}>{busy ? '서버 생성 중…' : '서버 만들기'}</button>
          <div style={{ textAlign: 'center', marginTop: 12 }}>
            <span style={{ fontSize: 10.5, color: 'var(--faint)', cursor: 'pointer' }} onClick={() => { setView2('remote') }}>대신 다른 팀 서버에 접속 →</span>
          </div>
        </div>
      )}

      {/* 다른 팀 서버에 접속(원격) — 이름 조회 */}
      {step === 2 && view2 === 'remote' && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>다른 팀 서버에 접속</div>
          <div className="tag">서버 이름과 비밀번호로 원격 서버에 접속합니다</div>
          <input placeholder="서버 이름" value={srvName} onChange={e => setSrvName(e.target.value)} />
          <input type="password" placeholder="서버 비밀번호" value={srvPw} onChange={e => setSrvPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doConnectRemote()} />
          {error && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{error}</div>}
          <button className="pri-w" disabled={busy || !srvName.trim() || !srvPw} onClick={doConnectRemote}>{busy ? '접속 중…' : '접속'}</button>
          <div style={{ textAlign: 'center', marginTop: 12 }}>
            <span style={{ fontSize: 10.5, color: 'var(--faint)', cursor: 'pointer' }} onClick={() => { setView2('auto') }}>← 이 컴퓨터의 서버</span>
          </div>
        </div>
      )}
    </section>
  )
}
