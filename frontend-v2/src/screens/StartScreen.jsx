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
    signInEmail, signInGoogle, connectToServer } = useAuth()

  const [step, setStep] = useState(1)
  const [email, setEmail] = useState('')
  const [pw, setPw] = useState('')
  const [localErr, setLocalErr] = useState('')
  const [srvName, setSrvName] = useState(serverName || '')
  const [srvPw, setSrvPw] = useState('')

  useEffect(() => {
    if (connected) navigate('/search')
    else if (firebaseUser && step === 1) setStep(2)
  }, [firebaseUser, connected]) // eslint-disable-line
  useEffect(() => { if (serverName) setSrvName(serverName) }, [serverName])

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
  const doConnect = async () => {
    const r = await connectToServer(srvName.trim(), srvPw)
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

      {step === 2 && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>서버 접속</div>
          <div className="tag">{firebaseUser?.email ? `${firebaseUser.email} 로 로그인됨` : '서버 이름과 비밀번호로 접속'}</div>
          <input placeholder="서버 이름" value={srvName} onChange={e => setSrvName(e.target.value)} />
          <input type="password" placeholder="서버 비밀번호" value={srvPw} onChange={e => setSrvPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doConnect()} />
          {error && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{error}</div>}
          <button className="pri-w" disabled={busy || !srvName.trim() || !srvPw} onClick={doConnect}>{busy ? '접속 중…' : '접속'}</button>
          <div style={{ fontSize: 10, color: 'var(--faint)', textAlign: 'center', marginTop: 12 }}>
            팀 초대를 받았다면 그 이메일로 로그인하면 자동으로 접속됩니다.<br />
            새 서버 생성·구매는 <a href="https://imagine.app" target="_blank" rel="noreferrer" style={{ color: '#93c5fd' }}>imagine.app</a> 에서.
          </div>
        </div>
      )}
    </section>
  )
}
