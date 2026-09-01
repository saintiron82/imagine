import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../state/AuthContext'
import { useLocale } from '../i18n'

/**
 * 시작 화면 — ① 로그인(Firebase) ② 서버 접속(/auth/connect).
 * 하드 게이트: 로그인·접속하지 않으면 앱의 어떤 화면도 못 본다.
 * 서버 생성/구매/초대 발송은 홈페이지(계정의 집) — 앱은 접속만 한다.
 */
export default function StartScreen() {
  const { t } = useLocale()
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
    catch (e) { setLocalErr(e.code === 'auth/invalid-credential' ? t('v2.start.err_bad_credential') : (e.message || t('v2.start.err_sign_in'))) }
  }
  const doGoogle = async () => {
    setLocalErr('')
    try { await signInGoogle(); setStep(2) }
    catch (e) { setLocalErr(e.message || t('v2.start.err_google')) }
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
          <div className="tag">{t('v2.start.tagline')}</div>
          <button className="start-opt" style={{ justifyContent: 'center', gap: 8 }} disabled={authLoading} onClick={doGoogle}>
            <span style={{ fontWeight: 700 }}>G</span><span className="t">{t('v2.start.continue_google')}</span>
          </button>
          <div className="start-div">{t('v2.start.or_email')}</div>
          <input placeholder={t('v2.start.email')} value={email} onChange={e => setEmail(e.target.value)} />
          <input type="password" placeholder={t('v2.start.password')} value={pw} onChange={e => setPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doEmail()} />
          {localErr && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{localErr}</div>}
          <button className="pri-w" disabled={authLoading || !email || !pw} onClick={doEmail}>{t('v2.start.sign_in')}</button>
          <div style={{ fontSize: 10.5, color: 'var(--faint)', textAlign: 'center', marginTop: 8 }}>
            {t('v2.start.no_account')} <a href="https://imagine.app" target="_blank" rel="noreferrer" style={{ color: '#93c5fd' }}>imagine.app</a> {t('v2.start.no_account_suffix')}
          </div>
        </div>
      )}

      {step === 2 && localInfo === null && (
        <div className="start-card"><div className="tag" style={{ padding: '20px 0' }}>{t('v2.start.checking_local')}</div></div>
      )}

      {/* 이 머신에 서버가 이미 있음 → 접속(이름 고정, 비번만) */}
      {step === 2 && view2 === 'auto' && localInfo?.initialized && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>{t('v2.start.connect_local_title')}</div>
          <div className="tag">🖥️ <b style={{ color: 'var(--text)' }}>{localInfo.group_name}</b> · {firebaseUser?.email || ''}</div>
          <input type="password" placeholder={t('v2.start.server_password')} value={srvPw} onChange={e => setSrvPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doConnectLocal()} autoFocus />
          {error && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{error}</div>}
          <button className="pri-w" disabled={busy || !srvPw} onClick={doConnectLocal}>{busy ? t('v2.start.connecting') : t('v2.start.connect')}</button>
          <div style={{ textAlign: 'center', marginTop: 12 }}>
            <span style={{ fontSize: 10.5, color: 'var(--faint)', cursor: 'pointer' }} onClick={() => { setView2('remote') }}>{t('v2.start.goto_remote')}</span>
          </div>
        </div>
      )}

      {/* 이 머신에 서버 없음 → 만들기 */}
      {step === 2 && view2 === 'auto' && localInfo && !localInfo.initialized && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>{t('v2.start.create_title')}</div>
          <div className="tag">{t('v2.start.create_desc')}</div>
          <input placeholder={t('v2.start.server_name_hint')} value={srvName} onChange={e => setSrvName(e.target.value)} />
          <input type="password" placeholder={t('v2.start.admin_password')} value={srvPw} onChange={e => setSrvPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doCreate()} />
          {error && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{error}</div>}
          <button className="pri-w" disabled={busy || !srvName.trim() || !srvPw} onClick={doCreate}>{busy ? t('v2.start.creating') : t('v2.start.create')}</button>
          <div style={{ textAlign: 'center', marginTop: 12 }}>
            <span style={{ fontSize: 10.5, color: 'var(--faint)', cursor: 'pointer' }} onClick={() => { setView2('remote') }}>{t('v2.start.goto_remote_alt')}</span>
          </div>
        </div>
      )}

      {/* 다른 팀 서버에 접속(원격) — 이름 조회 */}
      {step === 2 && view2 === 'remote' && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>{t('v2.start.remote_title')}</div>
          <div className="tag">{t('v2.start.remote_desc')}</div>
          <input placeholder={t('v2.start.server_name')} value={srvName} onChange={e => setSrvName(e.target.value)} />
          <input type="password" placeholder={t('v2.start.server_password')} value={srvPw} onChange={e => setSrvPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doConnectRemote()} />
          {error && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{error}</div>}
          <button className="pri-w" disabled={busy || !srvName.trim() || !srvPw} onClick={doConnectRemote}>{busy ? t('v2.start.connecting') : t('v2.start.connect')}</button>
          <div style={{ textAlign: 'center', marginTop: 12 }}>
            <span style={{ fontSize: 10.5, color: 'var(--faint)', cursor: 'pointer' }} onClick={() => { setView2('auto') }}>{t('v2.start.back_local')}</span>
          </div>
        </div>
      )}
    </section>
  )
}
