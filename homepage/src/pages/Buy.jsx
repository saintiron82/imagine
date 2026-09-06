import { useState, useEffect } from 'react'
import { auth } from '../firebase'
import { onAuthStateChanged, GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth'
import { PLANS } from '../plans'

/** 2장: 구매 — 플랜 선택 → (로그인) → Stripe 결제 또는 14일 체험 즉시 발급. */
export default function Buy() {
  const [user, setUser] = useState(null)
  const [group, setGroup] = useState('')
  const [busy, setBusy] = useState('')
  const [msg, setMsg] = useState(null)

  useEffect(() => onAuthStateChanged(auth, setUser), [])

  const login = async () => {
    try { await signInWithPopup(auth, new GoogleAuthProvider()) }
    catch (e) { setMsg({ type: 'warn', text: 'Google 로그인 실패: ' + e.message }) }
  }

  const choose = async (plan) => {
    setMsg(null)
    if (!user) { setMsg({ type: 'warn', text: '먼저 로그인하세요.' }); return }
    if (!group.trim()) { setMsg({ type: 'warn', text: '서버(그룹) 이름을 입력하세요.' }); return }
    setBusy(plan.id)
    try {
      const idToken = await user.getIdToken()
      const r = await fetch('/api/checkout', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan: plan.id, group_name: group.trim(), id_token: idToken }),
      })
      const data = await r.json()
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`)
      if (data.url) { window.location.href = data.url; return }   // Stripe Checkout
      if (data.trial) { setMsg({ type: 'ok', text: `체험 라이선스가 발급되었습니다 — '${group.trim()}'. 다운로드 후 이 그룹 이름으로 서버를 만드세요.` }) }
    } catch (e) {
      setMsg({ type: 'warn', text: '결제를 시작할 수 없습니다: ' + e.message + ' (Stripe/Firebase 설정 확인)' })
    } finally { setBusy('') }
  }

  return (
    <div className="wrap">
      <div className="section-title">플랜</div>
      <p className="muted">서버(그룹) 단위 라이선스 · 좌석 + 분석기 한도 · 설치비 + 연 갱신. 앱은 키만 있으면 홈페이지 없이 동작합니다.</p>

      <div className="card" style={{ marginTop: 20 }}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontWeight: 700 }}>{user ? `${user.email} 로 로그인됨` : '구매하려면 로그인'}</div>
            <div className="note">라이선스는 이 계정에 귀속됩니다.</div>
          </div>
          {user ? <button className="btn ghost" onClick={() => signOut(auth)}>로그아웃</button>
                : <button className="btn" onClick={login}>Google로 로그인</button>}
        </div>
        <div className="field" style={{ marginTop: 16, maxWidth: 360 }}>
          <label>서버(그룹) 이름 — 팀원이 접속할 때 쓰는 이름</label>
          <input value={group} onChange={e => setGroup(e.target.value)} placeholder="예: 우리팀 라이브러리" />
        </div>
      </div>

      {msg && <div className={`banner ${msg.type}`}>{msg.text}</div>}

      <div className="plans">
        {PLANS.map(p => (
          <div key={p.id} className={`plan ${p.highlight ? 'hl' : ''}`}>
            <div className="nm">{p.name}</div>
            <div className="pr">{p.price} <small>{p.checkout ? `/ ${p.period}` : p.period}</small></div>
            <div className="bl">{p.blurb}</div>
            <ul>{p.features.map(f => <li key={f}>{f}</li>)}</ul>
            <div className="pick">
              <button className={`btn ${p.highlight ? '' : 'ghost'}`} style={{ width: '100%' }}
                disabled={busy === p.id} onClick={() => choose(p)}>
                {busy === p.id ? '처리 중…' : p.checkout ? '구매' : '체험 시작'}
              </button>
            </div>
          </div>
        ))}
      </div>
      <p className="note" style={{ paddingBottom: 40 }}>결제는 Stripe로 안전하게 처리됩니다. 결제 완료 시 라이선스가 즉시 발급되어 [내 라이선스]에서 확인할 수 있습니다.</p>
    </div>
  )
}
