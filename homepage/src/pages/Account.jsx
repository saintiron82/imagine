import { useState, useEffect } from 'react'
import { auth, dbf } from '../firebase'
import { onAuthStateChanged, GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth'
import { collection, query, where, getDocs } from 'firebase/firestore'

/** 3장: 내 라이선스 — 구매자 포털(키·만료·좌석·갱신/업그레이드·영수증). */
export default function Account() {
  const [user, setUser] = useState(null)
  const [licenses, setLicenses] = useState(null) // null=loading
  const [err, setErr] = useState(null)

  useEffect(() => onAuthStateChanged(auth, setUser), [])

  useEffect(() => {
    if (!user) { setLicenses(null); return }
    (async () => {
      try {
        const q = query(collection(dbf, 'groups'), where('owner_uid', '==', user.uid))
        const snap = await getDocs(q)
        setLicenses(snap.docs.map(d => ({ key: d.id, ...d.data() })))
      } catch (e) {
        setErr(e.message); setLicenses([])
      }
    })()
  }, [user])

  const login = () => signInWithPopup(auth, new GoogleAuthProvider()).catch(e => setErr(e.message))
  const renew = async (lic) => {
    const idToken = await user.getIdToken()
    const r = await fetch('/api/checkout', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan: lic.plan_id || 'studio', group_name: lic.key, id_token: idToken, renew: true }),
    })
    const data = await r.json().catch(() => ({}))
    if (data.url) window.location.href = data.url
    else setErr(data.error || '갱신 결제를 시작할 수 없습니다 (Stripe 설정 확인)')
  }

  if (!user) {
    return (
      <div className="wrap">
        <div className="section-title">내 라이선스</div>
        <p className="muted">구매한 서버 라이선스를 관리합니다 — 키·갱신·업그레이드·영수증.</p>
        <div className="card"><button className="btn" onClick={login}>Google로 로그인</button></div>
      </div>
    )
  }

  return (
    <div className="wrap">
      <div className="section-title">내 라이선스</div>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <p className="muted">{user.email}</p>
        <button className="btn ghost" onClick={() => signOut(auth)}>로그아웃</button>
      </div>
      {err && <div className="banner warn">{err}</div>}
      {licenses === null && <div className="card">불러오는 중…</div>}
      {licenses && licenses.length === 0 && !err && (
        <div className="card">아직 라이선스가 없습니다. <a className="mono" href="/buy" style={{ color: 'var(--blue)' }}>플랜 보기 →</a></div>
      )}
      {(licenses || []).map(lic => {
        const expired = lic.expires_at && new Date(lic.expires_at) < new Date()
        return (
          <div className="card" key={lic.key}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <div style={{ fontWeight: 700, fontSize: 17 }}>{lic.key}</div>
              <span className="banner" style={{ margin: 0 }} >{lic.plan_id || 'studio'}</span>
            </div>
            <div className="kv"><span className="k">상태</span><span style={{ color: expired ? 'var(--amber)' : 'var(--emerald)' }}>{expired ? '만료됨' : '활성'}</span></div>
            <div className="kv"><span className="k">만료</span><span className="mono">{lic.expires_at || '—'}</span></div>
            <div className="kv"><span className="k">좌석 / 분석기</span><span className="mono">{lic.seat_limit ?? '—'} / {lic.analyzer_limit ?? '—'}</span></div>
            <div className="kv"><span className="k">라이선스 키</span><span className="mono">{lic.license_key || '(앱에 자동 적용)'}</span></div>
            <div className="row" style={{ marginTop: 14, gap: 10 }}>
              <button className="btn" onClick={() => renew(lic)}>갱신 / 업그레이드</button>
              <span className="note">갱신은 앱 접속 없이 여기서 — 결제 즉시 서버 입장이 복구됩니다.</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
