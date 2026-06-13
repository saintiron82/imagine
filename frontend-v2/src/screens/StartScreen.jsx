import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../state/AuthContext'

/**
 * 시작 화면 — 질문 하나씩: ① 너는 누구냐(Firebase 로그인) ② 어디에 연결하냐(내 서버).
 * 실인증: Firebase 로그인 → 서버 이름+비번 → connectToServer(/auth/connect).
 * 소프트 게이트: "데모로 둘러보기"로 로그인 없이도 앱을 탐색할 수 있다.
 * 가입·구매·초대 수락은 홈페이지(계정의 집) — 앱은 로그인만 한다.
 */
export default function StartScreen() {
  const navigate = useNavigate()
  const { firebaseUser, authLoading, connected, serverName, busy, error,
    signInEmail, signInGoogle, connectToServer } = useAuth()

  const [step, setStep] = useState(1)
  const [create, setCreate] = useState(0)
  const [email, setEmail] = useState('')
  const [pw, setPw] = useState('')
  const [localErr, setLocalErr] = useState('')

  // 이미 Firebase 로그인돼 있으면 서버 선택으로, 이미 연결돼 있으면 앱으로
  useEffect(() => {
    if (connected) navigate('/search')
    else if (firebaseUser && step === 1) setStep(2)
  }, [firebaseUser, connected]) // eslint-disable-line

  // 서버 연결 폼
  const [srvName, setSrvName] = useState(serverName || '')
  const [srvPw, setSrvPw] = useState('')
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
      {create === 0 && step === 1 && (
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
            계정이 없나요? <span style={{ color: '#93c5fd', cursor: 'pointer' }}>imagine.app에서 가입</span> — 초대 수락·구매도 거기서
          </div>
          <div style={{ textAlign: 'center', marginTop: 12 }}>
            <span style={{ fontSize: 10.5, color: 'var(--faint)', cursor: 'pointer' }} onClick={() => navigate('/search')}>로그인 없이 데모로 둘러보기 →</span>
          </div>
        </div>
      )}

      {create === 0 && step === 2 && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>내 서버</div>
          <div className="tag">{firebaseUser?.email ? `${firebaseUser.email} 로 로그인됨` : '서버 이름과 비밀번호로 접속'}</div>
          <input placeholder="서버 이름 — 예: 우리팀 라이브러리" value={srvName} onChange={e => setSrvName(e.target.value)} />
          <input type="password" placeholder="서버 비밀번호" value={srvPw} onChange={e => setSrvPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doConnect()} />
          {error && <div style={{ fontSize: 10.5, color: 'var(--red)', marginBottom: 8 }}>{error}</div>}
          <button className="pri-w" disabled={busy || !srvName.trim() || !srvPw} onClick={doConnect}>{busy ? '접속 중…' : '접속'}</button>
          <div className="start-div">또는</div>
          <button className="start-opt" onClick={() => setCreate(1)}>
            🖥️<div><div className="t">이 컴퓨터를 서버로 만들기</div><div className="d">계정에 <b style={{ color: '#93c5fd' }}>스튜디오 플랜</b> 보유 — 키 입력 불필요</div></div>
          </button>
          <div style={{ fontSize: 10, color: 'var(--faint)', textAlign: 'center', marginTop: 10 }}>
            팀 초대를 받았다면 메일 링크에서 수락하세요 — 수락 후 로그인하면 접속됩니다
          </div>
        </div>
      )}

      {create === 1 && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>서버 만들기 <span className="faint" style={{ fontSize: 10 }}>1 / 3</span></div>
          <div className="tag">이 컴퓨터가 팀의 라이브러리 서버가 됩니다</div>
          <input defaultValue="우리팀 라이브러리" />
          <div style={{ fontSize: 10, color: 'var(--emerald)', margin: '-4px 0 8px 2px' }}>✓ 사용 가능한 이름</div>
          <input type="password" placeholder="서버 관리 비밀번호" />
          <div style={{ fontSize: 10, color: 'var(--faint)', margin: '-4px 0 8px 2px' }}>서버 초기화 등 관리 작업에만 사용 — 멤버는 이메일 초대로</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'rgba(59,130,246,.07)', border: '1px solid rgba(59,130,246,.3)', borderRadius: 7, padding: '9px 12px', marginBottom: 8, fontSize: 11.5 }}>
            ✓ <b>스튜디오 플랜</b> 사용 — 좌석 10 · 분석기 5 <span className="faint">(계정 구매분)</span>
          </div>
          <div style={{ fontSize: 9.5, color: 'var(--amber)', marginBottom: 8 }}>※ 서버 생성(server/firebase-init)은 백엔드 연동 예정 — 현재 위저드는 미리보기</div>
          <button className="pri-w" onClick={() => setCreate(2)}>다음</button>
        </div>
      )}

      {create === 2 && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>무엇을 보관하나요? <span className="faint" style={{ fontSize: 10 }}>2 / 3</span></div>
          <div className="tag">분류·태그 기준이 됩니다 — 나중에 설정 &gt; 서버에서 변경 가능</div>
          <button className="start-opt" style={{ borderColor: 'var(--blue)' }} onClick={() => setCreate(3)}>
            🎮<div><div className="t">게임 에셋</div><div className="d">캐릭터·배경·UI·이펙트·아이콘 …</div></div>
          </button>
          <button className="start-opt" onClick={() => setCreate(3)}>
            🎨<div><div className="t">일반 일러스트/디자인</div><div className="d">일러스트·사진·시안·레퍼런스</div></div>
          </button>
          <button className="start-opt" onClick={() => setCreate(3)}>
            📐<div><div className="t">기본</div><div className="d">범용 분류 — 가장 단순</div></div>
          </button>
        </div>
      )}

      {create === 3 && (
        <div className="start-card">
          <div className="done-box">
            <div className="ic">🖥️✅</div>
            <h3 style={{ marginTop: 6 }}>서버가 켜졌습니다</h3>
            <p>우리팀 라이브러리 — 어디서나 접속됩니다<br /><b>주소를 알릴 필요 없음</b> · 팀원은 이메일 초대로 참여합니다</p>
          </div>
          <div style={{ display: 'flex', gap: 8, margin: '10px 0' }}>
            <input placeholder="팀원 이메일 — 쉼표로 여러 명" style={{ flex: 1, marginBottom: 0 }} />
            <button style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 7, padding: '0 16px', fontSize: 11.5, color: '#93c5fd' }}>초대 보내기</button>
          </div>
          <button className="pri-w" onClick={() => navigate('/search')}>시작하기 — 첫 폴더를 추가해 보세요</button>
        </div>
      )}
    </section>
  )
}
