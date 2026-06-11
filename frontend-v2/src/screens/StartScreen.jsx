import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

/**
 * 시작 화면 — 질문 하나씩: ① 너는 누구냐(로그인) ② 어디에 연결하냐(내 서버).
 * 가입·구매·초대 수락은 전부 홈페이지(계정의 집) — 앱은 로그인만 한다.
 * 재방문자는 ②를 건너뛰고 마지막 서버로 자동 접속한다 (실연동 시).
 */
export default function StartScreen() {
  const [step, setStep] = useState(1)
  const navigate = useNavigate()

  // 서버 만들기 위저드 (계정 라이선스 자동 사용 — 키 타이핑 없음)
  const [create, setCreate] = useState(0) // 0=꺼짐, 1~3=단계

  return (
    <section id="scr-start" className="screen active scr-center" style={{ height: '100vh' }}>
      {create === 0 && step === 1 && (
        <div className="start-card">
          <div className="lg"><span className="dot" />Imagine</div>
          <div className="tag">내 에셋을 자연어로 찾는 검색</div>
          <button className="start-opt" style={{ justifyContent: 'center', gap: 8 }} onClick={() => setStep(2)}>
            <span style={{ fontWeight: 700 }}>G</span><span className="t">Google로 계속하기</span>
          </button>
          <div className="start-div">또는 이메일로</div>
          <input placeholder="이메일" defaultValue="" />
          <input type="password" placeholder="비밀번호" />
          <button className="pri-w" onClick={() => setStep(2)}>로그인</button>
          <div style={{ fontSize: 10.5, color: 'var(--faint)', textAlign: 'center', marginTop: 8 }}>
            계정이 없나요? <span style={{ color: '#93c5fd', cursor: 'pointer' }}>imagine.app에서 가입</span> — 초대 수락·구매도 거기서
          </div>
        </div>
      )}

      {create === 0 && step === 2 && (
        <div className="start-card">
          <div className="lg" style={{ fontSize: 14 }}>내 서버</div>
          <div className="tag">멤버인 서버가 자동으로 표시됩니다</div>
          <button className="start-opt" onClick={() => navigate('/search')}>
            ⚡<div><div className="t">우리팀 라이브러리</div><div className="d">운영자 · 마지막 접속 어제 — 바로 접속</div></div>
          </button>
          <div className="start-div">또는</div>
          <button className="start-opt" onClick={() => setCreate(1)}>
            🖥️<div><div className="t">이 컴퓨터를 서버로 만들기</div><div className="d">계정에 <b style={{ color: '#93c5fd' }}>스튜디오 플랜</b> 보유 — 키 입력 불필요</div></div>
          </button>
          <div style={{ fontSize: 10, color: 'var(--faint)', textAlign: 'center', marginTop: 10 }}>
            팀 초대를 받았다면 메일 링크에서 수락하세요 — 수락 후 로그인하면 여기 나타납니다
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
