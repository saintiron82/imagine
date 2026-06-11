import { useApp } from '../state/AppContext'

/**
 * 설정 — 3스코프: 나 / 이 컴퓨터 / 서버.
 * 원칙: 항목은 한 곳에만, 스코프 태그로 "누구에게 적용되는가"를 구조로 말한다.
 * 서버 스코프는 운영자에게만 보인다.
 */
export default function SettingsScreen() {
  const { isOperator } = useApp()

  return (
    <section id="scr-settings" className="screen active">
      <div className="wrap">
        <div className="scope">
          <h3>나 <span className="scope-tag t-me">나에게만 적용</span></h3>
          <p>이 계정의 화면 표시 방식</p>
          <div className="srow"><div className="lab">언어</div><select defaultValue="한국어"><option>한국어</option><option>English</option></select></div>
          <div className="srow"><div className="lab">그리드 밀도<div className="d">검색 결과 카드 크기</div></div><select defaultValue="기본"><option>기본</option><option>조밀</option></select></div>
        </div>

        <div className="scope">
          <h3>이 컴퓨터 <span className="scope-tag t-pc">이 머신의 기여</span></h3>
          <p>이 컴퓨터가 분석에 참여하는 방식 — 누구나 자기 머신을 제공할 수 있습니다</p>
          <div className="srow"><div className="lab">분석 참여<div className="d">내 컴퓨터를 분석기로 제공</div></div><span className="toggle" /></div>
          <div className="srow"><div className="lab">활동 시간<div className="d">이 시간에만 분석 (예: 야간)</div></div><input type="text" defaultValue="22:00 – 07:00" style={{ width: 120 }} /></div>
          <div className="srow"><div className="lab">AI 모델 등급<div className="d">VRAM에 따라 자동 감지</div></div><select defaultValue="자동 (pro)"><option>자동 (pro)</option><option>standard</option></select></div>
          <div className="srow"><div className="lab">벤치마크<div className="d">마지막: S등급 · MC 7.9/m</div></div><div className="row-act"><button>다시 측정</button></div></div>
        </div>

        {isOperator && (
          <div className="scope">
            <h3>서버 <span className="scope-tag t-srv">전원에게 적용 · 운영자</span></h3>
            <p>이 라이브러리를 쓰는 모두에게 영향을 주는 정책</p>
            <div className="srow"><div className="lab">분석 도메인<div className="d">게임 에셋 — 분류·태그 기준</div></div><div className="row-act"><button>변경…</button></div></div>
            <div className="srow"><div className="lab">자동 분석 정책<div className="d">관리 &gt; 분석기에서 제어</div></div><div className="row-act"><button>이동 →</button></div></div>
            <div className="srow"><div className="lab">LAN 공개<div className="d">같은 네트워크에서 접속 허용</div></div><span className="toggle" /></div>
          </div>
        )}
      </div>
    </section>
  )
}
