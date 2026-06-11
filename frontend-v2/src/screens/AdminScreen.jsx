import { useState } from 'react'

/**
 * 관리 — 엔진룸. 기술 용어(MC/VV/MV)는 운영자 전용인 이 화면에만 허용된다.
 * 분석기: 병목 진단(명시) + 분석기별 현재/능력. 멤버: 이메일 초대(좌석).
 */
export default function AdminScreen() {
  const [tab, setTab] = useState('workers')

  return (
    <section id="scr-admin" className="screen active" style={{ height: '100%' }}>
      <aside className="adm-side">
        {[['workers', '분석기'], ['members', '멤버'], ['tools', '서버 도구']].map(([id, label]) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>
        ))}
      </aside>
      <div className="adm-main">
        {tab === 'workers' && <WorkersPanel />}
        {tab === 'members' && <MembersPanel />}
        {tab === 'tools' && <ToolsPanel />}
      </div>
    </section>
  )
}

function WorkersPanel() {
  const valves = [
    { ph: 'DL', n: '3,800/3,800', rate: '완료' },
    { ph: '파싱', n: '3,640/3,800', rate: '149/분' },
    { ph: 'MC', n: '2,710/3,800', rate: '88.9/분', bottleneck: true },
    { ph: 'VV', n: '3,205/3,800', rate: '156/분 가능' },
    { ph: 'MV', n: '2,560/3,800', rate: '115/분 가능' },
  ]
  return (
    <>
      <div className="panel">
        <h4>자동 분석 <span className="hint">서버 전역 정책 — 이 토글은 여기 한 곳에만 있습니다</span><span className="toggle" /></h4>
        <div className="bottleneck">
          현재 병목: <b>MC</b> — 대기 1,090장 · 클러스터 MC 처리력 <b>분당 88.9장</b> (이 단계가 전체 완료 속도를 결정)
          <div className="why">이유: MC 가능 분석기 2대뿐 (지민-MacBook은 VRAM 8GB로 불가) · VV·MV는 처리력 여유 — MC 산출을 기다리는 중</div>
        </div>
        <div className="valves">
          {valves.map(v => (
            <div className={`valve ${v.bottleneck ? 'bn' : ''}`} key={v.ph}>
              <div className="ph">{v.ph}{v.bottleneck && <span className="bn-tag">병목</span>}</div>
              <div className="n">{v.n}</div>
              <div className="rate">{v.rate}</div>
              <span className="sw on">ON</span>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <h4>분석기 <span className="hint">3대 온라인 — 정지=이번 세션 종료(재시작 가능) · 차단=재접속 거부</span></h4>
        <table>
          <thead>
            <tr><th>이름</th><th>MC <span style={{ fontWeight: 400 }}>현재/능력</span></th><th>VV <span style={{ fontWeight: 400 }}>현재/능력</span></th><th>MV <span style={{ fontWeight: 400 }}>현재/능력</span></th><th>발휘율</th><th style={{ textAlign: 'right' }}>제어</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>이 서버</b><div style={{ fontSize: 10, color: 'var(--faint)' }}>자동 · server-local · 벤치 A</div></td>
              <td><span className="cap full"><span className="cur">7.9</span><span className="max"> / 7.9</span><span className="note">만재 — 더 못 냄</span></span></td>
              <td><span className="cap idle"><span className="cur">0</span><span className="max"> / 95</span><span className="note">MC에 배정됨</span></span></td>
              <td><span className="cap idle"><span className="cur">0</span><span className="max"> / 130</span><span className="note">MC에 배정됨</span></span></td>
              <td><div className="util"><div className="ub"><i style={{ width: '100%' }} /></div><div className="ut">100%</div></div></td>
              <td><div className="row-act"><button>정지</button></div></td>
            </tr>
            <tr>
              <td><b>지민-MacBook</b><div style={{ fontSize: 10, color: 'var(--faint)' }}>client · electron · 벤치 B</div></td>
              <td><span className="cap"><span className="cur" style={{ color: 'var(--red)' }}>✕ 불가</span><span className="note">VRAM 8GB &lt; 필요 12GB</span></span></td>
              <td><span className="cap full"><span className="cur">78</span><span className="max"> / 78</span><span className="note">만재</span></span></td>
              <td><span className="cap"><span className="cur">37</span><span className="max"> / 115</span><span className="note">VV와 교대 중</span></span></td>
              <td><div className="util"><div className="ub"><i style={{ width: '96%' }} /></div><div className="ut">96%</div></div></td>
              <td><div className="row-act"><button>정지</button><button className="danger">차단</button></div></td>
            </tr>
            <tr>
              <td><b>gpu-box-01</b><div style={{ fontSize: 10, color: 'var(--faint)' }}>headless · cli · <span style={{ color: 'var(--amber)' }}>MC 핀</span> · 벤치 S</div></td>
              <td><span className="cap full"><span className="cur">81</span><span className="max"> / 81</span><span className="note">만재 (병목 단계 전담)</span></span></td>
              <td><span className="cap idle"><span className="cur">0</span><span className="max"> / 210</span><span className="note">핀으로 미사용</span></span></td>
              <td><span className="cap idle"><span className="cur">0</span><span className="max"> / 300</span><span className="note">핀으로 미사용</span></span></td>
              <td><div className="util"><div className="ub"><i style={{ width: '100%' }} /></div><div className="ut">100%</div></div></td>
              <td><div className="row-act"><button>핀 해제</button><button>정지</button><button className="danger">차단</button></div></td>
            </tr>
          </tbody>
        </table>
        <div className="blocked">
          ⛔ <b>old-laptop</b> <span className="faint">— 차단됨 (재접속 불가)</span>
          <button className="unb">차단 해제</button>
        </div>
      </div>

      <div className="panel">
        <h4>파생물 캐시 <span className="hint">같은 내용은 다시 계산하지 않습니다</span></h4>
        <div className="kpis">
          <div className="kpi"><div className="v" style={{ color: 'var(--emerald)' }}>312</div><div className="k">이번 잡 캐시 적중</div></div>
          <div className="kpi"><div className="v">17,726</div><div className="k">해시 보유 / 17,726 (백필 완료)</div></div>
          <div className="kpi"><div className="v" style={{ color: '#93c5fd' }}>3</div><div className="k">활성 모델 버전</div></div>
          <div className="kpi"><div className="v" style={{ color: 'var(--purple)' }}>파도 ▸</div><div className="k">모델 교체 재처리 시작</div></div>
        </div>
      </div>
    </>
  )
}

function MembersPanel() {
  return (
    <>
      <div className="panel">
        <h4>플랜 <span className="hint">서버 단위 라이선스 — 한도는 여기서 옴</span></h4>
        <div className="kpis">
          <div className="kpi"><div className="v" style={{ color: '#93c5fd' }}>스튜디오</div><div className="k">연 라이선스 · 2027-06-12 만료</div></div>
          <div className="kpi"><div className="v">5 <span className="faint" style={{ fontSize: 12 }}>/ 10</span></div><div className="k">좌석 — 멤버 3 + 대기 초대 2</div></div>
          <div className="kpi"><div className="v">3 <span className="faint" style={{ fontSize: 12 }}>/ 5</span></div><div className="k">분석기</div></div>
          <div className="kpi" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <button style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '7px 16px', borderRadius: 6 }}>갱신 / 업그레이드</button>
          </div>
        </div>
        <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 8 }}>사용 현황 보고: 멤버·분석기 수, 월 분석량 — 수치만 전송되며 파일 내용·이름은 절대 포함되지 않습니다</div>
        <div style={{ fontSize: 10, color: 'var(--amber)', marginTop: 4 }}>만료 시 이 서버 접속이 차단됩니다 (로그인은 되어도 입장 불가) · 데이터는 보존되며 갱신 즉시 복귀</div>
      </div>

      <div className="panel">
        <h4>초대 <span className="hint">이메일로 초대 → 메일의 링크로 수락 → 좌석 점유 · 초대는 그 이메일로만 수락 가능</span></h4>
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <input placeholder="이메일 주소 — 쉼표로 여러 명" style={{ flex: 1, background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 12px', color: 'var(--text)', fontSize: 12 }} />
          <button style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '0 18px', borderRadius: 6 }}>초대 보내기</button>
        </div>
        <table>
          <thead><tr><th>대기 중 초대</th><th>보냄</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            {[['minsu@studio.kr', '2일 전'], ['art-extern@partner.co', '5시간 전']].map(([email, when]) => (
              <tr key={email}>
                <td>{email} <span style={{ fontSize: 9, color: 'var(--amber)' }}>좌석 예약 중</span></td>
                <td className="mono">{when}</td>
                <td><div className="row-act"><button>재발송</button><button className="danger">취소</button></div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h4>멤버 <span className="hint">제거 = 좌석 반환 · 비활성 = 접속만 차단(좌석 유지)</span></h4>
        <table>
          <thead><tr><th>이름</th><th>역할</th><th>가입 경로</th><th>마지막 접속</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            <tr>
              <td>성철<div style={{ fontSize: 9.5, color: 'var(--faint)' }}>saintiron82@gmail.com</div></td>
              <td><span className="badge" style={{ background: 'rgba(192,132,252,.15)', color: 'var(--purple)' }}>운영자</span></td>
              <td className="mono" style={{ fontSize: 10 }}>서버 생성자</td><td className="mono">방금</td><td />
            </tr>
            <tr>
              <td>지민<div style={{ fontSize: 9.5, color: 'var(--faint)' }}>jimin@studio.kr</div></td>
              <td><span className="badge b-ok">사용자</span></td>
              <td className="mono" style={{ fontSize: 10 }}>초대 수락</td><td className="mono">2시간 전</td>
              <td><div className="row-act"><button>역할</button><button className="danger">비활성</button></div></td>
            </tr>
          </tbody>
        </table>
      </div>
    </>
  )
}

function ToolsPanel() {
  const rows = [
    ['외부 접속 (Cloudflare 터널)', '● 연결됨 — ourteam.imagine.app · 주소 공유 불필요', ['재연결'], 'var(--emerald)'],
    ['같은 네트워크 (LAN)', '192.168.0.5:8000 · 빠른 경로 (자동 발견)', []],
    ['정합성 감사', '미완성/잔여 작업 검사·복구', ['실행']],
    ['해시 백필', '파생물 캐시 자격 부여 (~16KB/파일)', ['실행']],
    ['DB 내보내기 / 가져오기', '', ['내보내기', '가져오기']],
    ['DB 초기화', '비밀번호 확인 필요', ['초기화…'], 'var(--red)', true],
  ]
  return (
    <div className="panel">
      <h4>서버 도구 <span className="hint">DB·정합성·연결 — 구 헤더 DB 메뉴가 여기로</span></h4>
      <table>
        <tbody>
          {rows.map(([name, desc, actions, color, danger]) => (
            <tr key={name}>
              <td style={danger ? { color: 'var(--red)' } : undefined}>{name}</td>
              <td style={{ color: color || 'var(--faint)', fontSize: 11 }}>{desc}</td>
              <td><div className="row-act">{actions.map(a => <button key={a} className={danger ? 'danger' : ''}>{a}</button>)}</div></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
