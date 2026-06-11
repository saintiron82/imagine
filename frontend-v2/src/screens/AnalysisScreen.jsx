/**
 * 분석 — "무엇이 되고 있나". 작업 리스트 + 지금 처리 중 라이브 모니터.
 * 원칙(확정): 전문 용어 0 (내부 단계명 노출 금지), ETA 없음, 완료 잔류 없음.
 * 화면의 주인공은 잡 메타데이터가 아니라 큐의 내용물이다.
 */
const ANALYZERS = [
  { name: '이 서버', file: 'knight_v3.psd 처리 중', rate: '7.9장/분' },
  { name: '지민-MacBook', file: 'bg_forest_07.psd 처리 중', rate: '78장/분' },
  { name: 'gpu-box-01', file: 'npc_smith.psd 처리 중', rate: '81장/분' },
]

const JOBS = [
  { name: 'NAS / 캐릭터', done: 1820, total: 3100, pct: 59, color: 'var(--blue)', failed: 3, actions: ['일시정지', '취소'] },
  { name: '배경 다시 분석', done: 410, total: 700, pct: 59, color: 'var(--purple)', actions: ['일시정지', '취소'] },
  { name: '신규 컨셉 2026-06', waiting: true, total: 412, actions: ['⚡ 우선 처리', '취소'] },
]

const FLOW = [
  { name: 'knight_v3.psd', busy: true, bg: 'linear-gradient(140deg,#1e3a5f,#2d4a73)' },
  { name: 'npc_smith.psd', busy: true, bg: 'linear-gradient(140deg,#33272b,#4a3a40)' },
  { name: 'bg_forest_07.psd', busy: true, bg: 'linear-gradient(140deg,#27344a,#3a4a66)' },
  { name: 'armor_silver.png', bg: 'linear-gradient(140deg,#1d3146,#28455e)' },
  { name: 'guard_set.psd', bg: 'linear-gradient(140deg,#2a2a3e,#3c3c5c)' },
  { name: 'tex_stone.png', bg: 'linear-gradient(140deg,#1e3a4a,#2a5468)' },
  { name: 'villager_12.psd', bg: 'linear-gradient(140deg,#3a2a3e,#503a56)' },
  { name: 'prop_chest.psd', bg: 'linear-gradient(140deg,#2e3a2a,#465640)' },
]

export default function AnalysisScreen() {
  return (
    <section id="scr-analysis" className="screen active">
      <div style={{ maxWidth: 880, margin: '0 auto', padding: '20px 24px' }}>
        <div className="sec-title">분석 리스트 <span className="sub">등록 순이 아니라 작업 간 공정 배분으로 처리</span></div>

        <div className="summary">
          <div className="sum-row">
            <span className="big">2,560</span><span className="of">/ 3,800 장 분석됨</span>
            <div className="bar"><i style={{ width: '67%' }} /></div>
            <span className="mono2" style={{ color: '#93c5fd', fontWeight: 700 }}>67%</span>
          </div>
          <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--dim)' }}>
            지금 <b style={{ color: 'var(--emerald)' }}>분당 14.2장</b>씩 처리 중 · 분석기 <b>3대</b> 참여
          </div>
          <div className="an-strip">
            {ANALYZERS.map(a => (
              <div className="an-card" key={a.name}>
                <span className="st run" />
                <div><div className="nm2">{a.name}</div><div className="sub2">{a.file}</div></div>
                <span className="rate2">{a.rate}</span>
              </div>
            ))}
            <button className="an-card" style={{ flex: '0 0 auto', cursor: 'pointer' }}>
              <span className="sub2">상세·제어 →</span>
            </button>
          </div>
        </div>

        <div className="jobs">
          {JOBS.map(j => (
            <div className={`jrow ${j.waiting ? 'waiting' : ''}`} key={j.name}>
              <span className={j.waiting ? 'st' : 'st run'} style={j.waiting ? { background: 'var(--faint)' } : undefined} />
              <span className="nm">{j.name}</span>
              {!j.waiting && <div className="bar"><i style={{ width: `${j.pct}%`, background: j.color }} /></div>}
              <span className="cnt">{j.waiting ? `${j.total}장 · 대기` : `${j.done.toLocaleString()} / ${j.total.toLocaleString()}`}</span>
              {j.failed && (
                <span className="plain-fail" style={{ margin: 0 }}>
                  <span className="n">실패 {j.failed}</span><button>보기</button>
                </span>
              )}
              <div className="acts2">
                {j.actions.map(a => <button key={a}>{a}</button>)}
              </div>
            </div>
          ))}
        </div>

        <div className="live">
          <div className="live-head">지금 처리 중 <span className="rate">분당 14.2장</span></div>
          <div className="flow">
            {FLOW.map(f => (
              <div className={`fl-item ${f.busy ? 'busy' : ''}`} key={f.name}>
                <div className="ph" style={{ background: f.bg }} />
                <div className="fn">{f.name}</div>
              </div>
            ))}
          </div>
          <div className="plain-fail">
            <span className="n">실패 3건</span>
            <span>그림 파일이 손상되어 읽을 수 없음 1 · 분석 시간 초과(자동 재시도 중) 2</span>
            <button>자세히</button>
          </div>
        </div>
      </div>
    </section>
  )
}
