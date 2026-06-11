/**
 * 검색 — 이식 대상 표면 (구 SearchPanel/FileGrid가 이 자리에 들어온다).
 * 여기 정적 마크업은 자리표시자: 레이아웃·배지 문법만 확정하고,
 * 실제 검색 로직은 U4에서 기존 컴포넌트를 이식한다. 재작성 금지.
 */
const RESULTS = [
  { name: 'knight_armor_v3.psd', cap: '푸른 판금 갑옷의 기사, 정면 입상', score: 0.84, axes: ['VV', 'MV'], bg: 'linear-gradient(140deg,#1e3a5f,#2d4a73)' },
  { name: 'new_knight_concept.psd', cap: '방금 추가됨 — 캡션 생성 중', score: 0.71, axes: ['VV'], wait: true, bg: 'linear-gradient(140deg,#27344a,#3a4a66)' },
  { name: 'paladin_sketch.png', cap: '은빛 갑옷 성기사 스케치', score: 0.69, axes: ['MV'], bg: 'linear-gradient(140deg,#1d3146,#28455e)' },
  { name: '기사단_로고_final.psd', cap: '레이어명 일치: knight_emblem', score: 0.66, axes: ['FTS'], bg: 'linear-gradient(140deg,#33272b,#4a3a40)' },
  { name: 'armor_texture_02.png', cap: '청색 금속 질감 타일', score: 0.63, axes: ['VV'], bg: 'linear-gradient(140deg,#1e3a4a,#2a5468)' },
  { name: 'guard_npc_set.psd', cap: '경비병 NPC 3종 세트', score: 0.61, axes: ['VV', 'MV'], bg: 'linear-gradient(140deg,#2a2a3e,#3c3c5c)' },
]

const AXIS_CLASS = { VV: 'b-vv', MV: 'b-mv', FTS: 'b-fts' }

export default function SearchScreen() {
  return (
    <section id="scr-search" className="screen active" style={{ height: '100%' }}>
      <div className="search-head">
        <div className="search-bar">
          <input defaultValue="" placeholder="자연어로 검색 — 예: 푸른 갑옷을 입은 기사 일러스트" />
          <button className="go">검색</button>
        </div>
        <div className="chips">
          <span className="chip">+ 조건 추가</span>
        </div>
      </div>
      <div className="results">
        {RESULTS.map(r => (
          <div className="card" key={r.name}>
            <div className="thumb" style={{ background: r.bg }}>
              <span className="score">{r.score.toFixed(2)}</span>
              <span className="badges">
                {r.wait && <span className="badge b-wait">분석 대기</span>}
                {r.axes.map(a => <span key={a} className={`badge ${AXIS_CLASS[a]}`}>{a}</span>)}
              </span>
            </div>
            <div className="meta">
              <div className="name">{r.name}</div>
              <div className="cap" style={r.wait ? { color: 'var(--amber)' } : undefined}>{r.cap}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
