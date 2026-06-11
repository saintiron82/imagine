import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

/**
 * + 추가 — 분석 작업 등록 플로우.
 * 원칙(확정): 폴더 지정형(파일 단위 없음) · 경로 타이핑 금지(시각 탐색) ·
 * 등록 의미론("시작"이 아니라 분석 리스트에 등록) · 뒤로 = 실제 지나온 길의
 * 역순(이력 스택) · 기본값 2클릭 완주 + 세부는 점진 공개.
 */
const VIEWS = { SOURCE: 'source', NAS: 'nas', BROWSE: 'browse', DETAIL: 'detail', DONE: 'done' }
const BAR = { source: 1, nas: 1, browse: 2, detail: 3, done: 4 }

const FOLDER_CARDS = [
  { id: 'concept', name: '신규 컨셉 2026-06', info: '412개 파일', tag: ['new', '미분석'], mosaic: null },
  { id: 'chars', name: '캐릭터', info: '3,100개', tag: ['done', '분석됨 98%'], mosaic: ['#1e3a5f,#2d4a73', '#33272b,#4a3a40', '#1d3146,#28455e', '#2a2a3e,#3c3c5c'] },
  { id: 'bg', name: '배경', info: '880개', tag: ['done', '분석 41%'], mosaic: ['#1e3a4a,#2a5468', '#27344a,#3a4a66', null, null] },
  { id: 'ref', name: '레퍼런스', info: '96개', tag: ['new', '미분석'], mosaic: null },
  { id: 'trash', name: '폐기예정', info: '12개', tag: null, mosaic: null },
]

export default function AddFlow({ onClose }) {
  const [view, setView] = useState(VIEWS.SOURCE)
  const [stack, setStack] = useState([])
  const [picked, setPicked] = useState('concept')
  const navigate = useNavigate()

  const navTo = v => { setStack(s => [...s, view]); setView(v) }
  const goBack = () => setStack(s => {
    if (!s.length) return s
    setView(s[s.length - 1])
    return s.slice(0, -1)
  })
  const showBack = stack.length > 0 && view !== VIEWS.DONE

  const pickedName = FOLDER_CARDS.find(c => c.id === picked)?.name

  return (
    <div className="overlay open" onClick={e => { if (e.target.classList.contains('overlay')) onClose() }}>
      <div className="modal">
        <div className="m-head">
          <button className={`m-back ${showBack ? '' : 'hide'}`} onClick={goBack}>←</button>
          <h3>분석 작업 등록</h3>
        </div>
        <div className="steps">
          {[1, 2, 3, 4].map(i => <i key={i} className={i <= BAR[view] ? 'on' : ''} />)}
        </div>

        {view === VIEWS.SOURCE && (
          <div>
            <p style={{ fontSize: 11, color: 'var(--faint)', marginBottom: 10 }}>
              분석은 <b style={{ color: 'var(--text)' }}>폴더 단위</b>로 지정합니다 — 파일을 하나씩 다루지 않습니다
            </p>
            <button className="src-opt" onClick={() => navTo(VIEWS.BROWSE)}>
              💻<div><div className="t">내 컴퓨터의 폴더</div><div className="d">이 컴퓨터에서 폴더 선택 (드래그&드롭 가능)</div></div>
            </button>
            <button className="src-opt" onClick={() => navTo(VIEWS.NAS)}>
              🌐<div><div className="t">NAS 폴더</div><div className="d">등록된 NAS에서 탐색 — 또는 새 NAS 연결</div></div>
            </button>
            <button className="src-opt" onClick={() => navTo(VIEWS.BROWSE)}>
              ♻️<div><div className="t">등록된 폴더 다시 분석</div><div className="d">로컬/컨셉아트 — 다시 분석 필요 <span style={{ color: 'var(--red)' }}>●</span></div></div>
            </button>
            <div className="m-acts"><button className="sec" onClick={onClose}>취소</button></div>
          </div>
        )}

        {view === VIEWS.NAS && (
          <div>
            <div className="nas-item">
              🌐<div><div className="t">synology-main</div><div className="d">https://nas.local:5006 · 연결됨 · 폴더 4개 사용 중</div></div>
              <button className="browse" onClick={() => navTo(VIEWS.BROWSE)}>탐색</button>
            </div>
            <div className="nas-form">
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>＋ 새 NAS 연결</div>
              <div className="fr"><input placeholder="주소 — 예: https://nas.local:5006 (WebDAV)" /></div>
              <div className="fr"><input placeholder="계정" /><input type="password" placeholder="비밀번호" /></div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <button className="sec" style={{ border: '1px solid var(--line)', borderRadius: 6, padding: '5px 14px', fontSize: 11.5 }}>연결 테스트</button>
              </div>
            </div>
          </div>
        )}

        {view === VIEWS.BROWSE && (
          <div>
            <div className="browse-wrap">
              <div className="btree">
                <div className="btn-node"><span className="tw">▾</span>🌐 synology-main</div>
                <div className="btn-node bt-1"><span className="tw">▾</span>📁 작업분</div>
                <div className="btn-node bt-2 sel"><span className="tw">▾</span>📁 2026</div>
                <div className="btn-node bt-2"><span className="tw">▸</span>📁 2025</div>
                <div className="btn-node bt-1"><span className="tw">▸</span>📁 원화</div>
                <div className="btn-node bt-1"><span className="tw">▸</span>📁 외주수령분</div>
              </div>
              <div className="bpane">
                <div className="crumb">
                  <button>synology-main</button><span className="sep">▸</span><button>작업분</button><span className="sep">▸</span><b>2026</b>
                  <span style={{ marginLeft: 'auto', fontSize: 10.5, color: 'var(--faint)' }}>하위 폴더 5개</span>
                </div>
                <div className="fgrid" style={{ maxHeight: 'none', flex: 1 }}>
                  {FOLDER_CARDS.map(c => (
                    <div key={c.id} className={`fcard ${picked === c.id ? 'sel' : ''}`} onClick={() => setPicked(c.id)}>
                      {c.mosaic ? (
                        <div className="mosaic">
                          {c.mosaic.map((m, i) => (
                            <div key={i} style={m ? { background: `linear-gradient(140deg,${m})` } : undefined} />
                          ))}
                        </div>
                      ) : (
                        <div className="mosaic noprev">📁</div>
                      )}
                      <div className="fname">{c.name}</div>
                      <div className="finfo">{c.info} {c.tag && <span className={`ftag ${c.tag[0]}`}>{c.tag[1]}</span>}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="pick-bar">
              <span className="sel-name">📁 {pickedName}</span>
              <label className="chk" style={{ fontSize: 11.5 }}><input type="checkbox" defaultChecked /> 하위 포함</label>
              <label className="chk" style={{ fontSize: 11.5 }}><input type="checkbox" defaultChecked /> 새 파일 자동 분석</label>
              <span style={{ flex: 1 }} />
              <button className="sec" onClick={() => navTo(VIEWS.DETAIL)}>세부 옵션</button>
              <button className="pri" onClick={() => navTo(VIEWS.DONE)}>작업 등록</button>
            </div>
          </div>
        )}

        {view === VIEWS.DETAIL && (
          <div>
            <div className="frow">
              <label>이 폴더에 주로 들어있는 것 <span style={{ color: 'var(--cyan)' }}>— 폴더명에서 '컨셉' 감지 → 추천 적용됨</span></label>
              <TypeChips />
              <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 4 }}>★ = 대표 유형 · 분류 정확도를 높입니다</div>
            </div>
            <div className="frow">
              <label>범위</label>
              <label className="radio"><input type="radio" name="md" defaultChecked /> <b>새 파일만</b> — 이미 분석된 파일은 건너뜀 (캐시 활용)</label>
              <label className="radio"><input type="radio" name="md" /> <b>전체 다시 분석</b> — 결과를 새로 덮어씀</label>
              <label className="radio"><input type="radio" name="md" /> <b>모델 파도</b> — 새 모델 버전 기준 재처리</label>
            </div>
            <div className="frow">
              <label className="chk"><input type="checkbox" /> ⚡ 우선 처리 — 다른 작업보다 먼저</label>
            </div>
            <div className="m-acts">
              <button className="pri" onClick={() => navTo(VIEWS.DONE)}>작업 등록</button>
            </div>
          </div>
        )}

        {view === VIEWS.DONE && (
          <div>
            <div className="done-box">
              <div className="ic">✅</div>
              <h3 style={{ marginTop: 6 }}>분석 리스트에 등록되었습니다</h3>
              <p>412개 파일 · <b>진행 중인 작업 2개와 공정하게 배분</b>되어 처리됩니다<br />캐시 적중분은 즉시 완료 · <b>파싱되는 대로 검색에 나타납니다</b></p>
            </div>
            <div className="m-acts">
              <button className="sec" onClick={onClose}>닫기</button>
              <button className="pri" onClick={() => { onClose(); navigate('/analysis') }}>분석에서 보기</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function TypeChips() {
  const [on, setOn] = useState(new Set(['일러스트 ★', '캐릭터']))
  const types = ['일러스트 ★', '캐릭터', '배경/BG', '소품', '이펙트', 'UI', '아이콘', '텍스처']
  return (
    <div className="type-grid">
      {types.map(t => (
        <span
          key={t}
          className={`tchip ${on.has(t) ? 'on' : ''}`}
          onClick={() => setOn(s => { const n = new Set(s); n.has(t) ? n.delete(t) : n.add(t); return n })}
        >{t}</span>
      ))}
    </div>
  )
}
