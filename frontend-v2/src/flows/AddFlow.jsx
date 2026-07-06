import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useFolders } from '../api/folders'
import { useRegisterJob } from '../api/jobs'
import { useSources, useSourceMutations, useBrowse } from '../api/browse'

/**
 * + 추가 — 분석 작업 등록 플로우.
 * 원칙(확정): 폴더 지정형(파일 단위 없음) · 경로 타이핑 금지(시각 탐색) ·
 * 등록 의미론(분석 리스트에 등록) · 뒤로 = 이력 스택 역순.
 *
 * 백엔드 연동:
 * - 등록 = POST /api/v1/discover/scan
 * - 신규 소스 시각 탐색 = /api/v1/browse/{webdav,local} (IMGV2-13) — 클릭 드릴다운
 * - 소스 관리 = /api/v1/sources (등록/테스트)
 * - "등록된 폴더 다시 분석" = archive/folders 카드(미연결 시 빈 목록)
 */
const VIEWS = { SOURCE: 'source', NAS: 'nas', LIVE: 'live', BROWSE: 'browse', DETAIL: 'detail', DONE: 'done' }
const BAR = { source: 1, nas: 1, live: 2, browse: 2, detail: 3, done: 4 }

function foldersToCards(folders) {
  return folders.map(f => ({
    id: f.path, name: f.name, path: f.path, info: `${f.total.toLocaleString()}개`,
    tag: f.fullyDone ? ['done', '완료'] : f.pct > 0 ? ['done', `분석 ${f.pct}%`] : ['new', '미분석'], mosaic: null,
  }))
}

export default function AddFlow({ onClose }) {
  const { disconnected, folders } = useFolders()
  const register = useRegisterJob()
  const navigate = useNavigate()

  const [view, setView] = useState(VIEWS.SOURCE)
  const [stack, setStack] = useState([])
  const [picked, setPicked] = useState('')
  const [priority, setPriority] = useState(false)
  const [types, setTypes] = useState(new Set())          // 분석 힌트 (expected_types)
  const [fullReanalyze, setFullReanalyze] = useState(false) // 전체 다시 분석
  const [browse, setBrowse] = useState({ kind: null, sourceId: null }) // 라이브 탐색 대상

  const cards = disconnected ? [] : foldersToCards(folders)
  const effId = picked || cards[0]?.id || ''
  const pickedCard = cards.find(c => c.id === effId) || null

  const navTo = v => { setStack(s => [...s, view]); setView(v) }
  const goBack = () => setStack(s => { if (!s.length) return s; setView(s[s.length - 1]); return s.slice(0, -1) })
  const showBack = stack.length > 0 && view !== VIEWS.DONE

  // 등록: 선택 폴더 경로로 실제 discover/scan.
  const doRegister = (folderPath, name) => {
    if (!folderPath) return
    register.mutate({
      folderPath, priority, name,
      analysisProfile: types.size ? { expected_types: [...types], source: 'user' } : null,
      forceReanalyze: fullReanalyze,
    }, { onSuccess: () => navTo(VIEWS.DONE) })
  }
  const onRegisterCard = () => doRegister(pickedCard?.path, pickedCard?.name)
  const registeredTotal = register.data?.total_files ?? 0

  const openLive = (kind, sourceId = null) => { setBrowse({ kind, sourceId }); navTo(VIEWS.LIVE) }

  return (
    <div className="overlay open" onClick={e => { if (e.target.classList.contains('overlay')) onClose() }}>
      <div className="modal">
        <div className="m-head">
          <button className={`m-back ${showBack ? '' : 'hide'}`} onClick={goBack}>←</button>
          <h3>분석 작업 등록</h3>
        </div>
        <div className="steps">{[1, 2, 3, 4].map(i => <i key={i} className={i <= BAR[view] ? 'on' : ''} />)}</div>

        {view === VIEWS.SOURCE && (
          <div>
            <p style={{ fontSize: 11, color: 'var(--faint)', marginBottom: 10 }}>
              분석은 <b style={{ color: 'var(--text)' }}>폴더 단위</b>로 지정합니다 — 파일을 하나씩 다루지 않습니다
            </p>
            <button className="src-opt" onClick={() => openLive('local')}>
              💻<div><div className="t">내 컴퓨터의 폴더</div><div className="d">서버에서 폴더를 클릭으로 탐색</div></div>
            </button>
            <button className="src-opt" onClick={() => navTo(VIEWS.NAS)}>
              🌐<div><div className="t">NAS 폴더</div><div className="d">등록된 NAS에서 탐색 — 또는 새 NAS 연결</div></div>
            </button>
            <button className="src-opt" onClick={() => navTo(VIEWS.BROWSE)}>
              ♻️<div><div className="t">등록된 폴더 다시 분석</div><div className="d">서버가 아는 폴더에서 선택</div></div>
            </button>
            <div className="m-acts"><button className="sec" onClick={onClose}>취소</button></div>
          </div>
        )}

        {view === VIEWS.NAS && <NasView onBrowse={(id) => openLive('webdav', id)} disconnected={disconnected} />}

        {view === VIEWS.LIVE && (
          <LiveBrowser kind={browse.kind} sourceId={browse.sourceId} registering={register.isPending} onRegister={doRegister} />
        )}

        {view === VIEWS.BROWSE && (
          <div>
            {!disconnected && <p style={{ fontSize: 10.5, color: 'var(--faint)', margin: '0 0 8px' }}>서버가 아는 폴더에서 선택</p>}
            <div className="fgrid" style={{ maxHeight: 360 }}>
              {cards.map(c => (
                <div key={c.id} className={`fcard ${effId === c.id ? 'sel' : ''}`} onClick={() => setPicked(c.id)}>
                  {c.mosaic ? (
                    <div className="mosaic">{c.mosaic.map((m, i) => <div key={i} style={m ? { background: `linear-gradient(140deg,${m})` } : undefined} />)}</div>
                  ) : <div className="mosaic noprev">📁</div>}
                  <div className="fname">{c.name}</div>
                  <div className="finfo">{c.info} {c.tag && <span className={`ftag ${c.tag[0]}`}>{c.tag[1]}</span>}</div>
                </div>
              ))}
              {cards.length === 0 && <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: 12 }}>아직 서버가 아는 폴더가 없습니다</div>}
            </div>
            <div className="pick-bar">
              <span className="sel-name">📁 {pickedCard?.name || '폴더 선택'}</span>
              <span style={{ flex: 1 }} />
              {register.isError && <span style={{ fontSize: 10.5, color: 'var(--red)', marginRight: 'auto' }}>등록 실패 — {register.error?.message || '서버 오류'}</span>}
              <button className="sec" onClick={() => navTo(VIEWS.DETAIL)}>세부 옵션</button>
              <button className="pri" disabled={!pickedCard || register.isPending} onClick={onRegisterCard}>{register.isPending ? '등록 중…' : '작업 등록'}</button>
            </div>
          </div>
        )}

        {view === VIEWS.DETAIL && (
          <div>
            <div className="frow">
              <label>이 폴더에 주로 들어있는 것 <span style={{ color: 'var(--faint)' }}>— 선택 시 분석 힌트로 전달 (선택 사항)</span></label>
              <TypeChips value={types} onChange={setTypes} />
            </div>
            <div className="frow">
              <label>범위</label>
              <label className="radio"><input type="radio" name="md" checked={!fullReanalyze} onChange={() => setFullReanalyze(false)} /> <b>새 파일만</b> — 이미 분석된 파일은 건너뜀 (캐시 활용)</label>
              <label className="radio"><input type="radio" name="md" checked={fullReanalyze} onChange={() => setFullReanalyze(true)} /> <b>전체 다시 분석</b> — 결과를 새로 덮어씀</label>
            </div>
            <div className="frow">
              <label className="chk"><input type="checkbox" checked={priority} onChange={e => setPriority(e.target.checked)} /> ⚡ 우선 처리 — 다른 작업보다 먼저</label>
            </div>
            <div className="m-acts">
              {register.isError && <span style={{ fontSize: 10.5, color: 'var(--red)', marginRight: 'auto' }}>등록 실패 — {register.error?.message || '서버 오류'}</span>}
              <button className="pri" disabled={!pickedCard || register.isPending} onClick={onRegisterCard}>{register.isPending ? '등록 중…' : '작업 등록'}</button>
            </div>
          </div>
        )}

        {view === VIEWS.DONE && (
          <div>
            <div className="done-box">
              <div className="ic">✅</div>
              <h3 style={{ marginTop: 6 }}>분석 리스트에 등록되었습니다</h3>
              <p>{Number(registeredTotal).toLocaleString()}개 파일 · <b>진행 중인 작업과 공정하게 배분</b>되어 처리됩니다<br />캐시 적중분은 즉시 완료 · <b>파싱되는 대로 검색에 나타납니다</b></p>
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

/** NAS 소스 목록 + 새 소스 연결(테스트/추가) */
function NasView({ onBrowse, disconnected }) {
  const { sources } = useSources()
  const { add, test } = useSourceMutations()
  const [form, setForm] = useState({ id: '', url: '', username: '', password: '', verify_ssl: true })
  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))
  const testMsg = test.data ? (test.data.success ? `✓ ${test.data.message}` : `✕ ${test.data.message}`) : test.isPending ? '테스트 중…' : null

  return (
    <div>
      {sources.map(s => (
        <div className="nas-item" key={s.id}>
          🌐<div><div className="t">{s.id}</div><div className="d">{s.url} · 연결됨</div></div>
          <button className="browse" onClick={() => onBrowse(s.id)}>탐색</button>
        </div>
      ))}
      {sources.length === 0 && (
        <div className="nas-item" style={{ opacity: .6 }}>
          🌐<div><div className="t">등록된 NAS 없음</div><div className="d">아래에서 새 NAS를 연결하세요{disconnected ? ' (서버 연결 시)' : ''}</div></div>
        </div>
      )}
      <div className="nas-form">
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>＋ 새 NAS 연결</div>
        <div className="fr"><input placeholder="이름(식별자) — 예: synology-main" value={form.id} onChange={set('id')} /></div>
        <div className="fr"><input placeholder="주소 — 예: https://nas.local:5006 (WebDAV)" value={form.url} onChange={set('url')} /></div>
        <div className="fr"><input placeholder="계정" value={form.username} onChange={set('username')} /><input type="password" placeholder="비밀번호" value={form.password} onChange={set('password')} /></div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button className="sec" style={{ border: '1px solid var(--line)', borderRadius: 6, padding: '5px 14px', fontSize: 11.5 }}
            disabled={!form.url || test.isPending} onClick={() => test.mutate(form)}>연결 테스트</button>
          <button className="pri" style={{ padding: '5px 14px', fontSize: 11.5 }}
            disabled={!form.id || !form.url || add.isPending} onClick={() => add.mutate(form)}>{add.isPending ? '추가 중…' : '추가'}</button>
          {testMsg && <span style={{ fontSize: 10.5, color: test.data?.success ? 'var(--emerald)' : 'var(--red)' }}>{testMsg}</span>}
          {add.isError && <span style={{ fontSize: 10.5, color: 'var(--red)' }}>추가 실패</span>}
        </div>
      </div>
    </div>
  )
}

/** 라이브 폴더 탐색기 — 클릭 드릴다운(경로 타이핑 없음) */
function LiveBrowser({ kind, sourceId, registering, onRegister }) {
  const initial = kind === 'webdav' ? '/' : ''
  const [stack, setStack] = useState([initial])
  const path = stack[stack.length - 1]
  const { folders, loading, error } = useBrowse({ kind, sourceId, path })

  const atRoot = stack.length === 1
  const isRootsList = kind === 'local' && path === ''   // 화이트리스트 루트 목록(가짜 폴더)
  const canRegister = !isRootsList
  const folderPath = kind === 'webdav' ? `webdav://${sourceId}${path}` : path
  const leaf = path.split('/').filter(Boolean).pop() || (kind === 'webdav' ? sourceId : '루트')

  return (
    <div>
      <div className="crumb" style={{ marginBottom: 8 }}>
        <button disabled={atRoot} onClick={() => setStack(s => s.length > 1 ? s.slice(0, -1) : s)}>▲ 상위</button>
        <span className="sep">▸</span>
        <b style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{isRootsList ? '시작 위치 선택' : (path || '/')}</b>
        <span style={{ marginLeft: 'auto', fontSize: 10.5, color: 'var(--faint)' }}>{loading ? '불러오는 중…' : `폴더 ${folders.length}개`}</span>
      </div>
      {error && <div style={{ fontSize: 11, color: 'var(--red)', padding: '8px 2px' }}>{error} — 서버 연결/권한을 확인하세요</div>}
      <div className="fgrid" style={{ maxHeight: 320 }}>
        {folders.map(f => (
          <div key={f.path} className="fcard" onClick={() => setStack(s => [...s, f.path])} title="열기">
            <div className="mosaic noprev">📁</div>
            <div className="fname">{f.name}</div>
          </div>
        ))}
        {!loading && !error && folders.length === 0 && (
          <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: 12 }}>하위 폴더가 없습니다 — 이 폴더를 등록할 수 있습니다</div>
        )}
      </div>
      <div className="pick-bar">
        <span className="sel-name">📁 {isRootsList ? '폴더를 열어 선택' : leaf}</span>
        <span style={{ flex: 1 }} />
        <button className="pri" disabled={!canRegister || registering} onClick={() => onRegister(folderPath, leaf)}>
          {registering ? '등록 중…' : '이 폴더 등록'}
        </button>
      </div>
    </div>
  )
}

// 화면 라벨(한글) ↔ 백엔드 analysis_profile.expected_types 정규 슬러그.
const TYPE_OPTIONS = [
  { label: '일러스트', value: 'illustration' },
  { label: '캐릭터', value: 'character' },
  { label: '배경/BG', value: 'background' },
  { label: '소품', value: 'item' },
  { label: '이펙트', value: 'effect' },
  { label: 'UI', value: 'ui_element' },
  { label: '아이콘', value: 'icon' },
  { label: '텍스처', value: 'texture' },
]

function TypeChips({ value, onChange }) {
  return (
    <div className="type-grid">
      {TYPE_OPTIONS.map(o => (
        <span key={o.value} className={`tchip ${value.has(o.value) ? 'on' : ''}`}
          onClick={() => onChange(s => { const n = new Set(s); n.has(o.value) ? n.delete(o.value) : n.add(o.value); return n })}>{o.label}</span>
      ))}
    </div>
  )
}
