import { useLocale } from '../i18n'
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

function foldersToCards(folders, t) {
  return folders.map(f => ({
    id: f.path, name: f.name, path: f.path, info: t('v2.add.count_files', { n: f.total.toLocaleString() }),
    tag: f.fullyDone ? ['done', t('v2.add.tag_done')] : f.pct > 0 ? ['done', t('v2.add.tag_pct', { pct: f.pct })] : ['new', t('v2.add.tag_new')], mosaic: null,
  }))
}

export default function AddFlow({ onClose }) {
  const { t } = useLocale()
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

  const cards = disconnected ? [] : foldersToCards(folders, t)
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
          <h3>{t('v2.add.title')}</h3>
        </div>
        <div className="steps">{[1, 2, 3, 4].map(i => <i key={i} className={i <= BAR[view] ? 'on' : ''} />)}</div>

        {view === VIEWS.SOURCE && (
          <div>
            <p style={{ fontSize: 11, color: 'var(--faint)', marginBottom: 10 }}>
              {t('v2.add.folder_unit')} <b style={{ color: 'var(--text)' }}>{t('v2.add.folder_unit_b')}</b>{t('v2.add.folder_unit_tail')}
            </p>
            <button className="src-opt" onClick={() => openLive('local')}>
              💻<div><div className="t">{t('v2.add.opt_local')}</div><div className="d">{t('v2.add.opt_local_d')}</div></div>
            </button>
            <button className="src-opt" onClick={() => navTo(VIEWS.NAS)}>
              🌐<div><div className="t">{t('v2.add.opt_nas')}</div><div className="d">{t('v2.add.opt_nas_d')}</div></div>
            </button>
            <button className="src-opt" onClick={() => navTo(VIEWS.BROWSE)}>
              ♻️<div><div className="t">{t('v2.add.opt_known')}</div><div className="d">{t('v2.add.opt_known_d')}</div></div>
            </button>
            <div className="m-acts"><button className="sec" onClick={onClose}>{t('v2.add.cancel')}</button></div>
          </div>
        )}

        {view === VIEWS.NAS && <NasView onBrowse={(id) => openLive('webdav', id)} disconnected={disconnected} />}

        {view === VIEWS.LIVE && (
          <LiveBrowser kind={browse.kind} sourceId={browse.sourceId} registering={register.isPending} onRegister={doRegister} />
        )}

        {view === VIEWS.BROWSE && (
          <div>
            {!disconnected && <p style={{ fontSize: 10.5, color: 'var(--faint)', margin: '0 0 8px' }}>{t('v2.add.opt_known_d')}</p>}
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
              {cards.length === 0 && <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: 12 }}>{t('v2.add.no_known')}</div>}
            </div>
            <div className="pick-bar">
              <span className="sel-name">📁 {pickedCard?.name || t('v2.add.pick_folder')}</span>
              <span style={{ flex: 1 }} />
              {register.isError && <span style={{ fontSize: 10.5, color: 'var(--red)', marginRight: 'auto' }}>{t('v2.add.register_failed')} {register.error?.message || t('v2.add.server_error')}</span>}
              <button className="sec" onClick={() => navTo(VIEWS.DETAIL)}>{t('v2.add.detail_options')}</button>
              <button className="pri" disabled={!pickedCard || register.isPending} onClick={onRegisterCard}>{register.isPending ? t('v2.add.registering') : t('v2.add.register')}</button>
            </div>
          </div>
        )}

        {view === VIEWS.DETAIL && (
          <div>
            <div className="frow">
              <label>{t('v2.add.types_label')} <span style={{ color: 'var(--faint)' }}>{t('v2.add.types_hint')}</span></label>
              <TypeChips value={types} onChange={setTypes} />
            </div>
            <div className="frow">
              <label>{t('v2.add.scope')}</label>
              <label className="radio"><input type="radio" name="md" checked={!fullReanalyze} onChange={() => setFullReanalyze(false)} /> <b>{t('v2.add.scope_new')}</b> {t('v2.add.scope_new_d')}</label>
              <label className="radio"><input type="radio" name="md" checked={fullReanalyze} onChange={() => setFullReanalyze(true)} /> <b>{t('v2.add.scope_all')}</b> {t('v2.add.scope_all_d')}</label>
            </div>
            <div className="frow">
              <label className="chk"><input type="checkbox" checked={priority} onChange={e => setPriority(e.target.checked)} /> {t('v2.add.priority')}</label>
            </div>
            <div className="m-acts">
              {register.isError && <span style={{ fontSize: 10.5, color: 'var(--red)', marginRight: 'auto' }}>{t('v2.add.register_failed')} {register.error?.message || t('v2.add.server_error')}</span>}
              <button className="pri" disabled={!pickedCard || register.isPending} onClick={onRegisterCard}>{register.isPending ? t('v2.add.registering') : t('v2.add.register')}</button>
            </div>
          </div>
        )}

        {view === VIEWS.DONE && (
          <div>
            <div className="done-box">
              <div className="ic">✅</div>
              <h3 style={{ marginTop: 6 }}>{t('v2.add.done_title')}</h3>
              <p>{t('v2.add.done_body', { n: Number(registeredTotal).toLocaleString() })}<b>{t('v2.add.done_body_b')}</b>{t('v2.add.done_body_tail')}<br />{t('v2.add.done_body2')}<b>{t('v2.add.done_body2_b')}</b></p>
            </div>
            <div className="m-acts">
              <button className="sec" onClick={onClose}>{t('v2.add.close')}</button>
              <button className="pri" onClick={() => { onClose(); navigate('/analysis') }}>{t('v2.add.view_in_analysis')}</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/** NAS 소스 목록 + 새 소스 연결(테스트/추가) */
function NasView({ onBrowse, disconnected }) {
  const { t } = useLocale()
  const { sources } = useSources()
  const { add, test } = useSourceMutations()
  const [form, setForm] = useState({ id: '', url: '', username: '', password: '', verify_ssl: true })
  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))
  const testMsg = test.data ? (test.data.success ? `✓ ${test.data.message}` : `✕ ${test.data.message}`) : test.isPending ? t('v2.add.testing') : null

  return (
    <div>
      {sources.map(s => (
        <div className="nas-item" key={s.id}>
          🌐<div><div className="t">{s.id}</div><div className="d">{s.url} · {t('v2.add.connected')}</div></div>
          <button className="browse" onClick={() => onBrowse(s.id)}>{t('v2.add.browse')}</button>
        </div>
      ))}
      {sources.length === 0 && (
        <div className="nas-item" style={{ opacity: .6 }}>
          🌐<div><div className="t">{t('v2.add.no_nas')}</div><div className="d">{t('v2.add.no_nas_d')}{disconnected ? t('v2.add.no_nas_d_off') : ''}</div></div>
        </div>
      )}
      <div className="nas-form">
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>{t('v2.add.new_nas')}</div>
        <div className="fr"><input placeholder={t('v2.add.nas_id')} value={form.id} onChange={set('id')} /></div>
        <div className="fr"><input placeholder="주소 — 예: https://nas.local:5006 (WebDAV)" value={form.url} onChange={set('url')} /></div>
        <div className="fr"><input placeholder={t('v2.add.nas_user')} value={form.username} onChange={set('username')} /><input type="password" placeholder={t('v2.add.nas_pw')} value={form.password} onChange={set('password')} /></div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button className="sec" style={{ border: '1px solid var(--line)', borderRadius: 6, padding: '5px 14px', fontSize: 11.5 }}
            disabled={!form.url || test.isPending} onClick={() => test.mutate(form)}>{t('v2.add.test_connection')}</button>
          <button className="pri" style={{ padding: '5px 14px', fontSize: 11.5 }}
            disabled={!form.id || !form.url || add.isPending} onClick={() => add.mutate(form)}>{add.isPending ? t('v2.add.adding') : t('v2.add.add')}</button>
          {testMsg && <span style={{ fontSize: 10.5, color: test.data?.success ? 'var(--emerald)' : 'var(--red)' }}>{testMsg}</span>}
          {add.isError && <span style={{ fontSize: 10.5, color: 'var(--red)' }}>{t('v2.add.add_failed')}</span>}
        </div>
      </div>
    </div>
  )
}

/** 라이브 폴더 탐색기 — 클릭 드릴다운(경로 타이핑 없음) */
function LiveBrowser({ kind, sourceId, registering, onRegister }) {
  const { t } = useLocale()
  const initial = kind === 'webdav' ? '/' : ''
  const [stack, setStack] = useState([initial])
  const path = stack[stack.length - 1]
  const { folders, loading, error } = useBrowse({ kind, sourceId, path })

  const atRoot = stack.length === 1
  const isRootsList = kind === 'local' && path === ''   // 화이트리스트 루트 목록(가짜 폴더)
  const canRegister = !isRootsList
  const folderPath = kind === 'webdav' ? `webdav://${sourceId}${path}` : path
  const leaf = path.split('/').filter(Boolean).pop() || (kind === 'webdav' ? sourceId : t('v2.add.root'))

  return (
    <div>
      <div className="crumb" style={{ marginBottom: 8 }}>
        <button disabled={atRoot} onClick={() => setStack(s => s.length > 1 ? s.slice(0, -1) : s)}>{t('v2.add.up')}</button>
        <span className="sep">▸</span>
        <b style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{isRootsList ? t('v2.add.pick_start') : (path || '/')}</b>
        <span style={{ marginLeft: 'auto', fontSize: 10.5, color: 'var(--faint)' }}>{loading ? t('v2.common.loading') : t('v2.add.folder_count', { n: folders.length })}</span>
      </div>
      {error && <div style={{ fontSize: 11, color: 'var(--red)', padding: '8px 2px' }}>{error} {t('v2.add.browse_error')}</div>}
      <div className="fgrid" style={{ maxHeight: 320 }}>
        {folders.map(f => (
          <div key={f.path} className="fcard" onClick={() => setStack(s => [...s, f.path])} title={t('v2.add.open')}>
            <div className="mosaic noprev">📁</div>
            <div className="fname">{f.name}</div>
          </div>
        ))}
        {!loading && !error && folders.length === 0 && (
          <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: 12 }}>{t('v2.add.no_subfolders')}</div>
        )}
      </div>
      <div className="pick-bar">
        <span className="sel-name">📁 {isRootsList ? t('v2.add.open_to_pick') : leaf}</span>
        <span style={{ flex: 1 }} />
        <button className="pri" disabled={!canRegister || registering} onClick={() => onRegister(folderPath, leaf)}>
          {registering ? t('v2.add.registering') : t('v2.add.register_this')}
        </button>
      </div>
    </div>
  )
}

// 화면 라벨(한글) ↔ 백엔드 analysis_profile.expected_types 정규 슬러그.
const TYPE_OPTIONS = [
  { key: 'v2.type.illustration', value: 'illustration' },
  { key: 'v2.type.character',    value: 'character' },
  { key: 'v2.type.background',   value: 'background' },
  { key: 'v2.type.item',         value: 'item' },
  { key: 'v2.type.effect',       value: 'effect' },
  { key: null, label: 'UI',      value: 'ui_element' },
  { key: 'v2.type.icon',         value: 'icon' },
  { key: 'v2.type.texture',      value: 'texture' },
]

function TypeChips({ value, onChange }) {
  const { t } = useLocale()
  return (
    <div className="type-grid">
      {TYPE_OPTIONS.map(o => (
        <span key={o.value} className={`tchip ${value.has(o.value) ? 'on' : ''}`}
          onClick={() => onChange(s => { const n = new Set(s); n.has(o.value) ? n.delete(o.value) : n.add(o.value); return n })}>{o.key ? t(o.key) : o.label}</span>
      ))}
    </div>
  )
}
