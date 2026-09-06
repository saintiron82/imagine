import { useLocale } from '../i18n'
import { useState, useEffect, useRef } from 'react'
import { useWorkers, useClusterValves, useWorkerControl, useHeadlessCommand, useFeedbackSummary, useAutoProcessing, useLogStream } from '../api/admin'
import { useConnectionInfo } from '../api/connection'
import { useMembersData, useMemberMutations } from '../api/members'
import { useDbAudit, useBackfill, useDbReset, useRepairParse, useDbExport, useDbImport } from '../api/tools'
import { useDomains, useActiveDomain, useDomainDetail, useSetActiveDomain, useSaveDomain, generateDomainPrompt } from '../api/classification'
import { useDbStats } from '../api/analysis'

/**
 * 관리 — 엔진룸. 기술 용어(MC/VV/MV)는 운영자 전용인 이 화면에만 허용된다.
 * 분석기: 병목 진단 + 분석기별 실시간(admin/workers) + 정지/차단/차단해제.
 * 멤버/플랜: members/invites/usage 실데이터. 미연결 시 빈 상태(가짜 데이터 없음).
 */
export default function AdminScreen() {
  const { t } = useLocale()
  const [tab, setTab] = useState('dbstatus')

  return (
    <section id="scr-admin" className="screen active" style={{ height: '100%' }}>
      <aside className="adm-side">
        {[['dbstatus', 'v2.ad.tab_dbstatus'], ['classification', 'v2.ad.tab_classification'], ['members', 'v2.ad.tab_members'], ['feedback', 'v2.ad.tab_feedback'], ['logs', 'v2.ad.tab_logs'], ['tools', 'v2.ad.tab_tools']].map(([id, key]) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{t(key)}</button>
        ))}
      </aside>
      <div className="adm-main">
        {tab === 'dbstatus' && <DbStatusPanel />}
        {tab === 'classification' && <ClassificationPanel />}
        {tab === 'members' && <MembersPanel />}
        {tab === 'feedback' && <FeedbackPanel />}
        {tab === 'logs' && <LogPanel />}
        {tab === 'tools' && <ToolsPanel />}
      </div>
    </section>
  )
}

const ORIGIN_LABEL = (w) => [w.origin, w.launcher].filter(Boolean).join(' · ') || '—'
const PHASE_LABEL = { mc: 'MC', vv: 'VV', mv: 'MV', parse: 'v2.ad.phase_parse', dl: 'DL' }
const phaseText = (m, t) => (PHASE_LABEL[m] || '').startsWith('v2.') ? t(PHASE_LABEL[m]) : (PHASE_LABEL[m] || m)

// 자동 처리 수치 설정(IMGV2-21) — 입력 중엔 로컬, 포커스 아웃/Enter 시에만 PATCH(스팸 방지).
function ApSetting({ label, suffix, value, min, max, disabled, onCommit }) {
  const [draft, setDraft] = useState('')
  const [editing, setEditing] = useState(false)
  const shown = editing ? draft : (value ?? '')
  const commit = () => {
    setEditing(false)
    const n = Math.max(min, Math.min(max, Number(draft)))
    if (draft !== '' && Number.isFinite(n) && n !== value) onCommit(n)
  }
  return (
    <label style={{ fontSize: 11, color: 'var(--faint)', display: 'flex', gap: 6, alignItems: 'center' }}>
      {label}
      <input type="number" min={min} max={max} disabled={disabled} value={shown}
        onFocus={() => { setEditing(true); setDraft(String(value ?? '')) }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
        style={{ width: 64, background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '4px 8px', color: 'var(--text)', fontSize: 12, fontFamily: 'monospace' }} />
      {suffix}
    </label>
  )
}

// ── DB 상태 ── 지금까지 분석되어 DB에 보관 중인 "실제" 수치(작업 합산이 아님).
function DbStatusPanel() {
  const { t } = useLocale()
  const { total, analyzed, pct, loading } = useDbStats()
  const remaining = Math.max(0, total - analyzed)
  return (
    <div className="panel">
      <h4>{t('v2.ad.db_title')} <span className="hint">{t('v2.ad.db_hint')}{loading ? t('v2.ad.db_loading') : ''}</span></h4>
      <div className="sum-row" style={{ marginTop: 4 }}>
        <span className="big">{analyzed.toLocaleString()}</span>
        <span className="of">{t('v2.ad.db_of', { n: total.toLocaleString() })}</span>
        <div className="bar"><i style={{ width: `${pct}%` }} /></div>
        <span className="mono2" style={{ color: '#93c5fd', fontWeight: 700 }}>{pct}%</span>
      </div>
      <div style={{ marginTop: 10, fontSize: 12, color: 'var(--dim)' }}>
        {t('v2.ad.db_stored')} <b style={{ color: 'var(--text)' }}>{t('v2.ad.n_images', { n: total.toLocaleString() })}</b> · {t('v2.ad.db_analyzed')} <b style={{ color: 'var(--emerald)' }}>{t('v2.ad.n_images', { n: analyzed.toLocaleString() })}</b> · {t('v2.ad.db_remaining')} <b style={{ color: 'var(--text)' }}>{t('v2.ad.n_images', { n: remaining.toLocaleString() })}</b>
      </div>
    </div>
  )
}

export function WorkersPanel() {
  const { t } = useLocale()
  const { disconnected, workers, globalMode } = useWorkers()
  const { valves, bottleneck } = useClusterValves()
  const { stop, block, unblock } = useWorkerControl()
  const ap = useAutoProcessing()
  const canControl = !disconnected
  const apOn = ap.config?.enabled ?? true
  const setAp = (patch) => { if (!disconnected) ap.update.mutate(patch) }

  const online = workers.filter(w => w.status === 'online')
  const blocked = workers.filter(w => w.status === 'blocked')

  return (
    <>
      <div className="panel">
        <h4>{t('v2.ad.ap_title')} <span className="hint">{t('v2.ad.ap_hint', { mode: globalMode })}{disconnected && t('v2.ad.disconnected_suffix')}</span>
          <span className={`toggle ${apOn ? '' : 'off'}`} title={apOn ? t('v2.ad.ap_on_title') : t('v2.ad.ap_off_title')}
            onClick={() => !disconnected && !ap.update.isPending && setAp({ enabled: !apOn })} /></h4>
        {bottleneck && bottleneck.pending > 0 ? (
          <div className="bottleneck">
            {t('v2.ad.bn_now')} <b>{bottleneck.label}</b> {t('v2.ad.bn_waiting', { n: bottleneck.pending.toLocaleString() })} <b>{t('v2.ad.bn_rate', { n: bottleneck.rate ?? 0 })}</b> {t('v2.ad.bn_tail')}
            <div className="why">{t('v2.ad.bn_why')}</div>
          </div>
        ) : (
          <div className="bottleneck" style={{ borderColor: 'var(--emerald-d)' }}>
            {t('v2.ad.bn_none')}
          </div>
        )}
        <div className="valves">
          {valves.map(v => {
            const isBn = bottleneck && v.phase === bottleneck.phase && v.pending > 0
            const rateTxt = v.rate != null ? t('v2.ad.per_min', { n: v.rate }) : (v.done >= v.total ? t('v2.ad.done') : t('v2.ad.in_progress'))
            return (
              <div className={`valve ${isBn ? 'bn' : ''}`} key={v.phase}>
                <div className="ph">{v.label}{isBn && <span className="bn-tag">{t('v2.ad.bn_tag')}</span>}</div>
                <div className="n">{v.pending > 0 ? t('v2.ad.pending_n', { n: v.pending.toLocaleString() }) : t('v2.ad.caught_up')}</div>
                <div className="rate">{rateTxt}</div>
                <span className="sw on">ON</span>
              </div>
            )
          })}
        </div>
        <div style={{ display: 'flex', gap: 18, alignItems: 'center', marginTop: 10, flexWrap: 'wrap' }}>
          <ApSetting label={t('v2.ad.rest_between')} suffix={t('v2.ad.unit_sec')} disabled={disconnected || ap.update.isPending}
            value={ap.config?.rest_after_batch_s} min={0} max={3600} onCommit={(n) => setAp({ rest_after_batch_s: n })} />
          <ApSetting label={t('v2.ad.batch_size')} suffix={t('v2.ad.unit_images')} disabled={disconnected || ap.update.isPending}
            value={ap.config?.batch_size} min={1} max={64} onCommit={(n) => setAp({ batch_size: n })} />
          <label style={{ fontSize: 11, color: 'var(--faint)', display: 'flex', gap: 6, alignItems: 'center', cursor: disconnected ? 'default' : 'pointer' }}>
            <input type="checkbox" disabled={disconnected || ap.update.isPending} checked={ap.config?.verbose_log ?? false} onChange={(e) => setAp({ verbose_log: e.target.checked })} /> {t('v2.ad.verbose_log')}
          </label>
        </div>
      </div>

      <div className="panel">
        <h4>{t('v2.ad.an_title')} <span className="hint">{t('v2.ad.an_hint', { n: online.length })}{disconnected && t('v2.ad.disconnected_suffix')}</span></h4>
        <table>
          <thead>
            <tr><th>{t('v2.ad.th_name')}</th><th>{t('v2.ad.th_status')}</th><th>{t('v2.ad.th_now')}</th><th>{t('v2.ad.th_rate')}</th><th>{t('v2.ad.th_total')}</th><th style={{ textAlign: 'right' }}>{t('v2.ad.th_control')}</th></tr>
          </thead>
          <tbody>
            {online.map(w => (
              <tr key={w.id}>
                <td><b>{w.worker_name}</b><div style={{ fontSize: 10, color: 'var(--faint)' }}>{ORIGIN_LABEL(w)}</div></td>
                <td><span className="badge b-ok">{t('v2.ad.online')}</span></td>
                <td>{w.current_file
                  ? <span>{w.current_file} {w.current_phase && <span style={{ color: 'var(--cyan)', fontSize: 10 }}>{phaseText(w.current_phase, t)}</span>}</span>
                  : <span style={{ color: 'var(--faint)' }}>{t('v2.ad.idle')}</span>}</td>
                <td className="mono">{w.throughput != null ? t('v2.ad.per_min', { n: Number(w.throughput).toFixed(Number(w.throughput) < 10 ? 1 : 0) }) : '—'}{w.throughput_mode && <span style={{ color: 'var(--faint)', fontSize: 10 }}> {phaseText(w.throughput_mode, t)}</span>}</td>
                <td className="mono">{(w.jobs_completed || 0).toLocaleString()}</td>
                <td><div className="row-act">
                  <button disabled={!canControl || stop.isPending} onClick={() => stop.mutate(w.id)}>{t('v2.ad.stop')}</button>
                  <button className="danger" disabled={!canControl || block.isPending} onClick={() => block.mutate(w.id)}>{t('v2.ad.block')}</button>
                </div></td>
              </tr>
            ))}
            {online.length === 0 && (
              <tr><td colSpan={6} style={{ color: 'var(--faint)' }}>{t('v2.ad.no_analyzers')}</td></tr>
            )}
          </tbody>
        </table>
        {blocked.map(w => (
          <div className="blocked" key={w.id}>
            ⛔ <b>{w.worker_name}</b> <span className="faint">{t('v2.ad.blocked_suffix')}</span>
            <button className="unb" disabled={!canControl || unblock.isPending} onClick={() => unblock.mutate(w.id)}>{t('v2.ad.unblock')}</button>
          </div>
        ))}
      </div>

      <EnrollWorkerPanel disabled={disconnected} />
    </>
  )
}

// ── 원격 분석기 등록 (IMGV2-18) ─────────────────────────────
// 다른 머신을 분석기로 붙이는 설치 명령(+토큰)을 발급한다. connect_mode 별로
// 명령 형태가 다르다(direct_lan/manual_external = 직접 URL, relay_session = 릴레이).
const MODE_LABEL = { direct_lan: 'v2.ad.mode_lan', manual_external: 'v2.ad.mode_manual', relay_session: 'v2.ad.mode_relay' }

function EnrollWorkerPanel({ disabled }) {
  const { t } = useLocale()
  const { requestOrigin, modes } = useConnectionInfo()
  const issue = useHeadlessCommand()
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('cloud-worker-1')
  const [launcher, setLauncher] = useState('cloud')
  const [expires, setExpires] = useState(1440)
  const [mode, setMode] = useState('direct_lan')
  const [overrideUrl, setOverrideUrl] = useState('')
  const [copied, setCopied] = useState(false)

  const result = issue.data
  const errMsg = issue.isError ? (issue.error?.detail || issue.error?.message || t('v2.ad.issue_failed')) : ''
  // 서버가 광고하는 connect_mode 중 명령 발급 대상 3종만 노출
  const available = new Set((modes || []).filter(m => m.available).map(m => m.mode))
  const modeOptions = ['direct_lan', 'manual_external', 'relay_session']

  const generate = () => {
    setCopied(false)
    issue.mutate({
      worker_name: name.trim() || 'headless-worker',
      launcher,
      expires_minutes: Number(expires) || 1440,
      server_url: mode === 'relay_session' ? null : (overrideUrl.trim() || null),
      connect_mode: mode,
    })
  }
  const copy = async () => {
    if (!result?.linux_command) return
    try { await navigator.clipboard.writeText(result.linux_command); setCopied(true); setTimeout(() => setCopied(false), 1800) } catch { /* 수동 복사용으로 텍스트는 보임 */ }
  }

  const inp = { background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '6px 9px', color: 'var(--text)', fontSize: 12, width: '100%', boxSizing: 'border-box' }
  const lbl = { display: 'block', fontSize: 11, color: 'var(--faint)', marginBottom: 4 }

  return (
    <div className="panel">
      <h4>{t('v2.ad.enroll_title')} <span className="hint">{t('v2.ad.enroll_hint')}{disabled && t('v2.ad.disconnected_suffix')}</span>
        <button style={{ marginLeft: 'auto' }} onClick={() => setOpen(v => !v)} disabled={disabled}>{open ? t('v2.ad.close') : t('v2.ad.gen_command')}</button></h4>
      {open && (
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div><label style={lbl}>{t('v2.ad.analyzer_name')}</label><input style={{ ...inp, fontFamily: 'monospace' }} value={name} onChange={e => setName(e.target.value)} /></div>
            <div><label style={lbl}>{t('v2.ad.launcher')}</label>
              <select style={inp} value={launcher} onChange={e => setLauncher(e.target.value)}>
                <option value="cloud">cloud</option><option value="cli">cli</option><option value="service">service</option>
              </select></div>
            <div><label style={lbl}>{t('v2.ad.token_expiry')}</label><input style={{ ...inp, fontFamily: 'monospace' }} type="number" min={15} max={43200} value={expires} onChange={e => setExpires(e.target.value)} /></div>
            <div><label style={lbl}>{t('v2.ad.access_mode')}</label>
              <select style={inp} value={mode} onChange={e => setMode(e.target.value)}>
                {modeOptions.map(m => <option key={m} value={m}>{t(MODE_LABEL[m])}{available.size && !available.has(m) ? t('v2.ad.not_advertised') : ''}</option>)}
              </select></div>
          </div>
          {mode !== 'relay_session' && (
            <div><label style={lbl}>{t('v2.ad.url_override')}</label>
              <input style={{ ...inp, fontFamily: 'monospace' }} placeholder={requestOrigin || 'https://...'} value={overrideUrl} onChange={e => setOverrideUrl(e.target.value)} /></div>
          )}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button disabled={issue.isPending} onClick={generate}>{issue.isPending ? t('v2.ad.issuing') : t('v2.ad.gen_install')}</button>
            {result?.linux_command && <button onClick={copy}>{copied ? t('v2.ad.copied') : t('v2.ad.copy_command')}</button>}
          </div>
          {errMsg && <div style={{ color: 'var(--red)', fontSize: 11 }}>{errMsg}</div>}
          {result?.linux_command && (
            <>
              <div style={{ fontSize: 11, color: 'var(--faint)' }}>
                {t('v2.ad.analyzer_account')} <b style={{ color: 'var(--text)' }}>{result.worker_username}</b> {t('v2.ad.token_expires_in', { n: result.expires_minutes })}
                {result.relay_endpoint ? t('v2.ad.relay_prefix', { url: result.relay_endpoint }) : result.server_url ? ` · ${result.server_url}` : ''}
                <br />{t('v2.ad.one_time_warn')}
              </div>
              <pre style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: 10, fontSize: 11, color: 'var(--emerald)', whiteSpace: 'pre-wrap', wordBreak: 'break-all', margin: 0 }}>{result.linux_command}</pre>
            </>
          )}
        </div>
      )}
    </div>
  )
}

// ── 분류/도메인 (IMGV2-17) ──────────────────────────────────
// 활성 도메인이 분석 분류 기준·태그 공간·검색 필터 옵션을 정한다.
// 백엔드는 생성 전용(POST /domains 는 기존 id 409) → 상세는 읽기 전용, 편집은 "새 도메인" 생성으로.

function Chip({ children }) {
  return <span style={{ background: 'rgba(96,165,250,.15)', color: 'var(--cyan)', borderRadius: 6, padding: '2px 8px', fontSize: 11 }}>{children}</span>
}

function HintRows({ obj }) {
  return Object.entries(obj).map(([k, v]) => (
    <div key={k} style={{ display: 'flex', gap: 8, fontSize: 11, padding: '2px 0' }}>
      <span style={{ color: 'var(--faint)', minWidth: 120, flexShrink: 0 }}>{k}</span>
      <span>{Array.isArray(v) ? v.join(', ') : String(v)}</span>
    </div>
  ))
}

function DomainDetailPanel({ detail }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(() => new Set())
  const toggle = (t) => setOpen(prev => { const n = new Set(prev); n.has(t) ? n.delete(t) : n.add(t); return n })
  const box = { border: '1px solid var(--line)', borderRadius: 8 }
  const sectionLabel = { fontSize: 11, color: 'var(--faint)', margin: '0 0 6px' }

  return (
    <div className="panel">
      <h4>{detail.name_ko || detail.name} <span className="hint">{detail.id} {t('v2.ad.dom_readonly')}</span></h4>
      <div style={{ marginBottom: 12 }}>
        <div style={sectionLabel}>{t('v2.ad.image_types', { n: detail.image_types?.length || 0 })}</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>{(detail.image_types || []).map(t => <Chip key={t}>{t}</Chip>)}</div>
      </div>
      {Object.keys(detail.type_hints || {}).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={sectionLabel}>{t('v2.ad.hints_by_type')}</div>
          <div style={box}>
            {Object.entries(detail.type_hints).map(([type, hints]) => (
              <div key={type} style={{ borderBottom: '1px solid var(--line)' }}>
                <button onClick={() => toggle(type)} style={{ width: '100%', display: 'flex', gap: 8, alignItems: 'center', background: 'none', border: 'none', color: 'var(--text)', padding: '8px 10px', cursor: 'pointer', fontSize: 12 }}>
                  <span style={{ color: 'var(--faint)' }}>{open.has(type) ? '▾' : '▸'}</span>
                  <b>{type}</b><span style={{ marginLeft: 'auto', color: 'var(--faint)', fontSize: 10 }}>{t('v2.ad.n_fields', { n: Object.keys(hints).length })}</span>
                </button>
                {open.has(type) && <div style={{ padding: '0 10px 8px 28px' }}><HintRows obj={hints} /></div>}
              </div>
            ))}
          </div>
        </div>
      )}
      {detail.common_hints && Object.keys(detail.common_hints).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={sectionLabel}>{t('v2.ad.common_hints')} <span style={{ color: 'var(--faint)' }}>{t('v2.ad.auto_suffix')}</span></div>
          <div style={{ ...box, padding: '8px 10px' }}><HintRows obj={detail.common_hints} /></div>
        </div>
      )}
      {detail.type_instructions && Object.keys(detail.type_instructions).length > 0 && (
        <div>
          <div style={sectionLabel}>{t('v2.ad.instructions_by_type')}</div>
          <div style={box}>
            {Object.entries(detail.type_instructions).map(([type, ins]) => (
              <div key={type} style={{ borderBottom: '1px solid var(--line)', padding: '8px 10px' }}>
                <b style={{ fontSize: 12 }}>{type}</b>
                <p style={{ fontSize: 11, color: 'var(--faint)', margin: '4px 0 0' }}>{ins}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function CreateDomainModal({ existingIds, onClose }) {
  const { t } = useLocale()
  const save = useSaveDomain()
  const [step, setStep] = useState(1)
  const [domainId, setDomainId] = useState('')
  const [nameEn, setNameEn] = useState('')
  const [nameKo, setNameKo] = useState('')
  const [description, setDescription] = useState('')
  const [prompt, setPrompt] = useState('')
  const [yamlInput, setYamlInput] = useState('')
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState('')

  const idError = domainId && !/^[a-z][a-z0-9_]*$/.test(domainId) ? t('v2.ad.err_snake')
    : domainId && existingIds.includes(domainId) ? t('v2.ad.err_id_exists') : ''
  const canNext1 = domainId && nameEn && nameKo && description && !idError

  const next1 = () => { setPrompt(generateDomainPrompt({ domainId, nameEn, nameKo, description })); setStep(2) }
  const copy = async () => { try { await navigator.clipboard.writeText(prompt); setCopied(true); setTimeout(() => setCopied(false), 1800) } catch { /* noop */ } }
  const doSave = () => {
    setError('')
    const cleaned = yamlInput.replace(/^```[\w]*\n?/, '').replace(/\n?```\s*$/, '').trim()
    save.mutate({ domainId, yamlContent: cleaned }, {
      onSuccess: (r) => { if (r?.success) onClose(); else setError(r?.detail || r?.error || t('v2.ad.err_save')) },
      onError: (e) => setError(e?.detail || e?.message || t('v2.ad.err_save')),
    })
  }

  const overlay = { position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,.6)' }
  const modal = { background: 'var(--panel)', border: '1px solid var(--line)', borderRadius: 12, width: 'min(640px, 92vw)', maxHeight: '82vh', overflowY: 'auto', padding: 20 }
  const inp = { width: '100%', background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 10px', color: 'var(--text)', fontSize: 12, boxSizing: 'border-box' }
  const lbl = { display: 'block', fontSize: 11, color: 'var(--faint)', margin: '0 0 4px' }
  const stepName = step === 1 ? t('v2.ad.step_define') : step === 2 ? t('v2.ad.step_prompt') : t('v2.ad.step_yaml')

  return (
    <div style={overlay} onClick={onClose}>
      <div style={modal} onClick={e => e.stopPropagation()}>
        <h4 style={{ marginTop: 0 }}>{t('v2.ad.new_domain')} <span className="hint">{step}/3 · {stepName}</span></h4>
        {step === 1 && <div style={{ display: 'grid', gap: 12 }}>
          <div><label style={lbl}>{t('v2.ad.domain_id')}</label>
            <input style={inp} value={domainId} onChange={e => setDomainId(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))} placeholder={t('v2.ad.domain_id_ph')} autoFocus />
            {idError && <div style={{ color: 'var(--red)', fontSize: 11, marginTop: 4 }}>{idError}</div>}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div><label style={lbl}>{t('v2.ad.name_en')}</label><input style={inp} value={nameEn} onChange={e => setNameEn(e.target.value)} placeholder="Medical Image" /></div>
            <div><label style={lbl}>{t('v2.ad.name_ko')}</label><input style={inp} value={nameKo} onChange={e => setNameKo(e.target.value)} placeholder={t('v2.ad.name_ko_ph')} /></div>
          </div>
          <div><label style={lbl}>{t('v2.ad.description')}</label><textarea style={{ ...inp, resize: 'none' }} rows={4} value={description} onChange={e => setDescription(e.target.value)} placeholder={t('v2.ad.description_ph')} /></div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button onClick={onClose}>{t('v2.ad.cancel')}</button>
            <button disabled={!canNext1} onClick={next1}>{t('v2.ad.next')}</button>
          </div>
        </div>}
        {step === 2 && <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ fontSize: 12, color: 'var(--faint)' }}>{t('v2.ad.prompt_help')}</div>
          <pre style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: 10, fontSize: 11, maxHeight: '40vh', overflowY: 'auto', whiteSpace: 'pre-wrap', margin: 0 }}>{prompt}</pre>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setStep(1)}>{t('v2.ad.prev')}</button>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={copy}>{copied ? t('v2.ad.copied') : t('v2.ad.copy_prompt')}</button>
              <button onClick={() => setStep(3)}>{t('v2.ad.next')}</button>
            </div>
          </div>
        </div>}
        {step === 3 && <div style={{ display: 'grid', gap: 12 }}>
          <label style={lbl}>{t('v2.ad.paste_yaml')}</label>
          <textarea style={{ ...inp, resize: 'none', fontFamily: 'monospace' }} rows={16} value={yamlInput} onChange={e => { setYamlInput(e.target.value); setError('') }} placeholder={'domain:\n  id: ...'} />
          {error && <div style={{ color: 'var(--red)', fontSize: 11, background: 'rgba(248,113,113,.08)', border: '1px solid rgba(248,113,113,.3)', borderRadius: 6, padding: '8px 10px' }}>{error}</div>}
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setStep(2)}>{t('v2.ad.prev')}</button>
            <button disabled={save.isPending || !yamlInput.trim()} onClick={doSave}>{save.isPending ? t('v2.ad.saving') : t('v2.ad.save_domain')}</button>
          </div>
        </div>}
      </div>
    </div>
  )
}

function ClassificationPanel() {
  const { t } = useLocale()
  const { data: domains = [], isLoading, isError } = useDomains()
  const { data: active } = useActiveDomain()
  const setActive = useSetActiveDomain()
  const [selectedId, setSelectedId] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const activeId = active?.active_domain || null
  const effectiveId = selectedId || activeId || domains[0]?.id || null
  const { data: detail } = useDomainDetail(effectiveId)

  if (isLoading) return <div className="panel"><span style={{ color: 'var(--faint)' }}>{t('v2.ad.loading_domains')}</span></div>
  if (isError) return <div className="panel"><span style={{ color: 'var(--faint)' }}>{t('v2.ad.not_connected')}</span></div>

  return (
    <>
      <div className="panel">
        <h4>{t('v2.ad.cls_title')} <span className="hint">{t('v2.ad.cls_hint')}</span>
          <button style={{ marginLeft: 'auto' }} onClick={() => setShowCreate(true)}>{t('v2.ad.new_domain_btn')}</button></h4>
        <table>
          <thead><tr><th>{t('v2.ad.th_domain')}</th><th>{t('v2.ad.th_type')}</th><th style={{ textAlign: 'right' }}>{t('v2.ad.th_status')}</th></tr></thead>
          <tbody>
            {domains.map(d => {
              const isActive = d.id === activeId
              const isSel = d.id === effectiveId
              return (
                <tr key={d.id} onClick={() => setSelectedId(d.id)} style={{ cursor: 'pointer', background: isSel ? 'var(--panel2)' : undefined }}>
                  <td><b>{d.name_ko || d.name}</b> <span style={{ color: 'var(--faint)', fontSize: 10 }}>{d.id}</span>
                    <div style={{ color: 'var(--faint)', fontSize: 11 }}>{d.description}</div></td>
                  <td className="mono">{d.image_types_count ?? d.image_types?.length ?? 0}</td>
                  <td style={{ textAlign: 'right' }}>{isActive
                    ? <span className="badge b-ok">{t('v2.ad.active')}</span>
                    : <button disabled={setActive.isPending} onClick={(e) => { e.stopPropagation(); setActive.mutate(d.id) }}>{t('v2.ad.activate')}</button>}</td>
                </tr>
              )
            })}
            {domains.length === 0 && <tr><td colSpan={3} style={{ color: 'var(--faint)' }}>{t('v2.ad.no_domains')}</td></tr>}
          </tbody>
        </table>
      </div>
      {detail && <DomainDetailPanel detail={detail} />}
      {showCreate && <CreateDomainModal existingIds={domains.map(d => d.id)} onClose={() => setShowCreate(false)} />}
    </>
  )
}

// ── 실시간 로그 (IMGV2-26) ──────────────────────────────────
const LOG_LEVEL_COLOR = { ERROR: 'var(--red)', CRITICAL: 'var(--red)', WARNING: '#fbbf24', INFO: 'var(--faint)', DEBUG: 'var(--faint)' }
const LOG_CATS = [['all', 'v2.ad.log_all'], ['job', 'v2.ad.log_job'], ['network', 'v2.ad.log_network'], ['general', 'v2.ad.log_general']]

function LogPanel() {
  const { t } = useLocale()
  const [cat, setCat] = useState('all')
  const { entries, clear, disconnected } = useLogStream(true)
  const scrollRef = useRef(null)
  const filtered = cat === 'all' ? entries : entries.filter(e => e.category === cat)
  useEffect(() => { const el = scrollRef.current; if (el) el.scrollTop = el.scrollHeight }, [filtered.length])

  const chip = (id) => ({ fontSize: 11, background: cat === id ? 'var(--panel2)' : 'none', border: '1px solid var(--line)', borderRadius: 6, padding: '3px 9px', color: cat === id ? 'var(--text)' : 'var(--dim)', cursor: 'pointer' })

  return (
    <div className="panel">
      <h4>{t('v2.ad.log_title')} <span className="hint">{t('v2.ad.log_hint')}{disconnected && t('v2.ad.disconnected_suffix')}</span>
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {LOG_CATS.map(([id, k]) => <button key={id} style={chip(id)} onClick={() => setCat(id)}>{t(k)}</button>)}
          <button style={{ fontSize: 11, border: '1px solid var(--line)', borderRadius: 6, padding: '3px 9px', background: 'none', color: 'var(--dim)', cursor: 'pointer' }} onClick={clear}>{t('v2.ad.clear')}</button>
        </span>
      </h4>
      <div ref={scrollRef} style={{ height: '62vh', overflowY: 'auto', background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>
        {filtered.length === 0 && <div style={{ color: 'var(--faint)' }}>{t('v2.ad.log_empty')}</div>}
        {filtered.map(e => (
          <div key={e.seq} style={{ display: 'flex', gap: 8, padding: '1px 0', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            <span style={{ color: 'var(--faint)', flexShrink: 0 }}>{new Date(e.ts * 1000).toLocaleTimeString('ko-KR', { hour12: false })}</span>
            <span style={{ color: LOG_LEVEL_COLOR[e.level] || 'var(--faint)', flexShrink: 0, width: 56 }}>{e.level}</span>
            <span style={{ color: 'var(--text)' }}>{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── 검색 피드백 대시보드 (IMGV2-23) ─────────────────────────
function FeedbackPanel() {
  const { t } = useLocale()
  const { disconnected, loading, total30d, topFiles, topQueries } = useFeedbackSummary()
  if (loading) return <div className="panel"><span style={{ color: 'var(--faint)' }}>{t('v2.ad.loading_feedback')}</span></div>
  if (disconnected) return <div className="panel"><span style={{ color: 'var(--faint)' }}>{t('v2.ad.not_connected')}</span></div>

  return (
    <>
      <div className="panel">
        <h4>{t('v2.ad.fb_title')} <span className="hint">{t('v2.ad.fb_hint')}</span></h4>
        <div style={{ fontSize: 28, fontWeight: 600 }}>{total30d.toLocaleString()}<span style={{ fontSize: 12, color: 'var(--faint)', fontWeight: 400 }}>{t('v2.ad.fb_count')}</span></div>
        {total30d === 0 && <div style={{ color: 'var(--faint)', fontSize: 12, marginTop: 6 }}>{t('v2.ad.fb_none')}</div>}
      </div>
      {topQueries.length > 0 && (
        <div className="panel">
          <h4>{t('v2.ad.fb_top_q')} <span className="hint">{t('v2.ad.fb_top_q_hint')}</span></h4>
          <table>
            <thead><tr><th>{t('v2.ad.th_query')}</th><th style={{ textAlign: 'right' }}>{t('v2.ad.th_flags')}</th></tr></thead>
            <tbody>{topQueries.map((q, i) => (
              <tr key={i}><td>{q.query}</td><td className="mono" style={{ textAlign: 'right' }}>{q.count.toLocaleString()}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {topFiles.length > 0 && (
        <div className="panel">
          <h4>{t('v2.ad.fb_top_f')} <span className="hint">{t('v2.ad.fb_top_f_hint')}</span></h4>
          <table>
            <thead><tr><th>{t('v2.ad.th_file_id')}</th><th style={{ textAlign: 'right' }}>{t('v2.ad.th_flags')}</th></tr></thead>
            <tbody>{topFiles.map((f, i) => (
              <tr key={i}><td className="mono">#{f.file_id}</td><td className="mono" style={{ textAlign: 'right' }}>{f.count.toLocaleString()}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </>
  )
}

function MembersPanel() {
  const { t } = useLocale()
  const { disconnected, members, invites, usage } = useMembersData()
  const { invite, revoke, remove, deactivate } = useMemberMutations()
  const [emails, setEmails] = useState('')
  const canMutate = !disconnected

  const sendInvites = () => {
    const list = emails.split(',').map(s => s.trim()).filter(Boolean)
    if (!list.length) return
    invite.mutate({ emails: list }, { onSuccess: () => setEmails('') })
  }
  const inviteMsg = invite.data
    ? `${t('v2.ad.inv_processed', { n: invite.data.results.filter(r => r.ok).length })}${invite.data.smtp_configured ? t('v2.ad.inv_mailed') : t('v2.ad.inv_link_only')}`
    : invite.isPending ? t('v2.ad.inviting') : null

  return (
    <>
      <div className="panel">
        <h4>{t('v2.ad.plan')} <span className="hint">{t('v2.ad.plan_hint')}{disconnected && t('v2.ad.disconnected_suffix')}</span></h4>
        <div className="kpis">
          <div className="kpi"><div className="v" style={{ color: usage.expired ? 'var(--red)' : '#93c5fd' }}>{usage.expired ? t('v2.ad.expired') : t('v2.ad.active_state')}</div><div className="k">{usage.expires_at ? t('v2.ad.expires_at', { date: usage.expires_at }) : t('v2.ad.no_expiry')}</div></div>
          <div className="kpi"><div className="v">{usage.seats_used} <span className="faint" style={{ fontSize: 12 }}>/ {usage.seat_limit || '∞'}</span></div><div className="k">{t('v2.ad.seats', { members: usage.members, pending: usage.pending_invites })}</div></div>
          <div className="kpi"><div className="v" style={{ color: usage.smtp_configured ? 'var(--emerald)' : 'var(--faint)' }}>{usage.smtp_configured ? t('v2.ad.mail_on') : t('v2.ad.mail_off')}</div><div className="k">{t('v2.ad.invite_method')}</div></div>
          <div className="kpi" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <button onClick={() => window.open('https://imagine.app/account', '_blank', 'noopener')}
              style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '7px 16px', borderRadius: 6 }}>{t('v2.ad.renew_upgrade')}</button>
          </div>
        </div>
        <div style={{ fontSize: 10, color: 'var(--amber)', marginTop: 6 }}>{t('v2.ad.expiry_warning')}</div>
      </div>

      <div className="panel">
        <h4>{t('v2.ad.invites')} <span className="hint">{t('v2.ad.invites_hint')}{!usage.smtp_configured && t('v2.ad.invites_hint_nosmtp')}</span></h4>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <input placeholder={t('v2.ad.email_ph')} value={emails} onChange={e => setEmails(e.target.value)}
            style={{ flex: 1, background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 12px', color: 'var(--text)', fontSize: 12 }} />
          <button onClick={sendInvites} disabled={!canMutate || invite.isPending || !emails.trim()}
            style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '0 18px', borderRadius: 6, opacity: canMutate ? 1 : .5 }}>{t('v2.ad.send_invites')}</button>
        </div>
        {inviteMsg && <div style={{ fontSize: 10.5, color: 'var(--cyan)', marginBottom: 8 }}>{inviteMsg}</div>}
        {disconnected && <div style={{ fontSize: 9.5, color: 'var(--faint)', marginBottom: 8 }}>{t('v2.ad.invite_needs_conn')}</div>}
        <table>
          <thead><tr><th>{t('v2.ad.th_pending_invite')}</th><th>{t('v2.ad.th_role')}</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            {invites.map(iv => (
              <tr key={iv.id}>
                <td>{iv.email} <span style={{ fontSize: 9, color: 'var(--amber)' }}>{t('v2.ad.seat_reserved')}</span></td>
                <td className="mono">{iv.role}</td>
                <td><div className="row-act"><button className="danger" disabled={!canMutate} onClick={() => revoke.mutate(iv.id)}>{t('v2.ad.cancel')}</button></div></td>
              </tr>
            ))}
            {invites.length === 0 && <tr><td colSpan={3} style={{ color: 'var(--faint)' }}>{t('v2.ad.no_pending')}</td></tr>}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h4>{t('v2.ad.members')} <span className="hint">{t('v2.ad.members_hint')}</span></h4>
        <table>
          <thead><tr><th>{t('v2.ad.th_name')}</th><th>{t('v2.ad.th_role')}</th><th>{t('v2.ad.th_joined_via')}</th><th>{t('v2.ad.th_last_seen')}</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            {members.map(mb => (
              <tr key={mb.id}>
                <td>{mb.username}<div style={{ fontSize: 9.5, color: 'var(--faint)' }}>{mb.email}</div></td>
                <td>{mb.role === 'admin'
                  ? <span className="badge" style={{ background: 'rgba(192,132,252,.15)', color: 'var(--purple)' }}>{t('v2.ad.role_operator')}</span>
                  : <span className="badge b-ok">{mb.is_active ? t('v2.ad.role_user') : t('v2.ad.role_inactive')}</span>}</td>
                <td className="mono" style={{ fontSize: 10 }}>{mb.via === 'firebase' ? t('v2.ad.via_invite') : t('v2.ad.via_password')}</td>
                <td className="mono">{mb.last_login_at || '—'}</td>
                <td>{mb.role !== 'admin' && (
                  <div className="row-act">
                    <button disabled={!canMutate} onClick={() => deactivate.mutate(mb.id)}>{t('v2.ad.deactivate')}</button>
                    <button className="danger" disabled={!canMutate} onClick={() => remove.mutate(mb.id)}>{t('v2.ad.remove')}</button>
                  </div>
                )}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function DesktopServerControl() {
  const { t } = useLocale()
  const d = window.imagineDesktop
  const [st, setSt] = useState(null) // {running, autostart}
  const [busy, setBusy] = useState(false)
  const refresh = () => d.serverStatus().then(setSt).catch(() => setSt({ running: false, autostart: false }))
  useEffect(() => { refresh() }, []) // eslint-disable-line

  const start = async () => { setBusy(true); try { await d.startServer() } finally { await refresh(); setBusy(false) } }
  const stop = async () => {
    if (!confirm(t('v2.ad.srv_stop_confirm'))) return
    setBusy(true); try { await d.stopServer() } finally { await refresh(); setBusy(false) }
  }
  const toggleAuto = async (e) => { setBusy(true); try { await d.setAutostart(e.target.checked) } finally { await refresh(); setBusy(false) } }

  const running = st?.running
  return (
    <div className="panel">
      <h4>{t('v2.ad.srv_title')} <span className="hint">{t('v2.ad.srv_hint')}</span></h4>
      <table><tbody>
        <tr>
          <td>{t('v2.ad.srv_process')}</td>
          <td style={{ color: running ? 'var(--emerald)' : 'var(--faint)', fontSize: 11 }}>
            {st === null ? t('v2.ad.checking') : running ? t('v2.ad.srv_running') : t('v2.ad.srv_off')}
          </td>
          <td><div className="row-act">
            {running
              ? <button className="danger" disabled={busy} onClick={stop}>{t('v2.ad.srv_stop')}</button>
              : <button disabled={busy} onClick={start}>{t('v2.ad.srv_start')}</button>}
          </div></td>
        </tr>
        <tr>
          <td>{t('v2.ad.autostart')}<div style={{ fontSize: 10, color: 'var(--faint)' }}>{t('v2.ad.autostart_d')}</div></td>
          <td />
          <td><label className="chk" style={{ justifyContent: 'flex-end' }}>
            <input type="checkbox" checked={!!st?.autostart} disabled={busy || st === null} onChange={toggleAuto} /> {t('v2.ad.autostart_chk')}
          </label></td>
        </tr>
      </tbody></table>
    </div>
  )
}

function DbToolsControl() {
  const { t } = useLocale()
  const audit = useDbAudit()
  const { status: bf, start: bfStart, stop: bfStop } = useBackfill()
  const { status: rp, start: rpStart } = useRepairParse()
  const reset = useDbReset()
  const dbExport = useDbExport()
  const dbImport = useDbImport()
  const fileRef = useRef(null)
  const [pwOpen, setPwOpen] = useState(false)
  const [pw, setPw] = useState('')
  const [impOpen, setImpOpen] = useState(false)
  const [impPw, setImpPw] = useState('')
  const [impFile, setImpFile] = useState(null)

  const auditMsg = audit.data
    ? t('v2.ad.audit_result', { total: audit.data.total_files?.toLocaleString() ?? '—', incomplete: audit.data.total_incomplete?.toLocaleString() ?? 0, folders: (audit.data.folders || []).length })
    : audit.isPending ? t('v2.ad.auditing') : audit.isError ? t('v2.ad.audit_failed') : t('v2.ad.audit_idle')
  const bfRunning = bf?.running
  const bfMsg = bf
    ? (bfRunning
        ? `${t('v2.ad.running_progress', { done: (bf.done ?? 0).toLocaleString(), total: (bf.total ?? 0).toLocaleString() })}${bf.failed ? t('v2.ad.failed_n', { n: bf.failed }) : ''}`
        : `${t('v2.ad.idle_last', { done: (bf.done ?? 0).toLocaleString() })}${bf.skipped ? t('v2.ad.skipped_n', { n: bf.skipped }) : ''}${bf.last_error ? t('v2.ad.had_errors') : ''}`)
    : t('v2.ad.bf_idle')
  const resetMsg = reset.data ? t('v2.ad.reset_done', { n: (reset.data.file_count ?? reset.data.deleted ?? 0).toLocaleString() }) : reset.isError ? (reset.error?.detail || t('v2.ad.reset_failed')) : t('v2.ad.reset_need_pw')
  const rpRunning = rp?.running
  const rpp = rp?.progress
  const rpMsg = rpp
    ? (rpRunning
        ? `${t('v2.ad.running_progress', { done: (rpp.done ?? 0).toLocaleString(), total: (rpp.total ?? 0).toLocaleString() })}${rpp.failed ? t('v2.ad.failed_n', { n: rpp.failed }) : ''}${rpp.current_file ? ` · ${rpp.current_file}` : ''}`
        : `${t('v2.ad.idle_last', { done: (rpp.done ?? 0).toLocaleString() })}${rpp.skipped ? t('v2.ad.skipped_n', { n: rpp.skipped }) : ''}${rpp.failed ? t('v2.ad.failed_n', { n: rpp.failed }) : ''}`)
    : t('v2.ad.rp_idle')

  const doReset = () => {
    if (!pw) return
    reset.mutate(pw, { onSuccess: () => { setPwOpen(false); setPw('') } })
  }
  // 내보내기/가져오기 둘 다 data 가 남으므로 더 최근(submittedAt) 액션을 표시.
  const recentImport = (dbImport.submittedAt || 0) >= (dbExport.submittedAt || 0)
  const ioError = recentImport ? dbImport.isError : dbExport.isError
  let ioMsg = t('v2.ad.io_idle')
  if (dbExport.isPending) ioMsg = t('v2.ad.exporting')
  else if (dbImport.isPending) ioMsg = t('v2.ad.restoring')
  else if (recentImport && dbImport.isError) ioMsg = dbImport.error?.detail || dbImport.error?.message || t('v2.ad.restore_failed')
  else if (recentImport && dbImport.data) ioMsg = t('v2.ad.restored', { n: (dbImport.data.file_count ?? 0).toLocaleString() })
  else if (!recentImport && dbExport.isError) ioMsg = dbExport.error?.message || t('v2.ad.export_failed')
  else if (!recentImport && dbExport.data) ioMsg = t('v2.ad.exported', { name: dbExport.data.filename, kb: Math.round((dbExport.data.bytes || 0) / 1024).toLocaleString() })
  const doImport = () => {
    if (!impPw || !impFile) return
    dbImport.mutate({ password: impPw, file: impFile }, {
      onSuccess: () => { setImpOpen(false); setImpPw(''); setImpFile(null) },
    })
  }

  return (
    <div className="panel">
      <h4>{t('v2.ad.tools_title')} <span className="hint">{t('v2.ad.tools_hint')}</span></h4>
      <table><tbody>
        <tr>
          <td>{t('v2.ad.audit_row')}</td>
          <td style={{ color: 'var(--faint)', fontSize: 11 }}>{auditMsg}</td>
          <td><div className="row-act"><button disabled={audit.isPending} onClick={() => audit.mutate()}>{t('v2.ad.run')}</button></div></td>
        </tr>
        <tr>
          <td>{t('v2.ad.hash_backfill')}</td>
          <td style={{ color: bfRunning ? 'var(--cyan)' : 'var(--faint)', fontSize: 11 }}>{bfMsg}</td>
          <td><div className="row-act">
            {bfRunning
              ? <button className="danger" disabled={bfStop.isPending} onClick={() => bfStop.mutate()}>{t('v2.ad.stop_btn')}</button>
              : <button disabled={bfStart.isPending} onClick={() => bfStart.mutate()}>{t('v2.ad.run')}</button>}
          </div></td>
        </tr>
        <tr>
          <td>{t('v2.ad.parse_recover')}</td>
          <td style={{ color: rpRunning ? 'var(--cyan)' : 'var(--faint)', fontSize: 11 }}>{rpMsg}</td>
          <td><div className="row-act"><button disabled={rpRunning || rpStart.isPending} onClick={() => rpStart.mutate()}>{rpRunning ? t('v2.ad.running') : t('v2.ad.run')}</button></div></td>
        </tr>
        <tr>
          <td>{t('v2.ad.db_io')}</td>
          <td style={{ color: ioError ? 'var(--red)' : 'var(--faint)', fontSize: 11 }}>{ioMsg}</td>
          <td><div className="row-act">
            <button disabled={dbExport.isPending} onClick={() => dbExport.mutate()}>{t('v2.ad.export')}</button>
            <button className="danger" onClick={() => setImpOpen(v => !v)}>{t('v2.ad.import')}</button>
          </div></td>
        </tr>
        {impOpen && (
          <tr>
            <td colSpan={3}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', background: 'rgba(248,113,113,.08)', border: '1px solid rgba(248,113,113,.3)', borderRadius: 6, padding: '8px 10px' }}>
                <span style={{ fontSize: 11, color: 'var(--red)' }}>{t('v2.ad.import_warn')}</span>
                <input type="password" value={impPw} onChange={e => setImpPw(e.target.value)} placeholder={t('v2.ad.password')} style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '6px 10px', color: 'var(--text)', fontSize: 12, width: 150 }} />
                <input ref={fileRef} type="file" accept=".db,.sqlite,.sqlite3,application/x-sqlite3" style={{ display: 'none' }} onChange={e => setImpFile(e.target.files?.[0] || null)} />
                <button onClick={() => fileRef.current?.click()}>{impFile ? impFile.name : t('v2.ad.pick_backup')}</button>
                <button className="danger" disabled={!impPw || !impFile || dbImport.isPending} onClick={doImport}>{dbImport.isPending ? t('v2.ad.restoring') : t('v2.ad.run_restore')}</button>
                <button onClick={() => { setImpOpen(false); setImpPw(''); setImpFile(null) }}>{t('v2.ad.cancel')}</button>
              </div>
            </td>
          </tr>
        )}
        <tr>
          <td style={{ color: 'var(--red)' }}>{t('v2.ad.db_reset')}</td>
          <td style={{ color: reset.isError ? 'var(--red)' : 'var(--faint)', fontSize: 11 }}>{resetMsg}</td>
          <td><div className="row-act"><button className="danger" onClick={() => setPwOpen(v => !v)}>{t('v2.ad.reset_open')}</button></div></td>
        </tr>
        {pwOpen && (
          <tr>
            <td colSpan={3}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', background: 'rgba(248,113,113,.08)', border: '1px solid rgba(248,113,113,.3)', borderRadius: 6, padding: '8px 10px' }}>
                <span style={{ fontSize: 11, color: 'var(--red)' }}>{t('v2.ad.reset_warn')}</span>
                <input type="password" value={pw} onChange={e => setPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doReset()} placeholder={t('v2.ad.password')} style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '6px 10px', color: 'var(--text)', fontSize: 12, width: 160 }} autoFocus />
                <button className="danger" disabled={!pw || reset.isPending} onClick={doReset}>{reset.isPending ? t('v2.ad.resetting') : t('v2.ad.run_reset')}</button>
                <button onClick={() => { setPwOpen(false); setPw('') }}>{t('v2.ad.cancel')}</button>
              </div>
            </td>
          </tr>
        )}
      </tbody></table>
    </div>
  )
}

function ToolsPanel() {
  const { t } = useLocale()
  const { disconnected, external, lan, refetch, refreshing } = useConnectionInfo()
  const desktop = typeof window !== 'undefined' && window.imagineDesktop // 데스크톱 셸에서만
  const extDesc = external?.available
    ? t('v2.ad.ext_on', { url: external.url || t('v2.ad.tunnel') })
    : t('v2.ad.ext_off')
  const lanDesc = lan?.available ? t('v2.ad.lan_on', { url: lan.url }) : t('v2.ad.lan_off')

  return (
    <>
      {desktop && <DesktopServerControl />}
      <div className="panel">
        <h4>{t('v2.ad.conn_title')} <span className="hint">{t('v2.ad.conn_hint')}{disconnected && t('v2.ad.disconnected_suffix')}</span></h4>
        <table><tbody>
          <tr>
            <td>{t('v2.ad.ext_row')}</td>
            <td style={{ color: external?.available ? 'var(--emerald)' : 'var(--faint)', fontSize: 11 }}>{extDesc}</td>
            <td><div className="row-act"><button onClick={() => refetch()} disabled={refreshing}>{refreshing ? t('v2.ad.checking') : t('v2.ad.reconnect')}</button></div></td>
          </tr>
          <tr>
            <td>{t('v2.ad.lan_row')}</td>
            <td style={{ color: 'var(--faint)', fontSize: 11 }}>{lanDesc}</td>
            <td />
          </tr>
        </tbody></table>
      </div>
      <DbToolsControl />
    </>
  )
}
