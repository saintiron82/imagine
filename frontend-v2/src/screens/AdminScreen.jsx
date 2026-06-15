import { useState, useEffect, useRef } from 'react'
import { useWorkers, useClusterValves, useWorkerControl, useHeadlessCommand, useFeedbackSummary, useAutoProcessing, useLogStream } from '../api/admin'
import { useConnectionInfo } from '../api/connection'
import { useMembersData, useMemberMutations } from '../api/members'
import { useDbAudit, useBackfill, useDbReset, useRepairParse, useDbExport, useDbImport } from '../api/tools'
import { useDomains, useActiveDomain, useDomainDetail, useSetActiveDomain, useSaveDomain, generateDomainPrompt } from '../api/classification'

/**
 * 관리 — 엔진룸. 기술 용어(MC/VV/MV)는 운영자 전용인 이 화면에만 허용된다.
 * 분석기: 병목 진단 + 분석기별 실시간(admin/workers) + 정지/차단/차단해제.
 * 멤버/플랜: members/invites/usage 실데이터. 미연결 시 빈 상태(가짜 데이터 없음).
 */
export default function AdminScreen() {
  const [tab, setTab] = useState('workers')

  return (
    <section id="scr-admin" className="screen active" style={{ height: '100%' }}>
      <aside className="adm-side">
        {[['workers', '분석기'], ['classification', '분류'], ['members', '멤버'], ['feedback', '피드백'], ['logs', '로그'], ['tools', '서버 도구']].map(([id, label]) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>
        ))}
      </aside>
      <div className="adm-main">
        {tab === 'workers' && <WorkersPanel />}
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
const PHASE_KR = { mc: 'MC', vv: 'VV', mv: 'MV', parse: '파싱', dl: 'DL' }

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

function WorkersPanel() {
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
        <h4>자동 분석 <span className="hint">서버 전역 정책 · 현재 모드 {globalMode}{disconnected && ' · 연결 끊김'}</span>
          <span className={`toggle ${apOn ? '' : 'off'}`} title={apOn ? '자동 분석 켜짐 — 끄려면 클릭' : '자동 분석 꺼짐 — 켜려면 클릭'}
            onClick={() => !disconnected && !ap.update.isPending && setAp({ enabled: !apOn })} /></h4>
        {bottleneck && bottleneck.pending > 0 ? (
          <div className="bottleneck">
            현재 병목: <b>{bottleneck.label}</b> — 대기 {bottleneck.pending.toLocaleString()}장 · 클러스터 처리력 <b>분당 {bottleneck.rate ?? 0}장</b> (이 단계가 전체 완료 속도를 결정)
            <div className="why">처리력 여유 단계는 이 단계 산출을 기다리는 중 — 병목 단계에 분석기를 더 붙이면 전체가 빨라집니다</div>
          </div>
        ) : (
          <div className="bottleneck" style={{ borderColor: 'var(--emerald-d)' }}>
            병목 없음 — 모든 단계가 따라잡았습니다
          </div>
        )}
        <div className="valves">
          {valves.map(v => {
            const isBn = bottleneck && v.phase === bottleneck.phase && v.pending > 0
            const rateTxt = v.rate != null ? `${v.rate}/분` : (v.done >= v.total ? '완료' : '진행 중')
            return (
              <div className={`valve ${isBn ? 'bn' : ''}`} key={v.phase}>
                <div className="ph">{v.label}{isBn && <span className="bn-tag">병목</span>}</div>
                <div className="n">{v.done.toLocaleString()}/{v.total.toLocaleString()}</div>
                <div className="rate">{rateTxt}</div>
                <span className="sw on">ON</span>
              </div>
            )
          })}
        </div>
        <div style={{ display: 'flex', gap: 18, alignItems: 'center', marginTop: 10, flexWrap: 'wrap' }}>
          <ApSetting label="배치 간 휴식" suffix="초" disabled={disconnected || ap.update.isPending}
            value={ap.config?.rest_after_batch_s} min={0} max={3600} onCommit={(n) => setAp({ rest_after_batch_s: n })} />
          <ApSetting label="배치 크기" suffix="장" disabled={disconnected || ap.update.isPending}
            value={ap.config?.batch_size} min={1} max={64} onCommit={(n) => setAp({ batch_size: n })} />
          <label style={{ fontSize: 11, color: 'var(--faint)', display: 'flex', gap: 6, alignItems: 'center', cursor: disconnected ? 'default' : 'pointer' }}>
            <input type="checkbox" disabled={disconnected || ap.update.isPending} checked={ap.config?.verbose_log ?? false} onChange={(e) => setAp({ verbose_log: e.target.checked })} /> 상세 로그
          </label>
        </div>
      </div>

      <div className="panel">
        <h4>분석기 <span className="hint">{online.length}대 온라인 — 정지=이번 세션 종료(재시작 가능) · 차단=재접속 거부{disconnected && ' · 연결 끊김'}</span></h4>
        <table>
          <thead>
            <tr><th>이름</th><th>상태</th><th>지금 처리 중</th><th>처리율</th><th>누적</th><th style={{ textAlign: 'right' }}>제어</th></tr>
          </thead>
          <tbody>
            {online.map(w => (
              <tr key={w.id}>
                <td><b>{w.worker_name}</b><div style={{ fontSize: 10, color: 'var(--faint)' }}>{ORIGIN_LABEL(w)}</div></td>
                <td><span className="badge b-ok">온라인</span></td>
                <td>{w.current_file
                  ? <span>{w.current_file} {w.current_phase && <span style={{ color: 'var(--cyan)', fontSize: 10 }}>{PHASE_KR[w.current_phase] || w.current_phase}</span>}</span>
                  : <span style={{ color: 'var(--faint)' }}>대기</span>}</td>
                <td className="mono">{w.throughput != null ? `${Number(w.throughput).toFixed(Number(w.throughput) < 10 ? 1 : 0)}/분` : '—'}{w.throughput_mode && <span style={{ color: 'var(--faint)', fontSize: 10 }}> {PHASE_KR[w.throughput_mode] || w.throughput_mode}</span>}</td>
                <td className="mono">{(w.jobs_completed || 0).toLocaleString()}</td>
                <td><div className="row-act">
                  <button disabled={!canControl || stop.isPending} onClick={() => stop.mutate(w.id)}>정지</button>
                  <button className="danger" disabled={!canControl || block.isPending} onClick={() => block.mutate(w.id)}>차단</button>
                </div></td>
              </tr>
            ))}
            {online.length === 0 && (
              <tr><td colSpan={6} style={{ color: 'var(--faint)' }}>온라인 분석기가 없습니다</td></tr>
            )}
          </tbody>
        </table>
        {blocked.map(w => (
          <div className="blocked" key={w.id}>
            ⛔ <b>{w.worker_name}</b> <span className="faint">— 차단됨 (재접속 불가)</span>
            <button className="unb" disabled={!canControl || unblock.isPending} onClick={() => unblock.mutate(w.id)}>차단 해제</button>
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
const MODE_LABEL = { direct_lan: '같은 네트워크 (LAN 직접)', manual_external: '외부 주소 직접 입력', relay_session: '릴레이 세션 (Cloudflare 터널 — 주소 비공유)' }

function EnrollWorkerPanel({ disabled }) {
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
  const errMsg = issue.isError ? (issue.error?.detail || issue.error?.message || '발급 실패') : ''
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
      <h4>분석기 추가 (원격) <span className="hint">다른 머신을 분석기로 붙이는 설치 명령·토큰 발급{disabled && ' · 연결 끊김'}</span>
        <button style={{ marginLeft: 'auto' }} onClick={() => setOpen(v => !v)} disabled={disabled}>{open ? '닫기' : '명령 생성…'}</button></h4>
      {open && (
        <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div><label style={lbl}>분석기 이름</label><input style={{ ...inp, fontFamily: 'monospace' }} value={name} onChange={e => setName(e.target.value)} /></div>
            <div><label style={lbl}>실행 방식 (launcher)</label>
              <select style={inp} value={launcher} onChange={e => setLauncher(e.target.value)}>
                <option value="cloud">cloud</option><option value="cli">cli</option><option value="service">service</option>
              </select></div>
            <div><label style={lbl}>토큰 만료 (분)</label><input style={{ ...inp, fontFamily: 'monospace' }} type="number" min={15} max={43200} value={expires} onChange={e => setExpires(e.target.value)} /></div>
            <div><label style={lbl}>접속 방식</label>
              <select style={inp} value={mode} onChange={e => setMode(e.target.value)}>
                {modeOptions.map(m => <option key={m} value={m}>{MODE_LABEL[m]}{available.size && !available.has(m) ? ' (미광고)' : ''}</option>)}
              </select></div>
          </div>
          {mode !== 'relay_session' && (
            <div><label style={lbl}>서버 주소 재정의 (선택)</label>
              <input style={{ ...inp, fontFamily: 'monospace' }} placeholder={requestOrigin || 'https://...'} value={overrideUrl} onChange={e => setOverrideUrl(e.target.value)} /></div>
          )}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button disabled={issue.isPending} onClick={generate}>{issue.isPending ? '발급 중…' : '설치 명령 생성'}</button>
            {result?.linux_command && <button onClick={copy}>{copied ? '복사됨 ✓' : '명령 복사'}</button>}
          </div>
          {errMsg && <div style={{ color: 'var(--red)', fontSize: 11 }}>{errMsg}</div>}
          {result?.linux_command && (
            <>
              <div style={{ fontSize: 11, color: 'var(--faint)' }}>
                분석기 계정 <b style={{ color: 'var(--text)' }}>{result.worker_username}</b> · 토큰 만료 {result.expires_minutes}분
                {result.relay_endpoint ? ` · 릴레이 ${result.relay_endpoint}` : result.server_url ? ` · ${result.server_url}` : ''}
                <br />⚠ 토큰이 포함된 1회용 명령입니다 — 워커 머신에서 그대로 실행하세요.
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
  const [open, setOpen] = useState(() => new Set())
  const toggle = (t) => setOpen(prev => { const n = new Set(prev); n.has(t) ? n.delete(t) : n.add(t); return n })
  const box = { border: '1px solid var(--line)', borderRadius: 8 }
  const sectionLabel = { fontSize: 11, color: 'var(--faint)', margin: '0 0 6px' }

  return (
    <div className="panel">
      <h4>{detail.name_ko || detail.name} <span className="hint">{detail.id} · 읽기 전용 (편집은 새 도메인 생성)</span></h4>
      <div style={{ marginBottom: 12 }}>
        <div style={sectionLabel}>이미지 타입 ({detail.image_types?.length || 0})</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>{(detail.image_types || []).map(t => <Chip key={t}>{t}</Chip>)}</div>
      </div>
      {Object.keys(detail.type_hints || {}).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={sectionLabel}>타입별 힌트</div>
          <div style={box}>
            {Object.entries(detail.type_hints).map(([type, hints]) => (
              <div key={type} style={{ borderBottom: '1px solid var(--line)' }}>
                <button onClick={() => toggle(type)} style={{ width: '100%', display: 'flex', gap: 8, alignItems: 'center', background: 'none', border: 'none', color: 'var(--text)', padding: '8px 10px', cursor: 'pointer', fontSize: 12 }}>
                  <span style={{ color: 'var(--faint)' }}>{open.has(type) ? '▾' : '▸'}</span>
                  <b>{type}</b><span style={{ marginLeft: 'auto', color: 'var(--faint)', fontSize: 10 }}>{Object.keys(hints).length} 필드</span>
                </button>
                {open.has(type) && <div style={{ padding: '0 10px 8px 28px' }}><HintRows obj={hints} /></div>}
              </div>
            ))}
          </div>
        </div>
      )}
      {detail.common_hints && Object.keys(detail.common_hints).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={sectionLabel}>공통 힌트 <span style={{ color: 'var(--faint)' }}>(_base.yaml — 자동)</span></div>
          <div style={{ ...box, padding: '8px 10px' }}><HintRows obj={detail.common_hints} /></div>
        </div>
      )}
      {detail.type_instructions && Object.keys(detail.type_instructions).length > 0 && (
        <div>
          <div style={sectionLabel}>타입별 지시문</div>
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

  const idError = domainId && !/^[a-z][a-z0-9_]*$/.test(domainId) ? '소문자 snake_case 만 가능'
    : domainId && existingIds.includes(domainId) ? '이미 존재하는 ID' : ''
  const canNext1 = domainId && nameEn && nameKo && description && !idError

  const next1 = () => { setPrompt(generateDomainPrompt({ domainId, nameEn, nameKo, description })); setStep(2) }
  const copy = async () => { try { await navigator.clipboard.writeText(prompt); setCopied(true); setTimeout(() => setCopied(false), 1800) } catch { /* noop */ } }
  const doSave = () => {
    setError('')
    const cleaned = yamlInput.replace(/^```[\w]*\n?/, '').replace(/\n?```\s*$/, '').trim()
    save.mutate({ domainId, yamlContent: cleaned }, {
      onSuccess: (r) => { if (r?.success) onClose(); else setError(r?.detail || r?.error || '저장 실패') },
      onError: (e) => setError(e?.detail || e?.message || '저장 실패'),
    })
  }

  const overlay = { position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,.6)' }
  const modal = { background: 'var(--panel)', border: '1px solid var(--line)', borderRadius: 12, width: 'min(640px, 92vw)', maxHeight: '82vh', overflowY: 'auto', padding: 20 }
  const inp = { width: '100%', background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 10px', color: 'var(--text)', fontSize: 12, boxSizing: 'border-box' }
  const lbl = { display: 'block', fontSize: 11, color: 'var(--faint)', margin: '0 0 4px' }
  const stepName = step === 1 ? '정의' : step === 2 ? 'AI 프롬프트 복사' : 'YAML 붙여넣기'

  return (
    <div style={overlay} onClick={onClose}>
      <div style={modal} onClick={e => e.stopPropagation()}>
        <h4 style={{ marginTop: 0 }}>새 도메인 <span className="hint">{step}/3 · {stepName}</span></h4>
        {step === 1 && <div style={{ display: 'grid', gap: 12 }}>
          <div><label style={lbl}>도메인 ID (snake_case)</label>
            <input style={inp} value={domainId} onChange={e => setDomainId(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))} placeholder="예: medical_image" autoFocus />
            {idError && <div style={{ color: 'var(--red)', fontSize: 11, marginTop: 4 }}>{idError}</div>}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div><label style={lbl}>이름 (영문)</label><input style={inp} value={nameEn} onChange={e => setNameEn(e.target.value)} placeholder="Medical Image" /></div>
            <div><label style={lbl}>이름 (한글)</label><input style={inp} value={nameKo} onChange={e => setNameKo(e.target.value)} placeholder="의료 이미지" /></div>
          </div>
          <div><label style={lbl}>설명 / 용도</label><textarea style={{ ...inp, resize: 'none' }} rows={4} value={description} onChange={e => setDescription(e.target.value)} placeholder="이 도메인이 다루는 이미지 종류와 용도" /></div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button onClick={onClose}>취소</button>
            <button disabled={!canNext1} onClick={next1}>다음</button>
          </div>
        </div>}
        {step === 2 && <div style={{ display: 'grid', gap: 12 }}>
          <div style={{ fontSize: 12, color: 'var(--faint)' }}>이 프롬프트를 LLM(Claude 등)에 붙여넣어 YAML 을 생성한 뒤, 다음 단계에 결과를 붙여넣으세요.</div>
          <pre style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: 10, fontSize: 11, maxHeight: '40vh', overflowY: 'auto', whiteSpace: 'pre-wrap', margin: 0 }}>{prompt}</pre>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setStep(1)}>이전</button>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={copy}>{copied ? '복사됨 ✓' : '프롬프트 복사'}</button>
              <button onClick={() => setStep(3)}>다음</button>
            </div>
          </div>
        </div>}
        {step === 3 && <div style={{ display: 'grid', gap: 12 }}>
          <label style={lbl}>생성된 YAML 붙여넣기</label>
          <textarea style={{ ...inp, resize: 'none', fontFamily: 'monospace' }} rows={16} value={yamlInput} onChange={e => { setYamlInput(e.target.value); setError('') }} placeholder={'domain:\n  id: ...'} />
          {error && <div style={{ color: 'var(--red)', fontSize: 11, background: 'rgba(248,113,113,.08)', border: '1px solid rgba(248,113,113,.3)', borderRadius: 6, padding: '8px 10px' }}>{error}</div>}
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setStep(2)}>이전</button>
            <button disabled={save.isPending || !yamlInput.trim()} onClick={doSave}>{save.isPending ? '저장 중…' : '도메인 저장'}</button>
          </div>
        </div>}
      </div>
    </div>
  )
}

function ClassificationPanel() {
  const { data: domains = [], isLoading, isError } = useDomains()
  const { data: active } = useActiveDomain()
  const setActive = useSetActiveDomain()
  const [selectedId, setSelectedId] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const activeId = active?.active_domain || null
  const effectiveId = selectedId || activeId || domains[0]?.id || null
  const { data: detail } = useDomainDetail(effectiveId)

  if (isLoading) return <div className="panel"><span style={{ color: 'var(--faint)' }}>도메인을 불러오는 중…</span></div>
  if (isError) return <div className="panel"><span style={{ color: 'var(--faint)' }}>서버에 연결되지 않았습니다</span></div>

  return (
    <>
      <div className="panel">
        <h4>분류 도메인 <span className="hint">활성 도메인이 분류 기준·태그 공간·검색 필터 옵션을 정한다</span>
          <button style={{ marginLeft: 'auto' }} onClick={() => setShowCreate(true)}>+ 새 도메인</button></h4>
        <table>
          <thead><tr><th>도메인</th><th>타입</th><th style={{ textAlign: 'right' }}>상태</th></tr></thead>
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
                    ? <span className="badge b-ok">활성</span>
                    : <button disabled={setActive.isPending} onClick={(e) => { e.stopPropagation(); setActive.mutate(d.id) }}>활성화</button>}</td>
                </tr>
              )
            })}
            {domains.length === 0 && <tr><td colSpan={3} style={{ color: 'var(--faint)' }}>도메인이 없습니다</td></tr>}
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
const LOG_CATS = [['all', '전체'], ['job', '작업'], ['network', '네트워크'], ['general', '일반']]

function LogPanel() {
  const [cat, setCat] = useState('all')
  const { entries, clear, disconnected } = useLogStream(true)
  const scrollRef = useRef(null)
  const filtered = cat === 'all' ? entries : entries.filter(e => e.category === cat)
  useEffect(() => { const el = scrollRef.current; if (el) el.scrollTop = el.scrollHeight }, [filtered.length])

  const chip = (id) => ({ fontSize: 11, background: cat === id ? 'var(--panel2)' : 'none', border: '1px solid var(--line)', borderRadius: 6, padding: '3px 9px', color: cat === id ? 'var(--text)' : 'var(--dim)', cursor: 'pointer' })

  return (
    <div className="panel">
      <h4>실시간 로그 <span className="hint">서버 처리 로그 · 1.5초 폴링{disconnected && ' · 연결 끊김'}</span>
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          {LOG_CATS.map(([id, l]) => <button key={id} style={chip(id)} onClick={() => setCat(id)}>{l}</button>)}
          <button style={{ fontSize: 11, border: '1px solid var(--line)', borderRadius: 6, padding: '3px 9px', background: 'none', color: 'var(--dim)', cursor: 'pointer' }} onClick={clear}>지우기</button>
        </span>
      </h4>
      <div ref={scrollRef} style={{ height: '62vh', overflowY: 'auto', background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>
        {filtered.length === 0 && <div style={{ color: 'var(--faint)' }}>로그 없음 — 처리 활동이 생기면 여기에 실시간으로 흐릅니다</div>}
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
  const { disconnected, loading, total30d, topFiles, topQueries } = useFeedbackSummary()
  if (loading) return <div className="panel"><span style={{ color: 'var(--faint)' }}>피드백을 불러오는 중…</span></div>
  if (disconnected) return <div className="panel"><span style={{ color: 'var(--faint)' }}>서버에 연결되지 않았습니다</span></div>

  return (
    <>
      <div className="panel">
        <h4>검색 피드백 <span className="hint">사용자가 '부적절'로 표시한 결과 — 검색 품질 진단</span></h4>
        <div style={{ fontSize: 28, fontWeight: 600 }}>{total30d.toLocaleString()}<span style={{ fontSize: 12, color: 'var(--faint)', fontWeight: 400 }}> 건 (지난 30일)</span></div>
        {total30d === 0 && <div style={{ color: 'var(--faint)', fontSize: 12, marginTop: 6 }}>최근 부적절 표시가 없습니다 — 검색 결과 만족도 양호.</div>}
      </div>
      {topQueries.length > 0 && (
        <div className="panel">
          <h4>최다 플래그 쿼리 <span className="hint">이 검색어들이 부적절 결과를 자주 냄</span></h4>
          <table>
            <thead><tr><th>쿼리</th><th style={{ textAlign: 'right' }}>플래그</th></tr></thead>
            <tbody>{topQueries.map((q, i) => (
              <tr key={i}><td>{q.query}</td><td className="mono" style={{ textAlign: 'right' }}>{q.count.toLocaleString()}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {topFiles.length > 0 && (
        <div className="panel">
          <h4>최다 플래그 파일 <span className="hint">이 파일들이 부적절 결과로 자주 지목됨</span></h4>
          <table>
            <thead><tr><th>파일 ID</th><th style={{ textAlign: 'right' }}>플래그</th></tr></thead>
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
    ? `${invite.data.results.filter(r => r.ok).length}건 처리${invite.data.smtp_configured ? ' · 메일 발송' : ' · 링크 생성(메일 미설정)'}`
    : invite.isPending ? '초대 중…' : null

  return (
    <>
      <div className="panel">
        <h4>플랜 <span className="hint">서버 단위 라이선스 — 한도는 여기서 옴{disconnected && ' · 연결 끊김'}</span></h4>
        <div className="kpis">
          <div className="kpi"><div className="v" style={{ color: usage.expired ? 'var(--red)' : '#93c5fd' }}>{usage.expired ? '만료됨' : '활성'}</div><div className="k">{usage.expires_at ? `${usage.expires_at} 만료` : '만료 없음'}</div></div>
          <div className="kpi"><div className="v">{usage.seats_used} <span className="faint" style={{ fontSize: 12 }}>/ {usage.seat_limit || '∞'}</span></div><div className="k">좌석 — 멤버 {usage.members} + 대기 초대 {usage.pending_invites}</div></div>
          <div className="kpi"><div className="v" style={{ color: usage.smtp_configured ? 'var(--emerald)' : 'var(--faint)' }}>{usage.smtp_configured ? '메일 ON' : '메일 OFF'}</div><div className="k">초대 발송 방식</div></div>
          <div className="kpi" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <button style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '7px 16px', borderRadius: 6 }}>갱신 / 업그레이드</button>
          </div>
        </div>
        <div style={{ fontSize: 10, color: 'var(--amber)', marginTop: 6 }}>만료 시 이 서버 접속이 차단됩니다 (로그인은 되어도 입장 불가) · 데이터는 보존되며 갱신 즉시 복귀</div>
      </div>

      <div className="panel">
        <h4>초대 <span className="hint">이메일로 초대 → 링크 수락 → 좌석 점유 · 그 이메일로만 수락 가능{!usage.smtp_configured && ' · 메일 미설정 시 링크 수동 공유'}</span></h4>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <input placeholder="이메일 주소 — 쉼표로 여러 명" value={emails} onChange={e => setEmails(e.target.value)}
            style={{ flex: 1, background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 12px', color: 'var(--text)', fontSize: 12 }} />
          <button onClick={sendInvites} disabled={!canMutate || invite.isPending || !emails.trim()}
            style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '0 18px', borderRadius: 6, opacity: canMutate ? 1 : .5 }}>초대 보내기</button>
        </div>
        {inviteMsg && <div style={{ fontSize: 10.5, color: 'var(--cyan)', marginBottom: 8 }}>{inviteMsg}</div>}
        {disconnected && <div style={{ fontSize: 9.5, color: 'var(--faint)', marginBottom: 8 }}>서버 연결 시 실제 초대가 발송됩니다</div>}
        <table>
          <thead><tr><th>대기 중 초대</th><th>역할</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            {invites.map(iv => (
              <tr key={iv.id}>
                <td>{iv.email} <span style={{ fontSize: 9, color: 'var(--amber)' }}>좌석 예약 중</span></td>
                <td className="mono">{iv.role}</td>
                <td><div className="row-act"><button className="danger" disabled={!canMutate} onClick={() => revoke.mutate(iv.id)}>취소</button></div></td>
              </tr>
            ))}
            {invites.length === 0 && <tr><td colSpan={3} style={{ color: 'var(--faint)' }}>대기 중 초대 없음</td></tr>}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h4>멤버 <span className="hint">제거 = 좌석 반환 · 비활성 = 접속만 차단(좌석 유지)</span></h4>
        <table>
          <thead><tr><th>이름</th><th>역할</th><th>가입 경로</th><th>마지막 접속</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            {members.map(mb => (
              <tr key={mb.id}>
                <td>{mb.username}<div style={{ fontSize: 9.5, color: 'var(--faint)' }}>{mb.email}</div></td>
                <td>{mb.role === 'admin'
                  ? <span className="badge" style={{ background: 'rgba(192,132,252,.15)', color: 'var(--purple)' }}>운영자</span>
                  : <span className="badge b-ok">{mb.is_active ? '사용자' : '비활성'}</span>}</td>
                <td className="mono" style={{ fontSize: 10 }}>{mb.via === 'firebase' ? '초대/Firebase' : '비밀번호'}</td>
                <td className="mono">{mb.last_login_at || '—'}</td>
                <td>{mb.role !== 'admin' && (
                  <div className="row-act">
                    <button disabled={!canMutate} onClick={() => deactivate.mutate(mb.id)}>비활성</button>
                    <button className="danger" disabled={!canMutate} onClick={() => remove.mutate(mb.id)}>제거</button>
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
  const d = window.imagineDesktop
  const [st, setSt] = useState(null) // {running, autostart}
  const [busy, setBusy] = useState(false)
  const refresh = () => d.serverStatus().then(setSt).catch(() => setSt({ running: false, autostart: false }))
  useEffect(() => { refresh() }, []) // eslint-disable-line

  const start = async () => { setBusy(true); try { await d.startServer() } finally { await refresh(); setBusy(false) } }
  const stop = async () => {
    if (!confirm('이 컴퓨터의 서버를 끕니다.\n끄면 팀원도 접속할 수 없습니다. 계속할까요?')) return
    setBusy(true); try { await d.stopServer() } finally { await refresh(); setBusy(false) }
  }
  const toggleAuto = async (e) => { setBusy(true); try { await d.setAutostart(e.target.checked) } finally { await refresh(); setBusy(false) } }

  const running = st?.running
  return (
    <div className="panel">
      <h4>이 컴퓨터의 서버 <span className="hint">독립·상주 — 앱 창을 닫아도 팀에 계속 서비스합니다</span></h4>
      <table><tbody>
        <tr>
          <td>서버 프로세스</td>
          <td style={{ color: running ? 'var(--emerald)' : 'var(--faint)', fontSize: 11 }}>
            {st === null ? '확인 중…' : running ? '● 실행 중 (상주)' : '○ 꺼짐'}
          </td>
          <td><div className="row-act">
            {running
              ? <button className="danger" disabled={busy} onClick={stop}>서버 끄기</button>
              : <button disabled={busy} onClick={start}>서버 켜기</button>}
          </div></td>
        </tr>
        <tr>
          <td>부팅 시 자동 실행<div style={{ fontSize: 10, color: 'var(--faint)' }}>로그인하면 서버가 조용히 켜짐(창은 안 뜸)</div></td>
          <td />
          <td><label className="chk" style={{ justifyContent: 'flex-end' }}>
            <input type="checkbox" checked={!!st?.autostart} disabled={busy || st === null} onChange={toggleAuto} /> 자동 실행
          </label></td>
        </tr>
      </tbody></table>
    </div>
  )
}

function DbToolsControl() {
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
    ? `전체 ${audit.data.total_files?.toLocaleString() ?? '—'}장 · 미완성 ${audit.data.total_incomplete?.toLocaleString() ?? 0}장 · 폴더 ${(audit.data.folders || []).length}개`
    : audit.isPending ? '검사 중…' : audit.isError ? '검사 실패' : '미완성/잔여 작업 검사'
  const bfRunning = bf?.running
  const bfMsg = bf
    ? (bfRunning
        ? `진행 중 ${(bf.done ?? 0).toLocaleString()} / ${(bf.total ?? 0).toLocaleString()}${bf.failed ? ` · 실패 ${bf.failed}` : ''}`
        : `대기 — 마지막: 완료 ${(bf.done ?? 0).toLocaleString()}${bf.skipped ? ` · 건너뜀 ${bf.skipped}` : ''}${bf.last_error ? ` · 오류 있음` : ''}`)
    : '파생물 캐시 자격 부여 (~16KB/파일)'
  const resetMsg = reset.data ? `초기화됨 — 파일 ${(reset.data.file_count ?? reset.data.deleted ?? 0).toLocaleString()}장 삭제` : reset.isError ? (reset.error?.detail || '초기화 실패(비번 확인)') : '비밀번호 확인 필요'
  const rpRunning = rp?.running
  const rpp = rp?.progress
  const rpMsg = rpp
    ? (rpRunning
        ? `진행 중 ${(rpp.done ?? 0).toLocaleString()} / ${(rpp.total ?? 0).toLocaleString()}${rpp.failed ? ` · 실패 ${rpp.failed}` : ''}${rpp.current_file ? ` · ${rpp.current_file}` : ''}`
        : `대기 — 마지막: 완료 ${(rpp.done ?? 0).toLocaleString()}${rpp.skipped ? ` · 건너뜀 ${rpp.skipped}` : ''}${rpp.failed ? ` · 실패 ${rpp.failed}` : ''}`)
    : '파싱 누락/손상 데이터 재처리'

  const doReset = () => {
    if (!pw) return
    reset.mutate(pw, { onSuccess: () => { setPwOpen(false); setPw('') } })
  }
  // 내보내기/가져오기 둘 다 data 가 남으므로 더 최근(submittedAt) 액션을 표시.
  const recentImport = (dbImport.submittedAt || 0) >= (dbExport.submittedAt || 0)
  const ioError = recentImport ? dbImport.isError : dbExport.isError
  let ioMsg = '전체 DB 스냅샷 백업/복원'
  if (dbExport.isPending) ioMsg = '내보내는 중…'
  else if (dbImport.isPending) ioMsg = '복원 중…'
  else if (recentImport && dbImport.isError) ioMsg = dbImport.error?.detail || dbImport.error?.message || '복원 실패'
  else if (recentImport && dbImport.data) ioMsg = `복원됨 — 파일 ${(dbImport.data.file_count ?? 0).toLocaleString()}장`
  else if (!recentImport && dbExport.isError) ioMsg = dbExport.error?.message || '내보내기 실패'
  else if (!recentImport && dbExport.data) ioMsg = `내보냄 — ${dbExport.data.filename} (${Math.round((dbExport.data.bytes || 0) / 1024).toLocaleString()} KB)`
  const doImport = () => {
    if (!impPw || !impFile) return
    dbImport.mutate({ password: impPw, file: impFile }, {
      onSuccess: () => { setImpOpen(false); setImpPw(''); setImpFile(null) },
    })
  }

  return (
    <div className="panel">
      <h4>DB·정합성 도구 <span className="hint">정합성 감사·해시 백필·초기화</span></h4>
      <table><tbody>
        <tr>
          <td>정합성 감사</td>
          <td style={{ color: 'var(--faint)', fontSize: 11 }}>{auditMsg}</td>
          <td><div className="row-act"><button disabled={audit.isPending} onClick={() => audit.mutate()}>실행</button></div></td>
        </tr>
        <tr>
          <td>해시 백필</td>
          <td style={{ color: bfRunning ? 'var(--cyan)' : 'var(--faint)', fontSize: 11 }}>{bfMsg}</td>
          <td><div className="row-act">
            {bfRunning
              ? <button className="danger" disabled={bfStop.isPending} onClick={() => bfStop.mutate()}>중지</button>
              : <button disabled={bfStart.isPending} onClick={() => bfStart.mutate()}>실행</button>}
          </div></td>
        </tr>
        <tr>
          <td>Parse 데이터 복구</td>
          <td style={{ color: rpRunning ? 'var(--cyan)' : 'var(--faint)', fontSize: 11 }}>{rpMsg}</td>
          <td><div className="row-act"><button disabled={rpRunning || rpStart.isPending} onClick={() => rpStart.mutate()}>{rpRunning ? '진행 중' : '실행'}</button></div></td>
        </tr>
        <tr>
          <td>DB 내보내기 / 가져오기</td>
          <td style={{ color: ioError ? 'var(--red)' : 'var(--faint)', fontSize: 11 }}>{ioMsg}</td>
          <td><div className="row-act">
            <button disabled={dbExport.isPending} onClick={() => dbExport.mutate()}>내보내기</button>
            <button className="danger" onClick={() => setImpOpen(v => !v)}>가져오기…</button>
          </div></td>
        </tr>
        {impOpen && (
          <tr>
            <td colSpan={3}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', background: 'rgba(248,113,113,.08)', border: '1px solid rgba(248,113,113,.3)', borderRadius: 6, padding: '8px 10px' }}>
                <span style={{ fontSize: 11, color: 'var(--red)' }}>⚠ 현재 DB를 업로드한 스냅샷으로 완전히 덮어씁니다(사용자/인증 포함). 운영자 비밀번호 확인:</span>
                <input type="password" value={impPw} onChange={e => setImpPw(e.target.value)} placeholder="비밀번호" style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '6px 10px', color: 'var(--text)', fontSize: 12, width: 150 }} />
                <input ref={fileRef} type="file" accept=".db,.sqlite,.sqlite3,application/x-sqlite3" style={{ display: 'none' }} onChange={e => setImpFile(e.target.files?.[0] || null)} />
                <button onClick={() => fileRef.current?.click()}>{impFile ? impFile.name : '백업 파일 선택…'}</button>
                <button className="danger" disabled={!impPw || !impFile || dbImport.isPending} onClick={doImport}>{dbImport.isPending ? '복원 중…' : '복원 실행'}</button>
                <button onClick={() => { setImpOpen(false); setImpPw(''); setImpFile(null) }}>취소</button>
              </div>
            </td>
          </tr>
        )}
        <tr>
          <td style={{ color: 'var(--red)' }}>DB 초기화</td>
          <td style={{ color: reset.isError ? 'var(--red)' : 'var(--faint)', fontSize: 11 }}>{resetMsg}</td>
          <td><div className="row-act"><button className="danger" onClick={() => setPwOpen(v => !v)}>초기화…</button></div></td>
        </tr>
        {pwOpen && (
          <tr>
            <td colSpan={3}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', background: 'rgba(248,113,113,.08)', border: '1px solid rgba(248,113,113,.3)', borderRadius: 6, padding: '8px 10px' }}>
                <span style={{ fontSize: 11, color: 'var(--red)' }}>⚠ 모든 파일 분석 데이터를 삭제합니다. 운영자 비밀번호 확인:</span>
                <input type="password" value={pw} onChange={e => setPw(e.target.value)} onKeyDown={e => e.key === 'Enter' && doReset()} placeholder="비밀번호" style={{ background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '6px 10px', color: 'var(--text)', fontSize: 12, width: 160 }} autoFocus />
                <button className="danger" disabled={!pw || reset.isPending} onClick={doReset}>{reset.isPending ? '초기화 중…' : '초기화 실행'}</button>
                <button onClick={() => { setPwOpen(false); setPw('') }}>취소</button>
              </div>
            </td>
          </tr>
        )}
      </tbody></table>
    </div>
  )
}

function ToolsPanel() {
  const { disconnected, external, lan } = useConnectionInfo()
  const desktop = typeof window !== 'undefined' && window.imagineDesktop // 데스크톱 셸에서만
  const extDesc = external?.available
    ? `● 연결됨 — ${external.url || '터널'} · 주소 공유 불필요`
    : '○ 외부 접속 꺼짐 — 터널 미연결'
  const lanDesc = lan?.available ? `${lan.url} · 빠른 경로 (자동 발견)` : 'LAN 경로 없음'

  return (
    <>
      {desktop && <DesktopServerControl />}
      <div className="panel">
        <h4>접속 <span className="hint">외부(터널)·LAN{disconnected && ' · 연결 끊김'}</span></h4>
        <table><tbody>
          <tr>
            <td>외부 접속 (Cloudflare 터널)</td>
            <td style={{ color: external?.available ? 'var(--emerald)' : 'var(--faint)', fontSize: 11 }}>{extDesc}</td>
            <td><div className="row-act"><button>재연결</button></div></td>
          </tr>
          <tr>
            <td>같은 네트워크 (LAN)</td>
            <td style={{ color: 'var(--faint)', fontSize: 11 }}>{lanDesc}</td>
            <td />
          </tr>
        </tbody></table>
      </div>
      <DbToolsControl />
    </>
  )
}
