import { useState } from 'react'
import { useAnalysisData, useJobControl, useJobHistory, useJobErrors, useIncompleteStats, useDbStats } from '../api/analysis'

/**
 * 분석 — "무엇이 되고 있나". 작업 리스트 + 지금 처리 중 라이브 모니터.
 * 원칙(확정): 전문 용어 0 (내부 단계명 노출 금지), ETA 없음, 완료 잔류 없음.
 * 화면의 주인공은 잡 메타데이터가 아니라 큐의 내용물이다.
 *
 * 데이터: useAnalysisData() (analysis-jobs + admin/workers 집계).
 * 미연결/오류 시 가짜 데이터 없이 빈 상태 + 정직한 표식.
 * 완료 잔류 없음 원칙: 히스토리(IMGV2-20)는 기본 숨김·opt-in 별도 뷰.
 */
// 작업 단위 상태 — 내부 단계명(MC/VV/MV) 노출 금지, 사용자 언어로.
const ACTIVITY = {
  running:  { label: '처리 중',       color: 'var(--emerald)' },
  queued:   { label: '분석 대기',     color: 'var(--cyan)' },
  upstream: { label: '다운로드 대기', color: 'var(--dim)' },
  paused:   { label: '일시정지',      color: 'var(--faint)' },
  waiting:  { label: '대기',          color: 'var(--faint)' },
  done:     { label: '완료',          color: 'var(--faint)' },
}

export default function AnalysisScreen() {
  const { disconnected, loading, summary, analyzers, jobs, totalFailed } = useAnalysisData()
  const db = useDbStats()
  const { pause, resume, cancel, retry, dismiss } = useJobControl()
  const [showHistory, setShowHistory] = useState(false)

  const jobActions = (j) => {
    if (j.status === 'paused') return [{ label: '재개', fn: () => resume.mutate(j.id) }, { label: '취소', fn: () => cancel.mutate(j.id) }]
    return [{ label: '일시정지', fn: () => pause.mutate(j.id) }, { label: '취소', fn: () => cancel.mutate(j.id) }]
  }

  const pausedJobs = jobs.filter(j => j.status === 'paused')

  return (
    <section id="scr-analysis" className="screen active">
      <div style={{ maxWidth: 880, margin: '0 auto', padding: '20px 24px' }}>
        <div className="sec-title" style={{ display: 'flex', alignItems: 'baseline' }}>
          <span>분석 리스트 <span className="sub">등록 순이 아니라 작업 간 공정 배분으로 처리</span></span>
          <button style={{ marginLeft: 'auto', fontSize: 11.5, background: 'none', border: '1px solid var(--line)', borderRadius: 6, padding: '4px 10px', color: showHistory ? 'var(--text)' : 'var(--dim)', cursor: 'pointer' }}
            onClick={() => setShowHistory(v => !v)}>{showHistory ? '히스토리 닫기' : '히스토리'}</button>
        </div>

        {!disconnected && pausedJobs.length > 0 && (
          <ResumeBanner pausedJobs={pausedJobs} onResumeAll={() => pausedJobs.forEach(j => resume.mutate(j.id))} resuming={resume.isPending} />
        )}

        {disconnected && (
          <div style={{
            margin: '8px 0 14px', padding: '8px 12px', borderRadius: 6, fontSize: 11.5,
            background: 'rgba(248,113,113,.10)', border: '1px solid rgba(248,113,113,.25)', color: 'var(--red)',
          }}>
            ● 서버 연결 끊김 {loading ? '— 재시도 중…' : '— 데이터를 불러올 수 없습니다'}
          </div>
        )}

        <div className="summary">
          <div className="sum-row">
            <span className="big">{db.analyzed.toLocaleString()}</span>
            <span className="of">/ {db.total.toLocaleString()} 장 분석 완료 · DB 보관</span>
            <div className="bar"><i style={{ width: `${db.pct}%` }} /></div>
            <span className="mono2" style={{ color: '#93c5fd', fontWeight: 700 }}>{db.pct}%</span>
          </div>
          <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--dim)' }}>
            활성 작업 <b style={{ color: 'var(--text)' }}>{summary.activeCount}개</b> · 잔여 <b style={{ color: 'var(--text)' }}>{summary.remainingTotal.toLocaleString()}장</b> · 지금 <b style={{ color: 'var(--emerald)' }}>분당 {summary.ratePerMin}장</b>
          </div>
        </div>

        <div className="sec-sub" style={{ margin: '18px 0 6px', fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
          작업 <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--faint)' }}>— 무엇을 · 얼마나 처리됐나</span>
        </div>
        <div className="jobs">
          {jobs.map(j => {
            const meta = ACTIVITY[j.activity] || ACTIVITY.queued
            const running = j.activity === 'running'
            return (
              <div className={`jrow ${running ? '' : 'waiting'}`} key={j.id}>
                <span className={running ? 'st run' : 'st'} style={running ? undefined : { background: 'var(--faint)' }} />
                <span className="nm">{j.name}</span>
                <span style={{ fontSize: 10, fontWeight: 700, color: meta.color, minWidth: 54, textAlign: 'right' }}>{meta.label}</span>
                <div className="bar"><i style={{ width: `${j.pct}%` }} /></div>
                <span className="cnt">
                  {j.done.toLocaleString()} / {j.total.toLocaleString()}
                  {j.remaining > 0 && <span style={{ color: 'var(--faint)' }}> · 잔여 {j.remaining.toLocaleString()}</span>}
                </span>
                {j.failed > 0 && (
                  <span className="plain-fail" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="n">실패 {j.failed}</span>
                    <button style={{ fontSize: 10 }} disabled={retry.isPending} onClick={() => retry.mutate(j.id)}>재시도</button>
                    <button style={{ fontSize: 10 }} disabled={dismiss.isPending}
                      onClick={() => { if (window.confirm('재시도를 소진한 영구 실패 파일을 실패 목록에서 제외할까요? (파일은 삭제되지 않음)')) dismiss.mutate(j.id) }}>무시</button>
                  </span>
                )}
                <div className="acts2">
                  {jobActions(j).map(a => <button key={a.label} onClick={a.fn}>{a.label}</button>)}
                </div>
              </div>
            )
          })}
          {jobs.length === 0 && !loading && (
            <div className="jrow" style={{ color: 'var(--faint)' }}>
              <span className="nm">등록된 분석 작업이 없습니다 — [＋ 추가]에서 폴더를 등록하세요</span>
            </div>
          )}
        </div>

        <div className="sec-sub" style={{ margin: '18px 0 6px', fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
          분석기 <b style={{ color: 'var(--emerald)' }}>{summary.analyzerCount}대</b> <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--faint)' }}>— 누가 · 지금 무엇을 · 처리율</span>
        </div>
        <div className="jobs">
          {analyzers.map(a => (
            <div className={`jrow ${a.busy ? '' : 'waiting'}`} key={a.id}>
              <span className={a.busy ? 'st run' : 'st'} style={a.busy ? undefined : { background: 'var(--faint)' }} />
              <span className="nm">{a.name}</span>
              <span className="cnt" style={{ flex: 1, minWidth: 0, color: a.busy ? 'var(--text)' : 'var(--faint)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.busy ? a.file : '대기 중'}</span>
              <span className="mono2" style={{ color: 'var(--emerald)', fontWeight: 700 }}>{a.rate}</span>
            </div>
          ))}
          {analyzers.length === 0 && (
            <div className="jrow" style={{ color: 'var(--faint)' }}>
              <span className="nm">연결된 분석기 없음</span>
            </div>
          )}
        </div>
        {totalFailed > 0 && (
          <div className="plain-fail" style={{ marginTop: 12 }}>
            <span className="n">실패 {totalFailed}건</span>
            <span>작업 행의 재시도/무시 또는 히스토리에서 상세 확인</span>
          </div>
        )}

        {showHistory && <HistorySection />}
      </div>
    </section>
  )
}

// ── 이어하기 제안 (IMGV2-25) ─────────────────────────────────
// 일시정지된 작업이 있으면 상단에 이어하기/무시 배너. 미완료 파일 수는 보조 정보.
function ResumeBanner({ pausedJobs, onResumeAll, resuming }) {
  const { totalIncomplete } = useIncompleteStats()
  const [dismissed, setDismissed] = useState(false)
  if (dismissed) return null
  return (
    <div style={{
      margin: '8px 0 14px', padding: '10px 14px', borderRadius: 8, fontSize: 12,
      background: 'rgba(96,165,250,.10)', border: '1px solid rgba(96,165,250,.30)',
      display: 'flex', alignItems: 'center', gap: 12,
    }}>
      <span style={{ color: 'var(--cyan)' }}>
        일시정지된 작업 <b>{pausedJobs.length}개</b>{totalIncomplete > 0 ? ` · 미완료 ${totalIncomplete.toLocaleString()}장` : ''} — 이어서 처리할까요?
      </span>
      <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
        <button disabled={resuming} onClick={onResumeAll}>{resuming ? '재개 중…' : '모두 이어하기'}</button>
        <button onClick={() => setDismissed(true)}>무시</button>
      </div>
    </div>
  )
}

// ── 완료 작업 히스토리 (IMGV2-20) ───────────────────────────
const HIST_STATUS = { completed: { label: '완료', color: 'var(--emerald)' }, cancelled: { label: '취소', color: 'var(--faint)' }, archived: { label: '보관', color: 'var(--dim)' } }

function HistorySection() {
  const { jobs, loading, isError } = useJobHistory(true)
  return (
    <div className="live" style={{ marginTop: 16 }}>
      <div className="live-head">완료/지난 작업 <span className="rate">{jobs.length}건</span></div>
      {loading && <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: '8px 2px' }}>불러오는 중…</div>}
      {isError && <div style={{ fontSize: 11.5, color: 'var(--red)', padding: '8px 2px' }}>히스토리를 불러올 수 없습니다</div>}
      {!loading && !isError && jobs.length === 0 && <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: '8px 2px' }}>종료된 작업이 없습니다</div>}
      {jobs.map(j => <HistoryRow key={j.id} job={j} />)}
    </div>
  )
}

function HistoryRow({ job }) {
  const [open, setOpen] = useState(false)
  const { errors, count, loading } = useJobErrors(job.id, open)
  const { retry, dismiss } = useJobControl()
  const s = HIST_STATUS[job.status] || { label: job.status, color: 'var(--faint)' }
  const when = job.createdAt ? new Date(job.createdAt).toLocaleString('ko-KR', { dateStyle: 'short', timeStyle: 'short' }) : ''
  return (
    <div style={{ borderTop: '1px solid var(--line)', padding: '8px 2px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12 }}>
        <span style={{ color: s.color, fontSize: 10, fontWeight: 700, minWidth: 28 }}>{s.label}</span>
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{job.name}</span>
        <span className="mono2" style={{ color: 'var(--dim)' }}>{job.done.toLocaleString()} / {job.total.toLocaleString()}</span>
        {when && <span style={{ color: 'var(--faint)', fontSize: 10 }}>{when}</span>}
        {job.failed > 0
          ? <button style={{ fontSize: 10 }} onClick={() => setOpen(v => !v)}>{open ? '오류 닫기' : `오류 ${job.failed}`}</button>
          : <span style={{ width: 56 }} />}
      </div>
      {open && (
        <div style={{ marginTop: 6, paddingLeft: 38 }}>
          {job.failed > 0 && (
            <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
              <button style={{ fontSize: 10 }} disabled={retry.isPending} onClick={() => retry.mutate(job.id)}>실패 재시도</button>
              <button style={{ fontSize: 10 }} disabled={dismiss.isPending}
                onClick={() => { if (window.confirm('영구 실패 파일을 실패 목록에서 제외할까요? (파일은 삭제되지 않음)')) dismiss.mutate(job.id) }}>무시</button>
            </div>
          )}
          {loading && <div style={{ fontSize: 11, color: 'var(--faint)' }}>불러오는 중…</div>}
          {!loading && count === 0 && <div style={{ fontSize: 11, color: 'var(--faint)' }}>상세 오류 기록이 없습니다</div>}
          {errors.map((e, i) => (
            <div key={i} style={{ fontSize: 11, color: 'var(--faint)', padding: '2px 0' }}>
              <b style={{ color: 'var(--text)' }}>{e.file_name}</b>
              {e.failed_phases?.length ? <span style={{ color: 'var(--red)' }}> [{e.failed_phases.join(', ')}]</span> : null}
              {e.permanent ? <span style={{ color: 'var(--red)' }}> · 영구 실패</span> : (e.retry_count != null ? ` · 재시도 ${e.retry_count}` : '')}
              {e.error ? <span> — {e.error}</span> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
