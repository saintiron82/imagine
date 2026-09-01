import { useState } from 'react'
import { useAnalysisData, useJobControl, useJobHistory, useJobErrors, useIncompleteStats } from '../api/analysis'
import { WorkersPanel } from './AdminScreen'
import { useLocale } from '../i18n'

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
  running:  { key: 'v2.an.st_running',  color: 'var(--emerald)' },
  queued:   { key: 'v2.an.st_queued',   color: 'var(--cyan)' },
  upstream: { key: 'v2.an.st_upstream', color: 'var(--dim)' },
  paused:   { key: 'v2.an.st_paused',   color: 'var(--faint)' },
  waiting:  { key: 'v2.an.st_waiting',  color: 'var(--faint)' },
  done:     { key: 'v2.an.st_done',     color: 'var(--faint)' },
}

export default function AnalysisScreen() {
  const { t } = useLocale()
  const { disconnected, loading, summary, jobs, totalFailed } = useAnalysisData()
  const { pause, resume, cancel, retry, dismiss } = useJobControl()
  const [showHistory, setShowHistory] = useState(false)

  const jobActions = (j) => {
    if (j.status === 'paused') return [{ label: t('v2.an.act_resume'), fn: () => resume.mutate(j.id) }, { label: t('v2.an.act_cancel'), fn: () => cancel.mutate(j.id) }]
    return [{ label: t('v2.an.act_pause'), fn: () => pause.mutate(j.id) }, { label: t('v2.an.act_cancel'), fn: () => cancel.mutate(j.id) }]
  }

  const pausedJobs = jobs.filter(j => j.status === 'paused')

  return (
    <section id="scr-analysis" className="screen active">
      <div style={{ maxWidth: 880, margin: '0 auto', padding: '20px 24px' }}>
        <div className="sec-title" style={{ display: 'flex', alignItems: 'baseline' }}>
          <span>{t('v2.an.list_title')} <span className="sub">{t('v2.an.list_sub')}</span></span>
          <button style={{ marginLeft: 'auto', fontSize: 11.5, background: 'none', border: '1px solid var(--line)', borderRadius: 6, padding: '4px 10px', color: showHistory ? 'var(--text)' : 'var(--dim)', cursor: 'pointer' }}
            onClick={() => setShowHistory(v => !v)}>{showHistory ? t('v2.an.history_close') : t('v2.an.history')}</button>
        </div>

        {!disconnected && pausedJobs.length > 0 && (
          <ResumeBanner pausedJobs={pausedJobs} onResumeAll={() => pausedJobs.forEach(j => resume.mutate(j.id))} resuming={resume.isPending} />
        )}

        {disconnected && (
          <div style={{
            margin: '8px 0 14px', padding: '8px 12px', borderRadius: 6, fontSize: 11.5,
            background: 'rgba(248,113,113,.10)', border: '1px solid rgba(248,113,113,.25)', color: 'var(--red)',
          }}>
            {t('v2.an.disconnected')} {loading ? t('v2.an.retrying') : t('v2.an.no_data')}
          </div>
        )}

        <div className="summary">
          <div style={{ fontSize: 12.5, color: 'var(--dim)' }}>
            {t('v2.an.summary')} <b style={{ color: 'var(--text)' }}>{t('v2.an.summary_count', { n: summary.activeCount })}</b> · {t('v2.an.remaining')} <b style={{ color: 'var(--text)' }}>{t('v2.an.remaining_n', { n: summary.remainingTotal.toLocaleString() })}</b> · {t('v2.an.now')} <b style={{ color: 'var(--emerald)' }}>{t('v2.an.rate', { n: summary.ratePerMin })}</b>
            <span style={{ marginLeft: 8, color: 'var(--faint)' }}>{t('v2.an.db_hint')}</span>
          </div>
        </div>

        <div className="sec-sub" style={{ margin: '18px 0 6px', fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
          {t('v2.an.jobs')} <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--faint)' }}>{t('v2.an.jobs_sub')}</span>
        </div>
        <div className="jobs">
          {jobs.map(j => {
            const meta = ACTIVITY[j.activity] || ACTIVITY.queued
            const running = j.activity === 'running'
            return (
              <div className={`jrow ${running ? '' : 'waiting'}`} key={j.id}>
                <span className={running ? 'st run' : 'st'} style={running ? undefined : { background: 'var(--faint)' }} />
                <span className="nm">{j.name}</span>
                <span style={{ fontSize: 10, fontWeight: 700, color: meta.color, minWidth: 54, textAlign: 'right' }}>{t(meta.key)}</span>
                <div className="bar"><i style={{ width: `${j.pct}%` }} /></div>
                <span className="cnt">
                  {j.done.toLocaleString()} / {j.total.toLocaleString()}
                  {j.remaining > 0 && <span style={{ color: 'var(--faint)' }}> {t('v2.an.remaining_inline', { n: j.remaining.toLocaleString() })}</span>}
                </span>
                {j.failed > 0 && (
                  <span className="plain-fail" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="n">{t('v2.an.failed_n', { n: j.failed })}</span>
                    <button style={{ fontSize: 10 }} disabled={retry.isPending} onClick={() => retry.mutate(j.id)}>{t('v2.an.retry')}</button>
                    <button style={{ fontSize: 10 }} disabled={dismiss.isPending}
                      onClick={() => { if (window.confirm(t('v2.an.dismiss_confirm'))) dismiss.mutate(j.id) }}>{t('v2.an.dismiss')}</button>
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
              <span className="nm">{t('v2.an.no_jobs')}</span>
            </div>
          )}
        </div>

        <div className="sec-sub" style={{ margin: '18px 0 6px', fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
          {t('v2.an.analyzers')} <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--faint)' }}>{t('v2.an.analyzers_sub')}</span>
        </div>
        <WorkersPanel />
        {totalFailed > 0 && (
          <div className="plain-fail" style={{ marginTop: 12 }}>
            <span className="n">{t('v2.an.failed_total', { n: totalFailed })}</span>
            <span>{t('v2.an.failed_where')}</span>
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
  const { t } = useLocale()
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
        {t('v2.an.paused_jobs')} <b>{t('v2.an.summary_count', { n: pausedJobs.length })}</b>{totalIncomplete > 0 ? ` ${t('v2.an.incomplete', { n: totalIncomplete.toLocaleString() })}` : ''} {t('v2.an.resume_ask')}
      </span>
      <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
        <button disabled={resuming} onClick={onResumeAll}>{resuming ? t('v2.an.resuming') : t('v2.an.resume_all')}</button>
        <button onClick={() => setDismissed(true)}>{t('v2.an.dismiss')}</button>
      </div>
    </div>
  )
}

// ── 완료 작업 히스토리 (IMGV2-20) ───────────────────────────
const HIST_STATUS = { completed: { key: 'v2.an.hist_completed', color: 'var(--emerald)' }, cancelled: { key: 'v2.an.hist_cancelled', color: 'var(--faint)' }, archived: { key: 'v2.an.hist_archived', color: 'var(--dim)' } }

function HistorySection() {
  const { t } = useLocale()
  const { jobs, loading, isError } = useJobHistory(true)
  return (
    <div className="live" style={{ marginTop: 16 }}>
      <div className="live-head">{t('v2.an.hist_title')} <span className="rate">{t('v2.an.hist_count', { n: jobs.length })}</span></div>
      {loading && <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: '8px 2px' }}>{t('v2.common.loading')}</div>}
      {isError && <div style={{ fontSize: 11.5, color: 'var(--red)', padding: '8px 2px' }}>{t('v2.an.hist_error')}</div>}
      {!loading && !isError && jobs.length === 0 && <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: '8px 2px' }}>{t('v2.an.hist_empty')}</div>}
      {jobs.map(j => <HistoryRow key={j.id} job={j} />)}
    </div>
  )
}

function HistoryRow({ job }) {
  const { t } = useLocale()
  const [open, setOpen] = useState(false)
  const { errors, count, loading } = useJobErrors(job.id, open)
  const { retry, dismiss } = useJobControl()
  const s = HIST_STATUS[job.status] || { label: job.status, color: 'var(--faint)' }
  const when = job.createdAt ? new Date(job.createdAt).toLocaleString('ko-KR', { dateStyle: 'short', timeStyle: 'short' }) : ''
  return (
    <div style={{ borderTop: '1px solid var(--line)', padding: '8px 2px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12 }}>
        <span style={{ color: s.color, fontSize: 10, fontWeight: 700, minWidth: 28 }}>{t(s.key)}</span>
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{job.name}</span>
        <span className="mono2" style={{ color: 'var(--dim)' }}>{job.done.toLocaleString()} / {job.total.toLocaleString()}</span>
        {when && <span style={{ color: 'var(--faint)', fontSize: 10 }}>{when}</span>}
        {job.failed > 0
          ? <button style={{ fontSize: 10 }} onClick={() => setOpen(v => !v)}>{open ? t('v2.an.err_close') : t('v2.an.err_open', { n: job.failed })}</button>
          : <span style={{ width: 56 }} />}
      </div>
      {open && (
        <div style={{ marginTop: 6, paddingLeft: 38 }}>
          {job.failed > 0 && (
            <div style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
              <button style={{ fontSize: 10 }} disabled={retry.isPending} onClick={() => retry.mutate(job.id)}>{t('v2.an.retry_failed')}</button>
              <button style={{ fontSize: 10 }} disabled={dismiss.isPending}
                onClick={() => { if (window.confirm(t('v2.an.dismiss_confirm2'))) dismiss.mutate(job.id) }}>{t('v2.an.dismiss')}</button>
            </div>
          )}
          {loading && <div style={{ fontSize: 11, color: 'var(--faint)' }}>{t('v2.common.loading')}</div>}
          {!loading && count === 0 && <div style={{ fontSize: 11, color: 'var(--faint)' }}>{t('v2.an.no_error_detail')}</div>}
          {errors.map((e, i) => (
            <div key={i} style={{ fontSize: 11, color: 'var(--faint)', padding: '2px 0' }}>
              <b style={{ color: 'var(--text)' }}>{e.file_name}</b>
              {e.failed_phases?.length ? <span style={{ color: 'var(--red)' }}> [{e.failed_phases.join(', ')}]</span> : null}
              {e.permanent ? <span style={{ color: 'var(--red)' }}> {t('v2.an.permanent_fail')}</span> : (e.retry_count != null ? ` ${t('v2.an.retry_count', { n: e.retry_count })}` : '')}
              {e.error ? <span> — {e.error}</span> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
