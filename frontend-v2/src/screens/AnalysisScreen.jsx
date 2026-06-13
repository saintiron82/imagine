import { useAnalysisData, useJobControl } from '../api/analysis'

/**
 * 분석 — "무엇이 되고 있나". 작업 리스트 + 지금 처리 중 라이브 모니터.
 * 원칙(확정): 전문 용어 0 (내부 단계명 노출 금지), ETA 없음, 완료 잔류 없음.
 * 화면의 주인공은 잡 메타데이터가 아니라 큐의 내용물이다.
 *
 * 데이터: useAnalysisData() (analysis-jobs + admin/workers 집계).
 * 서버 미연결 시 데모 데이터로 fallback 하고 상단에 표식.
 */
export default function AnalysisScreen() {
  const { isDemo, loading, summary, analyzers, jobs, flow, totalFailed } = useAnalysisData()
  const { pause, resume, cancel } = useJobControl()

  // 잡 상태별 실제 동작 가능한 버튼만 노출 (백엔드에 존재하는 엔드포인트)
  const jobActions = (j) => {
    if (j.status === 'paused') return [{ label: '재개', fn: () => resume.mutate(j.id) }, { label: '취소', fn: () => cancel.mutate(j.id) }]
    return [{ label: '일시정지', fn: () => pause.mutate(j.id) }, { label: '취소', fn: () => cancel.mutate(j.id) }]
  }
  const liveAction = (j) => !isDemo && j.id > 0  // 데모 행은 제어 비활성

  return (
    <section id="scr-analysis" className="screen active">
      <div style={{ maxWidth: 880, margin: '0 auto', padding: '20px 24px' }}>
        <div className="sec-title">
          분석 리스트 <span className="sub">등록 순이 아니라 작업 간 공정 배분으로 처리</span>
        </div>

        {isDemo && (
          <div style={{
            margin: '8px 0 14px', padding: '8px 12px', borderRadius: 6, fontSize: 11.5,
            background: 'rgba(251,191,36,.10)', border: '1px solid rgba(251,191,36,.25)', color: 'var(--amber)',
          }}>
            ● 데모 데이터 — 서버에 연결되지 않았습니다 {loading ? '(연결 시도 중…)' : '(예시를 표시 중)'}
          </div>
        )}

        <div className="summary">
          <div className="sum-row">
            <span className="big">{summary.complete.toLocaleString()}</span>
            <span className="of">/ {summary.total.toLocaleString()} 장 분석됨</span>
            <div className="bar"><i style={{ width: `${summary.pct}%` }} /></div>
            <span className="mono2" style={{ color: '#93c5fd', fontWeight: 700 }}>{summary.pct}%</span>
          </div>
          <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--dim)' }}>
            지금 <b style={{ color: 'var(--emerald)' }}>분당 {summary.ratePerMin}장</b>씩 처리 중 · 분석기 <b>{summary.analyzerCount}대</b> 참여
          </div>
          <div className="an-strip">
            {analyzers.map(a => (
              <div className="an-card" key={a.id}>
                <span className={a.busy ? 'st run' : 'st'} style={a.busy ? undefined : { background: 'var(--faint)' }} />
                <div><div className="nm2">{a.name}</div><div className="sub2">{a.file}</div></div>
                <span className="rate2">{a.rate}</span>
              </div>
            ))}
            {analyzers.length === 0 && (
              <div className="an-card" style={{ color: 'var(--faint)' }}><span className="sub2">연결된 분석기 없음</span></div>
            )}
          </div>
        </div>

        <div className="jobs">
          {jobs.map(j => (
            <div className={`jrow ${j.waiting ? 'waiting' : ''}`} key={j.id}>
              <span className={j.waiting ? 'st' : 'st run'} style={j.waiting ? { background: 'var(--faint)' } : undefined} />
              <span className="nm">{j.name}</span>
              {!j.waiting && <div className="bar"><i style={{ width: `${j.pct}%` }} /></div>}
              <span className="cnt">{j.waiting ? `${j.total.toLocaleString()}장 · 대기` : `${j.done.toLocaleString()} / ${j.total.toLocaleString()}`}</span>
              {j.failed > 0 && (
                <span className="plain-fail" style={{ margin: 0 }}>
                  <span className="n">실패 {j.failed}</span>
                </span>
              )}
              <div className="acts2">
                {liveAction(j)
                  ? jobActions(j).map(a => <button key={a.label} onClick={a.fn}>{a.label}</button>)
                  : <button disabled style={{ opacity: .4, cursor: 'default' }}>제어</button>}
              </div>
            </div>
          ))}
          {jobs.length === 0 && !loading && (
            <div className="jrow" style={{ color: 'var(--faint)' }}>
              <span className="nm">등록된 분석 작업이 없습니다 — [＋ 추가]에서 폴더를 등록하세요</span>
            </div>
          )}
        </div>

        <div className="live">
          <div className="live-head">지금 처리 중 <span className="rate">분당 {summary.ratePerMin}장</span></div>
          <div className="flow">
            {flow.map((f, i) => (
              <div className={`fl-item ${f.busy ? 'busy' : ''}`} key={`${f.name}-${i}`}>
                <div className="ph" style={{ background: 'linear-gradient(140deg,#1e3a5f,#2d4a73)' }} />
                <div className="fn">{f.name}</div>
              </div>
            ))}
            {flow.length === 0 && (
              <div style={{ fontSize: 11.5, color: 'var(--faint)', padding: '8px 2px' }}>처리 중인 파일이 없습니다</div>
            )}
          </div>
          {totalFailed > 0 && (
            <div className="plain-fail">
              <span className="n">실패 {totalFailed}건</span>
              <span>작업별 실패 상세는 관리에서 확인</span>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
