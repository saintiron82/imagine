/**
 * 분석 탭 데이터 계층 — TanStack Query 훅.
 *
 * 엔드포인트(백엔드 실측):
 *   GET  /api/v1/analysis-jobs?include_completed=false  → { success, jobs[] }
 *        각 job: { id, name, status, total_files, created_at, progress }
 *        progress: { total, complete, pct, dismissed, phases{}, failed{} }
 *   GET  /api/v1/admin/workers                          → { workers[], global_processing_mode }
 *        각 worker: { id, worker_name, status, current_file, current_phase,
 *                     throughput, throughput_mode, jobs_completed, ... }
 *   POST /api/v1/analysis-jobs/{id}/pause | /resume | /cancel  (admin)
 *
 * 미연결(서버 없음/401/네트워크)·오류 시 가짜 데이터 없이 빈 상태로 둔다
 * (disconnected 표식). 하드 게이트라 정상 흐름에선 인증 후에만 도달한다.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

// ── 원자 쿼리 ────────────────────────────────────────────────
function useJobsQuery() {
  return useQuery({
    queryKey: ['analysis-jobs'],
    queryFn: () => apiClient.get('/api/v1/analysis-jobs', { include_completed: false }),
  })
}

function useWorkersQuery() {
  return useQuery({
    queryKey: ['admin-workers'],
    queryFn: () => apiClient.get('/api/v1/admin/workers'),
  })
}

// ── 집계 셀렉터 — 화면이 쓰는 형태로 정규화 ───────────────────
function failedCount(progress) {
  const f = progress?.failed || {}
  return Object.values(f).reduce((a, b) => a + (Number(b) || 0), 0)
}

/**
 * 분석 화면 전체 데이터를 하나로 모은다.
 * 반환: { disconnected, connected, summary, analyzers, jobs, totalFailed }
 */
export function useAnalysisData() {
  const jq = useJobsQuery()
  const wq = useWorkersQuery()

  const connected = !jq.isError && !wq.isError && (jq.data || wq.data)
  const disconnected = !connected   // 미연결/오류 — 가짜 데이터 없이 빈 상태로 둔다

  const rawJobs = (connected && jq.data?.jobs) ? jq.data.jobs : []
  const rawWorkers = (connected && wq.data?.workers) ? wq.data.workers : []

  // 분석기 스트립: 온라인 워커만, 지금 처리 중인 파일 표시
  const analyzers = rawWorkers
    .filter(w => w.status === 'online')
    .map(w => ({
      id: w.id,
      name: w.worker_name,
      file: w.current_file ? `${w.current_file} 처리 중` : '대기 중',
      rate: w.throughput != null ? `${Number(w.throughput).toFixed(w.throughput < 10 ? 1 : 0)}장/분` : '—',
      busy: !!w.current_file,
    }))

  // 라이브 분석기 상태(전역): 지금 잡고 있는 파일 — 작업별 "처리 중" 표시에 귀속.
  const busyWorker = rawWorkers.find(w => w.status === 'online' && w.current_file)

  // 잡 행 — 작업 단위 진행. 핵심은 완료 누적이 아니라 "이 작업의 잔여가 지금
  // 어디서 처리/대기 중인지"다. progress.phases = 잔여 파일의 단계 분포
  // (download/parse=상류 준비, ai/mv=분석기 단계).
  const jobs = rawJobs.map(j => {
    const p = j.progress || {}
    const ph = p.phases || {}
    const total = p.total || j.total_files || 0
    const done = p.complete || 0
    const failed = failedCount(p)
    const remaining = Math.max(0, total - done)
    const analyzerQueued = (ph.ai || 0) + (ph.mv || 0)      // 분석기가 처리할 물량
    const upstreamQueued = (ph.download || 0) + (ph.parse || 0)  // 다운로드/준비 대기(상류)
    const waiting = done === 0 && j.status === 'active'
    // activity: 처리 중 / 분석 대기 / 다운로드 대기 / 일시정지 / 완료 / 대기
    let activity
    if (j.status === 'paused') activity = 'paused'
    else if (remaining === 0) activity = 'done'
    else if (analyzerQueued > 0) activity = analyzers.length > 0 ? 'running' : 'queued'
    else if (upstreamQueued > 0) activity = 'upstream'
    else if (waiting) activity = 'waiting'
    else activity = 'queued'
    return {
      id: j.id, name: j.name, status: j.status, waiting,
      done, total, remaining,
      pct: p.pct != null ? Math.round(p.pct) : (total ? Math.round((done / total) * 100) : 0),
      failed, activity, analyzerQueued, upstreamQueued,
    }
  })

  // 요약: 라이브러리 누적 + 지금 활성 작업/잔여 중심(완료 누적이 주인공이 아님).
  const totalAll = jobs.reduce((a, j) => a + j.total, 0)
  const completeAll = jobs.reduce((a, j) => a + j.done, 0)
  const ratePerMin = analyzers.filter(a => a.busy)
    .reduce((a, w) => a + (Number(rawWorkers.find(rw => rw.id === w.id)?.throughput) || 0), 0)
  const activeRows = jobs.filter(j => j.status === 'active' && j.remaining > 0)
  const summary = {
    total: totalAll,
    complete: completeAll,
    pct: totalAll ? Math.round((completeAll / totalAll) * 100) : 0,
    ratePerMin: Math.round(ratePerMin * 10) / 10,
    analyzerCount: analyzers.length,
    activeCount: activeRows.length,
    remainingTotal: activeRows.reduce((a, j) => a + j.remaining, 0),
  }

  // 라이브 현재 상태 — "처리 중"인 작업 행에 분당 처리율 + 현재 파일을 귀속.
  const current = busyWorker
    ? { file: busyWorker.current_file, rate: summary.ratePerMin }
    : null

  const totalFailed = jobs.reduce((a, j) => a + j.failed, 0)

  // 지금 처리 중: 워커가 현재 잡고 있는 파일들 (큐 미래분 API 없음 → busy 만 정직하게)
  const flow = analyzers.filter(a => a.busy).map(a => ({ name: a.file.replace(' 처리 중', ''), busy: true }))

  return {
    disconnected,
    connected: !!connected,
    loading: jq.isLoading || wq.isLoading,
    summary,
    analyzers,
    jobs,
    flow,
    current,
    totalFailed,
  }
}

// ── 잡 제어 mutation (admin) ─────────────────────────────────
export function useJobControl() {
  const qc = useQueryClient()
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['analysis-jobs'] })
    qc.invalidateQueries({ queryKey: ['analysis-jobs-history'] })
    qc.invalidateQueries({ queryKey: ['analysis-job-errors'] })
  }
  const mk = (verb) => useMutation({
    mutationFn: (jobId) => apiClient.post(`/api/v1/analysis-jobs/${jobId}/${verb}`),
    onSuccess: invalidate,
  })
  // retry = 실패 태스크 재큐잉 · dismiss = 영구 실패만 카운트에서 제외(백엔드 admin)
  return { pause: mk('pause'), resume: mk('resume'), cancel: mk('cancel'), retry: mk('retry'), dismiss: mk('dismiss') }
}

/**
 * 완료 작업 히스토리(IMGV2-20) — 완료/취소/보관된 과거 작업.
 *   GET /api/v1/analysis-jobs?include_completed=true → { jobs[] }
 * 진행 중(active/paused)은 메인 리스트에 있으므로 여기선 종료된 것만.
 * enabled 로 lazy — 히스토리 뷰를 열 때만 조회한다(완료 잔류 없음 원칙).
 */
export function useJobHistory(enabled) {
  const q = useQuery({
    queryKey: ['analysis-jobs-history'],
    queryFn: () => apiClient.get('/api/v1/analysis-jobs', { include_completed: true }),
    enabled: !!enabled,
  })
  const all = q.data?.jobs || []
  const TERMINAL = new Set(['completed', 'archived', 'cancelled'])
  const jobs = all.filter(j => TERMINAL.has(j.status)).map(j => {
    const p = j.progress || {}
    const failed = Object.values(p.failed || {}).reduce((a, b) => a + (Number(b) || 0), 0)
    return {
      id: j.id, name: j.name, status: j.status,
      done: p.complete || 0, total: p.total || j.total_files || 0,
      failed, createdAt: j.created_at,
    }
  })
  return { jobs, loading: q.isLoading, isError: q.isError }
}

/**
 * 작업별 실패 상세(IMGV2-20) — 히스토리 행 펼칠 때만 lazy 조회.
 *   GET /api/v1/analysis-jobs/{id}/errors → { errors[], count }
 *   error: { file_name, failed_phases[], error, retry_count, permanent }
 */
export function useJobErrors(jobId, enabled) {
  const q = useQuery({
    queryKey: ['analysis-job-errors', jobId],
    queryFn: () => apiClient.get(`/api/v1/analysis-jobs/${jobId}/errors`),
    enabled: !!enabled && !!jobId,
  })
  return { errors: q.data?.errors || [], count: q.data?.count || 0, loading: q.isLoading }
}

/**
 * 미완료 감지(IMGV2-25) — 이어하기 제안용. stats/incomplete 조회.
 *   GET /api/v1/stats/incomplete → { total_files, total_incomplete, folders[] }
 */
export function useIncompleteStats() {
  const q = useQuery({ queryKey: ['stats-incomplete'], queryFn: () => apiClient.get('/api/v1/stats/incomplete') })
  return {
    totalIncomplete: (!q.isError && q.data?.total_incomplete) || 0,
    folders: (!q.isError && q.data?.folders) || [],
  }
}
