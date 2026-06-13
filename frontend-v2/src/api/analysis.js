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
 * 미연결(서버 없음/401/네트워크)일 때는 데모 데이터를 fallback 으로 돌려준다
 * — 화면이 비지 않게 하되 `isDemo: true` 로 표식한다.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

// ── 데모 fallback (서버 미연결 시) ───────────────────────────
const DEMO = {
  jobs: [
    { id: -1, name: 'NAS / 캐릭터', status: 'active',
      progress: { total: 3100, complete: 1820, pct: 59, failed: { parse: 3 } } },
    { id: -2, name: '배경 다시 분석', status: 'active',
      progress: { total: 700, complete: 410, pct: 59, failed: {} } },
    { id: -3, name: '신규 컨셉 2026-06', status: 'active',
      progress: { total: 412, complete: 0, pct: 0, failed: {} } },
  ],
  workers: [
    { id: 1, worker_name: '이 서버', status: 'online', current_file: 'knight_v3.psd', throughput: 7.9 },
    { id: 2, worker_name: '지민-MacBook', status: 'online', current_file: 'bg_forest_07.psd', throughput: 78 },
    { id: 3, worker_name: 'gpu-box-01', status: 'online', current_file: 'npc_smith.psd', throughput: 81 },
  ],
}

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
 * 반환: { isDemo, connected, summary, analyzers, jobs, totalFailed }
 */
export function useAnalysisData() {
  const jq = useJobsQuery()
  const wq = useWorkersQuery()

  const connected = !jq.isError && !wq.isError && (jq.data || wq.data)
  const isDemo = !connected

  const rawJobs = (connected && jq.data?.jobs) ? jq.data.jobs : DEMO.jobs
  const rawWorkers = (connected && wq.data?.workers) ? wq.data.workers : DEMO.workers

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

  // 잡 행
  const jobs = rawJobs.map(j => {
    const p = j.progress || {}
    const failed = failedCount(p)
    const waiting = (p.complete || 0) === 0 && j.status === 'active'
    return {
      id: j.id,
      name: j.name,
      status: j.status,
      waiting,
      done: p.complete || 0,
      total: p.total || j.total_files || 0,
      pct: p.pct != null ? Math.round(p.pct) : 0,
      failed,
    }
  })

  // 요약: 전체 완료/전체, 처리율 합, 온라인 분석기 수
  const totalAll = jobs.reduce((a, j) => a + j.total, 0)
  const completeAll = jobs.reduce((a, j) => a + j.done, 0)
  const ratePerMin = analyzers.filter(a => a.busy)
    .reduce((a, w) => a + (Number(rawWorkers.find(rw => rw.id === w.id)?.throughput) || 0), 0)
  const summary = {
    total: totalAll,
    complete: completeAll,
    pct: totalAll ? Math.round((completeAll / totalAll) * 100) : 0,
    ratePerMin: Math.round(ratePerMin * 10) / 10,
    analyzerCount: analyzers.length,
  }

  const totalFailed = jobs.reduce((a, j) => a + j.failed, 0)

  // 지금 처리 중: 워커가 현재 잡고 있는 파일들 (큐 미래분 API 없음 → busy 만 정직하게)
  const flow = analyzers.filter(a => a.busy).map(a => ({ name: a.file.replace(' 처리 중', ''), busy: true }))

  return {
    isDemo,
    connected: !!connected,
    loading: jq.isLoading || wq.isLoading,
    summary,
    analyzers,
    jobs,
    flow,
    totalFailed,
  }
}

// ── 잡 제어 mutation (admin) ─────────────────────────────────
export function useJobControl() {
  const qc = useQueryClient()
  const invalidate = () => qc.invalidateQueries({ queryKey: ['analysis-jobs'] })
  const mk = (verb) => useMutation({
    mutationFn: (jobId) => apiClient.post(`/api/v1/analysis-jobs/${jobId}/${verb}`),
    onSuccess: invalidate,
  })
  return { pause: mk('pause'), resume: mk('resume'), cancel: mk('cancel') }
}
