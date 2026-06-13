/**
 * 관리(엔진룸) 데이터 계층 — 분석기(워커) + 클러스터 밸브/병목.
 * 이 화면은 운영자 전용 → 단계명(MC/VV/MV) 노출 허용(원칙상 유일한 예외).
 *
 * 엔드포인트(실측):
 *   GET  /api/v1/admin/workers → { workers[], global_processing_mode }
 *     worker: { id, worker_name, status, current_file, current_phase, throughput,
 *               throughput_mode, jobs_completed, jobs_failed, origin, launcher,
 *               pending_command, assigned_mode }
 *   POST /api/v1/admin/workers/{id}/stop | /block | /unblock
 *   GET  /api/v1/analysis-jobs → 각 job.progress { total, downloaded, parsed,
 *               mc_done, vv_done, mv_done } → 클러스터 밸브/병목 집계
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

const DEMO_WORKERS = {
  workers: [
    { id: 1, worker_name: '이 서버', status: 'online', current_file: 'knight_v3.psd', current_phase: 'mc', throughput: 7.9, throughput_mode: 'mc', jobs_completed: 210, origin: 'server-local', launcher: 'server' },
    { id: 2, worker_name: '지민-MacBook', status: 'online', current_file: 'bg_forest_07.psd', current_phase: 'vv', throughput: 78, throughput_mode: 'vv', jobs_completed: 880, origin: 'client', launcher: 'electron' },
    { id: 3, worker_name: 'gpu-box-01', status: 'online', current_file: 'npc_smith.psd', current_phase: 'mc', throughput: 81, throughput_mode: 'mc', jobs_completed: 1240, origin: 'headless', launcher: 'cli' },
    { id: 4, worker_name: 'old-laptop', status: 'blocked', current_file: null, throughput: 0, jobs_completed: 0, origin: 'client', launcher: 'electron' },
  ],
  global_processing_mode: 'auto',
}
const DEMO_JOBS = {
  jobs: [
    { id: -1, status: 'active', progress: { total: 3800, downloaded: 3800, parsed: 3640, mc_done: 2710, vv_done: 3205, mv_done: 2560 } },
  ],
}

const PHASE_LABEL = { dl: 'DL', parse: '파싱', mc: 'MC', vv: 'VV', mv: 'MV' }

export function useWorkers() {
  const q = useQuery({ queryKey: ['admin-workers'], queryFn: () => apiClient.get('/api/v1/admin/workers') })
  const connected = !q.isError && q.data
  const data = connected ? q.data : DEMO_WORKERS
  return {
    isDemo: !connected,
    loading: q.isLoading,
    workers: data.workers || [],
    globalMode: data.global_processing_mode || 'auto',
  }
}

/**
 * 클러스터 밸브 + 병목 — analysis-jobs progress 합산.
 * 밸브: 각 단계 done/total. 병목 = 가장 많이 밀린(pending 최대) AI 단계.
 * 단계별 처리율 = 해당 mode 로 도는 온라인 워커 throughput 합.
 */
export function useClusterValves() {
  const jq = useQuery({ queryKey: ['analysis-jobs'], queryFn: () => apiClient.get('/api/v1/analysis-jobs', { include_completed: false }) })
  const wq = useQuery({ queryKey: ['admin-workers'], queryFn: () => apiClient.get('/api/v1/admin/workers') })
  const connected = !jq.isError && !wq.isError && (jq.data || wq.data)
  const jobs = (connected && jq.data?.jobs) ? jq.data.jobs : DEMO_JOBS.jobs
  const workers = (connected && wq.data?.workers) ? wq.data.workers : DEMO_WORKERS.workers

  const sum = (key) => jobs.reduce((a, j) => a + (j.progress?.[key] || 0), 0)
  const total = sum('total')
  const done = { dl: sum('downloaded'), parse: sum('parsed'), mc: sum('mc_done'), vv: sum('vv_done'), mv: sum('mv_done') }

  const rateFor = (mode) => workers
    .filter(w => w.status === 'online' && (w.throughput_mode === mode || w.current_phase === mode))
    .reduce((a, w) => a + (Number(w.throughput) || 0), 0)

  const order = ['dl', 'parse', 'mc', 'vv', 'mv']
  const valves = order.map(ph => {
    const d = done[ph] || 0
    const pending = Math.max(0, total - d)
    const rate = ph === 'dl' || ph === 'parse' ? null : Math.round(rateFor(ph) * 10) / 10
    return { phase: ph, label: PHASE_LABEL[ph], done: d, total, pending, rate }
  })
  // 병목 = 파이프라인 순서(mc→vv→mv)에서 아직 밀린(pending>0) 가장 앞 단계.
  // 하류(mv)는 상류(mc) 산출을 기다려 pending 이 더 클 수 있으므로 "최대 pending"은
  // 오답 — 실제 제약은 가장 앞에서 막힌 단계다.
  const bottleneck = ['mc', 'vv', 'mv']
    .map(ph => valves.find(v => v.phase === ph))
    .find(v => v && v.pending > 0) || null

  return { isDemo: !connected, total, valves, bottleneck }
}

export function useWorkerControl() {
  const qc = useQueryClient()
  const invalidate = () => qc.invalidateQueries({ queryKey: ['admin-workers'] })
  const mk = (verb) => useMutation({
    mutationFn: (id) => apiClient.post(`/api/v1/admin/workers/${id}/${verb}`),
    onSuccess: invalidate,
  })
  return { stop: mk('stop'), block: mk('block'), unblock: mk('unblock') }
}
