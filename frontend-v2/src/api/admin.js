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
import { useRef, useState, useEffect } from 'react'
import { apiClient } from './client'

const PHASE_LABEL = { dl: 'DL', parse: '파싱', mc: 'MC', vv: 'VV', mv: 'MV' }

export function useWorkers() {
  const q = useQuery({ queryKey: ['admin-workers'], queryFn: () => apiClient.get('/api/v1/admin/workers') })
  const connected = !q.isError && q.data
  const data = connected ? q.data : {}
  return {
    disconnected: !connected,
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
  const jobs = (connected && jq.data?.jobs) ? jq.data.jobs : []
  const workers = (connected && wq.data?.workers) ? wq.data.workers : []

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

  return { disconnected: !connected, total, valves, bottleneck }
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

/**
 * 원격 분석기 등록(IMGV2-18) — 헤드리스 워커용 토큰 + Linux 부트스트랩 명령 발급.
 *   POST /api/v1/admin/workers/headless-command
 *     body: { worker_name, launcher(cli|service|cloud), expires_minutes(15..43200),
 *             server_url(direct 모드 override|null), connect_mode(direct_lan|manual_external|relay_session) }
 *     resp: { linux_command, worker_username, access_token, refresh_token, server_url,
 *             relay_endpoint, server_id, bootstrap_url, expires_minutes, ... }
 * 명령에 토큰이 박혀 있으므로 한 번만 보이고 운영자가 복사해 워커 머신에서 실행한다.
 */
export function useHeadlessCommand() {
  return useMutation({
    mutationFn: (payload) => apiClient.post('/api/v1/admin/workers/headless-command', payload),
  })
}

/**
 * 검색 피드백 대시보드(IMGV2-23) — 검색 품질 진단.
 *   GET /api/v1/admin/search-feedback/summary
 *     → { total_30d, top_files:[{file_id,count}], top_queries:[{query,count}] }
 * 사용자가 검색 결과에 '부적절' 라벨을 남긴 기록을 30일 집계.
 */
export function useFeedbackSummary() {
  const q = useQuery({ queryKey: ['feedback-summary'], queryFn: () => apiClient.get('/api/v1/admin/search-feedback/summary') })
  const connected = !q.isError && q.data
  const data = connected ? q.data : {}
  return {
    disconnected: !connected,
    loading: q.isLoading,
    total30d: data.total_30d || 0,
    topFiles: data.top_files || [],
    topQueries: data.top_queries || [],
  }
}

/**
 * 자동 처리 정책(IMGV2-21) — 서버 전역 auto-processing 읽기/갱신.
 *   GET   /api/v1/admin/workers/auto-processing → {enabled, rest_after_batch_s, batch_size, verbose_log}
 *   PATCH /api/v1/admin/workers/auto-processing  (부분 갱신) → {ok}
 */
/**
 * 실시간 로그(IMGV2-26) — 커서 폴링으로 서버 링버퍼를 따라간다.
 *   GET /api/v1/admin/logs?after=<seq> → { entries[], last_seq }
 * SSE 는 Authorization 헤더를 못 실어 JWT 인증과 안 맞으므로 폴링(1.5s) 채택.
 * 클라이언트에 최대 1000줄 누적(서버는 신규분만 반환).
 */
export function useLogStream(active) {
  const afterRef = useRef(0)
  const [entries, setEntries] = useState([])
  const q = useQuery({
    queryKey: ['admin-logs'],
    queryFn: () => apiClient.get('/api/v1/admin/logs', { after: afterRef.current, limit: 300 }),
    enabled: !!active,
    refetchInterval: active ? 1500 : false,
  })
  useEffect(() => {
    const d = q.data
    if (!d) return
    if (d.last_seq != null) afterRef.current = Math.max(afterRef.current, d.last_seq)
    if (d.entries?.length) {
      setEntries(prev => {
        const merged = prev.concat(d.entries)
        return merged.length > 1000 ? merged.slice(-1000) : merged
      })
    }
  }, [q.data])
  const clear = () => setEntries([])
  return { entries, clear, disconnected: q.isError }
}

export function useAutoProcessing() {
  const qc = useQueryClient()
  const q = useQuery({ queryKey: ['auto-processing'], queryFn: () => apiClient.get('/api/v1/admin/workers/auto-processing') })
  const update = useMutation({
    mutationFn: (patch) => apiClient.patch('/api/v1/admin/workers/auto-processing', patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['auto-processing'] }),
  })
  return { config: q.data, loading: q.isLoading, disconnected: q.isError, update }
}
