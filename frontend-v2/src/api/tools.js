/**
 * 서버 도구(관리) — 정합성 감사 / 해시 백필 / DB 초기화 (IMGV2-19).
 *   GET    /api/v1/stats/incomplete            정합성 감사(미완성 파일 집계)
 *   POST   /api/v1/admin/backfill-hashes        해시 백필 시작
 *   GET    /api/v1/admin/backfill-hashes        백필 상태 {running, total, done, skipped, failed}
 *   DELETE /api/v1/admin/backfill-hashes        백필 중지
 *   POST   /api/v1/admin/database/reset {password}  DB 초기화(비번 재확인)
 *
 * DB 내보내기/가져오기는 HTTP 엔드포인트가 없어 별도 백엔드 작업 필요(IMGV2-28).
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

export function useDbAudit() {
  return useMutation({ mutationFn: () => apiClient.get('/api/v1/stats/incomplete') })
}

export function useBackfill() {
  const qc = useQueryClient()
  const status = useQuery({
    queryKey: ['backfill-status'],
    queryFn: () => apiClient.get('/api/v1/admin/backfill-hashes'),
    refetchInterval: (q) => (q.state.data?.running ? 1500 : false),
  })
  const inval = () => qc.invalidateQueries({ queryKey: ['backfill-status'] })
  const start = useMutation({ mutationFn: () => apiClient.post('/api/v1/admin/backfill-hashes'), onSuccess: inval })
  const stop = useMutation({ mutationFn: () => apiClient.delete('/api/v1/admin/backfill-hashes'), onSuccess: inval })
  return { status: status.data, loading: status.isLoading, start, stop }
}

export function useDbReset() {
  return useMutation({ mutationFn: (password) => apiClient.post('/api/v1/admin/database/reset', { password }) })
}

/**
 * Parse 데이터 복구(IMGV2-24) — 파싱 누락/손상 재처리.
 *   POST /api/v1/admin/tools/repair-parse        시작 → {success, total, local_count, webdav_count}
 *   GET  /api/v1/admin/tools/repair-parse/status  {running, progress:{done,failed,skipped,total,current_file,phase}}
 * 백필과 유사하나 중지 엔드포인트는 없음(시작 + 진행률 폴링만).
 */
export function useRepairParse() {
  const qc = useQueryClient()
  const status = useQuery({
    queryKey: ['repair-parse-status'],
    queryFn: () => apiClient.get('/api/v1/admin/tools/repair-parse/status'),
    refetchInterval: (q) => (q.state.data?.running ? 1500 : false),
  })
  const start = useMutation({
    mutationFn: () => apiClient.post('/api/v1/admin/tools/repair-parse'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['repair-parse-status'] }),
  })
  return { status: status.data, start }
}
