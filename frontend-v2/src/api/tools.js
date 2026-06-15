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
  const inval = () => { qc.invalidateQueries({ queryKey: ['backfill-status'] }) }  // no promise return → mutation settles immediately
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
/**
 * DB 백업/복원(IMGV2-48). 내보내기는 JWT 인증 GET 이라 단순 링크 대신
 * 인증 fetch → blob → 다운로드. 복원은 비번 재확인 + 파일 업로드(파괴적).
 *   GET  /api/v1/admin/database/export          → application/x-sqlite3 스트림
 *   POST /api/v1/admin/database/import {password, file(multipart)}
 */
export function useDbExport() {
  return useMutation({
    mutationFn: async () => {
      const resp = await apiClient.raw('GET', '/api/v1/admin/database/export')
      if (!resp.ok) {
        let detail = resp.statusText
        try { detail = (await resp.json()).detail || detail } catch { /* keep statusText */ }
        throw new Error(detail || '내보내기 실패')
      }
      const blob = await resp.blob()
      const cd = resp.headers.get('Content-Disposition') || ''
      const m = cd.match(/filename="?([^"]+)"?/)
      const filename = m ? m[1] : 'imagine-backup.db'
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = filename
      document.body.appendChild(a); a.click(); a.remove()
      URL.revokeObjectURL(url)
      return { filename, bytes: blob.size }
    },
  })
}

export function useDbImport() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ password, file }) => {
      const fd = new FormData()
      fd.append('password', password)
      fd.append('file', file)
      return apiClient.upload('/api/v1/admin/database/import', fd)
    },
    // 전체 DB 교체됨 → 모든 캐시 무효화. 단 promise 를 반환하면 mutation 이
    // 리페치 완료까지 isPending 으로 묶이므로(불필요한 "복원 중…" 잔류) fire-and-forget.
    onSuccess: () => { qc.invalidateQueries() },
  })
}

export function useRepairParse() {
  const qc = useQueryClient()
  const status = useQuery({
    queryKey: ['repair-parse-status'],
    queryFn: () => apiClient.get('/api/v1/admin/tools/repair-parse/status'),
    refetchInterval: (q) => (q.state.data?.running ? 1500 : false),
  })
  const start = useMutation({
    mutationFn: () => apiClient.post('/api/v1/admin/tools/repair-parse'),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['repair-parse-status'] }) },
  })
  return { status: status.data, start }
}
