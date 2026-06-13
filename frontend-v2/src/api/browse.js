/**
 * 폴더 브라우즈 + 소스 관리 (IMGV2-13 백엔드 연동).
 *   GET    /api/v1/sources
 *   POST   /api/v1/sources           {id,url,username,password,verify_ssl}
 *   POST   /api/v1/sources/test      {url,username,password,verify_ssl,remote_path}
 *   DELETE /api/v1/sources/{id}
 *   GET    /api/v1/browse/webdav/{id}?path=  → {folders:[{name,path}], files:[...]}
 *   GET    /api/v1/browse/local?path=        → {folders:[{name,path}]}
 *
 * 경로 타이핑 금지 원칙: 사용자는 폴더를 클릭으로만 드릴다운한다.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

export function useSources() {
  const q = useQuery({ queryKey: ['sources'], queryFn: () => apiClient.get('/api/v1/sources') })
  return { sources: (q.data?.sources) || [], loading: q.isLoading, isError: q.isError }
}

/**
 * 한 단계 폴더 나열. kind='webdav'|'local'. webdav 는 sourceId 필요.
 * enabled=false 면 호출 안 함(소스 미선택 시).
 */
export function useBrowse({ kind, sourceId, path }) {
  const enabled = kind === 'local' || (kind === 'webdav' && !!sourceId)
  const q = useQuery({
    queryKey: ['browse', kind, sourceId || '', path || ''],
    enabled,
    queryFn: () => kind === 'webdav'
      ? apiClient.get(`/api/v1/browse/webdav/${encodeURIComponent(sourceId)}`, { path: path || '/' })
      : apiClient.get('/api/v1/browse/local', path ? { path } : undefined),
    retry: 0,
  })
  return {
    folders: (q.data?.folders) || [],
    loading: q.isFetching,
    error: q.isError ? (q.error?.detail || q.error?.message || '탐색 실패') : null,
  }
}

export function useSourceMutations() {
  const qc = useQueryClient()
  const inval = () => qc.invalidateQueries({ queryKey: ['sources'] })
  const add = useMutation({ mutationFn: (s) => apiClient.post('/api/v1/sources', s), onSuccess: inval })
  const test = useMutation({ mutationFn: (s) => apiClient.post('/api/v1/sources/test', s) })
  const remove = useMutation({ mutationFn: (id) => apiClient.delete(`/api/v1/sources/${encodeURIComponent(id)}`), onSuccess: inval })
  return { add, test, remove }
}
