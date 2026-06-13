/**
 * 작업 등록 — discover/scan 으로 폴더를 분석 작업으로 등록한다.
 *   POST /api/v1/discover/scan { folder_path, name?, priority?, analysis_profile? }
 *     → { success, job_id, total_files, scanned_path }
 * 우선 처리 체크 시 priority 를 높여 스케줄러가 먼저 잡도록 한다.
 */
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

export function useRegisterJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ folderPath, priority, name }) =>
      apiClient.post('/api/v1/discover/scan', {
        folder_path: folderPath,
        priority: priority ? 10 : 0,
        ...(name ? { name } : {}),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['analysis-jobs'] })
      qc.invalidateQueries({ queryKey: ['archive-folders'] })
    },
  })
}
