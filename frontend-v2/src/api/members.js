/**
 * 멤버/초대/좌석 (IMGV2-14 백엔드 연동).
 *   GET    /api/v1/members        / invites / usage
 *   POST   /api/v1/members/invite { emails[], role }
 *   DELETE /api/v1/members/invites/{id}
 *   DELETE /api/v1/members/{id}
 *   POST   /api/v1/members/{id}/deactivate
 * 미연결/오류 시 빈 상태(가짜 데이터 없음, disconnected 표식).
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

const EMPTY_USAGE = { seat_limit: 0, seats_used: 0, members: 0, pending_invites: 0, expires_at: null, expired: false, smtp_configured: false }

function useResource(key, path, emptyVal) {
  const q = useQuery({ queryKey: [key], queryFn: () => apiClient.get(path) })
  const connected = !q.isError && q.data
  return { connected: !!connected, disconnected: !connected, loading: q.isLoading, data: connected ? q.data : emptyVal }
}

export function useMembersData() {
  const mem = useResource('members', '/api/v1/members', { members: [] })
  const inv = useResource('member-invites', '/api/v1/members/invites', { invites: [] })
  const use = useResource('member-usage', '/api/v1/members/usage', EMPTY_USAGE)
  return {
    disconnected: mem.disconnected,
    members: mem.data.members || [],
    invites: inv.data.invites || [],
    usage: use.data || EMPTY_USAGE,
  }
}

export function useMemberMutations() {
  const qc = useQueryClient()
  const inval = () => {
    qc.invalidateQueries({ queryKey: ['members'] })
    qc.invalidateQueries({ queryKey: ['member-invites'] })
    qc.invalidateQueries({ queryKey: ['member-usage'] })
  }
  const invite = useMutation({ mutationFn: ({ emails, role }) => apiClient.post('/api/v1/members/invite', { emails, role: role || 'user' }), onSuccess: inval })
  const revoke = useMutation({ mutationFn: (id) => apiClient.delete(`/api/v1/members/invites/${id}`), onSuccess: inval })
  const remove = useMutation({ mutationFn: (id) => apiClient.delete(`/api/v1/members/${id}`), onSuccess: inval })
  const deactivate = useMutation({ mutationFn: (id) => apiClient.post(`/api/v1/members/${id}/deactivate`), onSuccess: inval })
  return { invite, revoke, remove, deactivate }
}
