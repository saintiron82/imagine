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

/**
 * 라이선스 만료 상태(IMGV2-22) — members/usage 의 expires_at/expired 로 배너 상태 계산.
 *   state: 'expired'(만료) | 'warning'(임박, ≤14일) | 'ok'
 * /members/usage 는 admin 전용이므로 운영자에게만 의미가 있다(일반 사용자는
 * 만료 시 백엔드 하드 게이트(IMGV2-14)로 입장 자체가 막힌다).
 */
export function useLicenseStatus() {
  const use = useResource('member-usage', '/api/v1/members/usage', EMPTY_USAGE)
  const u = use.data || EMPTY_USAGE
  const expiresAt = u.expires_at || null
  let daysLeft = null
  if (expiresAt) {
    const ms = new Date(expiresAt).getTime() - Date.now()
    daysLeft = Math.ceil(ms / 86400000)
  }
  const state = u.expired ? 'expired'
    : (daysLeft != null && daysLeft <= 14 ? 'warning' : 'ok')
  return { disconnected: use.disconnected, state, daysLeft, expiresAt, expired: !!u.expired }
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
