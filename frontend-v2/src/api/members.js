/**
 * 멤버/초대/좌석 (IMGV2-14 백엔드 연동).
 *   GET    /api/v1/members        / invites / usage
 *   POST   /api/v1/members/invite { emails[], role }
 *   DELETE /api/v1/members/invites/{id}
 *   DELETE /api/v1/members/{id}
 *   POST   /api/v1/members/{id}/deactivate
 * 미연결 시 데모 fallback + isDemo.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

const DEMO = {
  members: [
    { id: 1, username: '성철', email: 'saintiron82@gmail.com', role: 'admin', is_active: true, last_login_at: '방금', via: 'firebase' },
    { id: 2, username: '지민', email: 'jimin@studio.kr', role: 'user', is_active: true, last_login_at: '2시간 전', via: 'firebase' },
  ],
  invites: [
    { id: 11, email: 'minsu@studio.kr', role: 'user', created_at: '2일 전' },
    { id: 12, email: 'art-extern@partner.co', role: 'user', created_at: '5시간 전' },
  ],
  usage: { seat_limit: 10, seats_used: 5, members: 3, pending_invites: 2, expires_at: '2027-06-12', expired: false, smtp_configured: false },
}

function useResource(key, path, demoVal) {
  const q = useQuery({ queryKey: [key], queryFn: () => apiClient.get(path) })
  const connected = !q.isError && q.data
  return { connected: !!connected, isDemo: !connected, loading: q.isLoading, data: connected ? q.data : demoVal }
}

export function useMembersData() {
  const mem = useResource('members', '/api/v1/members', { members: DEMO.members })
  const inv = useResource('member-invites', '/api/v1/members/invites', { invites: DEMO.invites })
  const use = useResource('member-usage', '/api/v1/members/usage', DEMO.usage)
  return {
    isDemo: mem.isDemo,
    members: mem.data.members || [],
    invites: inv.data.invites || [],
    usage: use.data || DEMO.usage,
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
