/**
 * 폴더 탭 데이터 계층 — TanStack Query 훅.
 *
 * 엔드포인트(백엔드 실측):
 *   GET  /api/v1/archive/folders  → { success, folders[] }
 *        각 folder: { folder_path, total, mc, vv, mv, image_types{} }
 *   POST /api/v1/sync/scan         (admin) → { matched, moved, missing, new_files, moved_list[], missing_list[] }
 *   POST /api/v1/sync/apply-moves  (admin) → { updated }
 *   POST /api/v1/sync/delete-missing (admin) → { deleted }
 *   POST /api/v1/discover/scan     → 재분석 작업 등록 { job_id, total_files }
 *
 * 주의(용어 0): 폴더 탭은 일반 표면 — MC/VV/MV 라벨 노출 금지.
 *   내부적으로 "분석됨"은 mc(캡션 생성 완료) 수를 헤드라인으로 쓰되 UI 라벨은 "분석됨".
 * 폴더 목록은 평면 folder_path 문자열 → 경로 분할로 트리를 구성한다.
 * 미연결 시 데모 데이터 fallback + isDemo 표식.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from './client'

// ── 데모 fallback (real shape: folder_path/total/mc/vv/mv) ───
const DEMO = {
  folders: [
    { folder_path: 'NAS/캐릭터/주인공', total: 1800, mc: 1800, vv: 1750, mv: 1700 },
    { folder_path: 'NAS/캐릭터/NPC', total: 1300, mc: 760, vv: 700, mv: 640 },
    { folder_path: 'NAS/배경', total: 880, mc: 880, vv: 880, mv: 880 },
    { folder_path: '로컬/컨셉아트', total: 700, mc: 0, vv: 0, mv: 0 },
  ],
}

const lastSeg = (p) => { const s = p.split('/').filter(Boolean); return s[s.length - 1] || p }
const covClass = (pct, total) => (total === 0 ? 'r' : pct >= 99 ? 'g' : pct > 0 ? 'a' : 'r')

function normalize(f) {
  const total = f.total || 0
  const analyzed = f.mc != null ? f.mc : (f.analyzed || 0) // mc = "분석됨" 헤드라인
  const pct = total ? Math.round((analyzed / total) * 100) : 0
  return {
    path: f.folder_path || f.storage_root || '(미지정)',
    name: lastSeg(f.folder_path || f.storage_root || '(미지정)'),
    total, analyzed, pct,
    cov: covClass(pct, total),
    fullyDone: total > 0 && f.mc === total && f.vv === total && f.mv === total,
  }
}

/**
 * 평면 folder_path 목록 → 트리. 분기 노드는 하위 leaf 합으로 커버리지 집계.
 * 반환: 배열 of { name, path, depth, total, analyzed, pct, cov, isLeaf, leaf? }
 */
function buildTree(folders) {
  const root = { children: {} }
  for (const f of folders) {
    const segs = f.path.split('/').filter(Boolean)
    let node = root
    segs.forEach((seg, i) => {
      node.children[seg] = node.children[seg] || { name: seg, path: segs.slice(0, i + 1).join('/'), children: {}, leaf: null }
      node = node.children[seg]
    })
    node.leaf = f
  }
  // 분기 노드의 (total, analyzed) = 하위 모든 leaf 합
  const agg = (node) => {
    let t = node.leaf?.total || 0, a = node.leaf?.analyzed || 0
    for (const c of Object.values(node.children)) { const r = agg(c); t += r.t; a += r.a }
    return { t, a }
  }
  // 선순회(pre-order) → depth 부여한 평면 목록 (부모 먼저, 자식 들여쓰기)
  const out = []
  const emit = (node, depth) => {
    if (node.name) {
      const { t, a } = agg(node)
      const pct = t ? Math.round((a / t) * 100) : 0
      out.push({ name: node.name, path: node.path, depth, total: t, analyzed: a, pct, cov: covClass(pct, t), isLeaf: Object.keys(node.children).length === 0 })
    }
    Object.values(node.children).sort((x, y) => x.name.localeCompare(y.name)).forEach(c => emit(c, depth + 1))
  }
  emit(root, -1)
  return out
}

export function useFolders() {
  const q = useQuery({
    queryKey: ['archive-folders'],
    queryFn: () => apiClient.get('/api/v1/archive/folders'),
  })
  const connected = !q.isError && q.data
  const raw = (connected && q.data?.folders) ? q.data.folders : DEMO.folders
  const folders = raw.map(normalize)
  const tree = buildTree(folders)
  return { isDemo: !connected, loading: q.isLoading, folders, tree }
}

// ── 동기화 (admin) — scan 후 이동/삭제 자동 적용 ──────────────
export function useFolderSync() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (folderPath) => {
      const scan = await apiClient.post('/api/v1/sync/scan', { folder_path: folderPath })
      let updated = 0, deleted = 0
      if (scan.moved_list?.length) {
        const r = await apiClient.post('/api/v1/sync/apply-moves', {
          moves: scan.moved_list.map(m => ({ id: m.id, new_path: m.new_path })),
        })
        updated = r.updated || 0
      }
      return { ...scan, applied_moves: updated, applied_deletes: deleted }
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['archive-folders'] }),
  })
}

// ── 재분석 (작업 등록) ───────────────────────────────────────
export function useReanalyze() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (folderPath) => apiClient.post('/api/v1/discover/scan', { folder_path: folderPath }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['analysis-jobs'] })
      qc.invalidateQueries({ queryKey: ['archive-folders'] })
    },
  })
}
