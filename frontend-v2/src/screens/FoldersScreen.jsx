import { useState, useMemo } from 'react'
import { useFolders, useFolderSync, useReanalyze } from '../api/folders'

/**
 * 폴더 — "무엇이 있나". 등록된 폴더의 구조·분석률.
 * 작업(동적)은 분석 탭의 일 — 여기서는 보지 않는다.
 *
 * 데이터: useFolders()(archive/folders → 트리+현황). 미연결 시 데모 fallback.
 * 용어 0: "분석됨/전체"만 — 내부 단계명(MC/VV/MV) 노출 금지.
 */
function TreeNode({ node, selected, onSelect }) {
  return (
    <div
      className={`tnode ${node.depth > 0 ? 'lvl1' : ''}`}
      style={{
        paddingLeft: 8 + node.depth * 14,
        ...(selected === node.path ? { background: 'rgba(59,130,246,.12)' } : null),
      }}
      onClick={() => onSelect(node.path)}
    >
      <span className="tw">{node.isLeaf ? '▸' : '▾'}</span>
      <span className="ic">{node.depth === 0 ? '🌐' : '📁'}</span>
      <span className="nm">{node.name}</span>
      <span className={`cov ${node.cov}`} />
    </div>
  )
}

export default function FoldersScreen() {
  const { isDemo, loading, folders, tree } = useFolders()
  const sync = useFolderSync()
  const reanalyze = useReanalyze()
  const [selected, setSelected] = useState(null)

  // 기본 선택 = 첫 트리 노드
  const sel = selected || tree[0]?.path || null
  const selNode = useMemo(() => tree.find(n => n.path === sel) || null, [tree, sel])

  const syncMsg = sync.data
    ? `동기화 완료 · 이동 ${sync.data.moved || 0} · 누락 ${sync.data.missing || 0} · 신규 ${sync.data.new_files || 0}`
    : sync.isPending ? '동기화 중…' : sync.isError ? '동기화 실패' : null
  const reMsg = reanalyze.data
    ? `재분석 작업 등록됨 (${(reanalyze.data.total_files ?? 0).toLocaleString()}장)`
    : reanalyze.isPending ? '등록 중…' : reanalyze.isError ? '등록 실패' : null

  const canMutate = !isDemo && !!selNode

  return (
    <section id="scr-library" className="screen active" style={{ height: '100%' }}>
      <aside className="lib-side">
        <h3>폴더 {isDemo && <span style={{ fontSize: 9, color: 'var(--amber)' }}>● 데모</span>}</h3>
        <div className="tree">
          {tree.length === 0 && !loading && (
            <div style={{ fontSize: 11, color: 'var(--faint)', padding: 8 }}>등록된 폴더가 없습니다</div>
          )}
          {tree.map(n => <TreeNode key={n.path} node={n} selected={sel} onSelect={setSelected} />)}
        </div>

        {selNode && (
          <div className="src-detail">
            <div className="sd-head">📁 {selNode.name} <span className="sd-sub">폴더 속성</span></div>
            <div className="sd-row"><span>분석됨</span><span className="sd-val">{selNode.analyzed.toLocaleString()} / {selNode.total.toLocaleString()}장 · {selNode.pct}%</span></div>
            <div className="sd-row"><span>경로</span><span className="sd-val" style={{ fontSize: 10, opacity: .8 }}>{selNode.path}</span></div>
            {(syncMsg || reMsg) && (
              <div className="sd-row"><span style={{ fontSize: 10.5, color: 'var(--cyan)' }}>{reMsg || syncMsg}</span></div>
            )}
            <div className="sd-acts">
              <button disabled={!canMutate || sync.isPending} onClick={() => sync.mutate(selNode.path)}>동기화</button>
              <button disabled={!canMutate || reanalyze.isPending} onClick={() => reanalyze.mutate(selNode.path)}>다시 분석…</button>
            </div>
            {isDemo && <div style={{ fontSize: 9.5, color: 'var(--faint)', marginTop: 6 }}>서버 연결 시 동기화·재분석이 활성화됩니다</div>}
          </div>
        )}
      </aside>

      <div className="lib-main">
        <div className="sec-title">폴더 현황 <span className="sub">등록된 폴더가 얼마나 분석됐는지</span></div>
        <div className="jobs">
          {folders.map(r => (
            <div className="jrow" key={r.path}>
              <span className={`cov ${r.cov}`} style={{ width: 9, height: 9 }} />
              <span className="nm">{r.name}</span>
              <div className="bar"><i style={{ width: `${r.pct}%`, background: r.cov === 'r' ? 'var(--amber)' : undefined }} /></div>
              <span className="cnt">{r.analyzed.toLocaleString()} / {r.total.toLocaleString()} 장</span>
              <span style={{ fontSize: 10.5, color: r.fullyDone ? 'var(--faint)' : 'var(--amber)' }}>
                {r.fullyDone ? '완료' : r.pct === 0 ? '미분석' : `${r.pct}%`}
              </span>
            </div>
          ))}
          {folders.length === 0 && !loading && (
            <div className="jrow" style={{ color: 'var(--faint)' }}>
              <span className="nm">등록된 폴더가 없습니다 — [＋ 추가]에서 폴더를 등록하세요</span>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
