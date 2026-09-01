import { useState, useMemo } from 'react'
import { useFolders, useFolderSync, useReanalyze } from '../api/folders'
import { useLocale } from '../i18n'

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
  const { disconnected, loading, folders, tree } = useFolders()
  const sync = useFolderSync()
  const reanalyze = useReanalyze()
  const [selected, setSelected] = useState(null)

  // 기본 선택 = 첫 트리 노드
  const { t } = useLocale()
  const sel = selected || tree[0]?.path || null
  const selNode = useMemo(() => tree.find(n => n.path === sel) || null, [tree, sel])

  const syncMsg = sync.data
    ? t('v2.folders.sync_done', { moved: sync.data.moved || 0, missing: sync.data.missing || 0, new: sync.data.new_files || 0 })
    : sync.isPending ? t('v2.folders.syncing') : sync.isError ? t('v2.folders.sync_failed') : null
  const reMsg = reanalyze.data
    ? t('v2.folders.reanalyze_queued', { count: (reanalyze.data.total_files ?? 0).toLocaleString() })
    : reanalyze.isPending ? t('v2.folders.registering') : reanalyze.isError ? t('v2.folders.register_failed') : null

  const canMutate = !disconnected && !!selNode

  return (
    <section id="scr-library" className="screen active" style={{ height: '100%' }}>
      <aside className="lib-side">
        <h3>{t('v2.folders.title')} {disconnected && <span style={{ fontSize: 9, color: 'var(--amber)' }}>{t('v2.folders.disconnected')}</span>}</h3>
        <div className="tree">
          {tree.length === 0 && !loading && (
            <div style={{ fontSize: 11, color: 'var(--faint)', padding: 8 }}>{t('v2.folders.none')}</div>
          )}
          {tree.map(n => <TreeNode key={n.path} node={n} selected={sel} onSelect={setSelected} />)}
        </div>

        {selNode && (
          <div className="src-detail">
            <div className="sd-head">📁 {selNode.name} <span className="sd-sub">{t('v2.folders.properties')}</span></div>
            <div className="sd-row"><span>{t('v2.folders.analyzed')}</span><span className="sd-val">{t('v2.folders.count_images', { done: selNode.analyzed.toLocaleString(), total: selNode.total.toLocaleString(), pct: selNode.pct })}</span></div>
            <div className="sd-row"><span>{t('v2.folders.path')}</span><span className="sd-val" style={{ fontSize: 10, opacity: .8 }}>{selNode.path}</span></div>
            {(syncMsg || reMsg) && (
              <div className="sd-row"><span style={{ fontSize: 10.5, color: 'var(--cyan)' }}>{reMsg || syncMsg}</span></div>
            )}
            <div className="sd-acts">
              <button disabled={!canMutate || sync.isPending} onClick={() => sync.mutate(selNode.path)}>{t('v2.folders.sync')}</button>
              <button disabled={!canMutate || reanalyze.isPending} onClick={() => reanalyze.mutate(selNode.path)}>{t('v2.folders.reanalyze')}</button>
            </div>
            {disconnected && <div style={{ fontSize: 9.5, color: 'var(--faint)', marginTop: 6 }}>{t('v2.folders.need_connection')}</div>}
          </div>
        )}
      </aside>

      <div className="lib-main">
        <div className="sec-title">{t('v2.folders.overview')} <span className="sub">{t('v2.folders.overview_sub')}</span></div>
        <div className="jobs">
          {folders.map(r => (
            <div className="jrow" key={r.path}>
              <span className={`cov ${r.cov}`} style={{ width: 9, height: 9 }} />
              <span className="nm">{r.name}</span>
              <div className="bar"><i style={{ width: `${r.pct}%`, background: r.cov === 'r' ? 'var(--amber)' : undefined }} /></div>
              <span className="cnt">{t('v2.folders.row_count', { done: r.analyzed.toLocaleString(), total: r.total.toLocaleString() })}</span>
              <span style={{ fontSize: 10.5, color: r.fullyDone ? 'var(--faint)' : 'var(--amber)' }}>
                {r.fullyDone ? t('v2.folders.done') : r.pct === 0 ? t('v2.folders.not_analyzed') : `${r.pct}%`}
              </span>
            </div>
          ))}
          {folders.length === 0 && !loading && (
            <div className="jrow" style={{ color: 'var(--faint)' }}>
              <span className="nm">{t('v2.folders.empty_hint')}</span>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
