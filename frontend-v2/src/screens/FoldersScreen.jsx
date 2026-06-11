import { useState } from 'react'

/**
 * 폴더 — "무엇이 있나". 등록된 폴더의 구조·속성·분석률.
 * 작업(동적)은 분석 탭의 일 — 여기서는 보지 않는다.
 */
const TREE = [
  { id: 'nas-chars', icon: '🌐', name: 'NAS / 캐릭터', cov: 'a', children: [
    { id: 'heroes', icon: '📁', name: '주인공', cov: 'g' },
    { id: 'npc', icon: '📁', name: 'NPC', cov: 'a' },
  ]},
  { id: 'nas-bg', icon: '🌐', name: 'NAS / 배경', cov: 'g' },
  { id: 'local-concept', icon: '💻', name: '로컬 / 컨셉아트', cov: 'r' },
]

const STATUS_ROWS = [
  { icon: '🌐', name: 'NAS / 캐릭터', done: 2560, total: 3800, pct: 67, cov: 'a', note: '자동 분석 ON', actions: ['동기화', '다시 분석…'] },
  { icon: '🌐', name: 'NAS / 배경', done: 880, total: 880, pct: 100, cov: 'g', note: '자동 분석 ON · 동기화 3일 전', actions: ['동기화', '다시 분석…'] },
  { icon: '💻', name: '로컬 / 컨셉아트', done: 698, total: 700, pct: 99, cov: 'r', note: '다시 분석 필요', warn: true, actions: ['다시 분석…', '제거'] },
]

function TreeNode({ node, depth = 0, selected, onSelect }) {
  return (
    <>
      <div
        className={`tnode ${depth > 0 ? 'lvl1' : ''}`}
        style={selected === node.id ? { background: 'rgba(59,130,246,.12)' } : undefined}
        onClick={() => onSelect(node.id)}
      >
        <span className="tw">{node.children ? '▾' : '▸'}</span>
        <span className="ic">{node.icon}</span>
        <span className="nm">{node.name}</span>
        <span className={`cov ${node.cov}`} />
      </div>
      {node.children?.map(c => (
        <TreeNode key={c.id} node={c} depth={depth + 1} selected={selected} onSelect={onSelect} />
      ))}
    </>
  )
}

export default function FoldersScreen() {
  const [selected, setSelected] = useState('nas-chars')

  return (
    <section id="scr-library" className="screen active" style={{ height: '100%' }}>
      <aside className="lib-side">
        <h3>폴더 <button>+ 추가</button></h3>
        <div className="tree">
          {TREE.map(n => <TreeNode key={n.id} node={n} selected={selected} onSelect={setSelected} />)}
        </div>
        <div className="src-detail">
          <div className="sd-head">🌐 NAS / 캐릭터 <span className="sd-sub">폴더 속성</span></div>
          <div className="sd-row"><span>새 파일 자동 분석</span><span className="toggle" style={{ transform: 'scale(.85)' }} /></div>
          <div className="sd-row"><span>마지막 동기화</span><span className="sd-val">3일 전 · 이동 2건 감지</span></div>
          <div className="sd-acts">
            <button>동기화</button>
            <button>다시 분석…</button>
            <button className="danger">제거</button>
          </div>
        </div>
      </aside>
      <div className="lib-main">
        <div className="sec-title">폴더 현황 <span className="sub">등록된 폴더가 얼마나 분석됐는지</span></div>
        <div className="jobs">
          {STATUS_ROWS.map(r => (
            <div className="jrow" key={r.name}>
              <span className={`cov ${r.cov}`} style={{ width: 9, height: 9 }} />
              <span className="nm">{r.icon} {r.name}</span>
              <div className="bar"><i style={{ width: `${r.pct}%`, background: r.warn ? 'var(--amber)' : undefined }} /></div>
              <span className="cnt">{r.done.toLocaleString()} / {r.total.toLocaleString()} 장</span>
              <span style={{ fontSize: 10.5, color: r.warn ? 'var(--amber)' : 'var(--faint)' }}>{r.note}</span>
              <div className="acts2">
                {r.actions.map(a => <button key={a} className={a === '제거' ? 'danger' : ''}>{a}</button>)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
