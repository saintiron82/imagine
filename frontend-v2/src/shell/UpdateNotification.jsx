import { useEffect, useState } from 'react'

/**
 * 자동 업데이트 토스트 (IMGV2-27) — Electron 셸에서만.
 * window.imagineDesktop.onUpdateEvent 로 메인 프로세스 업데이트 이벤트를 받아
 * 우하단 토스트로 보여준다. 브라우저(window.imagineDesktop 부재)에선 null.
 *   event.type: checking | available | progress | downloaded | none | error
 */
export default function UpdateNotification() {
  const [evt, setEvt] = useState(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const d = typeof window !== 'undefined' && window.imagineDesktop
    if (!d || typeof d.onUpdateEvent !== 'function') return
    const off = d.onUpdateEvent((e) => { setEvt(e); setDismissed(false) })
    return () => { if (typeof off === 'function') off() }
  }, [])

  // 조용한 상태(확인 중 / 없음)는 토스트를 띄우지 않는다.
  if (!evt || dismissed || evt.type === 'checking' || evt.type === 'none') return null

  const install = () => { window.imagineDesktop?.installUpdate?.() }

  let title = '', body = '', action = null, tone = 'var(--cyan)'
  if (evt.type === 'available') { title = '새 버전 다운로드 중'; body = evt.version ? `v${evt.version}` : '백그라운드에서 받는 중…' }
  else if (evt.type === 'progress') { title = '업데이트 다운로드'; body = `${evt.percent ?? 0}%` }
  else if (evt.type === 'downloaded') { title = '업데이트 준비됨'; body = evt.version ? `v${evt.version} — 재시작하면 적용됩니다` : '재시작하면 적용됩니다'; action = '지금 재시작' }
  else if (evt.type === 'error') { title = '업데이트 확인 실패'; body = evt.message || '잠시 후 다시 시도합니다'; tone = 'var(--red)' }
  else return null

  return (
    <div style={{
      position: 'fixed', right: 18, bottom: 18, zIndex: 60, width: 300,
      background: 'var(--panel)', border: `1px solid ${tone}`, borderRadius: 10,
      padding: '12px 14px', boxShadow: '0 8px 30px rgba(0,0,0,.4)', fontSize: 12.5,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <b style={{ color: tone }}>{title}</b>
        <button onClick={() => setDismissed(true)} style={{ marginLeft: 'auto', background: 'none', border: 'none', color: 'var(--faint)', cursor: 'pointer', fontSize: 15, lineHeight: 1 }}>×</button>
      </div>
      <div style={{ color: 'var(--dim)', marginTop: 4 }}>{body}</div>
      {evt.type === 'progress' && (
        <div style={{ height: 4, background: 'var(--panel2)', borderRadius: 99, marginTop: 8, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${evt.percent ?? 0}%`, background: 'var(--cyan)' }} />
        </div>
      )}
      {action && (
        <button onClick={install} style={{ marginTop: 10, width: '100%' }}>{action}</button>
      )}
    </div>
  )
}
