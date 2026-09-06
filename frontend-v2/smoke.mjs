// 렌더 스모크 — preview 서버(4199) 대상. 화면 5개 + 추가 플로우 클릭 검증.
import { chromium } from 'playwright'
const b = await chromium.launch()
const p = await b.newPage()
const errs = []
p.on('pageerror', e => errs.push(e.message))
p.on('console', m => { if (m.type() === 'error') errs.push(m.text()) })
await p.goto('http://localhost:4199/#/search')
for (const r of ['search', 'folders', 'analysis', 'admin', 'settings']) {
  await p.goto(`http://localhost:4199/#/${r}`)
  await p.waitForTimeout(200)
  const visible = await p.isVisible('main section')
  if (!visible) errs.push(`screen not visible: ${r}`)
}
await p.goto('http://localhost:4199/#/search')
await p.click('text=＋ 추가')
await p.click('text=NAS 폴더')
await p.click('text=탐색')
await p.click('button.pri:has-text("작업 등록")')
const done = await p.isVisible('text=분석 리스트에 등록되었습니다')
if (!done) errs.push('add flow did not complete')
console.log(errs.length ? `FAIL:\n${errs.join('\n')}` : 'SMOKE PASS')
await b.close()
process.exit(errs.length ? 1 : 0)
