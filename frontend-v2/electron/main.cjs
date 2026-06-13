/**
 * Imagine v2 — 얇은 Electron 셸 (자체, 기존 frontend/electron 과 무관).
 *
 * 역할: ① Python 백엔드(uvicorn)를 spawn(이미 떠 있으면 재사용) ② v2 렌더러 창 로드.
 * 의도적으로 window.electron 을 노출하지 않는다 → 렌더러의 isElectron=false →
 * v2 가 순수 HTTP 모드로 동작(이식된 IPC 분기는 휴면). "embedded/IPC 제거, HTTP
 * 일원화" 방향(project_worker_unification_decision)과 일치.
 *
 * 백엔드는 stdin 파이프로 spawn — Electron 생존 동안 파이프가 열려 있어 parent-watchdog
 * 가 블록(정상), Electron 종료 시 파이프가 닫혀 백엔드가 깨끗이 종료된다.
 */
const { app, BrowserWindow, shell } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const http = require('http')

const isDev = !app.isPackaged
const PORT = Number(process.env.IMAGINE_PORT || 8000)
const DEV_URL = process.env.IMAGINE_DEV_URL || 'http://localhost:9275'
const projectRoot = path.resolve(__dirname, '..', '..') // frontend-v2/electron → repo root
let backendProc = null

function healthOnce() {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${PORT}/api/v1/health`, (r) => { r.resume(); resolve(r.statusCode === 200) })
    req.on('error', () => resolve(false))
    req.setTimeout(1000, () => { req.destroy(); resolve(false) })
  })
}
async function waitHealth(timeoutMs = 40000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    if (await healthOnce()) return true
    await new Promise((r) => setTimeout(r, 500))
  }
  return false
}

function spawnBackend() {
  const py = isDev
    ? path.join(projectRoot, '.venv', 'bin', 'python')
    : path.join(process.resourcesPath, 'python', process.platform === 'win32' ? 'python.exe' : 'python3')
  console.log(`[electron] spawning backend: ${py} -m uvicorn (cwd=${projectRoot})`)
  backendProc = spawn(py, ['-m', 'uvicorn', 'backend.server.app:app', '--host', '127.0.0.1', '--port', String(PORT)], {
    cwd: projectRoot,
    stdio: ['pipe', 'pipe', 'pipe'], // stdin open → parent-watchdog blocks; closes on quit → backend exits
    env: { ...process.env, PYTHONPATH: projectRoot },
  })
  backendProc.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`))
  backendProc.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`))
  backendProc.on('exit', (code) => console.log(`[electron] backend exited (${code})`))
}

function killBackend() {
  if (backendProc && !backendProc.killed) {
    try { backendProc.stdin.end() } catch { /* noop */ }
    try { backendProc.kill() } catch { /* noop */ }
    backendProc = null
  }
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 1440, height: 900, backgroundColor: '#0f1419',
    title: 'Imagine',
    webPreferences: { contextIsolation: true, nodeIntegration: false }, // no preload → no window.electron
  })
  // 외부 링크는 기본 브라우저로
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: 'deny' } })
  if (isDev) await win.loadURL(DEV_URL)
  else await win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  return win
}

app.whenReady().then(async () => {
  const already = await healthOnce()
  if (already) console.log('[electron] backend already running — reusing :' + PORT)
  else { spawnBackend(); const ok = await waitHealth(); if (!ok) console.error('[electron] backend did not become healthy') }
  await createWindow()
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
})

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
app.on('before-quit', killBackend)
process.on('exit', killBackend)
