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
const { app, BrowserWindow, shell, ipcMain, session } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const http = require('http')
const crypto = require('crypto')

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
    webPreferences: {
      contextIsolation: true, nodeIntegration: false,
      preload: path.join(__dirname, 'preload.cjs'), // exposes window.imagineDesktop only (NOT window.electron)
    },
  })
  // 외부 링크는 기본 브라우저로
  win.webContents.setWindowOpenHandler(({ url }) => { shell.openExternal(url); return { action: 'deny' } })

  // dev 편의: IMAGINE_DEV_TOKEN 이 있으면 렌더러 localStorage 에 토큰을 1회 주입 →
  // 로그인 없이 실데이터로 진입(개발 전용). 미설정 시 정상 로그인 화면.
  if (isDev && process.env.IMAGINE_DEV_TOKEN) {
    let seeded = false
    win.webContents.on('did-finish-load', async () => {
      if (seeded) return
      seeded = true
      const t = JSON.stringify(process.env.IMAGINE_DEV_TOKEN)
      await win.webContents.executeJavaScript(
        `localStorage.setItem('imagine-access-token', ${t}); localStorage.setItem('imagine-server-url','');`,
      )
      win.reload()
    })
  }

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

// ── Google OAuth (desktop) — signInWithPopup 은 Electron 에서 막히므로 시스템 OAuth
//    윈도로 id_token 을 받아 렌더러에서 signInWithCredential. (구 셸 핸들러 이식) ──
const AUTH_DOMAIN = 'imagine-b1e9c.firebaseapp.com'
const FB_API_KEY = 'AIzaSyDgpwrJbQ8MYkP3NFAOrp-K8R3e8kaWpCc'

ipcMain.handle('google-oauth', async () => {
  // 1) Firebase auth handler 에서 Google client_id 추출
  const clientId = await new Promise((resolve, reject) => {
    const hidden = new BrowserWindow({ width: 0, height: 0, show: false, webPreferences: { nodeIntegration: false, contextIsolation: true } })
    let found = false
    const to = setTimeout(() => { if (!found) { found = true; hidden.close(); reject(new Error('CLIENT_ID_TIMEOUT')) } }, 10000)
    const grab = (event, url) => {
      if (found) return
      if (url.includes('accounts.google.com') && url.includes('client_id=')) {
        event.preventDefault(); found = true; clearTimeout(to)
        try { resolve(new URL(url).searchParams.get('client_id')) } catch (e) { reject(e) }
        hidden.close()
      }
    }
    hidden.webContents.on('will-redirect', grab)
    hidden.webContents.on('will-navigate', grab)
    hidden.loadURL(`https://${AUTH_DOMAIN}/__/auth/handler?apiKey=${FB_API_KEY}&authType=signInViaPopup&providerId=google.com&scopes=profile%20email&eventId=${Date.now()}`)
  })

  // 2) 보이는 Google OAuth 윈도(implicit flow → id_token in hash) → 캡처
  const authSession = session.fromPartition('persist:google-auth')
  return new Promise((resolve, reject) => {
    const win = new BrowserWindow({ width: 460, height: 700, autoHideMenuBar: true, title: 'Google Sign-In', webPreferences: { nodeIntegration: false, contextIsolation: true, session: authSession } })
    let settled = false
    const finish = (idToken) => { if (settled) return; settled = true; resolve({ idToken }); try { win.close() } catch { /* noop */ } }
    const checkUrl = (url) => {
      if (settled || !url) return
      try {
        const u = new URL(url)
        if (u.hash && u.hash.includes('id_token=')) { const t = new URLSearchParams(u.hash.slice(1)).get('id_token'); if (t) return finish(t) }
        const qt = u.searchParams.get('id_token'); if (qt) finish(qt)
      } catch { /* noop */ }
    }
    const tryHash = async () => {
      if (settled) return
      try { const h = await win.webContents.executeJavaScript('location.hash'); if (h && h.includes('id_token=')) { const t = new URLSearchParams(h.slice(1)).get('id_token'); if (t) finish(t) } } catch { /* noop */ }
    }
    win.webContents.on('will-redirect', (_, u) => checkUrl(u))
    win.webContents.on('will-navigate', (_, u) => checkUrl(u))
    win.webContents.on('did-navigate', (_, u) => checkUrl(u))
    win.webContents.on('did-navigate-in-page', (_, u) => checkUrl(u))
    win.webContents.on('did-finish-load', () => { checkUrl(win.webContents.getURL()); setTimeout(tryHash, 300); setTimeout(tryHash, 1000); setTimeout(tryHash, 3000) })
    win.on('closed', () => { if (!settled) { settled = true; reject(new Error('AUTH_WINDOW_CLOSED')) } })
    const nonce = crypto.randomBytes(16).toString('hex')
    const redirectUri = `https://${AUTH_DOMAIN}/__/auth/handler`
    win.loadURL(`https://accounts.google.com/o/oauth2/v2/auth?client_id=${encodeURIComponent(clientId)}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=id_token&scope=openid%20email%20profile&nonce=${nonce}&prompt=select_account`)
  })
})
