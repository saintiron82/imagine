/**
 * Imagine v2 — 얇은 Electron 셸 (자체, 기존 frontend/electron 과 무관).
 *
 * 역할: ① Python 백엔드(서버)를 **분리·상주 프로세스**로 띄움(이미 떠 있으면 재사용)
 *        ② v2 렌더러 창 로드.
 *
 * 핵심 모델(사용자 결정): 서버는 독립 서비스, 앱은 그 서버의 조종석.
 * - 백엔드는 detached + unref 로 spawn → **창을 닫아도 서버는 계속 살아 팀에 서비스.**
 *   (IMAGINE_NO_PARENT_WATCHDOG=1 로 부모-사망 워치독 비활성 — 부모 없이 살아야 함.)
 * - 앱 종료 시 백엔드를 죽이지 않는다. 끄는 건 명시적 'server-stop'(관리 버튼)으로만.
 * - PID 파일로 앱 재시작 후에도 끌 수 있게 추적.
 * - window.electron 은 노출하지 않음(isElectron=false 유지) — Google OAuth·server-stop
 *   은 window.imagineDesktop 네임스페이스로만.
 */
const { app, BrowserWindow, shell, ipcMain, session } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')
const http = require('http')
const crypto = require('crypto')

const isDev = !app.isPackaged
const PORT = Number(process.env.IMAGINE_PORT || 8000)
const DEV_URL = process.env.IMAGINE_DEV_URL || 'http://localhost:9275'
const projectRoot = path.resolve(__dirname, '..', '..') // frontend-v2/electron → repo root
const PIDFILE = path.join(projectRoot, '.imagine_server.pid')
const LOGFILE = path.join(projectRoot, 'imagine_server.log')
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

// 서버를 분리·상주 프로세스로 띄운다 — 앱(부모)이 죽어도 살아남는다.
// dev: .venv 의 python 으로 uvicorn 실행. prod: PyInstaller 번들 backend_cli 의
// `server` 서브커맨드(구 frontend 와 동일한 검증된 패키징 — backend_cli.spec →
// dist/backend_cli → electron-builder extraResources 로 resources/backend 에 동봉).
function spawnBackend() {
  const out = fs.openSync(LOGFILE, 'a')
  const common = {
    cwd: projectRoot,
    detached: true,                 // 부모 프로세스 그룹에서 분리
    stdio: ['ignore', out, out],    // 로그 파일로 (창과 무관)
    env: { ...process.env, PYTHONPATH: projectRoot, IMAGINE_NO_PARENT_WATCHDOG: '1' },
  }
  if (isDev) {
    const py = path.join(projectRoot, '.venv', 'bin', 'python')
    console.log(`[electron] starting independent backend (dev): ${py} -m uvicorn`)
    backendProc = spawn(py, ['-m', 'uvicorn', 'backend.server.app:app', '--host', '127.0.0.1', '--port', String(PORT)], common)
  } else {
    const exe = process.platform === 'win32' ? 'backend_cli.exe' : 'backend_cli'
    const cli = path.join(process.resourcesPath, 'backend', exe)
    console.log(`[electron] starting independent backend (packaged): ${cli} server`)
    backendProc = spawn(cli, ['server', '--port', String(PORT), '--host', '127.0.0.1'], { ...common, cwd: process.resourcesPath })
  }
  try { fs.writeFileSync(PIDFILE, String(backendProc.pid)) } catch { /* noop */ }
  backendProc.unref()               // 앱이 이 자식을 기다리지 않게 → 앱 종료해도 서버 생존
}

// 명시적 서버 종료(관리 'server-stop' 에서만). PID 파일로 앱 재시작 후에도 끌 수 있음.
function stopBackend() {
  let pid = backendProc?.pid
  if (!pid) { try { pid = Number(fs.readFileSync(PIDFILE, 'utf8').trim()) } catch { /* none */ } }
  if (pid) { try { process.kill(pid) } catch { /* already gone */ } }
  try { fs.unlinkSync(PIDFILE) } catch { /* noop */ }
  backendProc = null
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
  setupAutoUpdater(win)
  return win
}

// ── 자동 업데이트 (electron-updater) — IMGV2-27 ──────────────────────────────
// 패키징된 앱에서만 동작. dev 에서는 require 자체를 하지 않으므로 electron-updater
// 미설치여도 dev 가 깨지지 않는다. 실제 확인/다운로드는 build.publish 피드가
// 구성돼야 의미가 있다(미구성 시 checkForUpdates 가 에러 → 토스트 없이 무시).
let _autoUpdater = null
function setupAutoUpdater(win) {
  if (!app.isPackaged) return
  try {
    ({ autoUpdater: _autoUpdater } = require('electron-updater'))
  } catch (e) {
    console.warn('[electron] electron-updater unavailable:', e.message)
    _autoUpdater = null
    return
  }
  _autoUpdater.autoDownload = true
  _autoUpdater.autoInstallOnAppQuit = true
  const send = (type, payload = {}) => {
    try { win.webContents.send('update-event', { type, ...payload }) } catch { /* window gone */ }
  }
  _autoUpdater.on('checking-for-update', () => send('checking'))
  _autoUpdater.on('update-available', (info) => send('available', { version: info?.version }))
  _autoUpdater.on('update-not-available', () => send('none'))
  _autoUpdater.on('download-progress', (p) => send('progress', { percent: Math.round(p?.percent || 0) }))
  _autoUpdater.on('update-downloaded', (info) => send('downloaded', { version: info?.version }))
  _autoUpdater.on('error', (err) => send('error', { message: String(err?.message || err) }))
  // 시작 5초 후 1회 + 6시간 주기 확인
  setTimeout(() => { _autoUpdater.checkForUpdates().catch(() => {}) }, 5000)
  setInterval(() => { _autoUpdater.checkForUpdates().catch(() => {}) }, 6 * 60 * 60 * 1000)
}

ipcMain.handle('update-check', async () => {
  if (!_autoUpdater) return { ok: false, reason: 'unavailable' }
  try { await _autoUpdater.checkForUpdates(); return { ok: true } }
  catch (e) { return { ok: false, reason: String(e?.message || e) } }
})
ipcMain.handle('update-install', async () => {
  if (!_autoUpdater) return { ok: false }
  setImmediate(() => _autoUpdater.quitAndInstall())
  return { ok: true }
})

async function ensureBackend() {
  if (await healthOnce()) return true
  spawnBackend()
  return waitHealth()
}

app.whenReady().then(async () => {
  // 자동 실행(로그인 항목)으로 숨겨서 뜬 경우 → 서버만 띄우고 창은 만들지 않는다.
  const li = app.getLoginItemSettings()
  const launchedHidden = li.wasOpenedAsHidden || li.wasOpenedAtLogin || process.argv.includes('--hidden')
  await ensureBackend()
  if (!launchedHidden) await createWindow()
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow() })
})

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
// 앱이 종료돼도 백엔드는 죽이지 않는다 — 서버는 detached·상주라 앱과 무관하게 산다.

// 서버 켜기/끄기/상태/자동실행은 모두 이 Electron 앱이 담당(window.imagineDesktop 채널).
ipcMain.handle('server-start', async () => ({ ok: await ensureBackend() }))
ipcMain.handle('server-stop', async () => { stopBackend(); return { ok: true } })
ipcMain.handle('server-status', async () => ({
  running: await healthOnce(),
  autostart: app.getLoginItemSettings().openAtLogin,
  desktop: true,
}))
ipcMain.handle('server-autostart', async (_e, enabled) => {
  app.setLoginItemSettings({ openAtLogin: !!enabled, openAsHidden: !!enabled })
  return { ok: true, autostart: app.getLoginItemSettings().openAtLogin }
})

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
