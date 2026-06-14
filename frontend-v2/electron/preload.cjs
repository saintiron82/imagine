/**
 * v2 데스크톱 preload — 의도적으로 최소.
 * window.electron 은 노출하지 않는다(노출하면 isElectron=true 가 되어 이식한
 * 검색면의 IPC 분기가 깨짐). Google OAuth 만 별도 네임스페이스로 노출 →
 * firebaseAuth.signInWithGoogle 이 popup 대신 이 채널을 쓴다.
 */
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('imagineDesktop', {
  googleOAuth: () => ipcRenderer.invoke('google-oauth'),
  // 서버는 독립·상주 프로세스 — 끄는 건 관리에서 명시적으로만.
  stopServer: () => ipcRenderer.invoke('server-stop'),
  serverStatus: () => ipcRenderer.invoke('server-status'),
})
