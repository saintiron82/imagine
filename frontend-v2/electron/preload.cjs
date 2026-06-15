/**
 * v2 데스크톱 preload — 의도적으로 최소.
 * window.electron 은 노출하지 않는다(노출하면 isElectron=true 가 되어 이식한
 * 검색면의 IPC 분기가 깨짐). Google OAuth 만 별도 네임스페이스로 노출 →
 * firebaseAuth.signInWithGoogle 이 popup 대신 이 채널을 쓴다.
 */
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('imagineDesktop', {
  googleOAuth: () => ipcRenderer.invoke('google-oauth'),
  // 서버 켜고/끄기/상태/자동실행은 이 Electron 앱이 담당.
  startServer: () => ipcRenderer.invoke('server-start'),
  stopServer: () => ipcRenderer.invoke('server-stop'),
  serverStatus: () => ipcRenderer.invoke('server-status'),
  setAutostart: (enabled) => ipcRenderer.invoke('server-autostart', enabled),
  // 자동 업데이트(IMGV2-27): 메인→렌더러 이벤트 구독 + 확인/설치 트리거.
  onUpdateEvent: (cb) => {
    const handler = (_e, data) => cb(data)
    ipcRenderer.on('update-event', handler)
    return () => ipcRenderer.removeListener('update-event', handler)
  },
  checkForUpdates: () => ipcRenderer.invoke('update-check'),
  installUpdate: () => ipcRenderer.invoke('update-install'),
})
