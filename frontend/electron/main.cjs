const { app, BrowserWindow, ipcMain, dialog, shell, Menu } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');
const isDev = process.env.NODE_ENV === 'development';

// Desktop startup log (visible even if app fails to start)
try {
    const _deskLog = path.join(app.getPath('desktop'), 'imagine-startup.log');
    fs.writeFileSync(_deskLog, `[${new Date().toISOString()}] Imagine starting...\nargv: ${process.argv.join(' ')}\nexecPath: ${process.execPath}\ncwd: ${process.cwd()}\nplatform: ${process.platform}\narch: ${process.arch}\n`);
    process.on('uncaughtException', (err) => {
        try { fs.appendFileSync(_deskLog, `[CRASH] ${err.stack || err}\n`); } catch {}
    });
} catch {}

// electron-updater (optional — skip if not available)
let autoUpdater = null;
try { autoUpdater = require('electron-updater').autoUpdater; } catch {}

// Suppress EPIPE errors from console.log when parent pipe is closed (background launch)
process.stdout?.on?.('error', (err) => { if (err.code !== 'EPIPE') throw err; });
process.stderr?.on?.('error', (err) => { if (err.code !== 'EPIPE') throw err; });

// ---------- File-based crash/error logging ----------
// Logs to <userData>/logs/main-<timestamp>.log per session. Old logs auto-cleaned.
const LOG_MAX_FILES = 5; // keep last 5 session logs
const logDir = path.join(app.getPath('userData'), 'logs');
try { fs.mkdirSync(logDir, { recursive: true }); } catch { /* ignore */ }

// Generate session-specific log filename
const _sessionTs = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19); // 2026-03-05T14-30-00
const logFilePath = path.join(logDir, `main-${_sessionTs}.log`);

// Clean old session logs (keep last N)
function _cleanOldLogs() {
    try {
        const files = fs.readdirSync(logDir)
            .filter(f => f.startsWith('main-') && f.endsWith('.log'))
            .sort()
            .reverse();
        // Also include legacy main.log / main.log.1
        const legacy = ['main.log', 'main.log.1'];
        for (const old of legacy) {
            try { fs.unlinkSync(path.join(logDir, old)); } catch { /* ok */ }
        }
        // Remove excess session logs
        for (const old of files.slice(LOG_MAX_FILES)) {
            try { fs.unlinkSync(path.join(logDir, old)); } catch { /* ok */ }
        }
    } catch { /* best effort */ }
}
_cleanOldLogs();

function writeLog(level, ...args) {
    const ts = new Date().toISOString();
    const msg = args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ');
    const line = `${ts} [${level}] ${msg}\n`;
    try {
        fs.appendFileSync(logFilePath, line, 'utf8');
    } catch { /* best effort */ }
}

// Crash handlers — log to file before process dies
process.on('uncaughtException', (err) => {
    writeLog('FATAL', 'uncaughtException:', err.stack || err.message || String(err));
    console.error('[FATAL] uncaughtException:', err);
});
process.on('unhandledRejection', (reason) => {
    writeLog('ERROR', 'unhandledRejection:', String(reason));
    console.error('[ERROR] unhandledRejection:', reason);
});

// V8 heap monitoring — warn before OOM
let _heapWarnedAt = 0;
const HEAP_CHECK_INTERVAL = 30_000; // 30s
const HEAP_WARN_THRESHOLD = 1.5 * 1024 * 1024 * 1024; // 1.5 GB
setInterval(() => {
    const mem = process.memoryUsage();
    if (mem.heapUsed > HEAP_WARN_THRESHOLD && Date.now() - _heapWarnedAt > 60_000) {
        _heapWarnedAt = Date.now();
        const mb = (mem.heapUsed / 1024 / 1024).toFixed(0);
        writeLog('WARN', `V8 heap high: ${mb} MB (rss: ${(mem.rss / 1024 / 1024).toFixed(0)} MB)`);
        console.warn(`[HEAP] V8 heap: ${mb} MB`);
    }
}, HEAP_CHECK_INTERVAL);

writeLog('INFO', `Imagine starting (pid: ${process.pid}, electron: ${process.versions.electron}, node: ${process.versions.node})`);
// Resolve project root where backend/ and config.yaml live.
// In dev mode: two levels up from electron/ directory.
// In built mode: first check resourcesPath (bundled production), then traverse
// up from the app location to find the source project tree (local testing).
const projectRoot = (() => {
    if (isDev) return path.resolve(__dirname, '../../');

    // Bundled production: backend/ included via extraResources
    if (fs.existsSync(path.join(process.resourcesPath, 'backend'))) {
        return process.resourcesPath;
    }

    // Local testing: traverse up from app to find project root
    let dir = process.resourcesPath;
    for (let i = 0; i < 10; i++) {
        dir = path.dirname(dir);
        if (dir === path.dirname(dir)) break; // filesystem root
        if (fs.existsSync(path.join(dir, 'backend')) &&
            fs.existsSync(path.join(dir, 'config.yaml'))) {
            return dir;
        }
    }

    return process.resourcesPath;
})();

// Config root: where config.yaml is stored per-instance.
// In dev mode: same as projectRoot.
// In built mode: process.resourcesPath (allows separate config per app instance).
const configRoot = isDev ? projectRoot : process.resourcesPath;

// User settings: personal per-user config (Tier, registered folders, etc.)
// Stored in OS app data directory, separate from system config.yaml.
const userSettingsPath = path.join(app.getPath('userData'), 'user-settings.yaml');

// Keys that belong to user-settings.yaml (personal, per-user)
const USER_SETTING_PREFIXES = [
    'ai_mode.override', 'ai_mode.auto_detect',
    'ai_mode.vlm_backend',
    'batch_processing.enabled', 'batch_processing.adaptive',
    'registered_folders', 'last_session', 'webdav_sources',
    'worker.claim_batch_size', 'worker.gpu_memory_percent',
    'worker.cpu_cores', 'worker.batch_capacity',
    'worker.schedule', 'worker.idle_unload_minutes',
    'worker.processing_mode',
    'ui.locale', 'ui.theme', 'ui.grid_density',
    'webdav.auto_process',
];

function isUserSetting(key) {
    return USER_SETTING_PREFIXES.some(p => key === p || key.startsWith(p + '.'));
}

/**
 * Deep merge two objects. Arrays are replaced (not merged).
 * Source values override target values.
 */
function deepMerge(target, source) {
    const result = { ...target };
    for (const key of Object.keys(source)) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])
            && target[key] && typeof target[key] === 'object' && !Array.isArray(target[key])) {
            result[key] = deepMerge(target[key], source[key]);
        } else {
            result[key] = source[key];
        }
    }
    return result;
}

/**
 * Read and parse a YAML file. Returns empty object if file doesn't exist.
 */
function readYamlFile(filePath) {
    const yaml = require('js-yaml');
    if (!fs.existsSync(filePath)) return {};
    try {
        return yaml.load(fs.readFileSync(filePath, 'utf8')) || {};
    } catch (err) {
        console.error(`[Config] Failed to read ${filePath}:`, err.message);
        return {};
    }
}

/**
 * Write an object to a YAML file, creating parent directories if needed.
 */
function writeYamlFile(filePath, data) {
    const yaml = require('js-yaml');
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(filePath, yaml.dump(data, { lineWidth: -1 }), 'utf8');
}

/**
 * Set a dotted key in an object (e.g., 'ai_mode.override' → obj.ai_mode.override).
 */
function setDottedKey(obj, key, value) {
    const keys = key.split('.');
    let current = obj;
    for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]] || typeof current[keys[i]] !== 'object') {
            current[keys[i]] = {};
        }
        current = current[keys[i]];
    }
    current[keys[keys.length - 1]] = value;
}

/**
 * Migrate personal settings from config.yaml to user-settings.yaml on first run.
 * config.yaml values are preserved as defaults (not deleted).
 */
function migrateUserSettings() {
    if (fs.existsSync(userSettingsPath)) return; // Already migrated

    const systemConfigPath = path.join(configRoot, 'config.yaml');
    const config = readYamlFile(systemConfigPath);
    if (!config || Object.keys(config).length === 0) return;

    const userSettings = {};

    // Extract personal settings
    if (config.ai_mode) {
        userSettings.ai_mode = {};
        if (config.ai_mode.auto_detect != null) userSettings.ai_mode.auto_detect = config.ai_mode.auto_detect;
        if (config.ai_mode.override != null) userSettings.ai_mode.override = config.ai_mode.override;
    }
    if (config.batch_processing) {
        userSettings.batch_processing = {};
        if (config.batch_processing.enabled != null) userSettings.batch_processing.enabled = config.batch_processing.enabled;
        if (config.batch_processing.adaptive) userSettings.batch_processing.adaptive = config.batch_processing.adaptive;
    }
    if (config.registered_folders) {
        userSettings.registered_folders = config.registered_folders;
    }
    if (config.last_session) {
        userSettings.last_session = config.last_session;
    }
    if (config.worker) {
        userSettings.worker = {};
        const workerKeys = ['claim_batch_size', 'gpu_memory_percent', 'cpu_cores', 'batch_capacity'];
        for (const k of workerKeys) {
            if (config.worker[k] != null) userSettings.worker[k] = config.worker[k];
        }
        if (Object.keys(userSettings.worker).length === 0) delete userSettings.worker;
    }

    if (Object.keys(userSettings).length === 0) return;

    writeYamlFile(userSettingsPath, userSettings);
    console.log('[Config] Migrated user settings to', userSettingsPath);
}

// Cross-platform Python path resolution
function getPythonPath() {
    const isWin = process.platform === 'win32';
    const venvDir = isWin ? 'Scripts' : 'bin';
    const pyExe = isWin ? 'python.exe' : 'python3';

    // 1. Check venv in project root
    const venvPath = path.join(projectRoot, '.venv', venvDir, pyExe);
    if (fs.existsSync(venvPath)) return venvPath;

    // 2. Check bundled Python (production)
    const bundledPath = path.join(process.resourcesPath, 'python', pyExe);
    if (fs.existsSync(bundledPath)) return bundledPath;

    return null;
}

function resolvePython() {
    const pythonPath = getPythonPath();
    return (pythonPath && fs.existsSync(pythonPath)) ? pythonPath : 'python3';
}

// ── Backend CLI (PyInstaller bundle) ─────────────────────────────
// In packaged mode, use backend_cli.exe instead of python.
// In dev mode, fall back to python.

let _backendCliPath = undefined; // cached

function getBackendCliPath() {
    if (_backendCliPath !== undefined) return _backendCliPath;
    if (isDev) { _backendCliPath = null; return null; }
    const exeName = process.platform === 'win32' ? 'backend_cli.exe' : 'backend_cli';
    const candidates = [
        path.join(process.resourcesPath || '', 'backend', exeName),
        path.join(process.resourcesPath || '', 'backend_cli', exeName),
    ];
    for (const p of candidates) {
        if (fs.existsSync(p)) {
            _backendCliPath = p;
            writeLog('INFO', `[BackendCLI] Found bundled backend: ${p}`);
            return p;
        }
    }
    _backendCliPath = null;
    writeLog('WARN', '[BackendCLI] No bundled backend found, using python fallback');
    return null;
}

/**
 * Spawn a backend process. In packaged mode uses backend_cli.exe,
 * in dev mode uses python with the original script.
 *
 * @param {string} subcmd - backend_cli subcommand (e.g., 'worker-ipc', 'search-daemon')
 * @param {string[]} subcmdArgs - arguments for the subcommand
 * @param {object} spawnOpts - additional spawn options (env, stdio, detached, etc.)
 * @param {string|null} devScript - relative script path from project root for dev mode (e.g., 'backend/api_stats.py')
 * @param {string[]|null} devArgs - override args for dev mode (default: same as subcmdArgs)
 * @returns {ChildProcess}
 */
function spawnBackend(subcmd, subcmdArgs = [], spawnOpts = {}, devScript = null, devArgs = null) {
    const cliPath = getBackendCliPath();
    if (cliPath) {
        // Packaged mode: use backend_cli.exe
        return spawn(cliPath, [subcmd, ...subcmdArgs], {
            cwd: projectRoot,
            ...spawnOpts,
        });
    } else if (devScript) {
        // Dev mode: use python with specific script
        const scriptPath = isDev
            ? path.resolve(__dirname, '../../', devScript)
            : path.join(projectRoot, devScript);
        const py = resolvePython();
        return spawn(py, [scriptPath, ...(devArgs || subcmdArgs)], {
            cwd: projectRoot,
            ...spawnOpts,
        });
    } else {
        // Dev fallback: python -m backend.xxx
        const py = resolvePython();
        return spawn(py, ['-u', '-m', `backend.${subcmd.replace(/-/g, '_')}`, ...subcmdArgs], {
            cwd: projectRoot,
            ...spawnOpts,
        });
    }
}

// ── Search Daemon (lazy-start, idle-kill) ────────────────────────
// Daemon is NOT started on app launch. It spawns on first search,
// stays alive for IDLE_TIMEOUT_MS after last search, then auto-kills.
const IDLE_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
let searchDaemon = null;
let searchReady = false;
let pendingRequests = [];
let responseBuffer = '';
let idleTimer = null;

/**
 * Kill residual processes from previous crashed sessions.
 *
 * Primary defense against residual processes is the parent_watchdog.py
 * (stdin pipe monitoring) in each Python subprocess.  This cleanup runs
 * as a safety net on app startup in case the watchdog failed.
 */
function cleanupOrphanDaemons() {
    const patterns = ['Imagine-Search', 'Imagine-Pipeline'];
    try {
        if (process.platform === 'win32') {
            // Windows: try taskkill by window title (works for console processes)
            for (const p of patterns) {
                try {
                    execSync(`taskkill /F /FI "WINDOWTITLE eq ${p}" 2>nul`, { stdio: 'ignore' });
                } catch { /* no match — fine */ }
            }
            // Also try wmic for piped (windowless) processes by command line
            const wmicPatterns = ['api_search.py', 'ingest_engine.py', 'uvicorn'];
            for (const pat of wmicPatterns) {
                try {
                    execSync(
                        `wmic process where "name='python.exe' and commandline like '%${pat}%'" call terminate 2>nul`,
                        { stdio: 'ignore', timeout: 5000 },
                    );
                } catch { /* wmic may be unavailable or no match */ }
            }
        } else {
            // macOS/Linux: pkill by process name (set via setproctitle) or command line
            for (const p of patterns) {
                try {
                    execSync(`pkill -f "${p}" 2>/dev/null || true`, { stdio: 'ignore' });
                } catch { /* no match */ }
            }
            // Also kill by script name for processes without setproctitle
            try {
                execSync('pkill -f "uvicorn.*backend.server.app" 2>/dev/null || true', { stdio: 'ignore' });
            } catch { /* no match */ }
        }
    } catch (e) {
        // Cleanup is best-effort — watchdog is the primary defense
    }
}

function resetIdleTimer() {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
        console.log(`[SearchDaemon] Idle for ${IDLE_TIMEOUT_MS / 1000}s — shutting down`);
        killSearchDaemon();
    }, IDLE_TIMEOUT_MS);
}

function spawnSearchDaemon() {
    if (searchDaemon) return;

    console.log('[SearchDaemon] Starting search process (lazy)...');

    searchDaemon = spawnBackend('search-daemon', [], {
        env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath },
        stdio: ['pipe', 'pipe', 'pipe'],
    }, 'backend/api_search.py', ['--daemon']);

    searchDaemon.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) console.error('[SearchDaemon:stderr]', msg);
    });

    searchDaemon.stdout.on('data', (chunk) => {
        responseBuffer += chunk.toString();
        let newlineIdx;
        while ((newlineIdx = responseBuffer.indexOf('\n')) !== -1) {
            const line = responseBuffer.slice(0, newlineIdx).trim();
            responseBuffer = responseBuffer.slice(newlineIdx + 1);
            if (!line) continue;

            try {
                const parsed = JSON.parse(line);
                // Daemon ready signal
                if (!searchReady && parsed.status === 'ok' && parsed.mode === 'daemon') {
                    searchReady = true;
                    console.log(`[SearchDaemon] Ready (PID: ${parsed.pid})`);
                    searchDaemon.stdin.write(JSON.stringify({ cmd: 'warmup' }) + '\n');
                    continue;
                }
                // Warmup complete — flush queued requests + auto-fix metadata
                if (parsed.status === 'ready') {
                    console.log(`[SearchDaemon] Models loaded (${parsed.warmup_ms}ms)`);
                    // Auto-fix missing relative_path on startup
                    searchDaemon.stdin.write(JSON.stringify({ cmd: 'fix_relative_paths' }) + '\n');
                    for (const req of pendingRequests) {
                        searchDaemon.stdin.write(JSON.stringify(req.data) + '\n');
                    }
                    continue;
                }
                // Auto-fix response (not a user request, just log it)
                if (parsed.fixed !== undefined && parsed.success) {
                    if (parsed.fixed > 0) {
                        console.log(`[SearchDaemon] Auto-fixed ${parsed.fixed} files with missing relative_path`);
                    }
                    continue;
                }
                // Search progress event — forward to renderer
                if (parsed.event === 'search_progress') {
                    const win = BrowserWindow.getAllWindows()[0];
                    if (win) win.webContents.send('search-progress', parsed.stage);
                    continue;
                }
                // Normal search response
                if (pendingRequests.length > 0) {
                    const { resolve: res, t0 } = pendingRequests.shift();
                    const jsonKB = (line.length / 1024).toFixed(0);
                    const ipcMs = t0 ? Date.now() - t0 : '?';
                    console.log(`[SearchDaemon] response: ${jsonKB}KB JSON, IPC total: ${ipcMs}ms (python elapsed: ${parsed.elapsed_ms || '?'}ms)`);
                    res(parsed);
                    resetIdleTimer();
                }
            } catch (e) {
                console.error('[SearchDaemon] JSON parse error:', e, line);
                if (pendingRequests.length > 0) {
                    const { resolve: res } = pendingRequests.shift();
                    res({ success: false, error: 'JSON parse error', results: [] });
                }
            }
        }
    });

    searchDaemon.on('close', (code) => {
        console.log(`[SearchDaemon] Exited (code: ${code})`);
        searchDaemon = null;
        searchReady = false;
        responseBuffer = '';
        if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
        const pending = pendingRequests.splice(0);
        for (const { resolve: res } of pending) {
            res({ success: false, error: 'Search daemon exited', results: [] });
        }
    });

    searchDaemon.on('error', (err) => {
        console.error('[SearchDaemon] Spawn error:', err);
        searchDaemon = null;
        searchReady = false;
    });
}

function sendSearchRequest(data) {
    return new Promise((resolve) => {
        if (!searchDaemon) {
            spawnSearchDaemon();
        }
        const t0 = Date.now();
        pendingRequests.push({ resolve, data, t0 });
        if (searchReady && searchDaemon) {
            searchDaemon.stdin.write(JSON.stringify(data) + '\n');
        }
    });
}

function killSearchDaemon() {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
    if (!searchDaemon) return;

    const proc = searchDaemon;
    searchDaemon = null;
    searchReady = false;
    responseBuffer = '';

    try {
        proc.stdin.write(JSON.stringify({ cmd: 'quit' }) + '\n');
    } catch (e) { /* ignore */ }
    setTimeout(() => {
        try { proc.kill('SIGTERM'); } catch (e) { /* already dead */ }
    }, 2000);
}

// ── Auto Updater ─────────────────────────────────────────────────

function sendUpdateEvent(channel, data) {
    try {
        const windows = BrowserWindow.getAllWindows();
        for (const win of windows) {
            if (!win.isDestroyed()) {
                win.webContents.send(channel, data);
            }
        }
    } catch (e) { /* window may be closed */ }
}

function initAutoUpdater() {
    if (!autoUpdater) { writeLog('WARN', 'electron-updater not available, skipping auto-update'); return; }
    autoUpdater.autoDownload = false;
    autoUpdater.autoInstallOnAppQuit = true;
    autoUpdater.allowPrerelease = false;

    autoUpdater.logger = {
        info: (...args) => writeLog('INFO', '[AutoUpdater]', ...args),
        warn: (...args) => writeLog('WARN', '[AutoUpdater]', ...args),
        error: (...args) => writeLog('ERROR', '[AutoUpdater]', ...args),
        debug: () => {}, // suppress verbose debug
    };

    autoUpdater.on('checking-for-update', () => {
        writeLog('INFO', '[AutoUpdater] Checking for update...');
        sendUpdateEvent('update-checking', {});
    });

    autoUpdater.on('update-available', (info) => {
        writeLog('INFO', '[AutoUpdater] Update available:', info.version);
        sendUpdateEvent('update-available', {
            version: info.version,
            releaseDate: info.releaseDate,
            releaseNotes: typeof info.releaseNotes === 'string'
                ? info.releaseNotes
                : (info.releaseNotes || []).map(n => n.note || '').filter(Boolean).join('\n'),
        });
    });

    autoUpdater.on('update-not-available', (info) => {
        writeLog('INFO', '[AutoUpdater] No update available (current:', info.version, ')');
        sendUpdateEvent('update-not-available', { version: info.version });
    });

    autoUpdater.on('download-progress', (progress) => {
        sendUpdateEvent('update-download-progress', {
            percent: Math.round(progress.percent),
            bytesPerSecond: progress.bytesPerSecond,
            transferred: progress.transferred,
            total: progress.total,
        });
    });

    autoUpdater.on('update-downloaded', (info) => {
        writeLog('INFO', '[AutoUpdater] Update downloaded:', info.version);
        sendUpdateEvent('update-downloaded', {
            version: info.version,
            releaseDate: info.releaseDate,
        });
    });

    autoUpdater.on('error', (err) => {
        writeLog('ERROR', '[AutoUpdater] Error:', err.message);
        sendUpdateEvent('update-error', { message: err.message });
    });

    // Initial check after 5 seconds (avoid blocking startup)
    setTimeout(() => {
        if (!isDev) {
            autoUpdater.checkForUpdates().catch(err => {
                writeLog('WARN', '[AutoUpdater] Initial check failed:', err.message);
            });
        }
    }, 5000);

    // Periodic check every 4 hours
    setInterval(() => {
        if (!isDev) {
            autoUpdater.checkForUpdates().catch(err => {
                writeLog('WARN', '[AutoUpdater] Periodic check failed:', err.message);
            });
        }
    }, 4 * 60 * 60 * 1000);
}

// ── IPC Handlers (global scope — registered once) ────────────────

// Google OAuth via BrowserWindow (bypasses signInWithPopup unauthorized-domain in Electron)
//
// Uses a persistent session so Google remembers accounts across app restarts.
// 2-step flow:
//   1. Load Firebase auth handler (hidden) → intercept redirect to capture Google client_id
//   2. Open visible Google OAuth with that client_id (response_type=id_token)
//   3. Capture id_token from redirect back → resolve to renderer
ipcMain.handle('google-oauth', async () => {
    const { session: electronSession } = require('electron');
    const crypto = require('crypto');
    const AUTH_DOMAIN = 'imagine-b1e9c.firebaseapp.com';
    const API_KEY = 'AIzaSyDgpwrJbQ8MYkP3NFAOrp-K8R3e8kaWpCc';

    // ── Step 1: Discover Google OAuth client_id from Firebase ──
    const clientId = await new Promise((resolve, reject) => {
        const hiddenWin = new BrowserWindow({
            width: 0, height: 0, show: false,
            webPreferences: { nodeIntegration: false, contextIsolation: true },
        });

        let found = false;
        const timeout = setTimeout(() => {
            if (!found) { found = true; hiddenWin.close(); reject(new Error('CLIENT_ID_TIMEOUT')); }
        }, 10000);

        hiddenWin.webContents.on('will-redirect', (event, url) => {
            if (found) return;
            if (url.includes('accounts.google.com') && url.includes('client_id=')) {
                event.preventDefault();
                found = true;
                clearTimeout(timeout);
                try {
                    resolve(new URL(url).searchParams.get('client_id'));
                } catch (e) {
                    reject(e);
                }
                hiddenWin.close();
            }
        });

        hiddenWin.webContents.on('will-navigate', (event, url) => {
            if (found) return;
            if (url.includes('accounts.google.com') && url.includes('client_id=')) {
                event.preventDefault();
                found = true;
                clearTimeout(timeout);
                try {
                    resolve(new URL(url).searchParams.get('client_id'));
                } catch (e) {
                    reject(e);
                }
                hiddenWin.close();
            }
        });

        const startUrl = `https://${AUTH_DOMAIN}/__/auth/handler?` +
            `apiKey=${API_KEY}&authType=signInViaPopup&providerId=google.com` +
            `&scopes=profile%20email&eventId=${Date.now()}`;
        hiddenWin.loadURL(startUrl);
    });

    writeLog('INFO', 'Google OAuth client_id discovered:', clientId?.substring(0, 20) + '...');

    // ── Step 2: Open visible Google OAuth window ──
    // persist: partition → Google remembers logged-in accounts across sessions
    const authSession = electronSession.fromPartition('persist:google-auth');

    return new Promise((resolve, reject) => {
        const authWindow = new BrowserWindow({
            width: 460,
            height: 700,
            autoHideMenuBar: true,
            title: 'Google Sign-In',
            webPreferences: {
                nodeIntegration: false,
                contextIsolation: true,
                session: authSession,
            },
        });

        let settled = false;
        const finish = (idToken) => {
            if (settled) return;
            settled = true;
            resolve({ idToken });
            try { authWindow.close(); } catch {}
        };

        // Extract id_token from URL (hash fragment or query)
        const checkUrl = (url) => {
            if (settled || !url) return;
            try {
                const u = new URL(url);
                // Hash fragment: #id_token=xxx&...
                if (u.hash && u.hash.includes('id_token=')) {
                    const params = new URLSearchParams(u.hash.substring(1));
                    const t = params.get('id_token');
                    if (t) { finish(t); return; }
                }
                // Query param fallback
                const qt = u.searchParams.get('id_token');
                if (qt) finish(qt);
            } catch {}
        };

        // Fallback: inject JS to read location.hash (in case navigation events lose the fragment)
        const tryReadHash = async () => {
            if (settled) return;
            try {
                const hash = await authWindow.webContents.executeJavaScript('location.hash');
                if (hash && hash.includes('id_token=')) {
                    const params = new URLSearchParams(hash.substring(1));
                    const t = params.get('id_token');
                    if (t) finish(t);
                }
            } catch {}
        };

        authWindow.webContents.on('will-redirect', (_, url) => checkUrl(url));
        authWindow.webContents.on('will-navigate', (_, url) => checkUrl(url));
        authWindow.webContents.on('did-navigate', (_, url) => checkUrl(url));
        authWindow.webContents.on('did-navigate-in-page', (_, url) => checkUrl(url));
        authWindow.webContents.on('did-finish-load', () => {
            checkUrl(authWindow.webContents.getURL());
            // Retry via JS injection (hash fragments may not appear in nav events)
            setTimeout(tryReadHash, 300);
            setTimeout(tryReadHash, 1000);
            setTimeout(tryReadHash, 3000);
        });

        authWindow.on('closed', () => {
            if (!settled) { settled = true; reject(new Error('AUTH_WINDOW_CLOSED')); }
        });

        // Build Google OAuth URL (implicit flow → id_token in hash)
        const nonce = crypto.randomBytes(16).toString('hex');
        const redirectUri = `https://${AUTH_DOMAIN}/__/auth/handler`;
        const oauthUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
            `client_id=${encodeURIComponent(clientId)}` +
            `&redirect_uri=${encodeURIComponent(redirectUri)}` +
            `&response_type=id_token` +
            `&scope=openid%20email%20profile` +
            `&nonce=${nonce}` +
            `&prompt=select_account`;

        authWindow.loadURL(oauthUrl);
    });
});

// ── Worker IPC (spawn worker_ipc.py, control via stdin/stdout JSON) ──

let workerProc = null;
let workerResponseBuffer = '';

function sendWorkerEvent(channel, data) {
    const windows = BrowserWindow.getAllWindows();
    for (const w of windows) {
        if (!w.isDestroyed()) w.webContents.send(channel, data);
    }
}

function processWorkerOutput(line) {
    if (!line.trim()) return;
    try {
        const parsed = JSON.parse(line);
        const event = parsed.event;
        if (event === 'status') sendWorkerEvent('worker-status', parsed);
        else if (event === 'log') sendWorkerEvent('worker-log', parsed);
        else if (event === 'job_done') sendWorkerEvent('worker-job-done', parsed);
        else if (event === 'stats') sendWorkerEvent('worker-stats', parsed);
        else if (event === 'batch_start') sendWorkerEvent('worker-batch-start', parsed);
        else if (event === 'batch_phase_start') sendWorkerEvent('worker-batch-phase-start', parsed);
        else if (event === 'batch_file_done') sendWorkerEvent('worker-batch-file-done', parsed);
        else if (event === 'batch_phase_complete') sendWorkerEvent('worker-batch-phase-complete', parsed);
        else if (event === 'batch_complete') sendWorkerEvent('worker-batch-complete', parsed);
        else if (event === 'processing_mode') sendWorkerEvent('worker-processing-mode', parsed);
    } catch {
        writeLog('WARN', '[Worker] unparseable:', line.substring(0, 200));
    }
}

function spawnWorkerIPC() {
    if (workerProc) return;
    workerProc = spawnBackend('worker-ipc', [], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
    }, 'backend/worker/worker_ipc.py');

    workerProc.stdout.on('data', (chunk) => {
        workerResponseBuffer += chunk.toString('utf8');
        const lines = workerResponseBuffer.split('\n');
        workerResponseBuffer = lines.pop();
        for (const line of lines) processWorkerOutput(line);
    });

    workerProc.stderr.on('data', (chunk) => {
        const msg = chunk.toString('utf8').trim();
        if (msg) writeLog('INFO', '[Worker stderr]', msg);
    });

    workerProc.on('exit', (code) => {
        writeLog('INFO', `[Worker] process exited (code=${code})`);
        workerProc = null;
        workerResponseBuffer = '';
        sendWorkerEvent('worker-status', { event: 'status', status: 'idle', jobs: [] });
    });
}

function sendWorkerCmd(cmd) {
    if (!workerProc || !workerProc.stdin.writable) return;
    workerProc.stdin.write(JSON.stringify(cmd) + '\n');
}

ipcMain.handle('worker-start', async (_, opts) => {
    try {
        spawnWorkerIPC();
        // Wait a moment for process to be ready
        await new Promise(r => setTimeout(r, 300));
        sendWorkerCmd({
            cmd: 'start',
            server_url: opts.serverUrl,
            access_token: opts.accessToken || '',
            refresh_token: opts.refreshToken || '',
        });
        return { success: true };
    } catch (e) {
        return { success: false, error: e.message };
    }
});

ipcMain.handle('worker-stop', async () => {
    try {
        if (workerProc) {
            sendWorkerCmd({ cmd: 'stop' });
        }
        return { success: true };
    } catch (e) {
        return { success: false, error: e.message };
    }
});

ipcMain.handle('worker-status', async () => {
    if (!workerProc) return { status: 'idle', jobs: [] };
    sendWorkerCmd({ cmd: 'status' });
    return { status: 'unknown' }; // actual status comes via event
});

// Auto Update IPC
ipcMain.handle('updater-check', async () => {
    if (isDev || !autoUpdater) return { available: false, reason: 'dev-mode' };
    try {
        const result = await autoUpdater.checkForUpdates();
        return { available: !!result?.updateInfo, info: result?.updateInfo };
    } catch (err) {
        return { available: false, error: err.message };
    }
});

ipcMain.handle('updater-download', async () => {
    if (!autoUpdater) return { success: false, error: 'updater not available' };
    try {
        await autoUpdater.downloadUpdate();
        return { success: true };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

ipcMain.on('updater-quit-and-install', () => {
    if (autoUpdater) autoUpdater.quitAndInstall(false, true);
});

ipcMain.on('app-quit', () => {
    app.quit();
});

ipcMain.handle('updater-get-version', () => {
    return app.getVersion();
});

// IPC Handler: Open Folder Dialog
ipcMain.handle('open-folder-dialog', async () => {
    const result = await dialog.showOpenDialog({
        properties: ['openDirectory'],
        title: 'Select Folder to Process'
    });
    if (result.canceled) return null;
    return result.filePaths[0];
});

// IPC Handler: Show file in OS file explorer
ipcMain.handle('show-item-in-folder', async (_, filePath) => {
    if (!fs.existsSync(filePath)) return { success: false, error: 'File not found' };
    shell.showItemInFolder(filePath);
    return { success: true };
});

// IPC Handler: Open file with OS default application
ipcMain.handle('open-file-native', async (_, filePath) => {
    if (!fs.existsSync(filePath)) return { success: false, error: 'File not found' };
    const errorMsg = await shell.openPath(filePath);
    if (errorMsg) return { success: false, error: errorMsg };
    return { success: true };
});

// IPC Handler: Read Metadata JSON
ipcMain.handle('read-metadata', async (_, filePath) => {
    try {
        const baseName = path.basename(filePath, path.extname(filePath));
        const outputDir = isDev
            ? path.join(projectRoot, 'output/json')
            : path.join(projectRoot, 'output/json');
        const jsonPath = path.join(outputDir, `${baseName}.json`);

        if (fs.existsSync(jsonPath)) {
            const content = fs.readFileSync(jsonPath, 'utf-8');
            return JSON.parse(content);
        }
        return null;
    } catch (err) {
        console.error('[Read Metadata Error]', err);
        return null;
    }
});

// IPC Handler: Check if metadata exists (batch)
ipcMain.handle('check-metadata-exists', async (_, filePaths) => {
    const outputDir = isDev
        ? path.join(projectRoot, 'output/json')
        : path.join(projectRoot, 'output/json');

    const results = {};
    for (const fp of filePaths) {
        const baseName = path.basename(fp, path.extname(fp));
        const jsonPath = path.join(outputDir, `${baseName}.json`);
        results[fp] = fs.existsSync(jsonPath);
    }
    return results;
});

// IPC Handler: Check per-file phase status (MC/VV/MV) via search daemon DB
ipcMain.handle('check-phase-status', async (_, filePaths) => {
    try {
        const result = await sendSearchRequest({ cmd: 'phase_status', file_paths: filePaths });
        return result?.status || {};
    } catch (e) {
        console.error('[check-phase-status] error:', e.message);
        return {};
    }
});

// IPC Handler: Fix missing relative_path via search daemon DB
ipcMain.handle('fix-relative-paths', async () => {
    try {
        const result = await sendSearchRequest({ cmd: 'fix_relative_paths' });
        return result || { success: false };
    } catch (e) {
        console.error('[fix-relative-paths] error:', e.message);
        return { success: false, error: e.message };
    }
});

// IPC Handler: Generate Thumbnail (single file)
ipcMain.handle('generate-thumbnail', async (_, filePath) => {
    return new Promise((resolve) => {
        const proc = spawnBackend('thumbnail', ['--file', filePath, '--size', '256'], {},
            'backend/utils/thumbnail_generator.py', [filePath, '--size', '256']);
        let output = '';
        let error = '';

        proc.stdout.on('data', (data) => {
            output += data.toString();
        });

        proc.stderr.on('data', (data) => {
            error += data.toString();
        });

        proc.on('close', (code) => {
            if (code === 0 && output.trim()) {
                resolve(`data:image/png;base64,${output.trim()}`);
            } else {
                console.error('[Thumbnail Error]', error);
                resolve(null);
            }
        });

        proc.on('error', (err) => {
            console.error('[Thumbnail Spawn Error]', err);
            resolve(null);
        });
    });
});

// IPC Handler: Generate Thumbnails Batch
ipcMain.handle('generate-thumbnails-batch', async (_, filePaths) => {
    return new Promise((resolve) => {
        const proc = spawnBackend('thumbnail', ['--batch', JSON.stringify(filePaths), '--size', '256', '--return-paths'], {},
            'backend/utils/thumbnail_generator.py', ['--batch', JSON.stringify(filePaths), '--size', '256', '--return-paths']);
        let output = '';
        let error = '';

        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());

        proc.on('close', (code) => {
            if (code === 0 && output.trim()) {
                try {
                    const results = JSON.parse(output.trim());
                    resolve(results);
                } catch {
                    resolve({});
                }
            } else {
                console.error('[Batch Thumbnail Error]', error);
                resolve({});
            }
        });

        proc.on('error', (err) => {
            console.error('[Batch Thumbnail Spawn Error]', err);
            resolve({});
        });
    });
});

// IPC Handler: Generate Thumbnails + Phase P Parse (preview_only)
ipcMain.handle('generate-thumbnails-and-parse', async (_, filePaths) => {
    return new Promise((resolve) => {
        const proc = spawnBackend('thumbnail', ['--batch', JSON.stringify(filePaths), '--size', '256', '--return-paths', '--parse'], {},
            'backend/utils/thumbnail_generator.py', ['--batch', JSON.stringify(filePaths), '--size', '256', '--return-paths', '--parse']);
        let output = '';
        let error = '';

        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());

        proc.on('close', (code) => {
            if (code === 0 && output.trim()) {
                try {
                    const results = JSON.parse(output.trim());
                    resolve(results);
                } catch {
                    resolve({});
                }
            } else {
                console.error('[Batch Thumbnail+Parse Error]', error);
                resolve({});
            }
        });

        proc.on('error', (err) => {
            console.error('[Batch Thumbnail+Parse Spawn Error]', err);
            resolve({});
        });
    });
});

// IPC Handler: Persistent thumbnail queue (survives app crash/restart)
const thumbQueuePath = path.join(app.getPath('userData'), 'thumb-queue.json');

ipcMain.handle('thumb-queue-load', async () => {
    try {
        if (!fs.existsSync(thumbQueuePath)) return [];
        const raw = fs.readFileSync(thumbQueuePath, 'utf8');
        const data = JSON.parse(raw);
        const queue = Array.isArray(data?.queue) ? data.queue : [];
        // Filter out files that no longer exist on disk
        return queue.filter(fp => {
            try { return fs.existsSync(fp); } catch { return false; }
        });
    } catch (err) {
        writeLog('WARN', 'Failed to load thumb queue:', err.message);
        return [];
    }
});

ipcMain.handle('thumb-queue-save', async (_, queue) => {
    try {
        const data = JSON.stringify({ queue: queue || [], updatedAt: Date.now() });
        fs.writeFileSync(thumbQueuePath, data, 'utf8');
        return { success: true };
    } catch (err) {
        writeLog('WARN', 'Failed to save thumb queue:', err.message);
        return { success: false };
    }
});

// Download cache IPC removed — downloads are queue-lifecycle managed by server.

// IPC Handler: Check if disk thumbnails exist (no Python needed)
ipcMain.handle('check-thumbnails-exist', async (_, filePaths) => {
    const thumbDir = isDev
        ? path.join(projectRoot, 'output', 'thumbnails')
        : path.join(projectRoot, 'output', 'thumbnails');

    const results = {};
    for (const fp of filePaths) {
        const stem = path.basename(fp, path.extname(fp));
        const thumbPath = path.join(thumbDir, `${stem}_thumb.png`);
        results[fp] = fs.existsSync(thumbPath) ? thumbPath : null;
    }
    return results;
});

// IPC Handler: Open file dialog for zip archive selection
ipcMain.handle('select-archive-file', async () => {
    const result = await dialog.showOpenDialog({
        properties: ['openFile'],
        title: 'Select Archive (.zip)',
        filters: [
            { name: 'ZIP Archive', extensions: ['zip'] },
            { name: 'Database', extensions: ['db'] },
        ],
    });
    if (result.canceled || !result.filePaths.length) return null;
    return result.filePaths[0];
});

// IPC Handler: Export database + thumbnails as zip archive
ipcMain.handle('export-database', async (_, { outputPath }) => {
    // If no outputPath, open save dialog
    if (!outputPath) {
        const result = await dialog.showSaveDialog({
            title: 'Export Database Archive',
            defaultPath: 'imageparser_archive.zip',
            filters: [{ name: 'ZIP Archive', extensions: ['zip'] }],
        });
        if (result.canceled) return { success: false, error: 'canceled' };
        outputPath = result.filePath;
    }

    return new Promise((resolve) => {
        const proc = spawnBackend('export', ['--output', outputPath], {},
            'backend/api_export.py');
        let output = '';
        let error = '';
        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());
        proc.on('close', (code) => {
            try {
                const result = JSON.parse(output.trim().split('\n').pop());
                resolve(result);
            } catch {
                resolve({ success: false, error: error || 'Export failed' });
            }
        });
        proc.on('error', (err) => resolve({ success: false, error: err.message }));
    });
});

// IPC Handler: Relink preview (dry-run)
ipcMain.handle('relink-preview', async (_, { packagePath, targetFolder }) => {
    return new Promise((resolve) => {
        const proc = spawnBackend('relink', ['--package', packagePath, '--folder', targetFolder, '--dry-run'], {},
            'backend/api_relink.py');
        let output = '';
        let error = '';
        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());
        proc.on('close', (code) => {
            try {
                const result = JSON.parse(output.trim().split('\n').pop());
                resolve(result);
            } catch {
                resolve({ success: false, error: error || 'Preview failed' });
            }
        });
        proc.on('error', (err) => resolve({ success: false, error: err.message }));
    });
});

// IPC Handler: Relink apply
ipcMain.handle('relink-apply', async (_, { packagePath, targetFolder, deleteMissing }) => {
    const relinkArgs = ['--package', packagePath, '--folder', targetFolder];
    if (deleteMissing) relinkArgs.push('--delete-missing');

    return new Promise((resolve) => {
        const proc = spawnBackend('relink', relinkArgs, {},
            'backend/api_relink.py');
        let output = '';
        let error = '';
        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());
        proc.on('close', (code) => {
            try {
                const result = JSON.parse(output.trim().split('\n').pop());
                resolve(result);
            } catch {
                resolve({ success: false, error: error || 'Relink failed' });
            }
        });
        proc.on('error', (err) => resolve({ success: false, error: err.message }));
    });
});

// ── mDNS Server Discovery (lazy-loaded, failure-safe) ─────────────────
let _mdnsBrowser = null;
function getMdnsBrowser() {
    if (_mdnsBrowser === null) {
        try {
            _mdnsBrowser = require('./mdns-browser.cjs');
        } catch {
            _mdnsBrowser = false; // mark as unavailable
        }
    }
    return _mdnsBrowser || null;
}

ipcMain.handle('mdns-start-browse', async (event) => {
    try {
        const mdns = getMdnsBrowser();
        if (!mdns || !mdns.isAvailable()) return { success: false, error: 'bonjour-service not available' };

        const win = BrowserWindow.fromWebContents(event.sender);
        mdns.startBrowsing((eventType, data) => {
            try {
                if (win && !win.isDestroyed()) {
                    win.webContents.send('mdns-server-event', { type: eventType, ...data });
                }
            } catch { /* window closed */ }
        });
        return { success: true };
    } catch (err) {
        console.warn('mDNS browse failed:', err.message);
        return { success: false, error: err.message };
    }
});

ipcMain.handle('mdns-stop-browse', async () => {
    try {
        const mdns = getMdnsBrowser();
        if (mdns) mdns.stopBrowsing();
    } catch { /* ignore */ }
    return { success: true };
});

ipcMain.handle('mdns-get-servers', async () => {
    try {
        const mdns = getMdnsBrowser();
        return mdns ? mdns.getDiscoveredServers() : [];
    } catch { return []; }
});

// IPC Handler: Sync folder — scan and compare disk vs DB
ipcMain.handle('sync-folder', async (_, { folderPath }) => {
    return new Promise((resolve) => {
        const proc = spawnBackend('sync', ['--folder', folderPath], {},
            'backend/api_sync.py');
        let output = '';
        let error = '';
        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());
        proc.on('close', (code) => {
            try {
                const result = JSON.parse(output.trim().split('\n').pop());
                resolve(result);
            } catch {
                resolve({ success: false, error: error || 'Sync scan failed' });
            }
        });
        proc.on('error', (err) => resolve({ success: false, error: err.message }));
    });
});

// IPC Handler: Sync apply moves — update paths for moved files
ipcMain.handle('sync-apply-moves', async (_, { moves }) => {
    // Pass first move's folder to get DB path, then apply all moves
    const folderPath = moves.length > 0 ? path.dirname(moves[0].new_path) : '.';

    return new Promise((resolve) => {
        const proc = spawnBackend('sync', ['--folder', folderPath, '--apply-moves'], {},
            'backend/api_sync.py');
        let output = '';
        let error = '';
        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());
        proc.on('close', (code) => {
            try {
                const lines = output.trim().split('\n');
                // Get the last JSON line (apply_moves result)
                const result = JSON.parse(lines[lines.length - 1]);
                resolve(result);
            } catch {
                resolve({ success: false, error: error || 'Apply moves failed' });
            }
        });
        proc.on('error', (err) => resolve({ success: false, error: err.message }));
    });
});

// IPC Handler: Sync delete missing — remove DB records for deleted files
ipcMain.handle('sync-delete-missing', async (_, { fileIds }) => {
    const idsStr = fileIds.join(',');

    return new Promise((resolve) => {
        const proc = spawnBackend('sync', ['--folder', '.', '--delete-missing', idsStr], {},
            'backend/api_sync.py');
        let output = '';
        let error = '';
        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());
        proc.on('close', (code) => {
            try {
                const result = JSON.parse(output.trim().split('\n').pop());
                resolve(result);
            } catch {
                resolve({ success: false, error: error || 'Delete missing failed' });
            }
        });
        proc.on('error', (err) => resolve({ success: false, error: err.message }));
    });
});

// IPC Handler: Fetch image from URL (bypasses CORS via Node.js)
ipcMain.handle('fetch-image-url', async (_, url) => {
    try {
        const response = await fetch(url, {
            headers: { 'User-Agent': 'ImageParser/1.0' },
            signal: AbortSignal.timeout(15000),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const contentType = response.headers.get('content-type') || '';
        if (!contentType.startsWith('image/')) {
            throw new Error('URL does not point to an image');
        }

        const buffer = Buffer.from(await response.arrayBuffer());
        const base64 = `data:${contentType};base64,${buffer.toString('base64')}`;
        return { success: true, data: base64 };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

// IPC Handler: Triaxis Search (Vector + FTS5 + Filters)
// Daemon spawns lazily on first search, auto-kills after idle timeout.
ipcMain.handle('search-vector', async (_, searchOptions) => {
    let inputData;
    if (typeof searchOptions === 'string') {
        inputData = { query: searchOptions, limit: 20, mode: 'triaxis' };
    } else {
        inputData = {
            query: searchOptions.query || '',
            query_image: searchOptions.queryImage || null,
            query_images: searchOptions.queryImages || null,
            image_search_mode: searchOptions.imageSearchMode || 'and',
            limit: searchOptions.limit || 20,
            mode: searchOptions.mode || 'triaxis',
            threshold: searchOptions.threshold ?? 0.0,
            filters: searchOptions.filters || null,
            query_file_id: searchOptions.queryFileId || null,
            use_codex: searchOptions.use_codex ?? true,
            effort: searchOptions.effort || 'low',
            file_ids: searchOptions.file_ids || null,
        };
    }

    return sendSearchRequest(inputData);
});

// IPC Handler: Database Stats (archived image count)
ipcMain.handle('get-db-stats', async () => {
    return new Promise((resolve) => {
        const proc = spawnBackend('stats', [], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        }, 'backend/api_stats.py');
        let output = '';
        proc.stdout.on('data', (d) => output += d.toString());
        proc.on('close', (code) => {
            if (code === 0) {
                try {
                    resolve(JSON.parse(output.trim()));
                } catch {
                    resolve({ success: false, total_files: 0 });
                }
            } else {
                resolve({ success: false, total_files: 0 });
            }
        });
        proc.on('error', () => resolve({ success: false, total_files: 0 }));
    });
});

// ── Job Queue IPC (server mode — direct DB, bypassing HTTP auth) ──

function spawnQueueCmd(cmd, data) {
    return new Promise((resolve) => {
        const proc = spawnBackend('queue', [cmd, JSON.stringify(data || {})], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        }, 'backend/api_queue.py');
        let output = '';
        let errOutput = '';
        proc.stdout.on('data', (d) => output += d.toString());
        proc.stderr.on('data', (d) => errOutput += d.toString());
        proc.on('close', (code) => {
            if (code === 0) {
                try {
                    resolve(JSON.parse(output.trim()));
                } catch {
                    // stdout may contain non-JSON lines (e.g. Python logging);
                    // try extracting the last JSON line
                    const lines = output.trim().split('\n');
                    for (let i = lines.length - 1; i >= 0; i--) {
                        const line = lines[i].trim();
                        if (line.startsWith('{')) {
                            try {
                                resolve(JSON.parse(line));
                                return;
                            } catch { /* continue searching */ }
                        }
                    }
                    writeLog('ERROR', `[QueueCmd:${cmd}] Failed to parse output: ${output.slice(0, 500)}`);
                    resolve({ success: false, error: 'Failed to parse output' });
                }
            } else {
                resolve({ success: false, error: errOutput || `Exit code ${code}` });
            }
        });
        proc.on('error', (e) => resolve({ success: false, error: e.message }));
    });
}

ipcMain.handle('queue-register-paths', async (_, { filePaths, priority }) => {
    return spawnQueueCmd('register-paths', { file_paths: filePaths, priority: parseInt(priority) || 0 });
});

ipcMain.handle('queue-scan-folder', async (_, { folderPath, priority }) => {
    const data = { folder_path: folderPath, priority: parseInt(priority) || 0 };
    // Resolve WebDAV credentials if needed
    if (folderPath.startsWith('webdav://')) {
        const { sourceId } = _parseWebDAVPath(folderPath);
        const userConfig = readYamlFile(userSettingsPath);
        const source = (userConfig.webdav_sources || []).find(s => s.id === sourceId);
        if (source) {
            data.webdav_configs = { [sourceId]: {
                url: source.url, username: source.username,
                password: _decryptWebdavPassword(source),
                remote_path: source.remote_path || '/', verify_ssl: source.verify_ssl !== false,
            }};
        }
    }
    return spawnQueueCmd('scan-folder', data);
});

ipcMain.handle('queue-scan-folders', async (_, { folderPaths, priority }) => {
    // Resolve WebDAV credentials for any webdav:// paths
    const webdavConfigs = {};
    for (const fp of folderPaths) {
        if (fp.startsWith('webdav://')) {
            const { sourceId } = _parseWebDAVPath(fp);
            if (!webdavConfigs[sourceId]) {
                const userConfig = readYamlFile(userSettingsPath);
                const source = (userConfig.webdav_sources || []).find(s => s.id === sourceId);
                if (source) {
                    webdavConfigs[sourceId] = {
                        url: source.url,
                        username: source.username,
                        password: _decryptWebdavPassword(source),
                        remote_path: source.remote_path || '/',
                        verify_ssl: source.verify_ssl !== false,
                    };
                }
            }
        }
    }
    return spawnQueueCmd('scan-folders', {
        folder_paths: folderPaths,
        priority: parseInt(priority) || 0,
        webdav_configs: Object.keys(webdavConfigs).length > 0 ? webdavConfigs : undefined,
    });
});

ipcMain.handle('queue-stats', async () => {
    return spawnQueueCmd('stats');
});

ipcMain.handle('queue-list-jobs', async (_, { status, limit, offset }) => {
    return spawnQueueCmd('list-jobs', { status: status || null, limit: limit || 50, offset: offset || 0 });
});

ipcMain.handle('queue-cancel-job', async (_, { jobId }) => {
    return spawnQueueCmd('cancel-job', { job_id: jobId });
});

ipcMain.handle('queue-retry-failed', async () => {
    return spawnQueueCmd('retry-failed');
});

ipcMain.handle('queue-clear-completed', async () => {
    return spawnQueueCmd('clear-completed');
});

// Work Request IPC handlers
ipcMain.handle('queue-list-work-requests', async (_, { includeCompleted }) => {
    return spawnQueueCmd('list-work-requests', { include_completed: !!includeCompleted });
});

ipcMain.handle('queue-work-request-detail', async (_, { wrId }) => {
    return spawnQueueCmd('work-request-detail', { wr_id: wrId });
});

ipcMain.handle('queue-pause-wr', async (_, { wrId }) => {
    return spawnQueueCmd('pause-wr', { wr_id: wrId });
});

ipcMain.handle('queue-resume-wr', async (_, { wrId }) => {
    return spawnQueueCmd('resume-wr', { wr_id: wrId });
});

ipcMain.handle('queue-cancel-wr', async (_, { wrId }) => {
    return spawnQueueCmd('cancel-wr', { wr_id: wrId });
});

// IPC Handler: Incomplete Stats (for resume dialog on startup)
ipcMain.handle('get-incomplete-stats', async () => {
    return new Promise((resolve) => {
        const proc = spawnBackend('incomplete-stats', [], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        }, 'backend/api_incomplete_stats.py');
        let output = '';
        proc.stdout.on('data', (d) => output += d.toString());
        proc.on('close', (code) => {
            if (code === 0) {
                try {
                    resolve(JSON.parse(output.trim()));
                } catch {
                    resolve({ success: false, total_incomplete: 0, folders: [] });
                }
            } else {
                resolve({ success: false, total_incomplete: 0, folders: [] });
            }
        });
        proc.on('error', () => resolve({ success: false, total_incomplete: 0, folders: [] }));
    });
});

// IPC Handler: Folder Phase Stats (MC/VV/MV per folder)
ipcMain.handle('get-folder-phase-stats', async (_, storageRoot) => {
    return new Promise((resolve) => {
        const proc = spawnBackend('folder-stats', [storageRoot], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        }, 'backend/api_folder_stats.py');
        let output = '';
        proc.stdout.on('data', (d) => output += d.toString());
        proc.on('close', (code) => {
            if (code === 0) {
                try {
                    resolve(JSON.parse(output.trim()));
                } catch {
                    resolve({ success: false, folders: [] });
                }
            } else {
                resolve({ success: false, folders: [] });
            }
        });
        proc.on('error', () => resolve({ success: false, folders: [] }));
    });
});

// ── Archive Browse IPC ──────────────────────────────────────

function spawnArchiveCmd(cmd, data) {
    return new Promise((resolve) => {
        const args = data ? [cmd, JSON.stringify(data)] : [cmd];
        const proc = spawnBackend('archive', args, {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        }, 'backend/api_archive.py');
        let output = '';
        let errOutput = '';
        proc.stdout.on('data', (d) => output += d.toString());
        proc.stderr.on('data', (d) => errOutput += d.toString());
        proc.on('close', (code) => {
            if (code === 0) {
                try {
                    resolve(JSON.parse(output.trim()));
                } catch {
                    // Try extracting last JSON line from mixed output
                    const lines = output.trim().split('\n');
                    for (let i = lines.length - 1; i >= 0; i--) {
                        const line = lines[i].trim();
                        if (line.startsWith('{')) {
                            try {
                                resolve(JSON.parse(line));
                                return;
                            } catch { /* continue */ }
                        }
                    }
                    writeLog('ERROR', `[ArchiveCmd:${cmd}] Failed to parse output: ${output.slice(0, 500)}`);
                    resolve({ success: false, error: 'Failed to parse output' });
                }
            } else {
                resolve({ success: false, error: errOutput || `Exit code ${code}` });
            }
        });
        proc.on('error', (e) => resolve({ success: false, error: e.message }));
    });
}

ipcMain.handle('archive-get-folders', async () => {
    return spawnArchiveCmd('folders');
});

ipcMain.handle('archive-get-files', async (_, params) => {
    return spawnArchiveCmd('files', params || {});
});

ipcMain.handle('archive-get-image-types', async () => {
    return spawnArchiveCmd('image-types');
});

// IPC Handler: Environment Check
ipcMain.handle('check-env', async () => {
    return new Promise((resolve) => {
        const proc = spawnBackend('installer', ['--check'], {},
            'backend/setup/installer.py');
        let output = '';
        proc.stdout.on('data', (d) => output += d.toString());
        proc.on('close', () => {
            try {
                resolve(JSON.parse(output.trim()));
            } catch {
                resolve({ dependencies_ok: false, error: "Parse Error" });
            }
        });
        proc.on('error', () => resolve({ dependencies_ok: false, error: "Spawn Error" }));
    });
});

// IPC Handler: Install Environment
ipcMain.on('install-env', (event) => {
    event.reply('install-log', { message: '🚀 Starting installation...', type: 'info' });

    // Packaged mode: pip install unnecessary (all deps bundled), download models only
    // Dev mode: pip install + model download
    const cliPath = getBackendCliPath();
    const installerArgs = cliPath
        ? ['--download-model']
        : ['--install', '--download-model'];
    const proc = spawnBackend('installer', installerArgs, {},
        'backend/setup/installer.py');

    proc.stdout.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) event.reply('install-log', { message: msg, type: 'info' });
    });
    proc.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) event.reply('install-log', { message: msg, type: 'warning' });
    });
    proc.on('close', (code) => {
        event.reply('install-log', {
            message: code === 0 ? '✅ Installation Complete!' : '❌ Installation Failed',
            type: code === 0 ? 'success' : 'error',
            done: true
        });
    });
});

// IPC Handler: Update User Metadata
ipcMain.handle('metadata:updateUserData', async (event, filePath, updates) => {
    return new Promise((resolve, reject) => {
        const proc = spawnBackend('metadata-update', [], {},
            'backend/api_metadata_update.py');

        let output = '';
        let errorOutput = '';

        proc.stdout.on('data', (data) => {
            output += data.toString();
        });

        proc.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });

        proc.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (err) {
                    reject(new Error('Failed to parse response: ' + err.message));
                }
            } else {
                reject(new Error('Python error: ' + errorOutput));
            }
        });

        proc.on('error', (err) => {
            reject(new Error('Failed to spawn Python process: ' + err.message));
        });

        const inputData = JSON.stringify({ file_path: filePath, updates });
        proc.stdin.write(inputData);
        proc.stdin.end();
    });
});

// IPC Handler: Run Python Pipeline (global — registered once)
// Guard: only one pipeline at a time
let activePipelineProc = null;
let activeBackfillProc = null;
let pipelineStoppedByUser = false;

ipcMain.on('run-pipeline', (event, { filePaths }) => {
    console.log('[run-pipeline] Received request:', filePaths.length, 'files');
    if (activePipelineProc) {
        console.log('[run-pipeline] BLOCKED: pipeline already running (pid:', activePipelineProc.pid, ')');
        event.reply('pipeline-log', { message: 'Pipeline already running. Wait for it to finish.', type: 'error' });
        return;
    }

    console.log('[run-pipeline] Starting pipeline...');
    event.reply('pipeline-log', { message: `Starting batch processing: ${filePaths.length} files...`, type: 'info' });

    let processedCount = 0;
    let skippedCount = 0;
    let batchDoneSent = false;
    const totalFiles = filePaths.length;
    pipelineStoppedByUser = false;

    // Cumulative phase tracking — each phase has independent progress
    let cumParse = 0, cumMC = 0, cumVV = 0, cumMV = 0;
    // Current active phase within mini-batch (for sub-progress tracking)
    let activePhase = 0; // 0=parse, 1=MC, 2=VV, 3=MV
    let phaseSubCount = 0, phaseSubTotal = 0;

    let batchInfo = '';

    function emitPhaseProgress(extraFields = {}) {
        if (extraFields.batchInfo !== undefined) batchInfo = extraFields.batchInfo;
        event.reply('pipeline-progress', {
            processed: processedCount,
            total: totalFiles,
            skipped: skippedCount,
            currentFile: extraFields.currentFile || '',
            // Cumulative per-phase counts
            cumParse, cumMC, cumVV, cumMV,
            // Active phase sub-progress (within mini-batch)
            activePhase,
            phaseSubCount, phaseSubTotal,
            // Current batch info (e.g. "8" or "8:VV")
            batchInfo,
        });
    }

    const proc = spawnBackend('pipeline', ['--files', JSON.stringify(filePaths)], {
        detached: true,  // Own process group for clean tree kill
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        stdio: ['pipe', 'pipe', 'pipe'],  // stdin pipe = watchdog lifeline
    }, 'backend/pipeline/ingest_engine.py');
    proc.stdin.on('error', () => {}); // Suppress EPIPE on process exit
    console.log('[run-pipeline] Spawned PID:', proc.pid);
    activePipelineProc = proc;

    proc.stdout.on('data', (data) => {
        const raw = data.toString().trim();
        if (!raw) return;

        // Handle multi-line output (batch mode emits multiple lines at once)
        const lines = raw.split('\n');
        for (const line of lines) {
            const message = line.trim();
            if (!message) continue;

            // Strip logger prefix for pattern matching
            const clean = message.replace(/^\d{4}-\d{2}-\d{2}\s[\d:,.]+ - [\w.]+ - \w+ - /, '');

            const processingMatch = clean.match(/^Processing: (.+)/);
            const stepMatch = clean.match(/^STEP (\d+)\/(\d+) (.+)/);
            const stepDoneMatch = clean.match(/^STEP (\d+)\/(\d+) completed/);
            // Phase sub-progress: [1/26] filename → type (may have leading whitespace from logger indent)
            const subProgressMatch = clean.match(/^\s*\[(\d+)\/(\d+)\]\s+(.+?)(?:\s+→|$)/);
            // Cumulative phase progress: [PHASE] P:40 MC:33 VV:30 MV:30 T:500 B:8
            const phaseMatch = clean.match(/^\[PHASE\]\s+P:(\d+)\s+MC:(\d+)\s+VV:(\d+)\s+MV:(\d+)\s+T:(\d+)(?:\s+B:(\S+))?/);

            // [PHASE] cumulative progress
            if (phaseMatch) {
                cumParse = parseInt(phaseMatch[1]);
                cumMC = parseInt(phaseMatch[2]);
                cumVV = parseInt(phaseMatch[3]);
                cumMV = parseInt(phaseMatch[4]);
                const batchInfo = phaseMatch[6] || '';
                emitPhaseProgress({ batchInfo });
            }

            // STEP x/y completed → phase finished within mini-batch
            if (stepDoneMatch) {
                activePhase = parseInt(stepDoneMatch[1]); // advance to next
                phaseSubCount = phaseSubTotal; // mark 100%
                emitPhaseProgress();
                phaseSubCount = 0;
                phaseSubTotal = 0;
            } else if (stepMatch) {
                // STEP x/y Name → phase started within mini-batch
                activePhase = parseInt(stepMatch[1]) - 1;
                phaseSubCount = 0;
                const countMatch = stepMatch[3].match(/\((\d+)/);
                phaseSubTotal = countMatch ? parseInt(countMatch[1]) : totalFiles;
                emitPhaseProgress();
            }

            // Per-file sub-progress within a phase: [3/26] file.psd → type
            if (subProgressMatch && !phaseMatch) {
                phaseSubCount = parseInt(subProgressMatch[1]);
                phaseSubTotal = parseInt(subProgressMatch[2]);
                const fileName = subProgressMatch[3].split(/\s+→/)[0].trim();
                emitPhaseProgress({ currentFile: fileName });
            }

            if (processingMatch) {
                emitPhaseProgress({ currentFile: path.basename(processingMatch[1]) });
            }

            // [OK] = file stored (Phase 4) or single-file parse success
            if (/\[OK\]/.test(clean)) {
                processedCount++;
                phaseSubCount = processedCount;
                emitPhaseProgress();
                event.reply('pipeline-file-done', {
                    processed: processedCount,
                    skipped: skippedCount
                });
            }

            // [SKIP] = smart skip (unchanged file)
            if (/\[SKIP\]/.test(clean) && !/files skipped/.test(clean)) {
                skippedCount++;
                emitPhaseProgress();
            }

            // [DONE] = batch complete
            if (/\[DONE\]/.test(clean) && !batchDoneSent) {
                batchDoneSent = true;
                event.reply('pipeline-batch-done', {
                    processed: processedCount,
                    skipped: skippedCount,
                    total: totalFiles
                });
            }

            // Log: show STEP, progress, errors, adaptive decisions (exclude noisy [PHASE])
            const isLogWorthy = /Processing:|STEP \d|\[OK\]|\[FAIL\]|\[DONE\]|\[SKIP\]|\[REBUILD\]|\[BATCH\]|\[REGEN\]|\[FALLBACK\]|\[MINI\s|\[TIER|\[ADAPTIVE:|\[\d+\/\d+\]/.test(clean) && !/^\[PHASE\]/.test(clean);
            if (isLogWorthy) {
                event.reply('pipeline-log', { message: clean, type: 'info' });
            }
        }
    });

    proc.stderr.on('data', (data) => {
        for (const line of data.toString().split('\n')) {
            const msg = line.trim();
            if (!msg) continue;
            // Try JSON structured log first (from json_log_formatter)
            try {
                const parsed = JSON.parse(msg);
                if (parsed.level) {
                    const type = parsed.level === 'ERROR' || parsed.level === 'CRITICAL' ? 'error'
                        : parsed.level === 'WARNING' ? 'warning' : 'info';
                    if (type === 'error' || type === 'warning') {
                        event.reply('pipeline-log', { message: parsed.message, type });
                    }
                    continue;
                }
            } catch { /* not JSON — fallback to regex */ }
            // Fallback: regex for non-Python output (e.g. native libraries)
            if (/\bERROR\b|Traceback|Exception:/i.test(msg)) {
                event.reply('pipeline-log', { message: msg, type: 'error' });
            }
        }
    });

    proc.on('close', (code) => {
        console.log('[run-pipeline] Process closed, code:', code, 'processed:', processedCount, 'skipped:', skippedCount, 'batchDoneSent:', batchDoneSent);
        activePipelineProc = null;
        const wasStopped = pipelineStoppedByUser;
        pipelineStoppedByUser = false;

        event.reply('pipeline-progress', {
            processed: processedCount,
            total: totalFiles,
            currentFile: '',
            skipped: skippedCount
        });

        event.reply('pipeline-log', {
            message: wasStopped
                ? 'Pipeline stopped by user.'
                : code === 0
                    ? `Pipeline complete! (${processedCount} processed, ${skippedCount} skipped)`
                    : `Pipeline exited with code ${code}`,
            type: wasStopped ? 'warning' : code === 0 ? 'success' : 'error'
        });

        if (!batchDoneSent) {
            batchDoneSent = true;
            event.reply('pipeline-batch-done', {
                success: code === 0 && !wasStopped,
                processed: processedCount,
                skipped: skippedCount,
                total: totalFiles
            });
        }
    });

    proc.on('error', (err) => {
        activePipelineProc = null;
        event.reply('pipeline-log', { message: `Pipeline error: ${err.message}`, type: 'error' });
    });
});

// IPC Handler: Stop running pipeline (kill entire process tree to avoid residual processes)
ipcMain.on('stop-pipeline', () => {
    if (activePipelineProc) {
        pipelineStoppedByUser = true;
        killProcessTree(activePipelineProc);
        // Don't send events here — proc.on('close') handles cleanup
    }
});

// IPC Handler: Run discover (DFS folder scan) (global — registered once)
// Guard: only one discover process at a time per folder
let activeDiscoverProcs = new Map(); // folderPath → proc

ipcMain.on('run-discover', (event, { folderPath, noSkip }) => {
    // Prevent duplicate discover for the same folder
    if (activeDiscoverProcs.has(folderPath)) {
        event.reply('discover-log', { message: `Already scanning: ${folderPath}`, type: 'info' });
        return;
    }

    const discoverArgs = ['--discover', folderPath];
    if (noSkip) discoverArgs.push('--no-skip');

    event.reply('discover-log', { message: `Scanning folder: ${folderPath}`, type: 'info' });

    let processedCount = 0;
    let skippedCount = 0;
    let totalFiles = 0;

    // Phase tracking (same as pipeline handler)
    let cumParse = 0, cumMC = 0, cumVV = 0, cumMV = 0;
    let activePhase = 0;
    let phaseSubCount = 0, phaseSubTotal = 0;
    let batchInfo = '';

    function emitDiscoverProgress(extraFields = {}) {
        if (extraFields.batchInfo !== undefined) batchInfo = extraFields.batchInfo;
        event.reply('discover-progress', {
            processed: processedCount,
            total: totalFiles,
            skipped: skippedCount,
            currentFile: extraFields.currentFile || '',
            cumParse, cumMC, cumVV, cumMV,
            activePhase, phaseSubCount, phaseSubTotal,
            batchInfo,
            folderPath
        });
    }

    const proc = spawnBackend('pipeline', discoverArgs, {
        detached: true,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        stdio: ['pipe', 'pipe', 'pipe'],  // stdin pipe = watchdog lifeline
    }, 'backend/pipeline/ingest_engine.py');
    proc.stdin.on('error', () => {}); // Suppress EPIPE on process exit
    activeDiscoverProcs.set(folderPath, proc);

    // Immediate feedback while Python loads modules (~15-30s)
    event.reply('discover-log', { message: 'Loading pipeline...', type: 'info' });

    proc.stdout.on('data', (data) => {
        const raw = data.toString();
        if (!raw.trim()) return;
        const lines = raw.split('\n').filter(l => l.trim());
        for (const line of lines) {
            const message = line.trim();
            if (!message) continue;

            const clean = message.replace(/^\d{4}-\d{2}-\d{2}\s[\d:,.]+ - [\w.]+ - \w+ - /, '');

            // Extract total file count
            const discoverMatch = clean.match(/\[DISCOVER\] Found (\d+)/);
            if (discoverMatch) totalFiles = parseInt(discoverMatch[1]);

            // [PHASE] P:40 MC:33 VV:30 MV:30 T:500 B:8
            const phaseMatch = clean.match(/^\[PHASE\]\s+P:(\d+)\s+MC:(\d+)\s+VV:(\d+)\s+MV:(\d+)\s+T:(\d+)(?:\s+B:(\S+))?/);
            if (phaseMatch) {
                cumParse = parseInt(phaseMatch[1]);
                cumMC = parseInt(phaseMatch[2]);
                cumVV = parseInt(phaseMatch[3]);
                cumMV = parseInt(phaseMatch[4]);
                emitDiscoverProgress({ batchInfo: phaseMatch[6] || '' });
            }

            const stepDoneMatch = clean.match(/^STEP (\d+)\/(\d+) completed/);
            const stepMatch = clean.match(/^STEP (\d+)\/(\d+) (.+)/);
            const subProgressMatch = clean.match(/^\s*\[(\d+)\/(\d+)\]\s+(.+?)(?:\s+→|$)/);
            const processingMatch = clean.match(/^Processing: (.+)/);

            if (stepDoneMatch) {
                activePhase = parseInt(stepDoneMatch[1]);
                phaseSubCount = phaseSubTotal;
                emitDiscoverProgress();
                phaseSubCount = 0;
                phaseSubTotal = 0;
            } else if (stepMatch) {
                activePhase = parseInt(stepMatch[1]) - 1;
                phaseSubCount = 0;
                const countMatch = stepMatch[3].match(/\((\d+)/);
                phaseSubTotal = countMatch ? parseInt(countMatch[1]) : totalFiles;
                emitDiscoverProgress();
            }

            if (subProgressMatch && !phaseMatch) {
                phaseSubCount = parseInt(subProgressMatch[1]);
                phaseSubTotal = parseInt(subProgressMatch[2]);
                const fileName = subProgressMatch[3].split(/\s+→/)[0].trim();
                emitDiscoverProgress({ currentFile: fileName });
            }

            if (processingMatch) {
                emitDiscoverProgress({ currentFile: path.basename(processingMatch[1]) });
            }

            if (/\[OK\]/.test(clean)) {
                processedCount++;
                emitDiscoverProgress();
            }
            if (/\[SKIP\]/.test(clean) && !/files skipped/.test(clean)) {
                skippedCount++;
                emitDiscoverProgress();
            }

            // Log key events (including per-file progress and adaptive batch decisions)
            const isLogWorthy = /Processing:|STEP \d|\[OK\]|\[FAIL\]|\[DONE\]|\[DISCOVER\]|\[SKIP\]|\[REBUILD\]|\[BATCH\]|\[REGEN\]|\[FALLBACK\]|\[TIER|\[ADAPTIVE:|\[\d+\/\d+\]/.test(clean) && !/^\[PHASE\]/.test(clean);
            if (isLogWorthy) {
                event.reply('discover-log', { message: clean, type: 'info' });
            }
        }
    });

    proc.stderr.on('data', (data) => {
        const raw = data.toString();
        if (!raw.trim()) return;
        for (const line of raw.split('\n')) {
            const msg = line.trim();
            if (!msg) continue;
            // Try JSON structured log first
            try {
                const parsed = JSON.parse(msg);
                if (parsed.level) {
                    const type = parsed.level === 'ERROR' || parsed.level === 'CRITICAL' ? 'error'
                        : parsed.level === 'WARNING' ? 'warning' : 'info';
                    if (type === 'error' || type === 'warning') {
                        event.reply('discover-log', { message: parsed.message, type });
                    }
                    continue;
                }
            } catch { /* not JSON */ }
            // Fallback: regex for non-Python output
            if (/\bERROR\b|Traceback|Exception:/i.test(msg)) {
                event.reply('discover-log', { message: msg, type: 'error' });
            }
        }
    });

    proc.on('close', (code) => {
        activeDiscoverProcs.delete(folderPath);
        event.reply('discover-log', {
            message: code === 0
                ? `Scan complete: ${folderPath} (${processedCount} files)`
                : `Scan failed: ${folderPath} (code ${code})`,
            type: code === 0 ? 'success' : 'error'
        });
        event.reply('discover-file-done', {
            success: code === 0,
            folderPath,
            processedCount
        });
    });

    proc.on('error', (err) => {
        activeDiscoverProcs.delete(folderPath);
        event.reply('discover-log', { message: `Discover error: ${err.message}`, type: 'error' });
        event.reply('discover-file-done', { success: false, folderPath, processedCount: 0 });
    });
});

// IPC Handler: List available classification domains
ipcMain.handle('list-domains', async () => {
    try {
        const domainsDir = path.join(configRoot, 'backend', 'vision', 'domains');
        if (!fs.existsSync(domainsDir)) return [];
        const files = fs.readdirSync(domainsDir)
            .filter(f => f.endsWith('.yaml') && !f.startsWith('_'));
        return files.map(f => {
            const data = readYamlFile(path.join(domainsDir, f));
            const meta = data.domain || {};
            return {
                id: meta.id || path.basename(f, '.yaml'),
                name: meta.name || path.basename(f, '.yaml'),
                name_ko: meta.name_ko || '',
                description: meta.description || '',
                image_types: data.image_types || [],
                image_types_count: (data.image_types || []).length,
            };
        });
    } catch (err) {
        console.error('[List Domains Error]', err);
        return [];
    }
});

// IPC Handler: Get classification domain detail (merged with _base.yaml)
ipcMain.handle('get-domain-detail', async (_, domainId) => {
    try {
        const domainsDir = path.join(configRoot, 'backend', 'vision', 'domains');
        const baseData = readYamlFile(path.join(domainsDir, '_base.yaml'));
        const data = readYamlFile(path.join(domainsDir, `${domainId}.yaml`));
        const meta = data.domain || {};
        return {
            id: meta.id || domainId,
            name: meta.name || domainId,
            name_ko: meta.name_ko || '',
            description: meta.description || '',
            image_types: data.image_types || [],
            type_hints: data.type_hints || {},
            type_instructions: data.type_instructions || {},
            common_hints: baseData.common_hints || {},
        };
    } catch (err) {
        console.error('[Get Domain Detail Error]', err);
        return null;
    }
});

// IPC Handler: Save new domain YAML file
ipcMain.handle('save-domain-yaml', async (_, domainId, yamlContent) => {
    try {
        const yaml = require('js-yaml');

        // 1. Validate domainId format
        if (!/^[a-z][a-z0-9_]*$/.test(domainId)) {
            return { success: false, error: 'Invalid domain ID: must be lowercase snake_case' };
        }

        // 2. Check if domain already exists
        const domainsDir = path.join(configRoot, 'backend', 'vision', 'domains');
        const targetPath = path.join(domainsDir, `${domainId}.yaml`);
        if (fs.existsSync(targetPath)) {
            return { success: false, error: `Domain '${domainId}' already exists` };
        }

        // 3. Parse YAML to validate structure
        let parsed;
        try {
            parsed = yaml.load(yamlContent);
        } catch (parseErr) {
            return { success: false, error: `YAML parse error: ${parseErr.message}` };
        }

        // 4. Structural validation
        if (!parsed || typeof parsed !== 'object') {
            return { success: false, error: 'YAML must be a valid object' };
        }
        if (!parsed.domain || !parsed.domain.id) {
            return { success: false, error: 'Missing required field: domain.id' };
        }
        if (parsed.domain.id !== domainId) {
            return { success: false, error: `domain.id mismatch: expected '${domainId}', got '${parsed.domain.id}'` };
        }
        if (!Array.isArray(parsed.image_types) || parsed.image_types.length === 0) {
            return { success: false, error: 'image_types must be a non-empty array' };
        }
        if (!parsed.type_hints || typeof parsed.type_hints !== 'object') {
            return { success: false, error: 'type_hints must be an object' };
        }

        // 5. Ensure domains directory exists
        if (!fs.existsSync(domainsDir)) {
            fs.mkdirSync(domainsDir, { recursive: true });
        }

        // 6. Write file with consistent formatting
        const formattedYaml = yaml.dump(parsed, {
            lineWidth: -1,
            noRefs: true,
            sortKeys: false,
        });
        fs.writeFileSync(targetPath, formattedYaml, 'utf8');

        return { success: true };
    } catch (err) {
        console.error('[Save Domain YAML Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Get config (system config.yaml merged with user-settings.yaml)
ipcMain.handle('get-config', async () => {
    try {
        const systemConfig = readYamlFile(path.join(configRoot, 'config.yaml'));
        const userConfig = readYamlFile(userSettingsPath);
        const config = deepMerge(systemConfig, userConfig);

        return { success: true, config };
    } catch (err) {
        console.error('[Get Config Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Get registered folders from user-settings.yaml (fallback: config.yaml)
ipcMain.handle('get-registered-folders', async () => {
    try {
        const userConfig = readYamlFile(userSettingsPath);
        const systemConfig = readYamlFile(path.join(configRoot, 'config.yaml'));
        const regFolders = userConfig.registered_folders || systemConfig.registered_folders || { folders: [], auto_scan: true };
        const folders = (regFolders.folders || []).map(fp => {
            const isWebDAV = fp.startsWith('webdav://');
            return {
                path: fp,
                exists: isWebDAV ? true : fs.existsSync(fp),
                isWebDAV,
            };
        });
        return { success: true, folders, autoScan: regFolders.auto_scan !== false };
    } catch (err) {
        console.error('[Get Registered Folders Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Add registered folders (opens multi-select dialog) — writes to user-settings.yaml
ipcMain.handle('add-registered-folder', async () => {
    try {
        const result = await dialog.showOpenDialog({
            properties: ['openDirectory', 'multiSelections'],
            title: 'Select Folders to Register'
        });
        if (result.canceled || result.filePaths.length === 0) {
            return { success: true, added: [] };
        }

        const userConfig = readYamlFile(userSettingsPath);
        if (!userConfig.registered_folders) userConfig.registered_folders = { folders: [], auto_scan: true };
        if (!userConfig.registered_folders.folders) userConfig.registered_folders.folders = [];

        const existing = new Set(userConfig.registered_folders.folders);
        const added = result.filePaths.filter(fp => !existing.has(fp));
        userConfig.registered_folders.folders.push(...added);

        writeYamlFile(userSettingsPath, userConfig);

        const folders = userConfig.registered_folders.folders.map(fp => ({
            path: fp,
            exists: fs.existsSync(fp),
        }));
        return { success: true, added, folders };
    } catch (err) {
        console.error('[Add Registered Folder Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Remove a registered folder — writes to user-settings.yaml
ipcMain.handle('remove-registered-folder', async (_, folderPath) => {
    try {
        const userConfig = readYamlFile(userSettingsPath);
        if (!userConfig.registered_folders || !userConfig.registered_folders.folders) {
            return { success: true, folders: [] };
        }
        userConfig.registered_folders.folders = userConfig.registered_folders.folders.filter(fp => fp !== folderPath);
        writeYamlFile(userSettingsPath, userConfig);

        const folders = userConfig.registered_folders.folders.map(fp => ({
            path: fp,
            exists: fs.existsSync(fp),
        }));
        return { success: true, folders };
    } catch (err) {
        console.error('[Remove Registered Folder Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Add a WebDAV folder path to registered folders (no OS dialog)
ipcMain.handle('add-registered-webdav-folder', async (_, webdavPath) => {
    try {
        if (!webdavPath || !webdavPath.startsWith('webdav://')) {
            return { success: false, error: 'Invalid WebDAV path' };
        }

        const userConfig = readYamlFile(userSettingsPath);
        if (!userConfig.registered_folders) userConfig.registered_folders = { folders: [], auto_scan: true };
        if (!userConfig.registered_folders.folders) userConfig.registered_folders.folders = [];

        const existing = new Set(userConfig.registered_folders.folders);
        if (existing.has(webdavPath)) {
            // Already registered — return current list without adding
            const folders = userConfig.registered_folders.folders.map(fp => {
                const isWebDAV = fp.startsWith('webdav://');
                return { path: fp, exists: isWebDAV ? true : fs.existsSync(fp), isWebDAV };
            });
            return { success: true, added: [], folders };
        }

        userConfig.registered_folders.folders.push(webdavPath);
        writeYamlFile(userSettingsPath, userConfig);

        const folders = userConfig.registered_folders.folders.map(fp => {
            const isWebDAV = fp.startsWith('webdav://');
            return { path: fp, exists: isWebDAV ? true : fs.existsSync(fp), isWebDAV };
        });
        return { success: true, added: [webdavPath], folders };
    } catch (err) {
        console.error('[Add WebDAV Folder Error]', err);
        return { success: false, error: err.message };
    }
});

// ── WebDAV Source Management ─────────────────────────────────────

const activeWebdavSyncs = new Map(); // sourceId → child process

ipcMain.handle('get-webdav-sources', async () => {
    try {
        const userConfig = readYamlFile(userSettingsPath);
        const sources = (userConfig.webdav_sources || []).map(s => ({
            id: s.id,
            name: s.name,
            url: s.url,
            remote_path: s.remote_path,
            verify_ssl: s.verify_ssl !== false,
            last_sync: s.last_sync || null,
            // password is NOT returned for security
        }));
        return { success: true, sources };
    } catch (err) {
        console.error('[Get WebDAV Sources Error]', err);
        return { success: false, error: err.message };
    }
});

ipcMain.handle('add-webdav-source', async (_, sourceConfig) => {
    try {
        const { safeStorage } = require('electron');
        const { randomUUID } = require('crypto');

        const userConfig = readYamlFile(userSettingsPath);
        if (!userConfig.webdav_sources) userConfig.webdav_sources = [];

        // Encrypt password
        let encryptedPassword = sourceConfig.password;
        if (safeStorage.isEncryptionAvailable()) {
            encryptedPassword = safeStorage.encryptString(sourceConfig.password).toString('base64');
        }

        const source = {
            id: sourceConfig.id || randomUUID().slice(0, 8),
            name: sourceConfig.name || 'WebDAV',
            url: sourceConfig.url,
            remote_path: sourceConfig.remote_path || '/',
            username: sourceConfig.username,
            encrypted_password: encryptedPassword,
            password_encrypted: safeStorage.isEncryptionAvailable(),
            verify_ssl: sourceConfig.verify_ssl !== false,
            last_sync: null,
        };

        userConfig.webdav_sources.push(source);
        writeYamlFile(userSettingsPath, userConfig);

        return { success: true, source: { ...source, encrypted_password: undefined } };
    } catch (err) {
        console.error('[Add WebDAV Source Error]', err);
        return { success: false, error: err.message };
    }
});

ipcMain.handle('remove-webdav-source', async (_, sourceId) => {
    try {
        const userConfig = readYamlFile(userSettingsPath);
        if (!userConfig.webdav_sources) return { success: true };

        userConfig.webdav_sources = userConfig.webdav_sources.filter(s => s.id !== sourceId);
        writeYamlFile(userSettingsPath, userConfig);

        // Clean up cache directory
        const homedir = require('os').homedir();
        const cachePath = path.join(homedir, '.imagine-cache', 'webdav', sourceId);
        if (fs.existsSync(cachePath)) {
            fs.rmSync(cachePath, { recursive: true, force: true });
        }

        return { success: true };
    } catch (err) {
        console.error('[Remove WebDAV Source Error]', err);
        return { success: false, error: err.message };
    }
});

ipcMain.handle('test-webdav-connection', async (_, config) => {
    return new Promise((resolve) => {
        const configJson = JSON.stringify(config);
        const proc = spawnBackend('remote.sync_cli', ['--test', configJson], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
            stdio: ['pipe', 'pipe', 'pipe'],
        }, 'backend/remote/sync_cli.py', ['--test', configJson]);

        let output = '';
        proc.stdout.on('data', (d) => output += d.toString());
        proc.stderr.on('data', (d) => console.error('[WebDAV Test]', d.toString()));
        proc.on('close', () => {
            try {
                resolve(JSON.parse(output.trim()));
            } catch {
                resolve({ success: false, message: 'Failed to parse response' });
            }
        });
        proc.on('error', (err) => {
            resolve({ success: false, message: err.message });
        });
    });
});

function _decryptWebdavPassword(source) {
    if (!source.encrypted_password) return '';
    if (!source.password_encrypted) return source.encrypted_password;
    try {
        const { safeStorage } = require('electron');
        const buf = Buffer.from(source.encrypted_password, 'base64');
        return safeStorage.decryptString(buf);
    } catch {
        return source.encrypted_password;
    }
}

// WebDAV Process: Run pipeline on remote files via FileContainer (fetch-and-process)
// Accepts either sourceId (string) or { sourceId, folderPath } (object with webdav:// path)
ipcMain.on('process-webdav-source', (event, arg) => {
    let sourceId, remotePath;

    if (typeof arg === 'string' && arg.startsWith('webdav://')) {
        // webdav://source-id/sub/path format
        const parsed = _parseWebDAVPath(arg);
        sourceId = parsed.sourceId;
        remotePath = parsed.subPath;
    } else if (typeof arg === 'object' && arg.sourceId) {
        sourceId = arg.sourceId;
        remotePath = arg.folderPath || null;
    } else {
        sourceId = arg;
        remotePath = null;
    }

    if (activeWebdavSyncs.has(sourceId)) {
        event.reply('webdav-sync-progress', { event: 'error', message: 'Processing already in progress' });
        return;
    }

    const userConfig = readYamlFile(userSettingsPath);
    const source = (userConfig.webdav_sources || []).find(s => s.id === sourceId);
    if (!source) {
        event.reply('webdav-sync-progress', { event: 'error', message: 'Source not found' });
        return;
    }

    const webdavConfig = JSON.stringify({
        id: source.id,
        url: source.url,
        username: source.username,
        password: _decryptWebdavPassword(source),
        remote_path: remotePath || source.remote_path || '/',
        verify_ssl: source.verify_ssl !== false,
    });

    // Use ingest_engine --webdav (FileContainer + WebDAVSupplier)
    const proc = spawnBackend('pipeline.ingest_engine', ['--webdav', webdavConfig], {
        env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath },
        stdio: ['pipe', 'pipe', 'pipe'],
    }, 'backend/pipeline/ingest_engine.py', ['--webdav', webdavConfig]);

    activeWebdavSyncs.set(sourceId, proc);

    // Reuse existing discover progress parsing (same stdout format)
    let processedCount = 0;
    let totalFiles = 0;
    let cumParse = 0, cumMC = 0, cumVV = 0, cumMV = 0;

    proc.stdout.on('data', (data) => {
        const lines = data.toString().split('\n').filter(l => l.trim());
        for (const line of lines) {
            // Parse pipeline progress from log lines (same as discover handler)
            if (line.includes('[DISCOVER] Found') || line.includes('[CONTAINER]')) {
                const m = line.match(/(\d+)\s+(files|stored)/);
                if (m) {
                    const n = parseInt(m[1]);
                    if (line.includes('Found')) totalFiles = n;
                }
            }

            // Forward as progress event
            event.reply('webdav-sync-progress', {
                event: 'progress',
                message: line,
                processedCount,
                totalFiles,
            });

            // Track phase completion for UI
            if (line.includes('STEP 1/4') && line.includes('completed')) cumParse++;
            if (line.includes('STEP 2/4') && line.includes('completed')) cumMC++;
            if (line.includes('STEP 3a') && line.includes('completed')) cumVV++;
            if (line.includes('STEP 3b') && line.includes('completed')) cumMV++;
            if (line.includes('[OK]')) processedCount++;
        }
    });

    proc.stderr.on('data', (d) => console.error('[WebDAV Process]', d.toString()));

    proc.on('close', (code) => {
        activeWebdavSyncs.delete(sourceId);
        // Update last_scan in config
        const uc = readYamlFile(userSettingsPath);
        const src = (uc.webdav_sources || []).find(s => s.id === sourceId);
        if (src) {
            src.last_scan = new Date().toISOString();
            writeYamlFile(userSettingsPath, uc);
        }
        event.reply('webdav-sync-complete', { sourceId, code, processedCount });
    });

    proc.on('error', (err) => {
        activeWebdavSyncs.delete(sourceId);
        event.reply('webdav-sync-complete', { sourceId, error: err.message });
    });
});

// WebDAV Folder Browser — accepts either direct config or webdav:// path
ipcMain.handle('browse-webdav-folders', async (_, config) => {
    return new Promise((resolve) => {
        let browseConfig;

        if (config.webdavPath) {
            // Parse webdav://source-id/some/path format — resolve source credentials
            const { sourceId, subPath } = _parseWebDAVPath(config.webdavPath);
            const userConfig = readYamlFile(userSettingsPath);
            const source = (userConfig.webdav_sources || []).find(s => s.id === sourceId);
            if (!source) {
                return resolve({ success: false, folders: [], message: 'Source not found' });
            }
            browseConfig = {
                url: source.url,
                username: source.username,
                password: _decryptWebdavPassword(source),
                remote_path: '/',
                verify_ssl: source.verify_ssl !== false,
                path: subPath || null,
            };
        } else {
            // Direct config (used by WebDAVConnectDialog / WebDAVFolderPicker)
            browseConfig = {
                url: config.url,
                username: config.username,
                password: config.password,
                remote_path: config.remote_path || '/',
                verify_ssl: config.verify_ssl !== false,
                path: config.path || null,
            };
        }

        const folderConfigJson = JSON.stringify(browseConfig);
        const proc = spawnBackend('remote.sync_cli', ['--folders', folderConfigJson], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
            stdio: ['pipe', 'pipe', 'pipe'],
        }, 'backend/remote/sync_cli.py', ['--folders', folderConfigJson]);

        let output = '';
        proc.stdout.on('data', (d) => { output += d.toString(); });
        proc.stderr.on('data', (d) => console.error('[WebDAV Folders]', d.toString()));

        proc.on('close', () => {
            try {
                const lines = output.trim().split('\n').filter(l => l.trim());
                const lastLine = lines[lines.length - 1];
                resolve(JSON.parse(lastLine));
            } catch {
                resolve({ success: false, folders: [] });
            }
        });
    });
});

// WebDAV Directory Listing — list files + folders (non-recursive PROPFIND)
ipcMain.handle('list-webdav-dir', async (_, config) => {
    console.log('[WebDAV ListDir]', config.webdavPath);
    return new Promise((resolve) => {
        let listConfig;

        if (config.webdavPath) {
            const { sourceId, subPath } = _parseWebDAVPath(config.webdavPath);
            const userConfig = readYamlFile(userSettingsPath);
            const source = (userConfig.webdav_sources || []).find(s => s.id === sourceId);
            if (!source) {
                return resolve({ success: false, folders: [], files: [], message: 'Source not found' });
            }
            listConfig = {
                url: source.url,
                username: source.username,
                password: _decryptWebdavPassword(source),
                remote_path: '/',
                verify_ssl: source.verify_ssl !== false,
                path: subPath || null,
            };
        } else {
            return resolve({ success: false, folders: [], files: [], message: 'webdavPath required' });
        }

        const configJson = JSON.stringify(listConfig);
        const proc = spawnBackend('remote.sync_cli', ['--list-dir', configJson], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
            stdio: ['pipe', 'pipe', 'pipe'],
        }, 'backend/remote/sync_cli.py', ['--list-dir', configJson]);

        let output = '';
        proc.stdout.on('data', (d) => { output += d.toString(); });
        proc.stderr.on('data', (d) => console.error('[WebDAV ListDir]', d.toString()));

        proc.on('close', () => {
            try {
                const lines = output.trim().split('\n').filter(l => l.trim());
                const lastLine = lines[lines.length - 1];
                resolve(JSON.parse(lastLine));
            } catch {
                resolve({ success: false, folders: [], files: [] });
            }
        });
    });
});

// WebDAV Thumbnail — download remote file, generate thumbnail, delete temp
ipcMain.handle('generate-webdav-thumbnail', async (_, config) => {
    // config: { webdavPath: 'webdav://source-id/path/to/file.psd' }
    console.log('[WebDAV Thumb] IPC received:', config.webdavPath);
    const { sourceId, subPath } = _parseWebDAVPath(config.webdavPath);
    console.log('[WebDAV Thumb] parsed: sourceId=', sourceId, 'subPath=', subPath);
    const userConfig = readYamlFile(userSettingsPath);
    const source = (userConfig.webdav_sources || []).find(s => s.id === sourceId);
    if (!source) {
        console.log('[WebDAV Thumb] ERROR: source not found for id:', sourceId);
        return null;
    }
    console.log('[WebDAV Thumb] source found:', source.name || source.url);

    const thumbConfig = JSON.stringify({
        url: source.url,
        username: source.username,
        password: _decryptWebdavPassword(source),
        remote_path: '/',
        verify_ssl: source.verify_ssl !== false,
        file_path: subPath,
        source_id: sourceId,
    });

    return new Promise((resolve) => {
        console.log('[WebDAV Thumb] spawning Python process for:', subPath);
        const proc = spawnBackend('remote.sync_cli', ['--thumbnail', thumbConfig], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
            stdio: ['pipe', 'pipe', 'pipe'],
        }, 'backend/remote/sync_cli.py', ['--thumbnail', thumbConfig]);

        let output = '';
        proc.stdout.on('data', (d) => {
            const chunk = d.toString();
            output += chunk;
            console.log('[WebDAV Thumb] stdout:', chunk.trim());
        });
        proc.stderr.on('data', (d) => console.error('[WebDAV Thumb] stderr:', d.toString().trim()));

        proc.on('close', (code) => {
            console.log('[WebDAV Thumb] process exited with code:', code, 'output length:', output.length);
            try {
                const lines = output.trim().split('\n').filter(l => l.trim());
                const lastLine = lines[lines.length - 1];
                const result = JSON.parse(lastLine);
                console.log('[WebDAV Thumb] result:', config.webdavPath, result.success ? result.thumb_path : ('FAIL: ' + result.message));
                resolve(result?.thumb_path || null);
            } catch (e) {
                console.log('[WebDAV Thumb] parse error for:', config.webdavPath, 'raw output:', output.substring(0, 500));
                resolve(null);
            }
        });

        proc.on('error', (err) => {
            console.error('[WebDAV Thumb] spawn error:', err);
            resolve(null);
        });
    });
});

// WebDAV Browse — single Python process per folder: PROPFIND + DB check + download+parse uncached
ipcMain.handle('browse-webdav-folder', async (event, config) => {
    // config: { sourceId, path }
    const { sourceId, path: browsePath } = config;
    console.log('[WebDAV Browse] IPC received: sourceId=', sourceId, 'path=', browsePath);
    const userConfig = readYamlFile(userSettingsPath);
    const source = (userConfig.webdav_sources || []).find(s => s.id === sourceId);
    if (!source) {
        console.log('[WebDAV Browse] ERROR: source not found for id:', sourceId);
        return { error: 'Source not found' };
    }

    const browseConfig = JSON.stringify({
        url: source.url,
        username: source.username,
        password: _decryptWebdavPassword(source),
        remote_path: '/',
        verify_ssl: source.verify_ssl !== false,
        source_id: sourceId,
        path: browsePath || '/',
    });

    return new Promise((resolve) => {
        const proc = spawnBackend('remote.sync_cli', ['--browse', browseConfig], {
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
            stdio: ['pipe', 'pipe', 'pipe'],
        }, 'backend/remote/sync_cli.py', ['--browse', browseConfig]);

        let buffer = '';
        proc.stdout.on('data', (d) => {
            buffer += d.toString();
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete last line
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const evt = JSON.parse(line);
                    console.log('[WebDAV Browse] event:', evt.event,
                        evt.event === 'cached' ? `${evt.files?.length} files` :
                        evt.event === 'processed' ? evt.name :
                        evt.event === 'done' ? `cached=${evt.cached} processed=${evt.processed}` : '');
                    event.sender.send('webdav-browse-event', evt);
                } catch (e) {
                    console.error('[WebDAV Browse] JSON parse error:', line.substring(0, 200));
                }
            }
        });
        proc.stderr.on('data', (d) => console.error('[WebDAV Browse] stderr:', d.toString().trim()));

        proc.on('close', (code) => {
            // Process remaining buffer
            if (buffer.trim()) {
                try {
                    const evt = JSON.parse(buffer);
                    event.sender.send('webdav-browse-event', evt);
                } catch {}
            }
            console.log('[WebDAV Browse] process exited with code:', code);
            resolve({ code });
        });

        proc.on('error', (err) => {
            console.error('[WebDAV Browse] spawn error:', err);
            resolve({ error: err.message });
        });
    });
});

/** Parse webdav://source-id/sub/path into { sourceId, subPath } */
function _parseWebDAVPath(webdavPath) {
    // webdav://source-id/sub/path → sourceId=source-id, subPath=/sub/path
    const withoutScheme = webdavPath.replace('webdav://', '');
    const slashIdx = withoutScheme.indexOf('/');
    if (slashIdx === -1) {
        return { sourceId: withoutScheme, subPath: '/' };
    }
    return {
        sourceId: withoutScheme.substring(0, slashIdx),
        subPath: withoutScheme.substring(slashIdx) || '/',
    };
}

// IPC Handler: Update config — routes personal keys to user-settings.yaml, system keys to config.yaml
ipcMain.handle('update-config', async (_, key, value) => {
    try {
        const targetPath = isUserSetting(key) ? userSettingsPath : path.join(configRoot, 'config.yaml');
        const config = readYamlFile(targetPath);
        setDottedKey(config, key, value);
        writeYamlFile(targetPath, config);

        return { success: true };
    } catch (err) {
        console.error('[Update Config Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Get user settings file path
ipcMain.handle('get-user-settings-path', () => userSettingsPath);

// ── Server Mode (embedded FastAPI) ────────────────────────────────
// Allows Electron app to run a local FastAPI server so other clients can connect.
let serverProc = null;
let serverMainWindow = null;
let serverPortCache = 8000;

/** Get all LAN IPv4 addresses (non-internal, non-VPN). */
function getLocalNetworkAddresses() {
    const os = require('os');
    const interfaces = os.networkInterfaces();
    const addresses = [];
    for (const [name, nets] of Object.entries(interfaces)) {
        for (const net of nets) {
            if (net.internal || net.family !== 'IPv4') continue;
            if (/^(utun|tun|tap)/.test(name)) continue;
            addresses.push({ name, address: net.address });
        }
    }
    return addresses;
}

/** Load config.yaml and return parsed config object (or null on error). */
function loadAppConfig() {
    try {
        const yaml = require('js-yaml');
        const configPath = path.join(configRoot, 'config.yaml');
        if (!fs.existsSync(configPath)) return null;
        return yaml.load(fs.readFileSync(configPath, 'utf8'));
    } catch (e) {
        console.error('[Config] Failed to load config.yaml:', e.message);
        return null;
    }
}

/** Check if a TCP port is available. */
function isPortAvailable(port) {
    return new Promise((resolve) => {
        const net = require('net');
        const tester = net.createServer()
            .once('error', () => resolve(false))
            .once('listening', () => { tester.close(); resolve(true); })
            .listen(port, '0.0.0.0');
    });
}

/** Start embedded FastAPI server. Returns { success, port } or { success: false, error }. */
async function startEmbeddedServer(port = 8000) {
    if (serverProc) return { success: false, error: 'Server already running' };

    // Check port availability before spawning to prevent restart loops
    const portFree = await isPortAvailable(port);
    if (!portFree) {
        console.warn(`[Server] Port ${port} is already in use`);
        return { success: false, error: `Port ${port} is already in use` };
    }

    console.log(`[Server] Starting FastAPI on port ${port}...`);

    // Serialize WebDAV source configs for DownloadAheadPool registration
    let webdavSourcesJson = '';
    try {
        const uc = readYamlFile(userSettingsPath);
        const sources = (uc.webdav_sources || []).map(s => ({
            id: s.id, url: s.url, username: s.username,
            password: _decryptWebdavPassword(s),
            remote_path: s.remote_path || '/',
            verify_ssl: s.verify_ssl !== false,
        }));
        if (sources.length > 0) {
            webdavSourcesJson = JSON.stringify(sources);
        }
    } catch (e) {
        console.warn('[Server] Failed to serialize WebDAV sources:', e.message);
    }

    // Firebase service account key: check bundled resources, then project root
    let firebaseSaKey = '';
    const saKeyCandidates = [
        path.join(process.resourcesPath, 'firebase-service-account.json'),
        path.join(projectRoot, 'firebase-service-account.json'),
    ];
    for (const candidate of saKeyCandidates) {
        if (fs.existsSync(candidate)) { firebaseSaKey = candidate; break; }
    }

    const serverEnv = {
        ...process.env,
        IMAGINE_USER_SETTINGS_PATH: userSettingsPath,
        ...(webdavSourcesJson ? { IMAGINE_WEBDAV_SOURCES: webdavSourcesJson } : {}),
        ...(firebaseSaKey ? { FIREBASE_SERVICE_ACCOUNT_KEY: firebaseSaKey } : {}),
    };

    const cliPath = getBackendCliPath();
    if (cliPath) {
        serverProc = spawn(cliPath, ['server', '--port', String(port), '--host', '0.0.0.0'], {
            cwd: projectRoot,
            env: serverEnv,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
    } else {
        const py = resolvePython();
        serverProc = spawn(py, [
            '-m', 'uvicorn', 'backend.server.app:app',
            '--host', '0.0.0.0', '--port', String(port),
        ], {
            cwd: projectRoot,
            env: { ...serverEnv, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
            stdio: ['pipe', 'pipe', 'pipe'],
        });
    }

    // Throttled server log forwarding: max 20 IPC messages/sec to renderer
    // Prevents OOM when server outputs high-volume warnings (e.g., EmbedAhead loops)
    let _serverLogCount = 0;
    let _serverLogWindowStart = Date.now();
    const SERVER_LOG_MAX_PER_SEC = 20;
    let _serverLogDropped = 0;

    function throttledServerLog(msg, type) {
        // Errors ALWAYS pass through — never throttle error messages
        if (type === 'error') {
            try {
                if (serverMainWindow && !serverMainWindow.isDestroyed()) {
                    serverMainWindow.webContents.send('server-log', { message: msg, type });
                }
            } catch (e) { /* window may be closed */ }
            return;
        }

        const now = Date.now();
        if (now - _serverLogWindowStart > 1000) {
            if (_serverLogDropped > 0) {
                writeLog('WARN', `Server log throttled: ${_serverLogDropped} messages dropped in last window`);
            }
            _serverLogCount = 0;
            _serverLogDropped = 0;
            _serverLogWindowStart = now;
        }
        _serverLogCount++;
        if (_serverLogCount > SERVER_LOG_MAX_PER_SEC) {
            _serverLogDropped++;
            return; // drop excess messages
        }
        try {
            if (serverMainWindow && !serverMainWindow.isDestroyed()) {
                serverMainWindow.webContents.send('server-log', { message: msg, type });
            }
        } catch (e) { /* window may be closed */ }
    }

    serverProc.stdout.on('data', (chunk) => {
        const msg = chunk.toString().trim();
        if (msg) {
            writeLog('INFO', '[Server:stdout]', msg);
            console.log('[Server]', msg);
            // stdout: only forward important messages to UI
            const isStdoutError = /\bERROR\b|\bCRITICAL\b/i.test(msg);
            const isStdoutWarning = /\bWARN(?:ING)?\b/i.test(msg);
            if (isStdoutError) {
                throttledServerLog(msg, 'error');
            } else if (isStdoutWarning) {
                throttledServerLog(msg, 'warning');
            }
        }
    });

    serverProc.stderr.on('data', (chunk) => {
        for (const line of chunk.toString().split('\n')) {
            const msg = line.trim();
            if (!msg) continue;

            // Try JSON structured log first (from json_log_formatter)
            try {
                const parsed = JSON.parse(msg);
                if (parsed.level) {
                    const type = parsed.level === 'ERROR' || parsed.level === 'CRITICAL' ? 'error'
                        : parsed.level === 'WARNING' ? 'warning' : 'info';
                    writeLog(type === 'error' ? 'ERROR' : 'INFO', '[Server]', parsed.message);
                    if (type === 'error' || type === 'warning') {
                        throttledServerLog(parsed.message, type);
                    } else {
                        const isImportant = /starting up|shutting down|processing mode|worker|License|Parse-ahead|Embed-ahead|builtin|Pool|startup cleanup|mDNS|Firebase/i.test(parsed.message);
                        if (isImportant) throttledServerLog(parsed.message, 'info');
                    }
                    continue;
                }
            } catch { /* not JSON — fallback to regex */ }

            // Fallback: regex for non-Python output (uvicorn startup, native libs)
            const isWarning = /\bWARN(?:ING)?\b/i.test(msg);
            const isError = !isWarning && /\bERROR\b|\bCRITICAL\b|Traceback|Exception:/i.test(msg);
            writeLog(isError ? 'ERROR' : 'INFO', '[Server:stderr]', msg);
            if (isError) {
                throttledServerLog(msg, 'error');
            } else if (isWarning) {
                throttledServerLog(msg, 'warning');
            }
        }
    });

    serverProc.on('close', (code) => {
        console.log(`[Server] Process exited (code: ${code})`);
        serverProc = null;
        try {
            if (serverMainWindow && !serverMainWindow.isDestroyed()) {
                serverMainWindow.webContents.send('server-status-change', { running: false });
            }
        } catch (e) { /* ignore */ }
    });

    serverProc.on('error', (err) => {
        console.error('[Server] Spawn error:', err);
        serverProc = null;
    });

    serverPortCache = port;
    const lanAddresses = getLocalNetworkAddresses();
    const primaryLan = lanAddresses[0]?.address || null;
    return {
        success: true, port,
        lanAddresses,
        primaryLanUrl: primaryLan ? `http://${primaryLan}:${port}` : null,
    };
}

/** Poll /api/v1/health until server responds or timeout. */
async function waitForServerReady(port = 8000, timeoutMs = 30000) {
    const http = require('http');
    const start = Date.now();
    const interval = 500;

    while (Date.now() - start < timeoutMs) {
        try {
            await new Promise((resolve, reject) => {
                const req = http.get(`http://127.0.0.1:${port}/api/v1/health`, (res) => {
                    if (res.statusCode === 200) resolve();
                    else reject(new Error(`Status ${res.statusCode}`));
                    res.resume();
                });
                req.on('error', reject);
                req.setTimeout(2000, () => { req.destroy(); reject(new Error('timeout')); });
            });
            console.log(`[Server] Ready on port ${port}`);
            return true;
        } catch {
            await new Promise(r => setTimeout(r, interval));
        }
    }
    console.warn(`[Server] Timed out waiting for server on port ${port}`);
    return false;
}

ipcMain.handle('server-start', async (event, opts) => {
    const port = opts?.port || 8000;
    serverMainWindow = BrowserWindow.fromWebContents(event.sender);
    return startEmbeddedServer(port);
});

ipcMain.handle('server-stop', async () => {
    stopTunnel(); // stop tunnel when server stops
    if (!serverProc) return { success: true };
    try {
        serverProc.kill('SIGTERM');
    } catch (e) { /* ignore */ }
    // Force kill after 5s if still alive
    const proc = serverProc;
    setTimeout(() => {
        try { proc?.kill('SIGKILL'); } catch (e) { /* already dead */ }
    }, 5000);
    serverProc = null;
    return { success: true };
});

ipcMain.handle('server-status', async () => {
    if (!serverProc) return { running: false };
    const lanAddresses = getLocalNetworkAddresses();
    const primaryLan = lanAddresses[0]?.address || null;
    return {
        running: true,
        lanAddresses,
        primaryLanUrl: primaryLan ? `http://${primaryLan}:${serverPortCache}` : null,
    };
});

function killServerProc() {
    if (!serverProc) return;
    stopTunnel(); // also stop tunnel when server stops
    try { serverProc.kill('SIGTERM'); } catch (e) { /* ignore */ }
    serverProc = null;
}

// ── Cloudflare Quick Tunnel ──────────────────────────────────────
// Exposes the local server to the internet via cloudflared (no account needed).
let tunnelProc = null;
let tunnelUrl = null;

const CLOUDFLARED_DIR = path.join(app.getPath('userData'), 'bin');

function getCloudflaredPath() {
    const bin = process.platform === 'win32' ? 'cloudflared.exe' : 'cloudflared';
    return path.join(CLOUDFLARED_DIR, bin);
}

function isCloudflaredInstalled() {
    return fs.existsSync(getCloudflaredPath());
}

/** Download cloudflared binary for the current platform from GitHub Releases. */
async function downloadCloudflared() {
    const https = require('https');
    const { execSync } = require('child_process');

    const platform = process.platform;
    const arch = process.arch === 'arm64' ? 'arm64' : 'amd64';

    const urls = {
        'darwin-arm64': 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz',
        'darwin-amd64': 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz',
        'win32-amd64': 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe',
        'win32-arm64': 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe',
        'linux-amd64': 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
        'linux-arm64': 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64',
    };

    const key = `${platform}-${arch}`;
    const url = urls[key];
    if (!url) throw new Error(`Unsupported platform: ${key}`);

    if (!fs.existsSync(CLOUDFLARED_DIR)) {
        fs.mkdirSync(CLOUDFLARED_DIR, { recursive: true });
    }

    console.log(`[Tunnel] Downloading cloudflared for ${key}...`);

    // Helper: follow redirects and download to file
    const downloadFile = (downloadUrl, destPath) => new Promise((resolve, reject) => {
        const follow = (url, redirectCount = 0) => {
            if (redirectCount > 5) return reject(new Error('Too many redirects'));
            const proto = url.startsWith('https') ? https : require('http');
            proto.get(url, { headers: { 'User-Agent': 'Imagine-App' } }, (res) => {
                if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                    return follow(res.headers.location, redirectCount + 1);
                }
                if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}`));
                const fileStream = fs.createWriteStream(destPath);
                res.pipe(fileStream);
                fileStream.on('finish', () => { fileStream.close(); resolve(); });
                fileStream.on('error', reject);
            }).on('error', reject);
        };
        follow(downloadUrl);
    });

    if (url.endsWith('.tgz')) {
        // macOS: download .tgz then extract
        const tgzPath = path.join(CLOUDFLARED_DIR, 'cloudflared.tgz');
        await downloadFile(url, tgzPath);
        execSync(`tar -xzf "${tgzPath}" -C "${CLOUDFLARED_DIR}"`, { stdio: 'ignore' });
        try { fs.unlinkSync(tgzPath); } catch (e) { /* ignore */ }
        fs.chmodSync(getCloudflaredPath(), 0o755);
    } else if (url.endsWith('.exe')) {
        // Windows: download .exe directly
        await downloadFile(url, getCloudflaredPath());
    } else {
        // Linux: download binary directly
        await downloadFile(url, getCloudflaredPath());
        fs.chmodSync(getCloudflaredPath(), 0o755);
    }

    console.log('[Tunnel] cloudflared downloaded successfully');
}

/** Start Cloudflare Quick Tunnel. Returns { success, url } or error. */
function startTunnel(port) {
    if (tunnelProc) return Promise.resolve({ success: false, error: 'Tunnel already running' });

    const bin = getCloudflaredPath();
    if (!fs.existsSync(bin)) {
        return Promise.resolve({ success: false, error: 'cloudflared not installed', needsInstall: true });
    }

    console.log(`[Tunnel] Starting cloudflared tunnel for port ${port}...`);

    tunnelProc = spawn(bin, [
        'tunnel', '--url', `http://localhost:${port}`, '--no-autoupdate',
    ], {
        stdio: ['pipe', 'pipe', 'pipe'],
    });

    return new Promise((resolve) => {
        let resolved = false;
        const timeout = setTimeout(() => {
            if (!resolved) {
                resolved = true;
                resolve({ success: false, error: 'Tunnel start timeout (30s)' });
            }
        }, 30000);

        const handleData = (chunk) => {
            const msg = chunk.toString();
            // cloudflared outputs URL to stderr
            const match = msg.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/);
            if (match && !resolved) {
                resolved = true;
                clearTimeout(timeout);
                tunnelUrl = match[0];
                console.log(`[Tunnel] URL: ${tunnelUrl}`);
                resolve({ success: true, url: tunnelUrl });
                try {
                    if (serverMainWindow && !serverMainWindow.isDestroyed()) {
                        serverMainWindow.webContents.send('tunnel-status-change', {
                            running: true, url: tunnelUrl
                        });
                    }
                } catch (e) { /* ignore */ }
                // Register tunnel URL to Firestore via server API
                if (serverPortCache) {
                    const http = require('http');
                    const body = JSON.stringify({ tunnel_url: tunnelUrl });
                    const req = http.request({
                        hostname: '127.0.0.1', port: serverPortCache,
                        path: '/api/v1/server/tunnel-url', method: 'PUT',
                        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
                    }, (res) => {
                        console.log(`[Tunnel] Firestore registration: HTTP ${res.statusCode}`);
                    });
                    req.on('error', (e) => console.warn('[Tunnel] Firestore registration failed:', e.message));
                    req.write(body);
                    req.end();
                }
            }
        };

        tunnelProc.stdout.on('data', handleData);
        tunnelProc.stderr.on('data', handleData);

        tunnelProc.on('close', (code) => {
            console.log(`[Tunnel] Process exited (code: ${code})`);
            tunnelProc = null;
            tunnelUrl = null;
            if (!resolved) {
                resolved = true;
                clearTimeout(timeout);
                resolve({ success: false, error: `Tunnel exited with code ${code}` });
            }
            try {
                if (serverMainWindow && !serverMainWindow.isDestroyed()) {
                    serverMainWindow.webContents.send('tunnel-status-change', { running: false });
                }
            } catch (e) { /* ignore */ }
        });

        tunnelProc.on('error', (err) => {
            console.error('[Tunnel] Spawn error:', err);
            if (!resolved) {
                resolved = true;
                clearTimeout(timeout);
                resolve({ success: false, error: err.message });
            }
        });
    });
}

function stopTunnel() {
    if (!tunnelProc) return { success: true };
    console.log('[Tunnel] Stopping...');
    try { tunnelProc.kill('SIGTERM'); } catch (e) { /* ignore */ }
    const proc = tunnelProc;
    setTimeout(() => {
        try { proc?.kill('SIGKILL'); } catch (e) { /* already dead */ }
    }, 5000);
    tunnelProc = null;
    tunnelUrl = null;
    return { success: true };
}

ipcMain.handle('tunnel-start', async (event, opts) => {
    serverMainWindow = BrowserWindow.fromWebContents(event.sender);

    // Auto-download if not installed
    if (!isCloudflaredInstalled()) {
        try {
            if (serverMainWindow && !serverMainWindow.isDestroyed()) {
                serverMainWindow.webContents.send('tunnel-status-change', { downloading: true });
            }
            await downloadCloudflared();
        } catch (e) {
            console.error('[Tunnel] Download failed:', e);
            return { success: false, error: `Failed to download cloudflared: ${e.message}` };
        }
    }

    return startTunnel(opts?.port || serverPortCache);
});

ipcMain.handle('tunnel-stop', async () => {
    return stopTunnel();
});

ipcMain.handle('tunnel-status', async () => {
    return {
        running: !!tunnelProc,
        url: tunnelUrl,
        installed: isCloudflaredInstalled(),
    };
});

// ── License IPC ─────────────────────────────────────────────────
const crypto = require('crypto');
const LICENSE_FILE = path.join(app.getPath('userData'), 'license.json');

function readLicenseFile() {
    try {
        if (fs.existsSync(LICENSE_FILE)) {
            return JSON.parse(fs.readFileSync(LICENSE_FILE, 'utf8'));
        }
    } catch { /* corrupt file — treat as empty */ }
    return {};
}

function writeLicenseFile(data) {
    fs.writeFileSync(LICENSE_FILE, JSON.stringify(data, null, 2), 'utf8');
}

ipcMain.handle('license-get', async () => {
    try {
        let data = readLicenseFile();
        // Auto-generate deviceId on first access
        if (!data.deviceId) {
            data.deviceId = crypto.randomUUID();
            writeLicenseFile(data);
        }
        return { success: true, data };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

ipcMain.handle('license-set', async (_event, updates) => {
    try {
        const data = { ...readLicenseFile(), ...updates };
        writeLicenseFile(data);
        return { success: true };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

ipcMain.handle('license-clear', async () => {
    try {
        if (fs.existsSync(LICENSE_FILE)) fs.unlinkSync(LICENSE_FILE);
        return { success: true };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

ipcMain.handle('license-get-config', async () => {
    try {
        const systemConfig = readYamlFile(path.join(configRoot, 'config.yaml'));
        return { success: true, license: systemConfig.license || { enabled: false } };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

// ── Window creation (pure UI — no IPC registration) ──────────────

function createWindow() {
    const _deskLog = path.join(app.getPath('desktop'), 'imagine-startup.log');
    try { fs.appendFileSync(_deskLog, `[createWindow] start\n`); } catch {}

    const iconPath = isDev
        ? path.join(__dirname, '../public/icon-512.png')
        : path.join(__dirname, '../dist/icon-512.png');

    try { fs.appendFileSync(_deskLog, `[createWindow] icon: ${iconPath} exists: ${fs.existsSync(iconPath)}\n`); } catch {}

    const preloadPath = path.join(__dirname, 'preload.cjs');
    try { fs.appendFileSync(_deskLog, `[createWindow] preload: ${preloadPath} exists: ${fs.existsSync(preloadPath)}\n`); } catch {}

    const mainWindow = new BrowserWindow({
        title: 'Imagine',
        icon: iconPath,
        width: 1280,
        height: 800,
        titleBarStyle: 'hidden',
        trafficLightPosition: { x: 12, y: 18 },
        webPreferences: {
            preload: preloadPath,
            nodeIntegration: false,
            contextIsolation: true,
            webSecurity: false,
            sandbox: false,
        },
    });

    // ── Strip COOP headers so Firebase signInWithPopup works in Electron ──
    mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
        const headers = { ...details.responseHeaders };
        delete headers['cross-origin-opener-policy'];
        delete headers['Cross-Origin-Opener-Policy'];
        callback({ responseHeaders: headers });
    });

    // ── Allow Google OAuth popups for Firebase Auth ──
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        if (url.includes('accounts.google.com') || url.includes('firebaseapp.com') || url.includes('googleapis.com')) {
            return { action: 'allow' };
        }
        const { shell } = require('electron');
        shell.openExternal(url);
        return { action: 'deny' };
    });

    // ── Application Menu ──
    const isMac = process.platform === 'darwin';
    const sendMenuAction = (action) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('menu-action', action);
        }
    };
    const menuTemplate = [
        // macOS app menu
        ...(isMac ? [{
            label: app.name,
            submenu: [
                { role: 'about' },
                { type: 'separator' },
                { role: 'services' },
                { type: 'separator' },
                { role: 'hide' },
                { role: 'hideOthers' },
                { role: 'unhide' },
                { type: 'separator' },
                { role: 'quit' },
            ],
        }] : []),
        // File
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Folder…',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => sendMenuAction('open-folder'),
                },
                { type: 'separator' },
                {
                    label: 'Export Database…',
                    click: () => sendMenuAction('export-db'),
                },
                {
                    label: 'Import Database…',
                    click: () => sendMenuAction('import-db'),
                },
                { type: 'separator' },
                isMac ? { role: 'close' } : { role: 'quit' },
            ],
        },
        // Edit
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectAll' },
            ],
        },
        // View
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' },
            ],
        },
        // Tools
        {
            label: 'Tools',
            submenu: [
                {
                    label: 'Toggle Server',
                    click: () => sendMenuAction('toggle-server'),
                },
                {
                    label: 'Toggle Worker',
                    click: () => sendMenuAction('toggle-worker'),
                },
            ],
        },
        // Window (macOS)
        ...(isMac ? [{
            label: 'Window',
            submenu: [
                { role: 'minimize' },
                { role: 'zoom' },
                { type: 'separator' },
                { role: 'front' },
            ],
        }] : []),
        // Help
        {
            label: 'Help',
            submenu: [
                {
                    label: 'Open Logs Folder',
                    click: () => {
                        const logsPath = path.join(app.getPath('userData'), 'logs');
                        shell.openPath(logsPath);
                    },
                },
            ],
        },
    ];
    Menu.setApplicationMenu(Menu.buildFromTemplate(menuTemplate));

    try { fs.appendFileSync(_deskLog, `[createWindow] BrowserWindow created\n`); } catch {}

    if (isDev) {
        mainWindow.loadURL('http://localhost:9274');
        mainWindow.webContents.openDevTools();
    } else {
        const htmlPath = path.join(__dirname, '../dist/index.html');
        try { fs.appendFileSync(_deskLog, `[createWindow] loadFile: ${htmlPath} exists: ${fs.existsSync(htmlPath)}\n`); } catch {}
        mainWindow.loadFile(htmlPath);
    }

    mainWindow.webContents.on('did-finish-load', () => {
        try { fs.appendFileSync(_deskLog, `[createWindow] page loaded OK\n`); } catch {}
    });
    mainWindow.webContents.on('did-fail-load', (e, code, desc) => {
        try { fs.appendFileSync(_deskLog, `[createWindow] LOAD FAILED: ${code} ${desc}\n`); } catch {}
    });

    try { fs.appendFileSync(_deskLog, `[createWindow] done\n`); } catch {}
}

// ── App lifecycle ────────────────────────────────────────────────

app.setName('Imagine');

app.whenReady().then(async () => {
    const _deskLog = path.join(app.getPath('desktop'), 'imagine-startup.log');
    try { fs.appendFileSync(_deskLog, `[ready] app.whenReady fired\n`); } catch {}

    // Migrate personal settings from config.yaml to user-settings.yaml (first run only)
    try { migrateUserSettings(); } catch (e) {
        try { fs.appendFileSync(_deskLog, `[ready] migrateUserSettings error: ${e.message}\n`); } catch {}
    }
    try { fs.appendFileSync(_deskLog, `[ready] migrateUserSettings done\n`); } catch {}

    // Set macOS dock icon
    if (process.platform === 'darwin' && app.dock) {
        const { nativeImage } = require('electron');
        const dockIconPath = isDev
            ? path.join(__dirname, '../public/icon-512.png')
            : path.join(__dirname, '../dist/icon-512.png');
        if (fs.existsSync(dockIconPath)) {
            app.dock.setIcon(nativeImage.createFromPath(dockIconPath));
        }
    }

    // Kill any residual search daemons from previous crashed sessions
    try { cleanupOrphanDaemons(); } catch (e) {
        try { fs.appendFileSync(_deskLog, `[ready] cleanupOrphanDaemons error: ${e.message}\n`); } catch {}
    }
    try { fs.appendFileSync(_deskLog, `[ready] cleanup done\n`); } catch {}

    // Ensure Codex CLI is installed (for search query decomposition)
    try {
        const { execSync } = require('child_process');
        execSync('codex --version', { stdio: 'pipe', timeout: 5000 });
        console.log('[Codex] CLI available');
    } catch {
        console.log('[Codex] CLI not found, installing...');
        try {
            const { execSync } = require('child_process');
            execSync('npm install -g @openai/codex', { stdio: 'pipe', timeout: 120000 });
            console.log('[Codex] CLI installed successfully');
        } catch (e) {
            console.warn('[Codex] CLI install failed (will use local MLX fallback):', e.message);
        }
    }

    try { fs.appendFileSync(_deskLog, `[ready] codex check done, calling createWindow\n`); } catch {}

    // Server is started by the React app via IPC (window.electron.server.start)
    // when user selects "관리" mode on SetupPage. No config.yaml auto-start.

    // Do NOT start search daemon here — it starts lazily on first search
    createWindow();

    // Initialize auto-updater (after window is ready to receive events)
    initAutoUpdater();
    try { fs.appendFileSync(_deskLog, `[ready] initAutoUpdater done\n`); } catch {}

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

/**
 * Kill a process and its entire child tree.
 * Windows: taskkill /T (tree kill) — kills all descendants.
 * Unix: process.kill(-pid) — kills process group (detached spawn creates own group).
 */
function killProcessTree(proc) {
    if (!proc || !proc.pid) return;
    if (process.platform === 'win32') {
        try {
            execSync(`taskkill /F /T /PID ${proc.pid} 2>nul`, { stdio: 'ignore', timeout: 5000 });
        } catch { /* already dead */ }
    } else {
        try {
            process.kill(-proc.pid, 'SIGKILL');
        } catch {
            try { proc.kill('SIGKILL'); } catch { /* already dead */ }
        }
    }
}

// Kill active pipeline/discover process trees (prevents residual processes on quit)
function killActivePipeline() {
    if (activePipelineProc) {
        killProcessTree(activePipelineProc);
        activePipelineProc = null;
    }
    for (const [, proc] of activeDiscoverProcs) {
        killProcessTree(proc);
    }
    activeDiscoverProcs.clear();

    if (activeBackfillProc) {
        killProcessTree(activeBackfillProc);
        activeBackfillProc = null;
    }
}

app.on('before-quit', () => {
    killActivePipeline();
    killSearchDaemon();
    killServerProc();
});

// Ensure daemon cleanup on unexpected termination signals
process.on('SIGINT', () => {
    killActivePipeline();
    killSearchDaemon();
    killServerProc();
    app.quit();
});

process.on('SIGTERM', () => {
    killActivePipeline();
    killSearchDaemon();
    killServerProc();
    app.quit();
});
