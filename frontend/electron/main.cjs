const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');
const { autoUpdater } = require('electron-updater');
const isDev = process.env.NODE_ENV === 'development';

// Suppress EPIPE errors from console.log when parent pipe is closed (background launch)
process.stdout?.on?.('error', (err) => { if (err.code !== 'EPIPE') throw err; });
process.stderr?.on?.('error', (err) => { if (err.code !== 'EPIPE') throw err; });

// ---------- File-based crash/error logging ----------
// Logs to <userData>/logs/main.log (survives crashes, rotated at 5MB)
const LOG_MAX_BYTES = 5 * 1024 * 1024; // 5 MB
const logDir = path.join(app.getPath('userData'), 'logs');
try { fs.mkdirSync(logDir, { recursive: true }); } catch { /* ignore */ }
const logFilePath = path.join(logDir, 'main.log');

function _rotateLogIfNeeded() {
    try {
        const stat = fs.statSync(logFilePath);
        if (stat.size > LOG_MAX_BYTES) {
            const prev = logFilePath + '.1';
            try { fs.unlinkSync(prev); } catch { /* ok */ }
            fs.renameSync(logFilePath, prev);
        }
    } catch { /* file doesn't exist yet */ }
}

function writeLog(level, ...args) {
    const ts = new Date().toISOString();
    const msg = args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ');
    const line = `${ts} [${level}] ${msg}\n`;
    try {
        _rotateLogIfNeeded();
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
    'registered_folders', 'last_session',
    'worker.claim_batch_size', 'worker.gpu_memory_percent',
    'worker.cpu_cores', 'worker.batch_capacity',
    'worker.schedule', 'worker.idle_unload_minutes',
    'worker.processing_mode',
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

// ── Search Daemon (lazy-start, idle-kill) ────────────────────────
// Daemon is NOT started on app launch. It spawns on first search,
// stays alive for IDLE_TIMEOUT_MS after last search, then auto-kills.
const IDLE_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
let searchDaemon = null;
let searchReady = false;
let pendingRequests = [];
let responseBuffer = '';
let idleTimer = null;

function getSearchScriptPath() {
    return isDev
        ? path.resolve(__dirname, '../../backend/api_search.py')
        : path.join(projectRoot, 'backend/api_search.py');
}

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

    const finalPython = resolvePython();
    const scriptPath = getSearchScriptPath();

    console.log('[SearchDaemon] Starting search process (lazy)...');

    searchDaemon = spawn(finalPython, [scriptPath, '--daemon'], {
        cwd: projectRoot,
        env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath },
        stdio: ['pipe', 'pipe', 'pipe'],
    });

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
                // Warmup complete — flush queued requests
                if (parsed.status === 'ready') {
                    console.log(`[SearchDaemon] Models loaded (${parsed.warmup_ms}ms)`);
                    for (const req of pendingRequests) {
                        searchDaemon.stdin.write(JSON.stringify(req.data) + '\n');
                    }
                    continue;
                }
                // Normal search response
                if (pendingRequests.length > 0) {
                    const { resolve: res } = pendingRequests.shift();
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
        pendingRequests.push({ resolve, data });
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

// Auto Update IPC
ipcMain.handle('updater-check', async () => {
    if (isDev) return { available: false, reason: 'dev-mode' };
    try {
        const result = await autoUpdater.checkForUpdates();
        return { available: !!result?.updateInfo, info: result?.updateInfo };
    } catch (err) {
        return { available: false, error: err.message };
    }
});

ipcMain.handle('updater-download', async () => {
    try {
        await autoUpdater.downloadUpdate();
        return { success: true };
    } catch (err) {
        return { success: false, error: err.message };
    }
});

ipcMain.on('updater-quit-and-install', () => {
    autoUpdater.quitAndInstall(false, true);
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

// IPC Handler: Generate Thumbnail (single file)
ipcMain.handle('generate-thumbnail', async (_, filePath) => {
    const finalPython = resolvePython();

    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/utils/thumbnail_generator.py')
        : path.join(projectRoot, 'backend/thumbnail_generator.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [scriptPath, filePath, '--size', '256'], { cwd: projectRoot });
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
    const finalPython = resolvePython();

    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/utils/thumbnail_generator.py')
        : path.join(projectRoot, 'backend/thumbnail_generator.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [scriptPath, '--batch', JSON.stringify(filePaths), '--size', '256', '--return-paths'], { cwd: projectRoot });
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
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_export.py')
        : path.join(projectRoot, 'backend/api_export.py');

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
        const proc = spawn(finalPython, [scriptPath, '--output', outputPath], { cwd: projectRoot });
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
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_relink.py')
        : path.join(projectRoot, 'backend/api_relink.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [
            scriptPath, '--package', packagePath, '--folder', targetFolder, '--dry-run'
        ], { cwd: projectRoot });
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
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_relink.py')
        : path.join(projectRoot, 'backend/api_relink.py');

    const args = [scriptPath, '--package', packagePath, '--folder', targetFolder];
    if (deleteMissing) args.push('--delete-missing');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, args, { cwd: projectRoot });
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
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_sync.py')
        : path.join(projectRoot, 'backend/api_sync.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [scriptPath, '--folder', folderPath], { cwd: projectRoot });
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
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_sync.py')
        : path.join(projectRoot, 'backend/api_sync.py');

    // Pass first move's folder to get DB path, then apply all moves
    const folderPath = moves.length > 0 ? path.dirname(moves[0].new_path) : '.';

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [
            scriptPath, '--folder', folderPath, '--apply-moves'
        ], { cwd: projectRoot });
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
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_sync.py')
        : path.join(projectRoot, 'backend/api_sync.py');

    const idsStr = fileIds.join(',');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [
            scriptPath, '--folder', '.', '--delete-missing', idsStr
        ], { cwd: projectRoot });
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
        };
    }

    return sendSearchRequest(inputData);
});

// IPC Handler: Database Stats (archived image count)
ipcMain.handle('get-db-stats', async () => {
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_stats.py')
        : path.join(projectRoot, 'backend/api_stats.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [scriptPath], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        });
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
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_queue.py')
        : path.join(projectRoot, 'backend/api_queue.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [scriptPath, cmd, JSON.stringify(data || {})], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        });
        let output = '';
        let errOutput = '';
        proc.stdout.on('data', (d) => output += d.toString());
        proc.stderr.on('data', (d) => errOutput += d.toString());
        proc.on('close', (code) => {
            if (code === 0) {
                try {
                    resolve(JSON.parse(output.trim()));
                } catch {
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
    return spawnQueueCmd('register-paths', { file_paths: filePaths, priority: priority || 0 });
});

ipcMain.handle('queue-scan-folder', async (_, { folderPath, priority }) => {
    return spawnQueueCmd('scan-folder', { folder_path: folderPath, priority: priority || 0 });
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

// IPC Handler: Incomplete Stats (for resume dialog on startup)
ipcMain.handle('get-incomplete-stats', async () => {
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_incomplete_stats.py')
        : path.join(projectRoot, 'backend/api_incomplete_stats.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [scriptPath], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        });
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
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_folder_stats.py')
        : path.join(projectRoot, 'backend/api_folder_stats.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [scriptPath, storageRoot], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath }
        });
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

// IPC Handler: Environment Check
ipcMain.handle('check-env', async () => {
    const finalPython = resolvePython();
    const scriptPath = isDev ? path.resolve(__dirname, '../../backend/setup/installer.py') : path.join(projectRoot, 'backend/setup/installer.py');

    return new Promise((resolve) => {
        const proc = spawn(finalPython, [scriptPath, '--check'], { cwd: projectRoot });
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
    const finalPython = resolvePython();
    const scriptPath = isDev ? path.resolve(__dirname, '../../backend/setup/installer.py') : path.join(projectRoot, 'backend/setup/installer.py');

    event.reply('install-log', { message: '🚀 Starting installation...', type: 'info' });

    const proc = spawn(finalPython, [scriptPath, '--install', '--download-model'], { cwd: projectRoot });

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
    const finalPython = resolvePython();

    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_metadata_update.py')
        : path.join(projectRoot, 'backend/api_metadata_update.py');

    return new Promise((resolve, reject) => {
        const proc = spawn(finalPython, [scriptPath], { cwd: projectRoot });

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
let pipelineStoppedByUser = false;

ipcMain.on('run-pipeline', (event, { filePaths }) => {
    console.log('[run-pipeline] Received request:', filePaths.length, 'files');
    if (activePipelineProc) {
        console.log('[run-pipeline] BLOCKED: pipeline already running (pid:', activePipelineProc.pid, ')');
        event.reply('pipeline-log', { message: 'Pipeline already running. Wait for it to finish.', type: 'error' });
        return;
    }

    const finalPython = resolvePython();
    console.log('[run-pipeline] Python:', finalPython);

    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/pipeline/ingest_engine.py')
        : path.join(projectRoot, 'backend/ingest_engine.py');

    console.log('[run-pipeline] Script:', scriptPath);
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

    const proc = spawn(finalPython, [scriptPath, '--files', JSON.stringify(filePaths)], {
        cwd: projectRoot,
        detached: true,  // Own process group for clean tree kill
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        stdio: ['pipe', 'pipe', 'pipe'],  // stdin pipe = watchdog lifeline
    });
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
        const message = data.toString().trim();
        if (!message) return;
        const isError = /\bERROR\b|Traceback|Exception:|raise\s|FAIL/i.test(message);
        if (isError) {
            event.reply('pipeline-log', { message, type: 'error' });
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

    const finalPython = resolvePython();

    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/pipeline/ingest_engine.py')
        : path.join(projectRoot, 'backend/ingest_engine.py');

    const args = [scriptPath, '--discover', folderPath];
    if (noSkip) args.push('--no-skip');

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

    const proc = spawn(finalPython, args, {
        cwd: projectRoot,
        detached: true,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        stdio: ['pipe', 'pipe', 'pipe'],  // stdin pipe = watchdog lifeline
    });
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
        // stderr: only forward errors (library output like transformers/torch is noisy)
        const lines = raw.split('\n').filter(l => l.trim());
        for (const line of lines) {
            const msg = line.trim();
            if (!msg) continue;
            if (/\bERROR\b|Traceback|Exception:|raise\s|FAIL/i.test(msg)) {
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
        const folders = (regFolders.folders || []).map(fp => ({
            path: fp,
            exists: fs.existsSync(fp),
        }));
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

// ── Worker Daemon (controlled via WorkerPage) ────────────────────
// Spawns backend/worker/worker_ipc.py and relays JSON events to renderer.
let workerProc = null;
let workerBuffer = '';
let workerMainWindow = null;  // BrowserWindow reference for sending events
let workerStartCmd = null;    // Queued start command (sent after 'ready' event)

function getWorkerScriptPath() {
    return isDev
        ? path.resolve(__dirname, '../../backend/worker/worker_ipc.py')
        : path.join(projectRoot, 'backend/worker/worker_ipc.py');
}

function sendWorkerEvent(channel, data) {
    try {
        if (workerMainWindow && !workerMainWindow.isDestroyed()) {
            workerMainWindow.webContents.send(channel, data);
        }
    } catch (e) { /* window may be closed */ }
}

function processWorkerOutput() {
    let newlineIdx;
    while ((newlineIdx = workerBuffer.indexOf('\n')) !== -1) {
        const line = workerBuffer.slice(0, newlineIdx).trim();
        workerBuffer = workerBuffer.slice(newlineIdx + 1);
        if (!line) continue;

        try {
            const parsed = JSON.parse(line);
            const evt = parsed.event;

            // 'ready' signal — send queued start command
            if (evt === 'ready') {
                console.log('[Worker] IPC ready');
                if (workerStartCmd && workerProc) {
                    workerProc.stdin.write(JSON.stringify(workerStartCmd) + '\n');
                    workerStartCmd = null;
                }
                continue;
            }

            // Relay events to renderer
            if (evt === 'status') {
                sendWorkerEvent('worker-status', parsed);
            } else if (evt === 'log') {
                sendWorkerEvent('worker-log', parsed);
            } else if (evt === 'job_done') {
                sendWorkerEvent('worker-job-done', parsed);
            } else if (evt === 'stats') {
                sendWorkerEvent('worker-stats', parsed);
            } else if (evt === 'batch_start') {
                console.log('[Worker:batch] START size=', parsed.batch_size);
                sendWorkerEvent('worker-batch-start', parsed);
            } else if (evt === 'batch_phase_start') {
                console.log('[Worker:batch] PHASE_START', parsed.phase, 'count=', parsed.count);
                sendWorkerEvent('worker-batch-phase-start', parsed);
            } else if (evt === 'batch_file_done') {
                console.log('[Worker:batch] FILE_DONE', parsed.phase, parsed.index, '/', parsed.count, parsed.file_name);
                sendWorkerEvent('worker-batch-file-done', parsed);
            } else if (evt === 'batch_phase_complete') {
                console.log('[Worker:batch] PHASE_COMPLETE', parsed.phase);
                sendWorkerEvent('worker-batch-phase-complete', parsed);
            } else if (evt === 'batch_job_upload') {
                console.log('[Worker:batch] JOB_UPLOAD', parsed.job_id, parsed.success);
                sendWorkerEvent('worker-batch-job-upload', parsed);
            } else if (evt === 'batch_complete') {
                console.log('[Worker:batch] COMPLETE', parsed.count, 'files in', parsed.elapsed_s, 's', parsed.files_per_min, '/min');
                sendWorkerEvent('worker-batch-complete', parsed);
            } else if (evt === 'processing_mode') {
                console.log('[Worker] Processing mode:', parsed.mode);
                sendWorkerEvent('worker-processing-mode', parsed);
            } else if (evt === 'worker_state') {
                console.log('[Worker] State:', parsed.state);
                sendWorkerEvent('worker-state', parsed);
            }
        } catch (e) {
            console.error('[Worker] JSON parse error:', e, line);
        }
    }
}

function killWorkerProc() {
    if (!workerProc) return;
    const proc = workerProc;
    workerProc = null;
    workerBuffer = '';
    workerStartCmd = null;

    // Try graceful exit first
    try {
        proc.stdin.write(JSON.stringify({ cmd: 'exit' }) + '\n');
    } catch (e) { /* ignore */ }
    // Force kill after 2 seconds (SIGKILL works on Windows, SIGTERM doesn't)
    setTimeout(() => {
        try { proc.kill('SIGKILL'); } catch (e) { /* already dead */ }
    }, 2000);
}

// IPC Handler: Start worker daemon
ipcMain.handle('worker-start', async (event, opts) => {
    const accessToken = opts.accessToken || '';
    const refreshToken = opts.refreshToken || '';
    const startCmd = {
        cmd: 'start',
        server_url: opts.serverUrl || 'http://localhost:8000',
        access_token: accessToken,
        refresh_token: refreshToken,
        username: opts.username || '',
        password: opts.password || '',
    };

    // If process is alive, send start command directly (restart after stop)
    if (workerProc) {
        console.log('[Worker] Process alive — sending start command to existing process');
        try {
            workerProc.stdin.write(JSON.stringify(startCmd) + '\n');
            return { success: true };
        } catch (e) {
            console.error('[Worker] Failed to write to existing process:', e);
            // Process is dead, fall through to spawn new one
            workerProc = null;
        }
    }

    const finalPython = resolvePython();

    // Store window reference for relaying events
    workerMainWindow = BrowserWindow.fromWebContents(event.sender);

    console.log('[Worker] Starting worker_ipc.py...');
    console.log(`[Worker] Auth: access=${accessToken ? accessToken.substring(0, 20) + '...' : '(none)'}, refresh=${refreshToken ? refreshToken.substring(0, 16) + '...' : '(none)'}`);

    workerProc = spawn(finalPython, ['-u', '-m', 'backend.worker.worker_ipc'], {
        cwd: projectRoot,
        env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath },
        stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Queue the start command — will be sent after 'ready' event
    workerStartCmd = startCmd;

    workerProc.stdout.on('data', (chunk) => {
        workerBuffer += chunk.toString();
        processWorkerOutput();
    });

    workerProc.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) {
            console.error('[Worker:stderr]', msg);
            // Forward ALL stderr to UI for debugging (not just errors)
            const isError = /\bERROR\b|Traceback|Exception:|FAIL/i.test(msg);
            sendWorkerEvent('worker-log', {
                message: `[stderr] ${msg}`,
                type: isError ? 'error' : 'warning',
            });
        }
    });

    workerProc.on('close', (code) => {
        console.log(`[Worker] Process exited (code: ${code})`);
        workerProc = null;
        workerBuffer = '';
        workerStartCmd = null;
        sendWorkerEvent('worker-status', { status: 'idle', jobs: [] });
        sendWorkerEvent('worker-log', {
            message: code === 0 ? 'Worker stopped' : `Worker exited (code: ${code})`,
            type: code === 0 ? 'info' : 'error',
        });
    });

    workerProc.on('error', (err) => {
        console.error('[Worker] Spawn error:', err);
        workerProc = null;
        sendWorkerEvent('worker-status', { status: 'error', jobs: [] });
        sendWorkerEvent('worker-log', { message: `Spawn error: ${err.message}`, type: 'error' });
    });

    return { success: true };
});

// IPC Handler: Stop worker daemon
ipcMain.handle('worker-stop', async () => {
    if (!workerProc) return { success: true };
    try {
        workerProc.stdin.write(JSON.stringify({ cmd: 'stop' }) + '\n');
    } catch (e) { /* ignore */ }
    return { success: true };
});

// IPC Handler: Forward refreshed tokens to the worker process
ipcMain.handle('worker-update-tokens', async (event, opts) => {
    if (!workerProc) return { success: false, error: 'Worker not running' };
    try {
        workerProc.stdin.write(JSON.stringify({
            cmd: 'update_tokens',
            access_token: opts.accessToken || '',
            refresh_token: opts.refreshToken || '',
        }) + '\n');
        return { success: true };
    } catch (e) {
        return { success: false, error: e.message };
    }
});

// IPC Handler: Query worker status
ipcMain.handle('worker-status', async () => {
    if (!workerProc) return { status: 'idle' };
    try {
        workerProc.stdin.write(JSON.stringify({ cmd: 'status' }) + '\n');
    } catch (e) { /* ignore */ }
    return { status: 'running' };
});

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

    const finalPython = resolvePython();
    console.log(`[Server] Starting FastAPI on port ${port}...`);

    serverProc = spawn(finalPython, [
        '-m', 'uvicorn', 'backend.server.app:app',
        '--host', '0.0.0.0', '--port', String(port),
    ], {
        cwd: projectRoot,
        env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8', IMAGINE_USER_SETTINGS_PATH: userSettingsPath },
        stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Throttled server log forwarding: max 20 IPC messages/sec to renderer
    // Prevents OOM when server outputs high-volume warnings (e.g., EmbedAhead loops)
    let _serverLogCount = 0;
    let _serverLogWindowStart = Date.now();
    const SERVER_LOG_MAX_PER_SEC = 20;
    let _serverLogDropped = 0;

    function throttledServerLog(msg, type) {
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
            throttledServerLog(msg, 'info');
        }
    });

    serverProc.stderr.on('data', (chunk) => {
        const msg = chunk.toString().trim();
        if (msg) {
            writeLog('INFO', '[Server:stderr]', msg);
            // uvicorn logs to stderr by default
            const type = /\bERROR\b|Traceback|Exception:/i.test(msg) ? 'error' : 'info';
            throttledServerLog(msg, type);
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

// ── Window creation (pure UI — no IPC registration) ──────────────

function createWindow() {
    const iconPath = isDev
        ? path.join(__dirname, '../public/icon-512.png')
        : path.join(__dirname, '../dist/icon-512.png');

    const mainWindow = new BrowserWindow({
        title: 'Imagine',
        icon: iconPath,
        width: 1280,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.cjs'),
            nodeIntegration: false,
            contextIsolation: true,
            webSecurity: false,
            sandbox: false,
        },
    });

    if (isDev) {
        mainWindow.loadURL('http://localhost:9274');
        mainWindow.webContents.openDevTools();
    } else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }
}

// ── App lifecycle ────────────────────────────────────────────────

app.setName('Imagine');

app.whenReady().then(async () => {
    // Migrate personal settings from config.yaml to user-settings.yaml (first run only)
    migrateUserSettings();

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

    // Kill any orphaned search daemons from previous crashed sessions
    cleanupOrphanDaemons();

    // Server is started by the React app via IPC (window.electron.server.start)
    // when user selects "관리" mode on SetupPage. No config.yaml auto-start.

    // Do NOT start search daemon here — it starts lazily on first search
    createWindow();

    // Initialize auto-updater (after window is ready to receive events)
    initAutoUpdater();

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
    killWorkerProc();
    killServerProc();
});

// Ensure daemon cleanup on unexpected termination signals
process.on('SIGINT', () => {
    killActivePipeline();
    killSearchDaemon();
    killWorkerProc();
    killServerProc();
    app.quit();
});

process.on('SIGTERM', () => {
    killActivePipeline();
    killSearchDaemon();
    killWorkerProc();
    killServerProc();
    app.quit();
});
