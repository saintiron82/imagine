const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');
const isDev = process.env.NODE_ENV === 'development';

// Suppress EPIPE errors from console.log when parent pipe is closed (background launch)
process.stdout?.on?.('error', (err) => { if (err.code !== 'EPIPE') throw err; });
process.stderr?.on?.('error', (err) => { if (err.code !== 'EPIPE') throw err; });
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

/** Kill any orphaned Imagine-Search processes from previous runs. */
function cleanupOrphanDaemons() {
    try {
        if (process.platform === 'win32') {
            // Windows: taskkill by window title or image name
            execSync('taskkill /F /FI "WINDOWTITLE eq Imagine-Search" 2>nul', { stdio: 'ignore' });
        } else {
            // macOS/Linux: pkill by process name set via setproctitle
            execSync('pkill -f "Imagine-Search" 2>/dev/null || true', { stdio: 'ignore' });
        }
    } catch (e) {
        // No orphan found — that's fine
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
        env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
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

// ── IPC Handlers (global scope — registered once) ────────────────

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
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' }
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
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' }
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
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' }
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
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' }
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
    });
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

// IPC Handler: Stop running pipeline (kill entire process group to avoid orphans)
ipcMain.on('stop-pipeline', () => {
    if (activePipelineProc) {
        pipelineStoppedByUser = true;
        const pid = activePipelineProc.pid;
        // Kill entire process group (detached=true gives child its own PGID=PID)
        try {
            process.kill(-pid, 'SIGKILL');
        } catch {
            // Fallback: kill just the main process
            try { activePipelineProc.kill('SIGKILL'); } catch { }
        }
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

    const proc = spawn(finalPython, args, { cwd: projectRoot, detached: true, env: { ...process.env, PYTHONUNBUFFERED: '1' } });
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

// IPC Handler: Get config.yaml
ipcMain.handle('get-config', async () => {
    try {
        const yaml = require('js-yaml');
        const configPath = path.join(configRoot, 'config.yaml');

        if (!fs.existsSync(configPath)) {
            return { success: false, error: 'config.yaml not found' };
        }

        const fileContents = fs.readFileSync(configPath, 'utf8');
        const config = yaml.load(fileContents);

        return { success: true, config };
    } catch (err) {
        console.error('[Get Config Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Get registered folders from config.yaml
ipcMain.handle('get-registered-folders', async () => {
    try {
        const yaml = require('js-yaml');
        const configPath = path.join(configRoot, 'config.yaml');
        if (!fs.existsSync(configPath)) {
            return { success: true, folders: [], autoScan: true };
        }
        const config = yaml.load(fs.readFileSync(configPath, 'utf8'));
        const regFolders = config.registered_folders || { folders: [], auto_scan: true };
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

// IPC Handler: Add registered folders (opens multi-select dialog)
ipcMain.handle('add-registered-folder', async () => {
    try {
        const result = await dialog.showOpenDialog({
            properties: ['openDirectory', 'multiSelections'],
            title: 'Select Folders to Register'
        });
        if (result.canceled || result.filePaths.length === 0) {
            return { success: true, added: [] };
        }

        const yaml = require('js-yaml');
        const configPath = path.join(configRoot, 'config.yaml');
        const config = yaml.load(fs.readFileSync(configPath, 'utf8'));
        if (!config.registered_folders) config.registered_folders = { folders: [], auto_scan: true };
        if (!config.registered_folders.folders) config.registered_folders.folders = [];

        const existing = new Set(config.registered_folders.folders);
        const added = result.filePaths.filter(fp => !existing.has(fp));
        config.registered_folders.folders.push(...added);

        fs.writeFileSync(configPath, yaml.dump(config, { lineWidth: -1 }), 'utf8');

        const folders = config.registered_folders.folders.map(fp => ({
            path: fp,
            exists: fs.existsSync(fp),
        }));
        return { success: true, added, folders };
    } catch (err) {
        console.error('[Add Registered Folder Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Remove a registered folder
ipcMain.handle('remove-registered-folder', async (_, folderPath) => {
    try {
        const yaml = require('js-yaml');
        const configPath = path.join(configRoot, 'config.yaml');
        const config = yaml.load(fs.readFileSync(configPath, 'utf8'));
        if (!config.registered_folders || !config.registered_folders.folders) {
            return { success: true, folders: [] };
        }
        config.registered_folders.folders = config.registered_folders.folders.filter(fp => fp !== folderPath);
        fs.writeFileSync(configPath, yaml.dump(config, { lineWidth: -1 }), 'utf8');

        const folders = config.registered_folders.folders.map(fp => ({
            path: fp,
            exists: fs.existsSync(fp),
        }));
        return { success: true, folders };
    } catch (err) {
        console.error('[Remove Registered Folder Error]', err);
        return { success: false, error: err.message };
    }
});

// IPC Handler: Update config.yaml
ipcMain.handle('update-config', async (_, key, value) => {
    try {
        const yaml = require('js-yaml');
        const configPath = path.join(configRoot, 'config.yaml');

        if (!fs.existsSync(configPath)) {
            return { success: false, error: 'config.yaml not found' };
        }

        const fileContents = fs.readFileSync(configPath, 'utf8');
        const config = yaml.load(fileContents);

        const keys = key.split('.');
        let current = config;
        for (let i = 0; i < keys.length - 1; i++) {
            if (!current[keys[i]]) current[keys[i]] = {};
            current = current[keys[i]];
        }
        current[keys[keys.length - 1]] = value;

        const newYaml = yaml.dump(config, { lineWidth: -1 });
        fs.writeFileSync(configPath, newYaml, 'utf8');

        return { success: true };
    } catch (err) {
        console.error('[Update Config Error]', err);
        return { success: false, error: err.message };
    }
});

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
            } else if (evt === 'phase_progress') {
                sendWorkerEvent('worker-phase-progress', parsed);
            } else if (evt === 'phase_done') {
                sendWorkerEvent('worker-phase-done', parsed);
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

    try {
        proc.stdin.write(JSON.stringify({ cmd: 'exit' }) + '\n');
    } catch (e) { /* ignore */ }
    setTimeout(() => {
        try { proc.kill('SIGTERM'); } catch (e) { /* already dead */ }
    }, 3000);
}

// IPC Handler: Start worker daemon
ipcMain.handle('worker-start', async (event, opts) => {
    if (workerProc) {
        return { success: false, error: 'Worker already running' };
    }

    const finalPython = resolvePython();
    const scriptPath = getWorkerScriptPath();

    // Store window reference for relaying events
    workerMainWindow = BrowserWindow.fromWebContents(event.sender);

    console.log('[Worker] Starting worker_ipc.py...');

    workerProc = spawn(finalPython, ['-m', 'backend.worker.worker_ipc'], {
        cwd: projectRoot,
        env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
        stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Queue the start command — will be sent after 'ready' event
    // Supports token mode (from existing session) or credential mode
    workerStartCmd = {
        cmd: 'start',
        server_url: opts.serverUrl || 'http://localhost:8000',
        access_token: opts.accessToken || '',
        refresh_token: opts.refreshToken || '',
        username: opts.username || '',
        password: opts.password || '',
    };

    workerProc.stdout.on('data', (chunk) => {
        workerBuffer += chunk.toString();
        processWorkerOutput();
    });

    workerProc.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg) {
            console.error('[Worker:stderr]', msg);
            // Forward errors as log events
            if (/\bERROR\b|Traceback|Exception:|FAIL/i.test(msg)) {
                sendWorkerEvent('worker-log', { message: msg, type: 'error' });
            }
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

/** Start embedded FastAPI server. Returns { success, port } or { success: false, error }. */
function startEmbeddedServer(port = 8000) {
    if (serverProc) return { success: false, error: 'Server already running' };

    const finalPython = resolvePython();
    console.log(`[Server] Starting FastAPI on port ${port}...`);

    serverProc = spawn(finalPython, [
        '-m', 'uvicorn', 'backend.server.app:app',
        '--host', '0.0.0.0', '--port', String(port),
    ], {
        cwd: projectRoot,
        env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' },
        stdio: ['pipe', 'pipe', 'pipe'],
    });

    serverProc.stdout.on('data', (chunk) => {
        const msg = chunk.toString().trim();
        if (msg) {
            console.log('[Server:stdout]', msg);
            try {
                if (serverMainWindow && !serverMainWindow.isDestroyed()) {
                    serverMainWindow.webContents.send('server-log', { message: msg, type: 'info' });
                }
            } catch (e) { /* window may be closed */ }
        }
    });

    serverProc.stderr.on('data', (chunk) => {
        const msg = chunk.toString().trim();
        if (msg) {
            console.log('[Server:stderr]', msg);
            try {
                if (serverMainWindow && !serverMainWindow.isDestroyed()) {
                    // uvicorn logs to stderr by default
                    const type = /\bERROR\b|Traceback|Exception:/i.test(msg) ? 'error' : 'info';
                    serverMainWindow.webContents.send('server-log', { message: msg, type });
                }
            } catch (e) { /* window may be closed */ }
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

    return { success: true, port };
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
    return { running: !!serverProc };
});

function killServerProc() {
    if (!serverProc) return;
    try { serverProc.kill('SIGTERM'); } catch (e) { /* ignore */ }
    serverProc = null;
}

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

// Kill active pipeline/discover process groups (prevents orphans on quit)
function killActivePipeline() {
    if (activePipelineProc) {
        try {
            process.kill(-activePipelineProc.pid, 'SIGKILL');
        } catch {
            try { activePipelineProc.kill('SIGKILL'); } catch { }
        }
        activePipelineProc = null;
    }
    for (const [, proc] of activeDiscoverProcs) {
        try {
            process.kill(-proc.pid, 'SIGKILL');
        } catch {
            try { proc.kill('SIGKILL'); } catch { }
        }
    }
    activeDiscoverProcs.clear();

    if (activeBackfillProc) {
        try {
            process.kill(-activeBackfillProc.pid, 'SIGKILL');
        } catch {
            try { activeBackfillProc.kill('SIGKILL'); } catch { }
        }
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
