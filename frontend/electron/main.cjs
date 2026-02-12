const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');
const isDev = process.env.NODE_ENV === 'development';
const projectRoot = isDev
    ? path.resolve(__dirname, '../../')
    : process.resourcesPath;

// Cross-platform Python path resolution
function getPythonPath() {
    const isWin = process.platform === 'win32';
    if (isDev) {
        const venvDir = isWin ? 'Scripts' : 'bin';
        const pyExe = isWin ? 'python.exe' : 'python3';
        return path.resolve(__dirname, `../../.venv/${venvDir}/${pyExe}`);
    }
    const pyExe = isWin ? 'python.exe' : 'python3';
    return path.join(process.resourcesPath, 'python', pyExe);
}

function resolvePython() {
    const pythonPath = getPythonPath();
    return fs.existsSync(pythonPath) ? pythonPath : 'python3';
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
        : path.join(process.resourcesPath, 'backend/api_search.py');
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
            : path.join(process.resourcesPath, 'output/json');
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
        : path.join(process.resourcesPath, 'output/json');

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
        : path.join(process.resourcesPath, 'backend/thumbnail_generator.py');

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
        : path.join(process.resourcesPath, 'backend/thumbnail_generator.py');

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
        : path.join(process.resourcesPath, 'output', 'thumbnails');

    const results = {};
    for (const fp of filePaths) {
        const stem = path.basename(fp, path.extname(fp));
        const thumbPath = path.join(thumbDir, `${stem}_thumb.png`);
        results[fp] = fs.existsSync(thumbPath) ? thumbPath : null;
    }
    return results;
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
        };
    }

    return sendSearchRequest(inputData);
});

// IPC Handler: Database Stats (archived image count)
ipcMain.handle('get-db-stats', async () => {
    const finalPython = resolvePython();
    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_stats.py')
        : path.join(process.resourcesPath, 'backend/api_stats.py');

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

// IPC Handler: Environment Check
ipcMain.handle('check-env', async () => {
    const finalPython = resolvePython();
    const scriptPath = isDev ? path.resolve(__dirname, '../../backend/setup/installer.py') : path.join(process.resourcesPath, 'backend/setup/installer.py');

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
    const scriptPath = isDev ? path.resolve(__dirname, '../../backend/setup/installer.py') : path.join(process.resourcesPath, 'backend/setup/installer.py');

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
        : path.join(process.resourcesPath, 'backend/api_metadata_update.py');

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
        : path.join(process.resourcesPath, 'backend/ingest_engine.py');

    console.log('[run-pipeline] Script:', scriptPath);
    event.reply('pipeline-log', { message: `Starting batch processing: ${filePaths.length} files...`, type: 'info' });

    let processedCount = 0;
    let skippedCount = 0;
    let batchDoneSent = false;
    const totalFiles = filePaths.length;
    pipelineStoppedByUser = false;

    // Cumulative phase tracking — each phase has independent progress
    let cumParse = 0, cumVision = 0, cumEmbed = 0, cumStore = 0;
    // Current active phase within mini-batch (for sub-progress tracking)
    let activePhase = 0; // 0=parse, 1=vision, 2=embed, 3=store
    let phaseSubCount = 0, phaseSubTotal = 0;

    function emitPhaseProgress(extraFields = {}) {
        event.reply('pipeline-progress', {
            processed: processedCount,
            total: totalFiles,
            skipped: skippedCount,
            currentFile: extraFields.currentFile || '',
            // Cumulative per-phase counts
            cumParse, cumVision, cumEmbed, cumStore,
            // Active phase sub-progress (within mini-batch)
            activePhase,
            phaseSubCount, phaseSubTotal,
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
            // Cumulative phase progress: [PHASE] P:40 V:33 E:30 S:30 T:500
            const phaseMatch = clean.match(/^\[PHASE\]\s+P:(\d+)\s+V:(\d+)\s+E:(\d+)\s+S:(\d+)\s+T:(\d+)/);

            // [PHASE] cumulative progress from mini-batch orchestrator
            if (phaseMatch) {
                cumParse = parseInt(phaseMatch[1]);
                cumVision = parseInt(phaseMatch[2]);
                cumEmbed = parseInt(phaseMatch[3]);
                cumStore = parseInt(phaseMatch[4]);
                emitPhaseProgress();
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

            // Log: show STEP, progress, errors, and key events (exclude noisy [PHASE])
            const isLogWorthy = /Processing:|STEP \d|(\[OK\])|(\[FAIL\])|(\[DONE\])|(\[SKIP\])|(\[BATCH\])|(\[MINI\s)|(\[TIER)|(\[\d+\/\d+\])/.test(clean) && !/^\[PHASE\]/.test(clean);
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
            try { activePipelineProc.kill('SIGKILL'); } catch {}
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
        : path.join(process.resourcesPath, 'backend/ingest_engine.py');

    const args = [scriptPath, '--discover', folderPath];
    if (noSkip) args.push('--no-skip');

    event.reply('discover-log', { message: `Scanning folder: ${folderPath}`, type: 'info' });

    let processedCount = 0;

    const proc = spawn(finalPython, args, { cwd: projectRoot, detached: true, env: { ...process.env, PYTHONUNBUFFERED: '1' } });
    activeDiscoverProcs.set(folderPath, proc);

    proc.stdout.on('data', (data) => {
        const message = data.toString().trim();
        if (!message) return;

        const processingMatch = message.match(/Processing: (.+)/);
        const stepMatch = message.match(/STEP (\d+)\/(\d+) (.+)/);
        if (processingMatch) {
            event.reply('discover-progress', {
                processed: processedCount,
                currentFile: path.basename(processingMatch[1]),
                folderPath
            });
        } else if (stepMatch) {
            event.reply('discover-progress', {
                processed: processedCount,
                step: parseInt(stepMatch[1]),
                totalSteps: parseInt(stepMatch[2]),
                stepName: stepMatch[3],
                folderPath
            });
        } else if (message.includes('[OK] Parsed successfully') || message.includes('[SKIP]')) {
            processedCount++;
            event.reply('discover-progress', {
                processed: processedCount,
                currentFile: '',
                folderPath
            });
        }

        const isLogWorthy = /^Processing:|^\[OK\]|^\[FAIL\]|^\[DONE\]|^\[DISCOVER\]|^\[SKIP\]|^\[BATCH\]|^\[TIER/.test(message);
        if (isLogWorthy) {
            event.reply('discover-log', { message, type: 'info' });
        }
    });

    proc.stderr.on('data', (data) => {
        const message = data.toString().trim();
        if (!message) return;
        const isError = /\bERROR\b|Traceback|Exception:|raise\s|FAIL/i.test(message);
        if (isError) {
            event.reply('discover-log', { message, type: 'error' });
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
        const configPath = path.join(projectRoot, 'config.yaml');

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
        const configPath = path.join(projectRoot, 'config.yaml');
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
        const configPath = path.join(projectRoot, 'config.yaml');
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
        const configPath = path.join(projectRoot, 'config.yaml');
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
        const configPath = path.join(projectRoot, 'config.yaml');

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

app.whenReady().then(() => {
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
            try { activePipelineProc.kill('SIGKILL'); } catch {}
        }
        activePipelineProc = null;
    }
    for (const [, proc] of activeDiscoverProcs) {
        try {
            process.kill(-proc.pid, 'SIGKILL');
        } catch {
            try { proc.kill('SIGKILL'); } catch {}
        }
    }
    activeDiscoverProcs.clear();
}

app.on('before-quit', () => {
    killActivePipeline();
    killSearchDaemon();
});

// Ensure daemon cleanup on unexpected termination signals
process.on('SIGINT', () => {
    killActivePipeline();
    killSearchDaemon();
    app.quit();
});

process.on('SIGTERM', () => {
    killActivePipeline();
    killSearchDaemon();
    app.quit();
});
