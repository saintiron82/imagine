const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
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
    const { spawn } = require('child_process');

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
    const { spawn } = require('child_process');

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
// Accepts either:
//   - string query (backward compatible): searchVector("dragon")
//   - object options: searchVector({query, limit, mode, filters})
ipcMain.handle('search-vector', async (_, searchOptions) => {
    const { spawn } = require('child_process');

    const finalPython = resolvePython();

    const scriptPath = isDev
        ? path.resolve(__dirname, '../../backend/api_search.py')
        : path.join(process.resourcesPath, 'backend/api_search.py');

    // Normalize input: string → object
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

    return new Promise((resolve) => {
        // Use stdin JSON protocol
        const proc = spawn(finalPython, [scriptPath], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONPATH: projectRoot, PYTHONIOENCODING: 'utf-8' }
        });

        let output = '';
        let error = '';

        proc.stdout.on('data', (data) => output += data.toString());
        proc.stderr.on('data', (data) => error += data.toString());

        proc.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (e) {
                    console.error('[Search JSON Parse Error]', e, output);
                    resolve({ success: false, error: 'JSON parse error', results: [] });
                }
            } else {
                console.error('[Search Execution Error]', error);
                resolve({ success: false, error: error || 'Search failed', results: [] });
            }
        });

        proc.on('error', (err) => {
            console.error('[Search Spawn Error]', err);
            resolve({ success: false, error: err.message, results: [] });
        });

        // Send search options via stdin
        proc.stdin.write(JSON.stringify(inputData));
        proc.stdin.end();
    });
});

// IPC Handler: Database Stats (archived image count)
ipcMain.handle('get-db-stats', async () => {
    const { spawn } = require('child_process');
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
    const { spawn } = require('child_process');
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
    const { spawn } = require('child_process');
    const finalPython = resolvePython();
    const scriptPath = isDev ? path.resolve(__dirname, '../../backend/setup/installer.py') : path.join(process.resourcesPath, 'backend/setup/installer.py');

    event.reply('install-log', { message: '🚀 Starting installation...', type: 'info' });

    // Run install + download model
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
    const { spawn } = require('child_process');

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

        // Send input data via stdin
        const inputData = JSON.stringify({ file_path: filePath, updates });
        proc.stdin.write(inputData);
        proc.stdin.end();
    });
});

function createWindow() {
    const mainWindow = new BrowserWindow({
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
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    } else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }

    // IPC Handler: Run Python Pipeline
    ipcMain.on('run-pipeline', (event, { filePaths }) => {
        const { spawn } = require('child_process');

        const finalPython = resolvePython();

        const scriptPath = isDev
            ? path.resolve(__dirname, '../../backend/pipeline/ingest_engine.py')
            : path.join(process.resourcesPath, 'backend/ingest_engine.py');

        event.reply('pipeline-log', { message: `Starting batch processing: ${filePaths.length} files...`, type: 'info' });

        // Track progress
        let processedCount = 0;
        const totalFiles = filePaths.length;

        const proc = spawn(finalPython, [scriptPath, '--files', JSON.stringify(filePaths)], { cwd: projectRoot });

        proc.stdout.on('data', (data) => {
            const message = data.toString().trim();
            if (!message) return;

            // Detect file processing progress
            const processingMatch = message.match(/Processing: (.+)/);
            const stepMatch = message.match(/STEP (\d+)\/(\d+) (.+)/);
            if (processingMatch) {
                event.reply('pipeline-progress', {
                    processed: processedCount,
                    total: totalFiles,
                    currentFile: path.basename(processingMatch[1])
                });
            } else if (stepMatch) {
                // Per-file step progress: STEP 1/5 Parsing
                event.reply('pipeline-step', {
                    step: parseInt(stepMatch[1]),
                    totalSteps: parseInt(stepMatch[2]),
                    stepName: stepMatch[3]
                });
            } else if (message.includes('[OK] Parsed successfully')) {
                processedCount++;
                event.reply('pipeline-progress', {
                    processed: processedCount,
                    total: totalFiles,
                    currentFile: ''
                });
            }

            // Only forward log-worthy messages (reduces ~22 lines/file to ~4)
            const isLogWorthy = /^Processing:|^\[OK\]|^\[FAIL\]|^\[DONE\]|^\[DISCOVER\]|^\[SKIP\]|^\[BATCH\]|^\[TIER/.test(message);
            if (isLogWorthy) {
                event.reply('pipeline-log', { message, type: 'info' });
            }
        });

        proc.stderr.on('data', (data) => {
            const message = data.toString().trim();
            if (!message) return;
            // Only forward actual errors, drop library noise (transformers/torch/tqdm warnings)
            const isError = /\bERROR\b|Traceback|Exception:|raise\s|FAIL/i.test(message);
            if (isError) {
                event.reply('pipeline-log', { message, type: 'error' });
            }
        });

        proc.on('close', (code) => {
            // Final progress update
            event.reply('pipeline-progress', {
                processed: totalFiles,
                total: totalFiles,
                currentFile: ''
            });

            event.reply('pipeline-log', {
                message: code === 0 ? '✅ Pipeline complete!' : `⚠️ Pipeline exited with code ${code}`,
                type: code === 0 ? 'success' : 'error'
            });

            // Dedicated completion signal for queue advancement
            event.reply('pipeline-file-done', {
                success: code === 0,
                filePaths: filePaths
            });
        });

        proc.on('error', (err) => {
            event.reply('pipeline-log', { message: `Pipeline error: ${err.message}`, type: 'error' });
        });
    });

    // IPC Handler: Run discover (DFS folder scan)
    ipcMain.on('run-discover', (event, { folderPath, noSkip }) => {
        const { spawn } = require('child_process');

        const finalPython = resolvePython();

        const scriptPath = isDev
            ? path.resolve(__dirname, '../../backend/pipeline/ingest_engine.py')
            : path.join(process.resourcesPath, 'backend/ingest_engine.py');

        const args = [scriptPath, '--discover', folderPath];
        if (noSkip) args.push('--no-skip');

        event.reply('discover-log', { message: `Scanning folder: ${folderPath}`, type: 'info' });

        let processedCount = 0;

        const proc = spawn(finalPython, args, { cwd: projectRoot });

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

            // Only forward log-worthy messages
            const isLogWorthy = /^Processing:|^\[OK\]|^\[FAIL\]|^\[DONE\]|^\[DISCOVER\]|^\[SKIP\]|^\[BATCH\]|^\[TIER/.test(message);
            if (isLogWorthy) {
                event.reply('discover-log', { message, type: 'info' });
            }
        });

        proc.stderr.on('data', (data) => {
            const message = data.toString().trim();
            if (!message) return;
            // Only forward actual errors, drop library noise
            const isError = /\bERROR\b|Traceback|Exception:|raise\s|FAIL/i.test(message);
            if (isError) {
                event.reply('discover-log', { message, type: 'error' });
            }
        });

        proc.on('close', (code) => {
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
            event.reply('discover-log', { message: `Discover error: ${err.message}`, type: 'error' });
            event.reply('discover-file-done', { success: false, folderPath, processedCount: 0 });
        });
    });
}

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

        // Update nested key (e.g., "ai_mode.override")
        const keys = key.split('.');
        let current = config;
        for (let i = 0; i < keys.length - 1; i++) {
            if (!current[keys[i]]) current[keys[i]] = {};
            current = current[keys[i]];
        }
        current[keys[keys.length - 1]] = value;

        // Write back to file
        const newYaml = yaml.dump(config, { lineWidth: -1 });
        fs.writeFileSync(configPath, newYaml, 'utf8');

        return { success: true };
    } catch (err) {
        console.error('[Update Config Error]', err);
        return { success: false, error: err.message };
    }
});

app.whenReady().then(() => {
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
