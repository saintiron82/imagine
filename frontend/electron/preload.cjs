const { contextBridge, ipcRenderer } = require('electron');
const fs = require('fs/promises');
const path = require('path');
const os = require('os');

const isDev = process.env.NODE_ENV === 'development';
const projectRoot = isDev
    ? path.resolve(__dirname, '../../')
    : process.resourcesPath;

contextBridge.exposeInMainWorld('electron', {
    versions: {
        node: process.versions.node,
        chrome: process.versions.chrome,
        electron: process.versions.electron,
    },
    projectRoot,
    fs: {
        // Get Home Directory
        getHomeDir: () => os.homedir(),

        // List Directory Contents
        listDir: async (dirPath) => {
            try {
                const items = await fs.readdir(dirPath, { withFileTypes: true });
                return items.map(item => ({
                    name: item.name,
                    isDirectory: item.isDirectory(),
                    path: path.join(dirPath, item.name),
                    extension: path.extname(item.name).toLowerCase(),
                })).filter(item => !item.name.startsWith('.')); // Hide dotfiles
            } catch (error) {
                console.error("Failed to read directory:", error);
                throw error;
            }
        },

        // Check if path exists
        exists: async (pathToCheck) => {
            try {
                await fs.access(pathToCheck);
                return true;
            } catch {
                return false;
            }
        },

        // Resolve path
        pathJoin: (...args) => path.join(...args),

        // Show file in OS file explorer (folder opens, file selected)
        showInFolder: (fullPath) => ipcRenderer.invoke('show-item-in-folder', fullPath),

        // Open file with OS default application
        openFile: (fullPath) => ipcRenderer.invoke('open-file-native', fullPath),
    },

    // Pipeline Bridge
    pipeline: {
        run: (filePaths) => ipcRenderer.send('run-pipeline', { filePaths }),
        stop: () => ipcRenderer.send('stop-pipeline'),
        onLog: (callback) => ipcRenderer.on('pipeline-log', (_, data) => callback(data)),
        offLog: () => ipcRenderer.removeAllListeners('pipeline-log'),
        onProgress: (callback) => ipcRenderer.on('pipeline-progress', (_, data) => callback(data)),
        offProgress: () => ipcRenderer.removeAllListeners('pipeline-progress'),
        onStep: (callback) => ipcRenderer.on('pipeline-step', (_, data) => callback(data)),
        offStep: () => ipcRenderer.removeAllListeners('pipeline-step'),
        onFileDone: (callback) => ipcRenderer.on('pipeline-file-done', (_, data) => callback(data)),
        offFileDone: () => ipcRenderer.removeAllListeners('pipeline-file-done'),
        onBatchDone: (callback) => ipcRenderer.on('pipeline-batch-done', (_, data) => callback(data)),
        offBatchDone: () => ipcRenderer.removeAllListeners('pipeline-batch-done'),
        openFolderDialog: () => ipcRenderer.invoke('open-folder-dialog'),
        generateThumbnail: (filePath) => ipcRenderer.invoke('generate-thumbnail', filePath),
        generateThumbnailsBatch: (filePaths) => ipcRenderer.invoke('generate-thumbnails-batch', filePaths),
        checkThumbnailsExist: (filePaths) => ipcRenderer.invoke('check-thumbnails-exist', filePaths),
        readMetadata: (filePath) => ipcRenderer.invoke('read-metadata', filePath),
        checkMetadataExists: (filePaths) => ipcRenderer.invoke('check-metadata-exists', filePaths),
        searchVector: (options) => ipcRenderer.invoke('search-vector', options),
        fetchImageUrl: (url) => ipcRenderer.invoke('fetch-image-url', url),
        getDbStats: () => ipcRenderer.invoke('get-db-stats'),
        getIncompleteStats: () => ipcRenderer.invoke('get-incomplete-stats'),
        getFolderPhaseStats: (storageRoot) => ipcRenderer.invoke('get-folder-phase-stats', storageRoot),

        // Installer
        checkEnv: () => ipcRenderer.invoke('check-env'),
        installEnv: () => ipcRenderer.send('install-env'),
        onInstallLog: (callback) => ipcRenderer.on('install-log', (_, data) => callback(data)),
        offInstallLog: () => ipcRenderer.removeAllListeners('install-log'),

        // Config Management
        getConfig: () => ipcRenderer.invoke('get-config'),
        updateConfig: (key, value) => ipcRenderer.invoke('update-config', key, value),

        // Registered Folders
        getRegisteredFolders: () => ipcRenderer.invoke('get-registered-folders'),
        addRegisteredFolder: () => ipcRenderer.invoke('add-registered-folder'),
        removeRegisteredFolder: (folderPath) => ipcRenderer.invoke('remove-registered-folder', folderPath),

        // Discover (DFS folder scan)
        runDiscover: (opts) => ipcRenderer.send('run-discover', opts),
        onDiscoverLog: (cb) => ipcRenderer.on('discover-log', (_, data) => cb(data)),
        offDiscoverLog: () => ipcRenderer.removeAllListeners('discover-log'),
        onDiscoverProgress: (cb) => ipcRenderer.on('discover-progress', (_, data) => cb(data)),
        offDiscoverProgress: () => ipcRenderer.removeAllListeners('discover-progress'),
        onDiscoverFileDone: (cb) => ipcRenderer.on('discover-file-done', (_, data) => cb(data)),
        offDiscoverFileDone: () => ipcRenderer.removeAllListeners('discover-file-done'),
    },

    // Job Queue (server mode — direct DB, bypassing HTTP auth)
    queue: {
        registerPaths: (filePaths, priority) =>
            ipcRenderer.invoke('queue-register-paths', { filePaths, priority }),
        scanFolder: (folderPath, priority) =>
            ipcRenderer.invoke('queue-scan-folder', { folderPath, priority }),
        getStats: () => ipcRenderer.invoke('queue-stats'),
        listJobs: (opts) => ipcRenderer.invoke('queue-list-jobs', opts || {}),
        cancelJob: (jobId) => ipcRenderer.invoke('queue-cancel-job', { jobId }),
        retryFailed: () => ipcRenderer.invoke('queue-retry-failed'),
        clearCompleted: () => ipcRenderer.invoke('queue-clear-completed'),
    },

    // DB Import/Export
    db: {
        selectArchive: () => ipcRenderer.invoke('select-archive-file'),
        exportDatabase: (outputPath) => ipcRenderer.invoke('export-database', { outputPath }),
        relinkPreview: (packagePath, targetFolder) =>
            ipcRenderer.invoke('relink-preview', { packagePath, targetFolder }),
        relinkApply: (packagePath, targetFolder, deleteMissing) =>
            ipcRenderer.invoke('relink-apply', { packagePath, targetFolder, deleteMissing }),
    },

    // User Metadata API
    metadata: {
        updateUserData: (filePath, updates) =>
            ipcRenderer.invoke('metadata:updateUserData', filePath, updates)
    },

    // Server Mode (embedded FastAPI)
    server: {
        start: (opts) => ipcRenderer.invoke('server-start', opts),
        stop: () => ipcRenderer.invoke('server-stop'),
        getStatus: () => ipcRenderer.invoke('server-status'),
        onLog: (cb) => ipcRenderer.on('server-log', (_, data) => cb(data)),
        offLog: () => ipcRenderer.removeAllListeners('server-log'),
        onStatusChange: (cb) => ipcRenderer.on('server-status-change', (_, data) => cb(data)),
        offStatusChange: () => ipcRenderer.removeAllListeners('server-status-change'),
    },

    // Worker Daemon (server-mode job processing)
    worker: {
        start: (opts) => ipcRenderer.invoke('worker-start', opts),
        stop: () => ipcRenderer.invoke('worker-stop'),
        getStatus: () => ipcRenderer.invoke('worker-status'),
        onStatus: (cb) => ipcRenderer.on('worker-status', (_, data) => cb(data)),
        offStatus: () => ipcRenderer.removeAllListeners('worker-status'),
        onLog: (cb) => ipcRenderer.on('worker-log', (_, data) => cb(data)),
        offLog: () => ipcRenderer.removeAllListeners('worker-log'),
        onJobDone: (cb) => ipcRenderer.on('worker-job-done', (_, data) => cb(data)),
        offJobDone: () => ipcRenderer.removeAllListeners('worker-job-done'),
    }
});
