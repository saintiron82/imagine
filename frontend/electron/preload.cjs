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
    app: {
        quit: () => ipcRenderer.send('app-quit'),
    },
    auth: {
        googleOAuth: () => ipcRenderer.invoke('google-oauth'),
    },
    worker: {
        start: (opts) => ipcRenderer.invoke('worker-start', opts),
        stop: () => ipcRenderer.invoke('worker-stop'),
        status: () => ipcRenderer.invoke('worker-status'),
        onStatus: (cb) => { ipcRenderer.removeAllListeners('worker-status'); ipcRenderer.on('worker-status', (_, d) => cb(d)); },
        onLog: (cb) => { ipcRenderer.removeAllListeners('worker-log'); ipcRenderer.on('worker-log', (_, d) => cb(d)); },
        onJobDone: (cb) => { ipcRenderer.removeAllListeners('worker-job-done'); ipcRenderer.on('worker-job-done', (_, d) => cb(d)); },
        onBatchFileDone: (cb) => { ipcRenderer.removeAllListeners('worker-batch-file-done'); ipcRenderer.on('worker-batch-file-done', (_, d) => cb(d)); },
        onProcessingMode: (cb) => { ipcRenderer.removeAllListeners('worker-processing-mode'); ipcRenderer.on('worker-processing-mode', (_, d) => cb(d)); },
        offStatus: () => ipcRenderer.removeAllListeners('worker-status'),
        offLog: () => ipcRenderer.removeAllListeners('worker-log'),
        offJobDone: () => ipcRenderer.removeAllListeners('worker-job-done'),
        offBatchFileDone: () => ipcRenderer.removeAllListeners('worker-batch-file-done'),
        offProcessingMode: () => ipcRenderer.removeAllListeners('worker-processing-mode'),
    },
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
        generateThumbnailsAndParse: (filePaths) => ipcRenderer.invoke('generate-thumbnails-and-parse', filePaths),
        checkThumbnailsExist: (filePaths) => ipcRenderer.invoke('check-thumbnails-exist', filePaths),
        thumbQueue: {
            load: () => ipcRenderer.invoke('thumb-queue-load'),
            save: (queue) => ipcRenderer.invoke('thumb-queue-save', queue),
        },
        // downloadCache removed — managed by server queue lifecycle
        readMetadata: (filePath) => ipcRenderer.invoke('read-metadata', filePath),
        checkMetadataExists: (filePaths) => ipcRenderer.invoke('check-metadata-exists', filePaths),
        checkPhaseStatus: (filePaths) => ipcRenderer.invoke('check-phase-status', filePaths),
        fixRelativePaths: () => ipcRenderer.invoke('fix-relative-paths'),
        searchVector: (options) => ipcRenderer.invoke('search-vector', options),
        onSearchProgress: (callback) => {
            const handler = (_, stage) => callback(stage);
            ipcRenderer.on('search-progress', handler);
            return () => ipcRenderer.removeListener('search-progress', handler);
        },
        fetchImageUrl: (url) => ipcRenderer.invoke('fetch-image-url', url),
        getDbStats: () => ipcRenderer.invoke('get-db-stats'),
        getIncompleteStats: () => ipcRenderer.invoke('get-incomplete-stats'),
        getFolderPhaseStats: (storageRoot) => ipcRenderer.invoke('get-folder-phase-stats', storageRoot),

        // Archive Browse
        getArchiveFolders: () => ipcRenderer.invoke('archive-get-folders'),
        getArchiveFiles: (params) => ipcRenderer.invoke('archive-get-files', params),
        getArchiveImageTypes: () => ipcRenderer.invoke('archive-get-image-types'),

        // Installer
        checkEnv: () => ipcRenderer.invoke('check-env'),
        installEnv: () => ipcRenderer.send('install-env'),
        onInstallLog: (callback) => ipcRenderer.on('install-log', (_, data) => callback(data)),
        offInstallLog: () => ipcRenderer.removeAllListeners('install-log'),

        // Config Management
        getConfig: () => ipcRenderer.invoke('get-config'),
        updateConfig: (key, value) => ipcRenderer.invoke('update-config', key, value),
        getUserSettingsPath: () => ipcRenderer.invoke('get-user-settings-path'),

        // Classification Domains
        listDomains: () => ipcRenderer.invoke('list-domains'),
        getDomainDetail: (domainId) => ipcRenderer.invoke('get-domain-detail', domainId),
        saveDomainYaml: (domainId, yamlContent) => ipcRenderer.invoke('save-domain-yaml', domainId, yamlContent),

        // Registered Folders
        getRegisteredFolders: () => ipcRenderer.invoke('get-registered-folders'),
        addRegisteredFolder: () => ipcRenderer.invoke('add-registered-folder'),
        addRegisteredWebDAVFolder: (webdavPath) => ipcRenderer.invoke('add-registered-webdav-folder', webdavPath),
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

    // Legacy queue IPC removed — all queue operations via HTTP API (Analysis Job System v1).

    // DB Import/Export
    db: {
        selectArchive: () => ipcRenderer.invoke('select-archive-file'),
        exportDatabase: (outputPath) => ipcRenderer.invoke('export-database', { outputPath }),
        relinkPreview: (packagePath, targetFolder) =>
            ipcRenderer.invoke('relink-preview', { packagePath, targetFolder }),
        relinkApply: (packagePath, targetFolder, deleteMissing) =>
            ipcRenderer.invoke('relink-apply', { packagePath, targetFolder, deleteMissing }),
    },

    // mDNS Server Discovery (LAN auto-discovery)
    mdns: {
        startBrowse: () => ipcRenderer.invoke('mdns-start-browse'),
        stopBrowse: () => ipcRenderer.invoke('mdns-stop-browse'),
        getServers: () => ipcRenderer.invoke('mdns-get-servers'),
        onServerEvent: (cb) => ipcRenderer.on('mdns-server-event', (_, data) => cb(data)),
        offServerEvent: () => ipcRenderer.removeAllListeners('mdns-server-event'),
    },

    // WebDAV Remote Sources
    webdav: {
        getSources: () => ipcRenderer.invoke('get-webdav-sources'),
        addSource: (config) => ipcRenderer.invoke('add-webdav-source', config),
        removeSource: (sourceId) => ipcRenderer.invoke('remove-webdav-source', sourceId),
        testConnection: (config) => ipcRenderer.invoke('test-webdav-connection', config),
        processSource: (sourceId) => ipcRenderer.send('process-webdav-source', sourceId),
        browseFolders: (config) => ipcRenderer.invoke('browse-webdav-folders', config),
        listDir: (config) => ipcRenderer.invoke('list-webdav-dir', config),
        generateThumbnail: (config) => ipcRenderer.invoke('generate-webdav-thumbnail', config),
        browseFolder: (config) => ipcRenderer.invoke('browse-webdav-folder', config),
        onBrowseEvent: (cb) => ipcRenderer.on('webdav-browse-event', (_, data) => cb(data)),
        offBrowseEvent: () => ipcRenderer.removeAllListeners('webdav-browse-event'),
        onSyncProgress: (cb) => ipcRenderer.on('webdav-sync-progress', (_, data) => cb(data)),
        offSyncProgress: () => ipcRenderer.removeAllListeners('webdav-sync-progress'),
        onSyncComplete: (cb) => ipcRenderer.on('webdav-sync-complete', (_, data) => cb(data)),
        offSyncComplete: () => ipcRenderer.removeAllListeners('webdav-sync-complete'),
    },

    // Folder Sync (DB ↔ disk reconciliation)
    sync: {
        scanFolder: (folderPath) => ipcRenderer.invoke('sync-folder', { folderPath }),
        applyMoves: (moves) => ipcRenderer.invoke('sync-apply-moves', { moves }),
        deleteMissing: (fileIds) => ipcRenderer.invoke('sync-delete-missing', { fileIds }),
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

    // Cloudflare Quick Tunnel (internet access)
    tunnel: {
        start: (opts) => ipcRenderer.invoke('tunnel-start', opts),
        stop: () => ipcRenderer.invoke('tunnel-stop'),
        getStatus: () => ipcRenderer.invoke('tunnel-status'),
        onStatusChange: (cb) => ipcRenderer.on('tunnel-status-change', (_, data) => cb(data)),
        offStatusChange: () => ipcRenderer.removeAllListeners('tunnel-status-change'),
    },

    // Network utilities (LAN IP for group registration)
    network: {
        getLocalIp: () => {
            const nets = os.networkInterfaces();
            for (const name of Object.keys(nets)) {
                for (const net of nets[name]) {
                    if (net.family === 'IPv4' && !net.internal) {
                        return net.address;
                    }
                }
            }
            return null;
        },
    },

    // License (deviceId generation + future license check)
    license: {
        get: () => ipcRenderer.invoke('license-get'),
        set: (data) => ipcRenderer.invoke('license-set', data),
        clear: () => ipcRenderer.invoke('license-clear'),
        getConfig: () => ipcRenderer.invoke('license-get-config'),
    },

    // Menu Actions (main → renderer)
    onMenuAction: (cb) => ipcRenderer.on('menu-action', (_, action) => cb(action)),
    offMenuAction: () => ipcRenderer.removeAllListeners('menu-action'),

    // Auto-Updater
    updater: {
        check: () => ipcRenderer.invoke('updater-check'),
        download: () => ipcRenderer.invoke('updater-download'),
        quitAndInstall: () => ipcRenderer.send('updater-quit-and-install'),
        getVersion: () => ipcRenderer.invoke('updater-get-version'),

        onChecking: (cb) => ipcRenderer.on('update-checking', (_, data) => cb(data)),
        offChecking: () => ipcRenderer.removeAllListeners('update-checking'),
        onAvailable: (cb) => ipcRenderer.on('update-available', (_, data) => cb(data)),
        offAvailable: () => ipcRenderer.removeAllListeners('update-available'),
        onNotAvailable: (cb) => ipcRenderer.on('update-not-available', (_, data) => cb(data)),
        offNotAvailable: () => ipcRenderer.removeAllListeners('update-not-available'),
        onProgress: (cb) => ipcRenderer.on('update-download-progress', (_, data) => cb(data)),
        offProgress: () => ipcRenderer.removeAllListeners('update-download-progress'),
        onDownloaded: (cb) => ipcRenderer.on('update-downloaded', (_, data) => cb(data)),
        offDownloaded: () => ipcRenderer.removeAllListeners('update-downloaded'),
        onError: (cb) => ipcRenderer.on('update-error', (_, data) => cb(data)),
        offError: () => ipcRenderer.removeAllListeners('update-error'),
    },
});
