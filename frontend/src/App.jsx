import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import FileGrid from './components/FileGrid';
import StatusBar from './components/StatusBar';
import SearchPanel from './components/SearchPanel';
import FolderInfoBar from './components/FolderInfoBar';
import ResumeDialog from './components/ResumeDialog';
import ImportDbDialog from './components/ImportDbDialog';
import LoginPage from './pages/LoginPage';
import AdminPage from './pages/AdminPage';
import WorkerPage from './pages/WorkerPage';
import SetupPage from './pages/SetupPage';
import { FolderOpen, Play, Search, Archive, Globe, Database, Upload, Download, Settings, LogOut, User, Server, Power, Copy, Monitor } from 'lucide-react';
import { useLocale } from './i18n';
import { useAuth } from './contexts/AuthContext';
import { isElectron } from './api/client';
import { registerPaths, scanFolder } from './api/admin';

function App() {
  const { t, locale, setLocale, availableLocales } = useLocale();
  const { user, loading: authLoading, isAuthenticated, isAdmin, skipAuth, logout } = useAuth();
  const [currentTab, setCurrentTab] = useState('search'); // 'search' | 'archive' | 'worker' | 'admin'
  const [currentPath, setCurrentPath] = useState('');
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [logs, setLogs] = useState([]);
  const [pendingSearch, setPendingSearch] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processProgress, setProcessProgress] = useState({
    processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0,
    // Cumulative per-phase counts
    cumParse: 0, cumMC: 0, cumVV: 0, cumMV: 0,
    // Active phase sub-progress
    activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0,
    batchInfo: '',
  });
  const [fileStep, setFileStep] = useState({ step: 0, totalSteps: 5, stepName: '' });
  const etaRef = useRef({ startTime: null, lastFileTime: null, emaMs: null });
  const phaseEtaRef = useRef({ phase: -1, startTime: null, startCount: 0 });
  const discoverQueueRef = useRef({ folders: [], index: 0, scanning: false });
  const discoverEtaRef = useRef({ phase: -1, startTime: null, startCount: 0 });
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [discoverProgress, setDiscoverProgress] = useState({
    processed: 0, total: 0, skipped: 0, currentFile: '', folderPath: '',
    cumParse: 0, cumMC: 0, cumVV: 0, cumMV: 0,
    activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '',
    etaMs: null
  });
  const [selectedPaths, setSelectedPaths] = useState(new Set());
  const [resumeStats, setResumeStats] = useState(null);
  const [showResumeDialog, setShowResumeDialog] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showDbMenu, setShowDbMenu] = useState(false);
  const [folderStatsVersion, setFolderStatsVersion] = useState(0);

  // App mode: 'server' | 'client' | null (not configured)
  const [appMode, setAppMode] = useState(isElectron ? null : 'web');
  const [appModeLoading, setAppModeLoading] = useState(isElectron);

  // Server mode state (Electron only)
  const [serverRunning, setServerRunning] = useState(false);
  const [serverPort, setServerPort] = useState(8000);

  // Load app mode from config.yaml (Electron only)
  useEffect(() => {
    if (!isElectron) return;
    const loadMode = async () => {
      try {
        const result = await window.electron?.pipeline?.getConfig();
        if (result?.success && result.config?.app?.mode) {
          setAppMode(result.config.app.mode);
        }
        // If no mode set, appMode stays null → show SetupPage
      } catch (e) {
        console.error('Failed to load app mode:', e);
      }
      setAppModeLoading(false);
    };
    loadMode();
  }, []);

  const handleSetupComplete = (mode, serverUrl) => {
    setAppMode(mode);
    // Reload the app to apply the new mode (server needs to start, etc.)
    if (window.electron?.app?.relaunch) {
      window.electron.app.relaunch();
    } else {
      window.location.reload();
    }
  };

  const MAX_LOGS = 200;

  const appendLog = (data) => {
    const entry = { ...data, timestamp: new Date().toLocaleTimeString() };
    setLogs((prev) => {
      const next = [...prev, entry];
      return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
    });
  };

  // Initialize with Home Directory & stable IPC listeners (never removed during app lifetime)
  useEffect(() => {
    if (window.electron) {
      if (window.electron.fs) {
        setCurrentPath(window.electron.fs.getHomeDir());
      }
    } else {
      // Web mode: start at server root
      setCurrentPath('/');
    }

    if (window.electron) {

      if (window.electron.pipeline) {
        // Stable log listener - feeds StatusBar (never removed mid-session)
        window.electron.pipeline.onLog((data) => {
          appendLog(data);
        });

        // Per-file step progress listener
        window.electron.pipeline.onStep((data) => {
          setFileStep(data);
        });

        // Progress updates (cumulative phase counts + sub-progress + phase-based ETA)
        window.electron.pipeline.onProgress((data) => {
          setProcessProgress(prev => {
            const next = {
              ...prev,
              processed: data.processed ?? prev.processed,
              skipped: data.skipped ?? prev.skipped,
              currentFile: data.currentFile ?? prev.currentFile,
              cumParse: data.cumParse ?? prev.cumParse,
              cumMC: data.cumMC ?? prev.cumMC,
              cumVV: data.cumVV ?? prev.cumVV,
              cumMV: data.cumMV ?? prev.cumMV,
              activePhase: data.activePhase ?? prev.activePhase,
              phaseSubCount: data.phaseSubCount ?? prev.phaseSubCount,
              phaseSubTotal: data.phaseSubTotal ?? prev.phaseSubTotal,
              batchInfo: data.batchInfo ?? prev.batchInfo,
            };

            // Phase-based ETA: track progress rate per active phase
            const pe = phaseEtaRef.current;
            const ap = next.activePhase;
            const counts = [next.cumParse, next.cumMC, next.cumVV, next.cumMV];
            const count = counts[ap] || 0;
            const effectiveTotal = next.total - (next.skipped || 0);

            // Reset timer on phase transition
            if (pe.phase !== ap) {
              pe.phase = ap;
              pe.startTime = Date.now();
              pe.startCount = count;
            }

            const done = count - pe.startCount;
            const remaining = effectiveTotal - count;
            if (pe.startTime && done > 0 && remaining > 0) {
              const elapsed = Date.now() - pe.startTime;
              next.etaMs = (elapsed / done) * remaining;
            } else {
              next.etaMs = null;
            }

            return next;
          });
        });

        // File-done: update ETA using EMA
        window.electron.pipeline.onFileDone((data) => {
          const eta = etaRef.current;
          if (eta.startTime) {
            const now = Date.now();
            const fileDuration = now - eta.lastFileTime;
            eta.lastFileTime = now;
            const ALPHA = 0.3;
            eta.emaMs = eta.emaMs === null ? fileDuration : ALPHA * fileDuration + (1 - ALPHA) * eta.emaMs;
          }
          setProcessProgress(prev => {
            const newProcessed = data.processed ?? prev.processed;
            const remaining = prev.total - (prev.skipped ?? 0) - newProcessed;
            const etaMs = (eta.emaMs && remaining > 0) ? remaining * eta.emaMs : null;
            return { ...prev, processed: newProcessed, etaMs };
          });
        });

        // Batch done: reset all processing state
        window.electron.pipeline.onBatchDone((data) => {
          setIsProcessing(false);
          setProcessProgress({ processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0, cumParse: 0, cumMC: 0, cumVV: 0, cumMV: 0, activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '' });
          setFileStep({ step: 0, totalSteps: 5, stepName: '' });
          etaRef.current = { startTime: null, lastFileTime: null, emaMs: null };
          const msg = data.skipped > 0
            ? `All done! ${data.processed} processed, ${data.skipped} skipped (total: ${data.total})`
            : `All ${data.processed} files processed!`;
          appendLog({ message: msg, type: 'success' });
          setFolderStatsVersion(v => v + 1);
        });

        // Discover event listeners (for auto-scan)
        window.electron.pipeline.onDiscoverLog((data) => {
          appendLog(data);
        });
        window.electron.pipeline.onDiscoverProgress((data) => {
          setDiscoverProgress(prev => {
            const next = { ...prev, ...data };

            // Phase-based ETA (same logic as pipeline mode)
            const de = discoverEtaRef.current;
            const ap = next.activePhase ?? 0;
            const counts = [next.cumParse || 0, next.cumMC || 0, next.cumVV || 0, next.cumMV || 0];
            const count = counts[ap];
            const effectiveTotal = (next.total || 0) - (next.skipped || 0);

            if (de.phase !== ap) {
              de.phase = ap;
              de.startTime = Date.now();
              de.startCount = count;
            }

            const done = count - de.startCount;
            const remaining = effectiveTotal - count;
            if (de.startTime && done > 0 && remaining > 0) {
              const elapsed = Date.now() - de.startTime;
              next.etaMs = (elapsed / done) * remaining;
            } else {
              next.etaMs = null;
            }

            return next;
          });
        });
        window.electron.pipeline.onDiscoverFileDone((data) => {
          setFolderStatsVersion(v => v + 1);
          // Auto-scan processes folders sequentially via discoverQueueRef
          const ref = discoverQueueRef.current;
          if (ref.scanning && ref.index < ref.folders.length - 1) {
            discoverQueueRef.current.index++;
            const nextFolder = ref.folders[ref.index + 1];
            window.electron.pipeline.runDiscover({ folderPath: nextFolder, noSkip: false });
          } else {
            setIsDiscovering(false);
            setDiscoverProgress({
              processed: 0, total: 0, skipped: 0, currentFile: '', folderPath: '',
              cumParse: 0, cumMC: 0, cumVV: 0, cumMV: 0,
              activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '',
              etaMs: null
            });
            discoverEtaRef.current = { phase: -1, startTime: null, startCount: 0 };
            // Clear session: all folders processed successfully
            window.electron.pipeline.updateConfig('last_session.folders', []);
          }
        });
      }
    }

    // Check for incomplete work from last session on startup
    const checkIncompleteOnStartup = async () => {
      try {
        // Read last session folders from config
        const configResult = await window.electron?.pipeline?.getConfig();
        const lastFolders = configResult?.config?.last_session?.folders || [];
        if (lastFolders.length === 0) return; // No interrupted session

        const stats = await window.electron?.pipeline?.getIncompleteStats();
        if (!stats?.success || stats.total_incomplete === 0) return;

        // Filter: only folders under last session targets
        const filtered = stats.folders.filter(f => {
          const sr = f.storage_root.normalize('NFC');
          return lastFolders.some(lf => {
            const nLf = lf.normalize('NFC');
            return sr === nLf || sr.startsWith(nLf.endsWith('/') ? nLf : nLf + '/');
          });
        });
        if (filtered.length === 0) return;

        const totalInc = filtered.reduce((s, f) => s + f.incomplete, 0);
        const totalFiles = filtered.reduce((s, f) => s + f.total, 0);
        setResumeStats({ total_incomplete: totalInc, total_files: totalFiles, folders: filtered });
        setShowResumeDialog(true);
      } catch (e) {
        console.error('Incomplete check failed:', e);
      }
    };
    checkIncompleteOnStartup();

    return () => {
      if (window.electron?.pipeline) {
        window.electron.pipeline.offLog();
        window.electron.pipeline.offStep();
        window.electron.pipeline.offProgress();
        window.electron.pipeline.offFileDone();
        window.electron.pipeline.offBatchDone();
        window.electron.pipeline.offDiscoverLog();
        window.electron.pipeline.offDiscoverProgress();
        window.electron.pipeline.offDiscoverFileDone();
      }
    };
  }, []);

  // Server mode IPC listeners (Electron only)
  useEffect(() => {
    if (!isElectron || !window.electron?.server) return;
    // Check initial status
    window.electron.server.getStatus().then(s => setServerRunning(s.running));
    // Listen for status changes (e.g. server process exit)
    window.electron.server.onStatusChange((data) => setServerRunning(data.running));
    return () => window.electron.server.offStatusChange();
  }, []);

  const handleServerToggle = async () => {
    if (!isElectron) return;
    if (serverRunning) {
      await window.electron.server.stop();
      setServerRunning(false);
    } else {
      const result = await window.electron.server.start({ port: serverPort });
      if (result.success) {
        setServerRunning(true);
      }
    }
  };

  const getLocalIp = () => {
    // Simple heuristic — return hostname for display
    return `${window.location.hostname || 'localhost'}:${serverPort}`;
  };

  const handleFolderSelect = (path) => {
    setCurrentPath(path);
    setSelectedPaths(new Set()); // Clear Ctrl-selection on normal click
    setSelectedFiles(new Set()); // Clear selection on folder change
  };

  const handleFolderToggle = (path) => {
    setSelectedPaths(prev => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
    setSelectedFiles(new Set());
  };

  // Send all selected files to backend in one batch call
  const handleProcess = async () => {
    if (selectedFiles.size === 0 || isProcessing) return;

    const fileArray = Array.from(selectedFiles);

    // Guardrail: selected-file processing may not cover whole folder/subfolders.
    // If counts differ, warn that folder-level rebuild badge can remain.
    try {
      if (currentPath && window.electron?.pipeline?.getFolderPhaseStats) {
        const statsResult = await window.electron.pipeline.getFolderPhaseStats(currentPath);
        if (statsResult?.success && Array.isArray(statsResult.folders)) {
          const folderTotal = statsResult.folders.reduce((acc, f) => acc + (f.total || 0), 0);
          if (folderTotal > fileArray.length) {
            appendLog({
              message: `Selected ${fileArray.length}/${folderTotal} files only. Folder rebuild status may remain. Use "Process All (incl. subfolders)".`,
              type: 'warning'
            });
          }
        }
      }
    } catch { }

    setIsProcessing(true);
    etaRef.current = { startTime: Date.now(), lastFileTime: Date.now(), emaMs: null };
    setProcessProgress({ processed: 0, total: fileArray.length, currentFile: '', etaMs: null, skipped: 0, cumParse: 0, cumMC: 0, cumVV: 0, cumMV: 0, activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '' });
    setSelectedFiles(new Set());

    appendLog({
      message: `Starting batch: ${fileArray.length} files`,
      type: 'info'
    });

    // Server/Client mode: register files via API → queue for workers
    if (appMode === 'server' || appMode === 'client') {
      try {
        const result = await registerPaths(fileArray);
        appendLog({
          message: `Queued ${result.jobs_created || 0} jobs (${result.registered || 0} files registered)`,
          type: 'success'
        });
        setIsProcessing(false);
        setProcessProgress(prev => ({ ...prev, processed: 0, total: 0 }));
      } catch (e) {
        appendLog({ message: `Queue registration failed: ${e.message}`, type: 'error' });
        setIsProcessing(false);
      }
      return;
    }

    // Standalone mode: direct pipeline spawn
    if (window.electron?.pipeline) {
      window.electron.pipeline.run(fileArray);
    }
  };

  const handleStopProcess = () => {
    // Kill the actual Python process
    window.electron?.pipeline?.stop();
    setIsProcessing(false);
    setProcessProgress({ processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0, cumParse: 0, cumMC: 0, cumVV: 0, cumMV: 0, activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '' });
    setFileStep({ step: 0, totalSteps: 5, stepName: '' });
    etaRef.current = { startTime: null, lastFileTime: null, emaMs: null };
  };

  // Process entire folder recursively (discover mode)
  const handleProcessFolder = async (folderPath, options = {}) => {
    if (isProcessing || isDiscovering) return;
    const noSkip = !!options.noSkip;
    setIsDiscovering(true);
    setCurrentTab('archive');
    setCurrentPath(folderPath);
    appendLog({
      message: noSkip
        ? `Processing folder (no-skip): ${folderPath}`
        : `Processing folder: ${folderPath}`,
      type: 'info'
    });

    // Server/Client mode: scan folder via API → queue for workers
    if (appMode === 'server' || appMode === 'client') {
      try {
        const result = await scanFolder(folderPath);
        appendLog({
          message: `Discovered ${result.discovered || 0} files, queued ${result.jobs_created || 0} jobs (${result.skipped || 0} skipped)`,
          type: 'success'
        });
      } catch (e) {
        appendLog({ message: `Folder scan failed: ${e.message}`, type: 'error' });
      }
      setIsDiscovering(false);
      return;
    }

    // Standalone mode: direct discover spawn
    window.electron?.pipeline?.updateConfig('last_session.folders', [folderPath]);
    window.electron?.pipeline?.runDiscover({ folderPath, noSkip });
  };

  // Resume incomplete work: discover only folders that have incomplete files
  const handleResume = () => {
    setShowResumeDialog(false);
    if (!resumeStats?.folders?.length) return;

    // Use individual storage_root paths (only incomplete folders, not parent roots)
    const incompleteFolders = resumeStats.folders.map(f => f.storage_root.normalize('NFC'));
    if (incompleteFolders.length === 0) return;

    setIsDiscovering(true);
    setCurrentTab('archive');
    setCurrentPath(incompleteFolders[0]);
    discoverQueueRef.current = { folders: incompleteFolders, index: 0, scanning: true };
    // Save session target for resume on next startup
    window.electron.pipeline.updateConfig('last_session.folders', incompleteFolders);
    appendLog({
      message: `Resuming: ${resumeStats.total_incomplete} incomplete files in ${incompleteFolders.length} folder(s)`,
      type: 'info'
    });
    window.electron.pipeline.runDiscover({ folderPath: incompleteFolders[0], noSkip: false });
  };

  const handleDismissResume = () => {
    setShowResumeDialog(false);
  };

  // Scan folders from RegisteredFoldersPanel (via Settings → SearchPanel)
  const handleScanFolders = (folderPaths) => {
    if (isProcessing || isDiscovering || !folderPaths?.length) return;
    setIsDiscovering(true);
    setCurrentTab('archive');
    setCurrentPath(folderPaths[0]);
    discoverQueueRef.current = { folders: folderPaths, index: 0, scanning: folderPaths.length > 1 };
    window.electron?.pipeline?.updateConfig('last_session.folders', folderPaths);
    appendLog({ message: `Scanning ${folderPaths.length} folder(s)...`, type: 'info' });
    window.electron?.pipeline?.runDiscover({ folderPath: folderPaths[0], noSkip: false });
  };

  const handleExportDb = async () => {
    setShowDbMenu(false);
    try {
      const result = await window.electron?.db?.exportDatabase();
      if (result?.success) {
        appendLog({ message: `DB exported: ${result.file_count} files, ${result.size_mb} MB → ${result.output_path}`, type: 'success' });
      } else if (result?.error) {
        appendLog({ message: `Export failed: ${result.error}`, type: 'error' });
      }
    } catch (e) {
      appendLog({ message: `Export error: ${e.message}`, type: 'error' });
    }
  };

  const handleImportProcessNew = (folderPath) => {
    handleProcessFolder(folderPath);
  };

  const handleFindSimilar = (searchParams) => {
    setPendingSearch(searchParams);
    setCurrentTab('search');
  };

  const clearLogs = () => setLogs([]);

  const localeLabel = locale === 'ko-KR' ? 'KR' : 'EN';

  // App mode loading state (Electron only)
  if (appModeLoading || authLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-gray-400 text-sm">{t('status.loading')}</div>
      </div>
    );
  }

  // First-run setup: no mode selected yet (Electron only)
  if (isElectron && !appMode) {
    return <SetupPage onComplete={handleSetupComplete} />;
  }

  // Show login page for web mode when not authenticated
  if (!skipAuth && !isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <div className="flex h-screen bg-gray-900 text-white overflow-hidden flex-col">
      {/* Resume Dialog */}
      {showResumeDialog && (
        <ResumeDialog
          stats={resumeStats}
          onResume={handleResume}
          onDismiss={handleDismissResume}
        />
      )}

      {showImportDialog && (
        <ImportDbDialog
          onClose={() => setShowImportDialog(false)}
          onProcessNew={handleImportProcessNew}
        />
      )}

      {/* Header Bar */}
      <div className="h-14 border-b border-gray-700 flex items-center px-4 justify-between bg-gray-800 shadow-sm z-10 shrink-0">
        {/* Left: App Name */}
        <div className="flex items-center space-x-2">
          <Search className="text-blue-400" size={20} />
          <h1 className="font-bold text-lg">{t('app.title')}</h1>
        </div>

        {/* Right: Tab Buttons + Process (in archive mode) + Language */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setCurrentTab('search')}
            className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${currentTab === 'search'
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
          >
            <Search size={16} />
            <span>{t('tab.search')}</span>
          </button>
          <button
            onClick={() => setCurrentTab('archive')}
            className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${currentTab === 'archive'
              ? 'bg-gray-700 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
          >
            <Archive size={16} />
            <span>{t('tab.archive')}</span>
          </button>

          {/* Worker tab (authenticated users) */}
          {!skipAuth && isAuthenticated && (
            <button
              onClick={() => setCurrentTab('worker')}
              className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${currentTab === 'worker'
                ? 'bg-emerald-700 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
            >
              <Server size={16} />
              <span>{t('tab.worker')}</span>
            </button>
          )}

          {/* Admin tab (admin users only, web mode) */}
          {isAdmin && !skipAuth && (
            <button
              onClick={() => setCurrentTab('admin')}
              className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${currentTab === 'admin'
                ? 'bg-purple-700 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
            >
              <Settings size={16} />
              <span>{t('tab.admin')}</span>
            </button>
          )}

          {currentTab === 'archive' && (
            <>
              <div className="w-px h-6 bg-gray-600 mx-1" />
              <div className="text-xs text-gray-500">
                {t('status.selected', { count: selectedFiles.size })}
              </div>
              <button
                onClick={handleProcess}
                disabled={selectedFiles.size === 0 || isProcessing}
                className={`
                  flex items-center space-x-1 px-4 py-1.5 rounded text-sm font-medium transition-colors
                  ${selectedFiles.size > 0 && !isProcessing
                    ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/50'
                    : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  }
                `}
              >
                <Play size={14} fill="currentColor" />
                <span>{t('action.process', { count: selectedFiles.size })}</span>
              </button>
            </>
          )}

          {/* DB Import/Export Menu */}
          <div className="w-px h-6 bg-gray-600 mx-1" />
          <div className="relative">
            <button
              onClick={() => setShowDbMenu(prev => !prev)}
              className="flex items-center space-x-1 px-2 py-1.5 rounded text-xs font-medium text-gray-400 hover:text-white hover:bg-gray-700/50 transition-colors"
              title="Database"
            >
              <Database size={14} />
              <span>DB</span>
            </button>
            {showDbMenu && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setShowDbMenu(false)} />
                <div className="absolute right-0 top-full mt-1 z-50 bg-gray-800 border border-gray-600 rounded-lg shadow-xl py-1 min-w-[180px]">
                  <button
                    onClick={handleExportDb}
                    className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                  >
                    <Download size={14} />
                    {t('action.export_db')}
                  </button>
                  <button
                    onClick={() => { setShowDbMenu(false); setShowImportDialog(true); }}
                    className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                  >
                    <Upload size={14} />
                    {t('action.import_db')}
                  </button>
                </div>
              </>
            )}
          </div>

          {/* Server Mode Toggle (Electron only) */}
          {isElectron && (
            <>
              <div className="w-px h-6 bg-gray-600 mx-1" />
              <div className="flex items-center gap-1.5">
                <button
                  onClick={handleServerToggle}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium transition-colors ${
                    serverRunning
                      ? 'bg-green-700/60 text-green-300 hover:bg-green-600/60'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                  }`}
                  title={serverRunning ? t('server.stop') : t('server.start')}
                >
                  <Monitor size={14} />
                  <span>{t('server.mode')}</span>
                  <Power size={12} className={serverRunning ? 'text-green-400' : 'text-gray-500'} />
                </button>
                {serverRunning && (
                  <button
                    onClick={() => navigator.clipboard?.writeText(`http://localhost:${serverPort}`)}
                    className="flex items-center gap-1 px-1.5 py-1 rounded text-[10px] text-green-400 hover:bg-green-900/30 transition-colors"
                    title={t('server.copy_url')}
                  >
                    <span className="font-mono">:{serverPort}</span>
                    <Copy size={10} />
                  </button>
                )}
              </div>
            </>
          )}

          {/* Language Switcher */}
          <div className="w-px h-6 bg-gray-600 mx-1" />
          <button
            onClick={() => setLocale(locale === 'en-US' ? 'ko-KR' : 'en-US')}
            className="flex items-center space-x-1 px-2 py-1.5 rounded text-xs font-medium text-gray-400 hover:text-white hover:bg-gray-700/50 transition-colors"
            title="Switch language"
          >
            <Globe size={14} />
            <span>{localeLabel}</span>
          </button>

          {/* User info + Logout (web mode only) */}
          {!skipAuth && user && (
            <>
              <div className="w-px h-6 bg-gray-600 mx-1" />
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <User size={14} />
                <span>{user.username}</span>
              </div>
              <button
                onClick={logout}
                className="flex items-center space-x-1 px-2 py-1.5 rounded text-xs font-medium text-gray-400 hover:text-red-400 hover:bg-gray-700/50 transition-colors"
                title={t('auth.logout')}
              >
                <LogOut size={14} />
              </button>
            </>
          )}
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Folder Tree (only in archive mode) */}
        {currentTab === 'archive' && (
          <div className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
            <div className="p-4 border-b border-gray-700 flex items-center space-x-2">
              <FolderOpen className="text-blue-400" size={20} />
              <span className="font-medium text-sm text-gray-300">{t('label.folders')}</span>
            </div>
            <div className="flex-1 overflow-y-auto p-2">
              <Sidebar
                currentPath={currentPath}
                onFolderSelect={handleFolderSelect}
                selectedPaths={selectedPaths}
                onFolderToggle={handleFolderToggle}
                reloadSignal={folderStatsVersion}
              />
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="flex-1 flex flex-col bg-gray-900 relative">
          {/* Content Area */}
          <div className="flex-1 overflow-hidden">
            {currentTab === 'admin' && isAdmin ? (
              <AdminPage />
            ) : currentTab === 'worker' ? (
              <WorkerPage appMode={appMode} />
            ) : currentTab === 'search' ? (
              <SearchPanel
                onScanFolder={handleScanFolders}
                isBusy={isProcessing || isDiscovering}
                initialSearch={pendingSearch}
                onSearchConsumed={() => setPendingSearch(null)}
                reloadSignal={folderStatsVersion}
              />
            ) : (
              <div className="h-full flex flex-col">
                <FolderInfoBar
                  currentPath={currentPath}
                  onProcessFolder={handleProcessFolder}
                  isProcessing={isProcessing || isDiscovering}
                  reloadSignal={folderStatsVersion}
                />
                <div className="flex-1 overflow-y-auto p-4 pb-16">
                  <FileGrid
                    currentPath={currentPath}
                    selectedFiles={selectedFiles}
                    setSelectedFiles={setSelectedFiles}
                    selectedPaths={selectedPaths}
                    onFindSimilar={handleFindSimilar}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Logs Overlay / Bottom Panel */}
          <StatusBar
            logs={logs}
            clearLogs={clearLogs}
            isProcessing={isProcessing}
            isDiscovering={isDiscovering}
            discoverProgress={discoverProgress}
            processed={processProgress.processed}
            total={processProgress.total}
            skipped={processProgress.skipped}
            currentFile={processProgress.currentFile}
            etaMs={processProgress.etaMs}
            cumParse={processProgress.cumParse}
            cumMC={processProgress.cumMC}
            cumVV={processProgress.cumVV}
            cumMV={processProgress.cumMV}
            activePhase={processProgress.activePhase}
            phaseSubCount={processProgress.phaseSubCount}
            phaseSubTotal={processProgress.phaseSubTotal}
            batchInfo={processProgress.batchInfo}
            fileStep={fileStep}
            onStop={handleStopProcess}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
