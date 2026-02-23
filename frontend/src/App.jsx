import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import StatusBar from './components/StatusBar';
import SearchPanel from './components/SearchPanel';
import ResumeDialog from './components/ResumeDialog';
import ImportDbDialog from './components/ImportDbDialog';
import ServerArchiveView from './components/ServerArchiveView';
import ClientWorkerView from './components/ClientWorkerView';
import LoginPage from './pages/LoginPage';
import AdminPage from './pages/AdminPage';
import SetupPage from './pages/SetupPage';
import DownloadPage from './pages/DownloadPage';
import AppDownloadBanner from './components/AppDownloadBanner';
import { FolderOpen, Play, Search, Archive, Zap, Globe, Database, Upload, Download, Settings, LogOut, User, Server, Power, Copy, Monitor, Wifi, Info } from 'lucide-react';
import ServerInfoPanel from './components/ServerInfoPanel';
import { useLocale } from './i18n';
import { useAuth } from './contexts/AuthContext';
import { isElectron, setServerUrl, getServerUrl, getAccessToken, getRefreshToken, clearTokens } from './api/client';
import { getWorkerCredentials } from './api/auth';
import { setUseLocalBackend } from './services/bridge';
import { registerPaths, scanFolder, getJobStats } from './api/admin';

function App() {
  const { t, locale, setLocale, availableLocales } = useLocale();
  const { user, loading: authLoading, isAuthenticated, isAdmin, skipAuth, logout, login, configureAuth } = useAuth();
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
  const [queueReloadSignal, setQueueReloadSignal] = useState(0);
  const [showDownloadPage, setShowDownloadPage] = useState(false);

  // Worker progress state (client mode)
  const [isWorkerRunning, setIsWorkerRunning] = useState(false);
  const [workerProgress, setWorkerProgress] = useState({
    batchSize: 0,
    currentPhase: '',       // "parse" | "vision" | "embed_vv" | "embed_mv" | "uploading"
    phaseIndex: 0,          // 1-based progress within current phase
    phaseCount: 0,          // total files in current phase
    currentFile: '',
    completed: 0,
    totalQueue: 0, pending: 0,
    etaMs: null, throughput: 0,  // overall items/min
    processingMode: 'full',     // "full" | "mc_only" — controls phase pill dimming
    workerState: 'active',      // "active" | "idle" | "resting" — from state machine
    // Per-phase speed (files/min) — updated on phase_complete
    phaseFpm: { parse: 0, vision: 0, embed_vv: 0, embed_mv: 0 },
    // Per-phase elapsed (seconds) — updated on phase_complete
    phaseElapsed: { parse: 0, vision: 0, embed_vv: 0, embed_mv: 0 },
  });
  const workerThroughputRef = useRef({ windowTimes: [] });

  // App mode: 'server' | 'client' | null (show SetupPage) | 'web'
  // Electron: always starts null → SetupPage shown every launch
  const [appMode, setAppMode] = useState(isElectron ? null : 'web');

  // Server mode state (Electron only)
  const [serverRunning, setServerRunning] = useState(false);
  const [serverPort, setServerPort] = useState(8000);
  const [serverLanUrl, setServerLanUrl] = useState(null);
  const [serverLanAddresses, setServerLanAddresses] = useState([]);
  const [showServerInfo, setShowServerInfo] = useState(false);
  const [tunnelRunning, setTunnelRunning] = useState(false);
  const [tunnelUrl, setTunnelUrl] = useState(null);
  const [tunnelDownloading, setTunnelDownloading] = useState(false);

  // Load server port from config.yaml (Electron only, mode is NOT loaded — SetupPage decides)
  useEffect(() => {
    if (!isElectron) return;
    const loadConfig = async () => {
      try {
        const result = await window.electron?.pipeline?.getConfig();
        if (result?.success) {
          const port = result.config?.server?.port || 8000;
          setServerPort(port);
        }
      } catch (e) {
        console.error('Failed to load config:', e);
      }
    };
    loadConfig();
  }, []);

  const handleSetupComplete = (mode, serverUrl) => {
    setAppMode(mode);
    setUseLocalBackend(mode === 'server');

    if (mode === 'server') {
      setServerUrl(`http://localhost:${serverPort}`);
    } else if (mode === 'client' && serverUrl) {
      setServerUrl(serverUrl);
    }

    // Switch auth: server → local bypass, client → JWT required
    configureAuth(mode);
  };

  const handleModeReset = () => {
    setAppMode(null); // Show SetupPage
    setUseLocalBackend(false);
    configureAuth(null); // Reset to local bypass
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
    // Check initial status (includes LAN addresses)
    window.electron.server.getStatus().then(s => {
      setServerRunning(s.running);
      if (s.running) {
        setServerLanUrl(s.primaryLanUrl || null);
        setServerLanAddresses(s.lanAddresses || []);
      }
    });
    // Listen for status changes (e.g. server process exit)
    window.electron.server.onStatusChange((data) => {
      setServerRunning(data.running);
      if (data.running) {
        serverStartAttemptRef.current = 0; // Reset cooldown on success
      } else {
        setServerLanUrl(null);
        setServerLanAddresses([]);
        setTunnelRunning(false);
        setTunnelUrl(null);
      }
    });

    // Tunnel status
    if (window.electron?.tunnel) {
      window.electron.tunnel.getStatus().then(s => {
        setTunnelRunning(s.running);
        setTunnelUrl(s.url || null);
      });
      window.electron.tunnel.onStatusChange((data) => {
        if (data.downloading) {
          setTunnelDownloading(true);
        } else {
          setTunnelDownloading(false);
          setTunnelRunning(data.running);
          setTunnelUrl(data.url || null);
        }
      });
    }

    return () => {
      window.electron.server.offStatusChange();
      window.electron?.tunnel?.offStatusChange();
    };
  }, []);

  // Auto-start server when entering server mode (with cooldown to prevent restart loops)
  const serverStartAttemptRef = useRef(0);
  useEffect(() => {
    if (!isElectron || appMode !== 'server' || serverRunning) return;
    if (!window.electron?.server) return;

    // Cooldown: max 3 attempts within 30 seconds
    const now = Date.now();
    const MAX_ATTEMPTS = 3;
    const COOLDOWN_MS = 30000;
    if (serverStartAttemptRef.current >= MAX_ATTEMPTS) return;

    const autoStart = async () => {
      serverStartAttemptRef.current++;
      try {
        const status = await window.electron.server.getStatus();
        if (status.running) {
          setServerRunning(true);
          return;
        }
        const result = await window.electron.server.start({ port: serverPort });
        if (result.success) {
          setServerRunning(true);
          setServerLanUrl(result.primaryLanUrl || null);
          setServerLanAddresses(result.lanAddresses || []);
          serverStartAttemptRef.current = 0; // Reset on success
        }
      } catch (e) {
        console.warn('Server auto-start failed:', e);
      }
    };

    // Delay restart to avoid tight loops
    const delay = serverStartAttemptRef.current > 1 ? 3000 : 500;
    const timer = setTimeout(autoStart, delay);
    return () => clearTimeout(timer);
  }, [appMode, serverPort, serverRunning]);

  // Worker IPC event listeners (Electron client mode)
  useEffect(() => {
    if (!isElectron || appMode !== 'client') return;
    const w = window.electron?.worker;
    if (!w) return;

    const onStatus = (data) => {
      if (data.status === 'error') {
        setIsWorkerRunning(false);
      } else {
        setIsWorkerRunning(data.status === 'running');
      }
    };

    const onBatchStart = (data) => {
      setWorkerProgress(prev => ({
        ...prev,
        batchSize: data.batch_size,
        currentPhase: 'starting',
        phaseIndex: 0,
      }));
    };

    const onBatchPhaseStart = (data) => {
      setWorkerProgress(prev => ({
        ...prev,
        currentPhase: data.phase,
        phaseIndex: 0,
        phaseCount: data.count,
      }));
    };

    const onBatchFileDone = (data) => {
      setWorkerProgress(prev => ({
        ...prev,
        currentPhase: data.phase,
        phaseIndex: data.index,
        phaseCount: data.count,
        currentFile: data.file_name || prev.currentFile,
      }));
    };

    const onBatchPhaseComplete = (data) => {
      // Update per-phase speed when a phase finishes
      setWorkerProgress(prev => ({
        ...prev,
        phaseFpm: { ...prev.phaseFpm, [data.phase]: data.files_per_min || 0 },
        phaseElapsed: { ...prev.phaseElapsed, [data.phase]: data.elapsed_s || 0 },
      }));
    };

    const onBatchComplete = (data) => {
      // Batch complete — update overall throughput from backend timing
      if (data.phase_fpm) {
        setWorkerProgress(prev => ({
          ...prev,
          phaseFpm: { ...prev.phaseFpm, ...data.phase_fpm },
        }));
      }
    };

    const onJobDone = (data) => {
      if (!data.success) return;
      const now = Date.now();
      const ref = workerThroughputRef.current;
      ref.windowTimes.push(now);
      // Keep 60-second sliding window for items/min calculation
      ref.windowTimes = ref.windowTimes.filter(t => now - t < 60000);
      const elapsedSec = ref.windowTimes.length > 1
        ? (now - ref.windowTimes[0]) / 1000
        : 0;
      // Throughput in items/min
      const throughput = elapsedSec > 0
        ? (ref.windowTimes.length / elapsedSec) * 60
        : 0;

      setWorkerProgress(prev => {
        const newCompleted = prev.completed + 1;
        const remaining = Math.max(0, prev.pending - 1);
        const perSec = throughput / 60;
        const etaMs = perSec > 0 && remaining > 0 ? (remaining / perSec) * 1000 : null;
        return { ...prev, completed: newCompleted, pending: remaining, throughput, etaMs };
      });
    };

    const onProcessingMode = (data) => {
      setWorkerProgress(prev => ({ ...prev, processingMode: data.mode || 'full' }));
    };

    const onWorkerState = (data) => {
      setWorkerProgress(prev => ({ ...prev, workerState: data.state || 'active' }));
    };

    w.onStatus(onStatus);
    w.onBatchStart?.(onBatchStart);
    w.onBatchPhaseStart?.(onBatchPhaseStart);
    w.onBatchFileDone?.(onBatchFileDone);
    w.onBatchPhaseComplete?.(onBatchPhaseComplete);
    w.onBatchComplete?.(onBatchComplete);
    w.onJobDone(onJobDone);
    w.onProcessingMode?.(onProcessingMode);
    w.onWorkerState?.(onWorkerState);

    return () => {
      w.offStatus();
      w.offBatchStart?.();
      w.offBatchPhaseStart?.();
      w.offBatchFileDone?.();
      w.offBatchPhaseComplete?.();
      w.offBatchComplete?.();
      w.offJobDone();
      w.offProcessingMode?.();
      w.offWorkerState?.();
    };
  }, [appMode]);

  // Worker queue stats polling (client mode, 5s interval)
  useEffect(() => {
    if (appMode !== 'client' || !isWorkerRunning) return;
    const fetchStats = async () => {
      try {
        const data = await getJobStats();
        if (data && data.success !== false) {
          setWorkerProgress(prev => ({
            ...prev,
            totalQueue: data.total || 0,
            pending: (data.pending || 0) + (data.assigned || 0) + (data.processing || 0),
          }));
        }
      } catch { /* ignore */ }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, [appMode, isWorkerRunning]);

  // Manual worker start handler (called from ClientWorkerView)
  const handleWorkerStart = async () => {
    if (!isElectron || isWorkerRunning) return;
    const w = window.electron?.worker;
    if (!w) return;

    try {
      const creds = getWorkerCredentials();
      const result = await w.start({
        serverUrl: getServerUrl() || `http://localhost:${serverPort}`,
        accessToken: creds ? '' : (getAccessToken() || ''),   // Only use tokens if no credentials
        refreshToken: creds ? '' : (getRefreshToken() || ''), // Credentials = independent login
        username: creds?.username || '',
        password: creds?.password || '',
      });
      if (result?.success) {
        setIsWorkerRunning(true);
      } else if (result?.error?.includes('already running')) {
        setIsWorkerRunning(true);
      } else if (result?.success === false) {
        appendLog({ message: result.error || 'Failed to start worker', type: 'error' });
      }
    } catch (e) {
      appendLog({ message: `Worker start failed: ${e.message}`, type: 'error' });
    }
  };

  const handleTunnelStart = async () => {
    if (!window.electron?.tunnel) return;
    setTunnelDownloading(true);
    const result = await window.electron.tunnel.start({ port: serverPort });
    setTunnelDownloading(false);
    if (result?.success) {
      setTunnelRunning(true);
      setTunnelUrl(result.url);
    }
  };

  const handleTunnelStop = async () => {
    if (!window.electron?.tunnel) return;
    await window.electron.tunnel.stop();
    setTunnelRunning(false);
    setTunnelUrl(null);
  };

  const handleWorkerStop = async () => {
    if (!isElectron || !window.electron?.worker) return;
    try {
      await window.electron.worker.stop();
      setIsWorkerRunning(false);
      setWorkerProgress({
        batchSize: 0, currentPhase: '', phaseIndex: 0, phaseCount: 0,
        currentFile: '', completed: 0, totalQueue: 0, pending: 0,
        etaMs: null, throughput: 0, processingMode: 'full', workerState: 'active',
        phaseFpm: { parse: 0, vision: 0, embed_vv: 0, embed_mv: 0 },
        phaseElapsed: { parse: 0, vision: 0, embed_vv: 0, embed_mv: 0 },
      });
      workerThroughputRef.current = { windowTimes: [] };
    } catch (e) {
      appendLog({ message: `Worker stop failed: ${e.message}`, type: 'error' });
    }
  };

  const handleServerToggle = async () => {
    if (!isElectron) return;
    if (serverRunning) {
      await window.electron.server.stop();
      setServerRunning(false);
      setServerLanUrl(null);
      setServerLanAddresses([]);
      clearTokens();
      logout(); // reset user state → LoginPage reappears
    } else {
      const result = await window.electron.server.start({ port: serverPort });
      if (result.success) {
        setServerRunning(true);
        setServerLanUrl(result.primaryLanUrl || null);
        setServerLanAddresses(result.lanAddresses || []);
      }
    }
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

    // Server mode Electron: register files into job queue via IPC (direct DB)
    if (appMode === 'server' && isElectron && window.electron?.queue) {
      try {
        const result = await window.electron.queue.registerPaths(fileArray);
        appendLog({
          message: t('archive.queue_registered', { jobs: result.jobs_created || 0 }),
          type: 'success'
        });
        setQueueReloadSignal(prev => prev + 1);
      } catch (e) {
        appendLog({ message: `Queue registration failed: ${e.message}`, type: 'error' });
      }
      setIsProcessing(false);
      setProcessProgress(prev => ({ ...prev, processed: 0, total: 0 }));
      return;
    }

    // Web client mode: register files via server API → queue for workers
    if (!isElectron) {
      try {
        const result = await registerPaths(fileArray);
        appendLog({
          message: t('archive.queue_registered', { jobs: result.jobs_created || 0 }),
          type: 'success'
        });
        setQueueReloadSignal(prev => prev + 1);
      } catch (e) {
        appendLog({ message: `Queue registration failed: ${e.message}`, type: 'error' });
      }
      setIsProcessing(false);
      setProcessProgress(prev => ({ ...prev, processed: 0, total: 0 }));
      return;
    }

    // Electron client mode: direct pipeline spawn (local processing)
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

    // Server mode Electron: scan folder → create jobs in queue via IPC (direct DB)
    if (appMode === 'server' && isElectron && window.electron?.queue) {
      try {
        const result = await window.electron.queue.scanFolder(folderPath);
        appendLog({
          message: t('archive.queue_registered', { jobs: result.jobs_created || 0 }),
          type: 'success'
        });
        setQueueReloadSignal(prev => prev + 1);
      } catch (e) {
        appendLog({ message: `Folder scan failed: ${e.message}`, type: 'error' });
      }
      setIsDiscovering(false);
      return;
    }

    // Web client mode: scan folder via server API → queue for workers
    if (!isElectron) {
      try {
        const result = await scanFolder(folderPath);
        appendLog({
          message: t('archive.queue_registered', { jobs: result.jobs_created || 0 }),
          type: 'success'
        });
        setQueueReloadSignal(prev => prev + 1);
      } catch (e) {
        appendLog({ message: `Folder scan failed: ${e.message}`, type: 'error' });
      }
      setIsDiscovering(false);
      return;
    }

    // Electron client mode: direct discover spawn (local processing)
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

  // Auth loading state
  if (authLoading) {
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

  // Show login page when auth required but not authenticated
  // skipAuth: null=undetermined (still loading), true=bypass, false=JWT required
  if (skipAuth === false && !isAuthenticated) {
    if (showDownloadPage) {
      return <DownloadPage onBack={() => setShowDownloadPage(false)} />;
    }
    return <LoginPage
      onShowDownload={() => setShowDownloadPage(true)}
      serverRunning={serverRunning}
      serverPort={serverPort}
    />;
  }

  // Download page overlay (web mode, authenticated)
  if (showDownloadPage && appMode === 'web') {
    return <DownloadPage onBack={() => setShowDownloadPage(false)} />;
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
        {/* Left: App Name + Mode Badge */}
        <div className="flex items-center space-x-2">
          <Search className="text-blue-400" size={20} />
          <h1 className="font-bold text-lg">{t('app.title')}</h1>
          {isElectron && appMode && (
            <button
              onClick={handleModeReset}
              className={`text-[10px] font-medium px-1.5 py-0.5 rounded cursor-pointer transition-colors ${
                appMode === 'server'
                  ? 'bg-blue-900/50 text-blue-300 hover:bg-blue-800/60'
                  : 'bg-emerald-900/50 text-emerald-300 hover:bg-emerald-800/60'
              }`}
              title={t('setup.changeable_later')}
            >
              {appMode === 'server' ? t('setup.server_title') : t('setup.client_title')}
            </button>
          )}
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
          {/* Archive/Worker tab — role-based: admin=archive, user=worker */}
          <button
            onClick={() => setCurrentTab('archive')}
            className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${currentTab === 'archive'
              ? (isAdmin ? 'bg-gray-700 text-white' : 'bg-emerald-700 text-white')
              : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
          >
            {isAdmin ? <Archive size={16} /> : <Zap size={16} />}
            <span>{isAdmin ? t('tab.archive_server') : t('tab.archive_worker')}</span>
          </button>

          {/* Admin tab — admin role only */}
          {isAdmin && (
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

          {currentTab === 'archive' && isAdmin && (
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
                <span>{t('action.queue_files', { count: selectedFiles.size })}</span>
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
                  <div className="relative">
                    <button
                      onClick={() => setShowServerInfo(!showServerInfo)}
                      className="flex items-center gap-1 px-1.5 py-1 rounded text-[10px] text-green-400 hover:bg-green-900/30 transition-colors"
                      title={t('server.info_title')}
                    >
                      {serverLanUrl ? (
                        <>
                          <Wifi size={10} />
                          <span className="font-mono">{serverLanUrl.replace('http://', '')}</span>
                        </>
                      ) : (
                        <span className="font-mono">:{serverPort}</span>
                      )}
                      <Info size={10} />
                    </button>
                    {showServerInfo && (
                      <ServerInfoPanel
                        serverPort={serverPort}
                        serverLanUrl={serverLanUrl}
                        serverLanAddresses={serverLanAddresses}
                        tunnelUrl={tunnelUrl}
                        tunnelRunning={tunnelRunning}
                        tunnelDownloading={tunnelDownloading}
                        onTunnelStart={handleTunnelStart}
                        onTunnelStop={handleTunnelStop}
                        onClose={() => setShowServerInfo(false)}
                      />
                    )}
                  </div>
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
                onClick={() => { if (isWorkerRunning) handleWorkerStop(); logout(); }}
                className="flex items-center space-x-1 px-2 py-1.5 rounded text-xs font-medium text-gray-400 hover:text-red-400 hover:bg-gray-700/50 transition-colors"
                title={t('auth.logout')}
              >
                <LogOut size={14} />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Download Desktop App Banner (web mode only) */}
      {/* Download banner for browser users (not Electron) */}
      {!isElectron && <AppDownloadBanner onShowDownload={() => setShowDownloadPage(true)} />}

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Folder Tree (admin only) */}
        {currentTab === 'archive' && isAdmin && (
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
            ) : currentTab === 'search' ? (
              <SearchPanel
                onScanFolder={handleScanFolders}
                isBusy={isProcessing || isDiscovering}
                initialSearch={pendingSearch}
                onSearchConsumed={() => setPendingSearch(null)}
                reloadSignal={folderStatsVersion}
              />
            ) : currentTab === 'archive' && !isAdmin ? (
              <ClientWorkerView
                appMode={appMode}
                isWorkerRunning={isWorkerRunning}
                workerProgress={workerProgress}
                onWorkerStart={handleWorkerStart}
                onWorkerStop={handleWorkerStop}
              />
            ) : currentTab === 'archive' && isAdmin ? (
              <ServerArchiveView
                currentPath={currentPath}
                selectedFiles={selectedFiles}
                setSelectedFiles={setSelectedFiles}
                selectedPaths={selectedPaths}
                onProcessFolder={handleProcessFolder}
                onFindSimilar={handleFindSimilar}
                isProcessing={isProcessing || isDiscovering}
                reloadSignal={folderStatsVersion}
                appMode={appMode}
                queueReloadSignal={queueReloadSignal}
              />
            ) : null}
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
            isWorkerProcessing={isWorkerRunning && appMode === 'client'}
            workerProgress={workerProgress}
            onWorkerStop={handleWorkerStop}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
