import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import StatusBar from './components/StatusBar';
import SearchPanel from './components/SearchPanel';
import ResumeDialog from './components/ResumeDialog';
import ImportDbDialog from './components/ImportDbDialog';
import ServerArchiveView from './components/ServerArchiveView';
import ClientWorkerView from './components/ClientWorkerView';
import LoginPage from './pages/LoginPage';
import LoginPageV2 from './pages/LoginPageV2';
import AdminPage from './pages/AdminPage';
// SetupPage removed — LoginPage is now the first screen
const USE_LOGIN_V2 = true; // Toggle to switch between login page versions
import DownloadPage from './pages/DownloadPage';
import AppDownloadBanner from './components/AppDownloadBanner';
import UpdateNotification from './components/UpdateNotification';
import EolBanner from './components/EolBanner';
import { FolderOpen, Play, Search, Archive, Zap, Globe, Database, Upload, Download, Settings, LogOut, User, Power, Monitor, Wifi, Info, Trash2, ShieldCheck, RotateCcw } from 'lucide-react';
import ServerInfoPanel from './components/ServerInfoPanel';
import { useLocale } from './i18n';
import { useAuth } from './contexts/AuthContext';
import { isElectron, setServerUrl, getServerUrl, getAccessToken, getRefreshToken, clearTokens } from './api/client';
import { getWorkerCredentials } from './api/auth';
import { setUseLocalBackend, getActiveDomainConfig } from './services/bridge';
import { registerPaths, scanFolder, getJobStats, resetDatabase, auditIntegrity, retryFailedJobs } from './api/admin';
import DomainSelectModal from './components/DomainSelectModal';
import SubscriptionBanner from './components/SubscriptionBanner';

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
  const [showDomainSelect, setShowDomainSelect] = useState(false);
  const [showResetDialog, setShowResetDialog] = useState(false);
  const [resetPassword, setResetPassword] = useState('');
  const [resetLoading, setResetLoading] = useState(false);
  const [resetError, setResetError] = useState('');
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditResult, setAuditResult] = useState(null);
  const [retryLoading, setRetryLoading] = useState(false);

  // Worker progress state (client mode)
  const [isWorkerRunning, setIsWorkerRunning] = useState(false);
  const [workerProgress, setWorkerProgress] = useState({
    batchSize: 0,
    currentPhase: '',       // "parse" | "vision" | "embed_vv" | "embed_mv" | "uploading"
    phaseIndex: 0,          // 1-based progress within current phase
    phaseCount: 0,          // total files in current phase
    currentFile: '',
    completed: 0, failed: 0,
    totalQueue: 0, pending: 0,
    etaMs: null, throughput: 0,  // overall items/min
    processingMode: 'full',     // "full" | "mc_only" | "embed_only" — controls phase pill dimming
    workerState: 'active',      // "active" | "idle" | "resting" — from state machine
    // Per-phase speed (files/min) — updated on phase_complete
    phaseFpm: { parse: 0, vision: 0, embed_vv: 0, embed_mv: 0 },
    // Per-phase elapsed (seconds) — updated on phase_complete
    phaseElapsed: { parse: 0, vision: 0, embed_vv: 0, embed_mv: 0 },
  });
  const workerThroughputRef = useRef({ windowTimes: [] });

  // App mode: 'server' | 'client' | 'web'
  // Electron: restore from localStorage, default null (LoginPage decides after login)
  const [appMode, setAppMode] = useState(() => {
    if (!isElectron) return 'web';
    return localStorage.getItem('imagine-app-mode') || null;
  });

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
  // Web mode: restrict tabs to search/download only
  useEffect(() => {
    if (!isElectron && (currentTab === 'archive' || currentTab === 'admin')) {
      setCurrentTab('search');
    }
  }, [currentTab]);

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

  // Ensure deviceId exists on startup (license infra — currently disabled)
  useEffect(() => {
    if (!isElectron) return;
    window.electron?.license?.get().catch(() => {});
  }, []);

  // Electron: auto-start local server on mount (V1 only — V2 handles its own startup)
  useEffect(() => {
    if (!isElectron || USE_LOGIN_V2) return;

    const startLocalServer = async () => {
      const port = serverPort || 8000;
      try {
        const status = await window.electron?.server?.getStatus();
        if (!status?.running) {
          await window.electron?.server?.start({ port });
          await new Promise(r => setTimeout(r, 1500));
        }
        setServerRunning(true);
        setServerUrl(`http://localhost:${port}`);
        const newStatus = await window.electron?.server?.getStatus();
        if (newStatus?.lanAddresses) setServerLanAddresses(newStatus.lanAddresses);
        if (newStatus?.primaryLanUrl) setServerLanUrl(newStatus.primaryLanUrl);
      } catch (e) {
        console.error('Failed to auto-start server:', e);
      }

      // Restore mode if previously set
      if (appMode) {
        setUseLocalBackend(appMode === 'server');
        configureAuth(appMode);
      }
    };

    startLocalServer();
  }, []); // Run once on mount

  /**
   * Called after successful login/init from LoginPage.
   * Determines mode based on user role: admin → server, otherwise → client.
   */
  const handleLoginComplete = async (mode) => {
    setAppMode(mode);
    localStorage.setItem('imagine-app-mode', mode);
    setUseLocalBackend(mode === 'server');

    if (mode === 'server') {
      const port = serverPort || 8000;
      setServerUrl(`http://localhost:${port}`);
      setServerRunning(true);
      try {
        const status = await window.electron?.server?.getStatus();
        if (status?.lanAddresses) setServerLanAddresses(status.lanAddresses);
        if (status?.primaryLanUrl) setServerLanUrl(status.primaryLanUrl);
      } catch { /* ignore */ }
    }

    configureAuth(mode);
  };

  const handleModeReset = async () => {
    // Stop ALL running backend processes before returning to LoginPage
    if (isWorkerRunning) {
      try { await window.electron?.worker?.stop(); } catch { /* ignore */ }
      setIsWorkerRunning(false);
      setWorkerProgress({
        batchSize: 0, currentPhase: '', phaseIndex: 0, phaseCount: 0,
        currentFile: '', completed: 0, failed: 0, totalQueue: 0, pending: 0,
        etaMs: null, throughput: 0, processingMode: 'full', workerState: 'active',
        phaseFpm: { parse: 0, vision: 0, embed_vv: 0, embed_mv: 0 },
        phaseElapsed: { parse: 0, vision: 0, embed_vv: 0, embed_mv: 0 },
      });
      workerThroughputRef.current = { windowTimes: [] };
    }
    if (isProcessing) {
      window.electron?.pipeline?.stop();
      setIsProcessing(false);
      setProcessProgress({ processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0, cumParse: 0, cumMC: 0, cumVV: 0, cumMV: 0, activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '' });
      setFileStep({ step: 0, totalSteps: 5, stepName: '' });
      etaRef.current = { startTime: null, lastFileTime: null, emaMs: null };
    }
    if (isDiscovering) {
      window.electron?.pipeline?.stop();
      setIsDiscovering(false);
      setDiscoverProgress({
        processed: 0, total: 0, skipped: 0, currentFile: '', folderPath: '',
        cumParse: 0, cumMC: 0, cumVV: 0, cumMV: 0,
        activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '',
        etaMs: null
      });
      discoverEtaRef.current = { phase: -1, startTime: null, startCount: 0 };
      discoverQueueRef.current = { folders: [], index: 0, scanning: false };
    }

    setAppMode(null);
    localStorage.removeItem('imagine-app-mode');
    setUseLocalBackend(false);
    clearTokens();
    logout();
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

    // Menu action listener (Electron menu → React handler)
    if (window.electron?.onMenuAction) {
      window.electron.onMenuAction((action) => {
        window.dispatchEvent(new CustomEvent('electron-menu-action', { detail: action }));
      });
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
      window.electron?.offMenuAction?.();
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
      if (!data.success) {
        setWorkerProgress(prev => ({ ...prev, failed: prev.failed + 1 }));
        return;
      }
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
            failed: data.failed || prev.failed,
            etaMs: data.eta_seconds != null ? data.eta_seconds * 1000 : prev.etaMs,
            throughput: data.throughput || prev.throughput,
          }));
        }
      } catch { /* ignore */ }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, [appMode, isWorkerRunning]);

  // Domain setup check — show selection modal if no domain configured
  const domainCheckedRef = useRef(false);
  useEffect(() => {
    if (!appMode || (skipAuth === false && !isAuthenticated)) return;
    if (domainCheckedRef.current) return;
    domainCheckedRef.current = true;

    const checkDomain = async () => {
      try {
        const config = await getActiveDomainConfig();
        if (!config?.active_domain) {
          setShowDomainSelect(true);
        }
      } catch {
        // Config not available yet — ignore
      }
    };
    checkDomain();
  }, [appMode, skipAuth, isAuthenticated]);

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
        currentFile: '', completed: 0, failed: 0, totalQueue: 0, pending: 0,
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

    // WebDAV folder: route through WebDAV processSource
    if (folderPath.startsWith('webdav://') && window.electron?.webdav) {
      window.electron.webdav.processSource(folderPath);
      return;
    }

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

    // Electron client mode with active worker: route through server API → queue for workers
    if (appMode === 'client' && isWorkerRunning) {
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

    // Electron standalone mode: direct discover spawn (local processing)
    window.electron?.pipeline?.updateConfig('last_session.folders', [folderPath]);
    window.electron?.pipeline?.runDiscover({ folderPath, noSkip });
  };

  // Resume incomplete work: discover only folders that have incomplete files
  const handleResume = async () => {
    setShowResumeDialog(false);
    if (!resumeStats?.folders?.length) return;

    // Use individual storage_root paths (only incomplete folders, not parent roots)
    const incompleteFolders = resumeStats.folders.map(f => f.storage_root.normalize('NFC'));
    if (incompleteFolders.length === 0) return;

    setIsDiscovering(true);
    setCurrentTab('archive');
    setCurrentPath(incompleteFolders[0]);

    // Client mode with active worker: route all folders through server API
    if (appMode === 'client' && isWorkerRunning) {
      let totalJobs = 0;
      try {
        for (const folder of incompleteFolders) {
          const result = await scanFolder(folder);
          totalJobs += result.jobs_created || 0;
        }
        appendLog({
          message: t('archive.queue_registered', { jobs: totalJobs }),
          type: 'success'
        });
        setQueueReloadSignal(prev => prev + 1);
      } catch (e) {
        appendLog({ message: `Resume scan failed: ${e.message}`, type: 'error' });
      }
      setIsDiscovering(false);
      return;
    }

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
  const handleScanFolders = async (folderPaths) => {
    if (isProcessing || isDiscovering || !folderPaths?.length) return;
    setIsDiscovering(true);
    setCurrentTab('archive');
    setCurrentPath(folderPaths[0]);

    // Client mode with active worker: route all folders through server API
    if (appMode === 'client' && isWorkerRunning) {
      let totalJobs = 0;
      try {
        for (const folder of folderPaths) {
          const result = await scanFolder(folder);
          totalJobs += result.jobs_created || 0;
        }
        appendLog({
          message: t('archive.queue_registered', { jobs: totalJobs }),
          type: 'success'
        });
        setQueueReloadSignal(prev => prev + 1);
      } catch (e) {
        appendLog({ message: `Folder scan failed: ${e.message}`, type: 'error' });
      }
      setIsDiscovering(false);
      return;
    }

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

  const handleResetDb = async () => {
    if (!resetPassword) return;
    setResetLoading(true);
    setResetError('');
    try {
      const result = await resetDatabase(resetPassword);
      appendLog({
        message: t('reset_db.success', { files: result.files, vectors: result.vectors, jobs: result.jobs }),
        type: 'success',
      });
      setShowResetDialog(false);
      setResetPassword('');
      setFolderStatsVersion(v => v + 1);
    } catch (e) {
      setResetError(e.message || 'Reset failed');
    } finally {
      setResetLoading(false);
    }
  };

  const handleAuditIntegrity = async ({ silent = false } = {}) => {
    setAuditLoading(true);
    setShowDbMenu(false);
    try {
      const result = await auditIntegrity();
      const hasIssues = result.incomplete_files > 0;
      if (hasIssues) {
        setAuditResult(result);
        appendLog({
          message: t('audit.result_repaired', {
            total: result.total_files,
            incomplete: result.incomplete_files,
          }),
          type: 'warn',
        });
      } else if (!silent) {
        setAuditResult(result);
        appendLog({
          message: t('audit.result_ok', { total: result.total_files }),
          type: 'success',
        });
      }
    } catch (e) {
      if (!silent) {
        appendLog({ message: `Audit error: ${e.message}`, type: 'error' });
      }
    } finally {
      setAuditLoading(false);
    }
  };

  // Auto-audit on login: admin이 서버에 로그인하면 자동 정합성 검사
  // skipAuth=false → 서버 모드 (web 또는 Electron server/client)
  useEffect(() => {
    if (isAuthenticated && isAdmin && skipAuth === false) {
      handleAuditIntegrity({ silent: true });
    }
  }, [isAuthenticated, isAdmin]);

  const handleRetryFailed = async () => {
    setRetryLoading(true);
    setShowDbMenu(false);
    try {
      const result = await retryFailedJobs();
      const count = result.retried || 0;
      if (count > 0) {
        appendLog({
          message: t('audit.retry_result', { count }),
          type: 'warn',
        });
      } else {
        appendLog({
          message: t('audit.retry_none'),
          type: 'success',
        });
      }
    } catch (e) {
      appendLog({ message: `Retry error: ${e.message}`, type: 'error' });
    } finally {
      setRetryLoading(false);
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

  // Electron menu action handler
  useEffect(() => {
    if (!isElectron) return;
    const handler = (e) => {
      const action = e.detail;
      switch (action) {
        case 'open-folder':
          window.electron?.pipeline?.openFolderDialog().then((folderPath) => {
            if (folderPath) {
              setCurrentPath(folderPath);
              setCurrentTab('archive');
            }
          });
          break;
        case 'export-db':
          handleExportDb();
          break;
        case 'import-db':
          setShowImportDialog(true);
          break;
        case 'toggle-server':
          handleServerToggle();
          break;
        case 'toggle-worker':
          if (isWorkerRunning) handleWorkerStop();
          else handleWorkerStart();
          break;
      }
    };
    window.addEventListener('electron-menu-action', handler);
    return () => window.removeEventListener('electron-menu-action', handler);
  }, [isElectron, isWorkerRunning]);

  const localeLabel = locale === 'ko-KR' ? 'KR' : 'EN';

  // Auth loading state
  if (authLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="text-gray-400 text-sm">{t('status.loading')}</div>
      </div>
    );
  }

  // Show login page when not authenticated (all modes)
  if (skipAuth === false && !isAuthenticated) {
    if (showDownloadPage) {
      return <DownloadPage onBack={() => setShowDownloadPage(false)} />;
    }
    if (USE_LOGIN_V2) {
      return <LoginPageV2
        onLoginComplete={handleLoginComplete}
        serverPort={serverPort}
      />;
    }
    return <LoginPage
      onShowDownload={!isElectron ? () => setShowDownloadPage(true) : undefined}
      onLoginComplete={handleLoginComplete}
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
      {/* EOL Warning Banner */}
      <EolBanner />
      {/* Subscription status banner (grace period, expired, etc.) */}
      {isAuthenticated && <SubscriptionBanner />}

      {/* Update Notification Toast (top-right overlay) */}
      <UpdateNotification />

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

      {showDomainSelect && (
        <DomainSelectModal onClose={() => setShowDomainSelect(false)} />
      )}

      {showResetDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl max-w-md w-full mx-4 p-6">
            <h2 className="text-lg font-semibold text-red-400 mb-3">
              {t('reset_db.title')}
            </h2>
            <p className="text-sm text-neutral-300 mb-4">
              {t('reset_db.warning')}
            </p>
            <label className="block text-sm text-neutral-400 mb-1.5">
              {t('reset_db.password_label')}
            </label>
            <input
              type="password"
              value={resetPassword}
              onChange={(e) => { setResetPassword(e.target.value); setResetError(''); }}
              placeholder={t('reset_db.password_placeholder')}
              className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-sm text-white placeholder-neutral-500 focus:outline-none focus:border-red-500 mb-2"
              onKeyDown={(e) => e.key === 'Enter' && handleResetDb()}
              autoFocus
            />
            {resetError && (
              <p className="text-xs text-red-400 mb-2">{t('reset_db.error', { error: resetError })}</p>
            )}
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => { setShowResetDialog(false); setResetPassword(''); setResetError(''); }}
                className="px-4 py-2 text-sm text-neutral-400 hover:text-white transition-colors"
                disabled={resetLoading}
              >
                {t('import.cancel_btn')}
              </button>
              <button
                onClick={handleResetDb}
                disabled={!resetPassword || resetLoading}
                className="px-4 py-2 text-sm bg-red-600 hover:bg-red-500 disabled:bg-neutral-700 disabled:text-neutral-500 text-white rounded-lg transition-colors"
              >
                {resetLoading ? (
                  <span className="flex items-center gap-2">
                    <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    {t('reset_db.confirm')}
                  </span>
                ) : t('reset_db.confirm')}
              </button>
            </div>
          </div>
        </div>
      )}

      {auditResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl max-w-lg w-full mx-4 p-6">
            <h2 className="text-lg font-semibold text-blue-400 mb-3 flex items-center gap-2">
              <ShieldCheck size={20} />
              {t('audit.title')}
            </h2>
            <div className="space-y-2 text-sm text-neutral-300 mb-4">
              {/* Summary stats */}
              <div className="grid grid-cols-3 gap-2 text-xs bg-neutral-800 rounded-lg p-3">
                <div>{t('audit.total_files')}: <span className="text-white font-medium">{auditResult.total_files}</span></div>
                <div>{t('audit.complete')}: <span className="text-green-400 font-medium">{auditResult.complete_files}</span></div>
                <div>{t('audit.incomplete')}: <span className={auditResult.incomplete_files > 0 ? 'text-yellow-400 font-medium' : 'text-neutral-500'}>{auditResult.incomplete_files}</span></div>
              </div>

              {auditResult.incomplete_files > 0 ? (
                <>
                  <p className="text-yellow-400 text-xs">
                    {t('audit.repaired_files', { count: auditResult.repaired_files || auditResult.incomplete_files })}
                  </p>
                  {auditResult.details?.length > 0 && (
                    <div className="max-h-48 overflow-y-auto mt-2 space-y-1">
                      {auditResult.details.map((d, i) => (
                        <div key={i} className="text-xs text-neutral-400 bg-neutral-800 px-2 py-1.5 rounded flex items-center gap-2">
                          <span className="text-red-400 shrink-0">[{d.missing.join(', ')}]</span>
                          <span className="truncate flex-1">{d.file_path?.split('/').pop()}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <p className="text-green-400 font-medium">{t('audit.all_ok')}</p>
              )}
            </div>
            <div className="flex justify-end gap-2">
              {isAdmin && (
                <button
                  onClick={async () => {
                    setRetryLoading(true);
                    try {
                      const result = await retryFailedJobs();
                      const count = result.retried || 0;
                      appendLog({
                        message: count > 0
                          ? t('audit.retry_result', { count })
                          : t('audit.retry_none'),
                        type: count > 0 ? 'warn' : 'success',
                      });
                    } catch (e) {
                      appendLog({ message: `Retry error: ${e.message}`, type: 'error' });
                    } finally {
                      setRetryLoading(false);
                    }
                  }}
                  disabled={retryLoading}
                  className="flex items-center gap-1.5 px-4 py-2 text-sm bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-white rounded-lg transition-colors"
                >
                  {retryLoading ? (
                    <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <RotateCcw size={14} />
                  )}
                  {t('action.retry_failed')}
                </button>
              )}
              <button
                onClick={() => setAuditResult(null)}
                className="px-4 py-2 text-sm bg-neutral-700 hover:bg-neutral-600 text-white rounded-lg transition-colors"
              >
                {t('action.close')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header Bar */}
      <div
        className="h-14 border-b border-gray-700 flex items-center pr-4 justify-between bg-gray-800 shadow-sm z-10 shrink-0"
        style={{ WebkitAppRegion: isElectron ? 'drag' : undefined, paddingLeft: isElectron ? 80 : 16 }}
      >
        {/* Left: App Name + Mode Badge */}
        <div className="flex items-center space-x-2" style={{ WebkitAppRegion: 'no-drag' }}>
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
        <div className="flex items-center space-x-2" style={{ WebkitAppRegion: 'no-drag' }}>
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
          {/* Archive/Worker tab — role-based: admin=archive, user=worker (Electron only) */}
          {isElectron && (
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
          )}

          {/* Admin tab — admin role only, Electron only */}
          {isElectron && isAdmin && (
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

          {/* Download tab — web mode only */}
          {!isElectron && (
            <button
              onClick={() => setCurrentTab('download')}
              className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${currentTab === 'download'
                ? 'bg-emerald-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
            >
              <Download size={16} />
              <span>{t('tab.download')}</span>
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

          {/* DB Import/Export Menu (Electron only) */}
          {isElectron && (
          <>
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
                  {isAdmin && (
                    <>
                      <div className="border-t border-gray-600 my-1" />
                      <button
                        onClick={handleRetryFailed}
                        disabled={retryLoading}
                        className="w-full flex items-center gap-2 px-4 py-2 text-sm text-orange-400 hover:bg-orange-900/30 hover:text-orange-300 transition-colors disabled:opacity-50"
                      >
                        {retryLoading ? (
                          <span className="w-3.5 h-3.5 border-2 border-orange-400 border-t-transparent rounded-full animate-spin" />
                        ) : (
                          <RotateCcw size={14} />
                        )}
                        {t('action.retry_failed')}
                      </button>
                      <button
                        onClick={handleAuditIntegrity}
                        disabled={auditLoading}
                        className="w-full flex items-center gap-2 px-4 py-2 text-sm text-blue-400 hover:bg-blue-900/30 hover:text-blue-300 transition-colors disabled:opacity-50"
                      >
                        {auditLoading ? (
                          <span className="w-3.5 h-3.5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                        ) : (
                          <ShieldCheck size={14} />
                        )}
                        {t('action.audit_integrity')}
                      </button>
                      <button
                        onClick={() => { setShowDbMenu(false); setShowResetDialog(true); }}
                        className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-400 hover:bg-red-900/30 hover:text-red-300 transition-colors"
                      >
                        <Trash2 size={14} />
                        {t('action.reset_db')}
                      </button>
                    </>
                  )}
                </div>
              </>
            )}
          </div>
          </>
          )}

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
            {currentTab === 'download' && !isElectron ? (
              <DownloadPage onBack={() => setCurrentTab('search')} />
            ) : currentTab === 'admin' && isAdmin ? (
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
