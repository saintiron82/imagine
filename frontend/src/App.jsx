import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import FileGrid from './components/FileGrid';
import StatusBar from './components/StatusBar';
import SearchPanel from './components/SearchPanel';
import FolderInfoBar from './components/FolderInfoBar';
import ResumeDialog from './components/ResumeDialog';
import { FolderOpen, Play, Search, Archive, Globe } from 'lucide-react';
import { useLocale } from './i18n';

function App() {
  const { t, locale, setLocale, availableLocales } = useLocale();
  const [currentTab, setCurrentTab] = useState('search'); // 'search' or 'archive'
  const [currentPath, setCurrentPath] = useState('');
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [logs, setLogs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processProgress, setProcessProgress] = useState({
    processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0,
    // Cumulative per-phase counts
    cumParse: 0, cumVision: 0, cumEmbed: 0, cumStore: 0,
    // Active phase sub-progress
    activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0,
    batchInfo: '',
  });
  const [fileStep, setFileStep] = useState({ step: 0, totalSteps: 5, stepName: '' });
  const etaRef = useRef({ startTime: null, lastFileTime: null, emaMs: null });
  const phaseEtaRef = useRef({ phase: -1, startTime: null, startCount: 0 });
  const discoverQueueRef = useRef({ folders: [], index: 0, scanning: false });
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [discoverProgress, setDiscoverProgress] = useState({
    processed: 0, total: 0, currentFile: '', step: 0, totalSteps: 4, folderPath: ''
  });
  const [selectedPaths, setSelectedPaths] = useState(new Set());
  const [resumeStats, setResumeStats] = useState(null);
  const [showResumeDialog, setShowResumeDialog] = useState(false);

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
              cumVision: data.cumVision ?? prev.cumVision,
              cumEmbed: data.cumEmbed ?? prev.cumEmbed,
              cumStore: data.cumStore ?? prev.cumStore,
              activePhase: data.activePhase ?? prev.activePhase,
              phaseSubCount: data.phaseSubCount ?? prev.phaseSubCount,
              phaseSubTotal: data.phaseSubTotal ?? prev.phaseSubTotal,
              batchInfo: data.batchInfo ?? prev.batchInfo,
            };

            // Phase-based ETA: track progress rate per active phase
            const pe = phaseEtaRef.current;
            const ap = next.activePhase;
            const counts = [next.cumParse, next.cumVision, next.cumEmbed, next.cumStore];
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
          setProcessProgress({ processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0, cumParse: 0, cumVision: 0, cumEmbed: 0, cumStore: 0, activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '' });
          setFileStep({ step: 0, totalSteps: 5, stepName: '' });
          etaRef.current = { startTime: null, lastFileTime: null, emaMs: null };
          const msg = data.skipped > 0
            ? `All done! ${data.processed} processed, ${data.skipped} skipped (total: ${data.total})`
            : `All ${data.processed} files processed!`;
          appendLog({ message: msg, type: 'success' });
        });

        // Discover event listeners (for auto-scan)
        window.electron.pipeline.onDiscoverLog((data) => {
          appendLog(data);
        });
        window.electron.pipeline.onDiscoverProgress((data) => {
          setDiscoverProgress(prev => ({ ...prev, ...data }));
        });
        window.electron.pipeline.onDiscoverFileDone((data) => {
          // Auto-scan processes folders sequentially via discoverQueueRef
          const ref = discoverQueueRef.current;
          if (ref.scanning && ref.index < ref.folders.length - 1) {
            discoverQueueRef.current.index++;
            const nextFolder = ref.folders[ref.index + 1];
            window.electron.pipeline.runDiscover({ folderPath: nextFolder, noSkip: false });
          } else {
            setIsDiscovering(false);
            setDiscoverProgress({ processed: 0, total: 0, currentFile: '', step: 0, totalSteps: 4, folderPath: '' });
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
  const handleProcess = () => {
    if (selectedFiles.size === 0 || isProcessing) return;

    const fileArray = Array.from(selectedFiles);

    setIsProcessing(true);
    etaRef.current = { startTime: Date.now(), lastFileTime: Date.now(), emaMs: null };
    setProcessProgress({ processed: 0, total: fileArray.length, currentFile: '', etaMs: null, skipped: 0, cumParse: 0, cumVision: 0, cumEmbed: 0, cumStore: 0, activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '' });
    setSelectedFiles(new Set());

    appendLog({
      message: `Starting batch: ${fileArray.length} files`,
      type: 'info'
    });

    // Single batch call — backend handles smart skip + phase pipeline
    if (window.electron?.pipeline) {
      window.electron.pipeline.run(fileArray);
    }
  };

  const handleStopProcess = () => {
    // Kill the actual Python process
    window.electron?.pipeline?.stop();
    setIsProcessing(false);
    setProcessProgress({ processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0, cumParse: 0, cumVision: 0, cumEmbed: 0, cumStore: 0, activePhase: 0, phaseSubCount: 0, phaseSubTotal: 0, batchInfo: '' });
    setFileStep({ step: 0, totalSteps: 5, stepName: '' });
    etaRef.current = { startTime: null, lastFileTime: null, emaMs: null };
  };

  // Process entire folder recursively (discover mode)
  const handleProcessFolder = (folderPath) => {
    if (isProcessing || isDiscovering) return;
    setIsDiscovering(true);
    setCurrentTab('archive');
    setCurrentPath(folderPath);
    appendLog({ message: `Processing folder: ${folderPath}`, type: 'info' });
    // Save session target for resume on next startup
    window.electron?.pipeline?.updateConfig('last_session.folders', [folderPath]);
    window.electron?.pipeline?.runDiscover({ folderPath, noSkip: false });
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

  const clearLogs = () => setLogs([]);

  const localeLabel = locale === 'ko-KR' ? 'KR' : 'EN';

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
            className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${
              currentTab === 'search'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
            }`}
          >
            <Search size={16} />
            <span>{t('tab.search')}</span>
          </button>
          <button
            onClick={() => setCurrentTab('archive')}
            className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${
              currentTab === 'archive'
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
            }`}
          >
            <Archive size={16} />
            <span>{t('tab.archive')}</span>
          </button>

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
                  ${
                    selectedFiles.size > 0 && !isProcessing
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
              />
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="flex-1 flex flex-col bg-gray-900 relative">
          {/* Content Area */}
          <div className="flex-1 overflow-hidden">
            {currentTab === 'search' ? (
              <SearchPanel />
            ) : (
              <div className="h-full flex flex-col">
                <FolderInfoBar
                  currentPath={currentPath}
                  onProcessFolder={handleProcessFolder}
                  isProcessing={isProcessing || isDiscovering}
                />
                <div className="flex-1 overflow-y-auto p-4 pb-16">
                  <FileGrid
                    currentPath={currentPath}
                    selectedFiles={selectedFiles}
                    setSelectedFiles={setSelectedFiles}
                    selectedPaths={selectedPaths}
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
            cumVision={processProgress.cumVision}
            cumEmbed={processProgress.cumEmbed}
            cumStore={processProgress.cumStore}
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
