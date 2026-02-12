import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import FileGrid from './components/FileGrid';
import StatusBar from './components/StatusBar';
import SearchPanel from './components/SearchPanel';
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
    processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0, batchPct: 0
  });
  const [fileStep, setFileStep] = useState({ step: 0, totalSteps: 5, stepName: '' });
  const etaRef = useRef({ startTime: null, lastFileTime: null, emaMs: null });
  const discoverQueueRef = useRef({ folders: [], index: 0, scanning: false });
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [discoverProgress, setDiscoverProgress] = useState('');
  const [selectedPaths, setSelectedPaths] = useState(new Set());

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

        // Progress updates (processed count, current file, skipped, batchPct)
        window.electron.pipeline.onProgress((data) => {
          setProcessProgress(prev => {
            const newProcessed = data.processed ?? prev.processed;
            const newSkipped = data.skipped ?? prev.skipped;
            const currentFile = data.currentFile ?? prev.currentFile;
            const batchPct = data.batchPct ?? prev.batchPct;
            return { ...prev, processed: newProcessed, skipped: newSkipped, currentFile, batchPct };
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
          setProcessProgress({ processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0, batchPct: 0 });
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
          setDiscoverProgress(data.message);
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
            setDiscoverProgress('');
          }
        });
      }
    }

    // Auto-scan registered folders on startup
    const autoScanRegisteredFolders = async () => {
      try {
        const result = await window.electron?.pipeline?.getRegisteredFolders();
        if (!result?.success) return;
        if (result.autoScan === false) return;
        const validFolders = (result.folders || []).filter(f => f.exists).map(f => f.path);
        if (validFolders.length === 0) return;

        setIsDiscovering(true);
        discoverQueueRef.current = { folders: validFolders, index: 0, scanning: true };
        appendLog({
          message: `Auto-scanning ${validFolders.length} registered folder(s)...`,
          type: 'info'
        });
        window.electron.pipeline.runDiscover({ folderPath: validFolders[0], noSkip: false });
      } catch (e) {
        console.error('Auto-scan failed:', e);
      }
    };
    autoScanRegisteredFolders();

    return () => {
      if (window.electron?.pipeline) {
        window.electron.pipeline.offLog();
        window.electron.pipeline.offStep();
        window.electron.pipeline.offProgress();
        window.electron.pipeline.offFileDone();
        window.electron.pipeline.offBatchDone();
        window.electron.pipeline.offDiscoverLog();
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
    setProcessProgress({ processed: 0, total: fileArray.length, currentFile: '', etaMs: null, skipped: 0, batchPct: 0 });
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
    setProcessProgress({ processed: 0, total: 0, currentFile: '', etaMs: null, skipped: 0, batchPct: 0 });
    setFileStep({ step: 0, totalSteps: 5, stepName: '' });
    etaRef.current = { startTime: null, lastFileTime: null, emaMs: null };
  };

  const clearLogs = () => setLogs([]);

  const localeLabel = locale === 'ko-KR' ? 'KR' : 'EN';

  return (
    <div className="flex h-screen bg-gray-900 text-white overflow-hidden flex-col">
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
              <div className="h-full overflow-y-auto p-4 pb-16">
                <FileGrid
                  currentPath={currentPath}
                  selectedFiles={selectedFiles}
                  setSelectedFiles={setSelectedFiles}
                  selectedPaths={selectedPaths}
                />
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
            batchPct={processProgress.batchPct}
            fileStep={fileStep}
            onStop={handleStopProcess}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
