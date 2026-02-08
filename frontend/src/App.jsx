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
  const [processProgress, setProcessProgress] = useState({ processed: 0, total: 0, currentFile: '' });
  const [fileStep, setFileStep] = useState({ step: 0, totalSteps: 5, stepName: '' });
  const [processQueue, setProcessQueue] = useState([]); // Processing queue
  const [queueIndex, setQueueIndex] = useState(0); // Current processing index
  const queueRef = useRef({ queue: [], index: 0, processing: false });

  // Initialize with Home Directory & stable IPC listeners (never removed during app lifetime)
  useEffect(() => {
    if (window.electron) {
      if (window.electron.fs) {
        setCurrentPath(window.electron.fs.getHomeDir());
      }

      if (window.electron.pipeline) {
        // Stable log listener - feeds StatusBar (never removed mid-session)
        window.electron.pipeline.onLog((data) => {
          setLogs((prev) => [...prev, data]);
        });

        // Per-file step progress listener
        window.electron.pipeline.onStep((data) => {
          setFileStep(data);
        });

        // Stable file-done listener - advances queue via ref
        window.electron.pipeline.onFileDone((data) => {
          const { processing, index, queue } = queueRef.current;
          if (processing && index < queue.length) {
            setQueueIndex((prev) => prev + 1);
          }
        });
      }
    }

    return () => {
      if (window.electron?.pipeline) {
        window.electron.pipeline.offLog();
        window.electron.pipeline.offStep();
        window.electron.pipeline.offFileDone();
      }
    };
  }, []);

  // Keep ref in sync with state
  useEffect(() => {
    queueRef.current = {
      queue: processQueue,
      index: queueIndex,
      processing: isProcessing,
    };
  }, [processQueue, queueIndex, isProcessing]);

  const handleFolderSelect = (path) => {
    setCurrentPath(path);
    setSelectedFiles(new Set()); // Clear selection on folder change
  };

  const handleProcess = () => {
    if (selectedFiles.size === 0) return;

    const fileArray = Array.from(selectedFiles);

    // Add to queue
    setProcessQueue((prev) => [...prev, ...fileArray]);
    setLogs((prev) => [...prev, {
      message: `Added ${fileArray.length} files to queue (Total: ${processQueue.length + fileArray.length})`,
      type: 'info'
    }]);

    // Clear selection after adding to queue
    setSelectedFiles(new Set());

    // Start processing if not already running
    if (!isProcessing) {
      setIsProcessing(true);
    }
  };

  // Queue processor - only dispatches pipeline.run(), no listener management
  useEffect(() => {
    if (!isProcessing || processQueue.length === 0 || queueIndex >= processQueue.length) {
      if (queueIndex >= processQueue.length && processQueue.length > 0) {
        // Queue finished
        setLogs((prev) => [...prev, {
          message: `All ${processQueue.length} files processed!`,
          type: 'success'
        }]);
        setProcessQueue([]);
        setQueueIndex(0);
        setIsProcessing(false);
        setProcessProgress({ processed: 0, total: 0, currentFile: '' });
      }
      return;
    }

    const currentFile = processQueue[queueIndex];

    setProcessProgress({
      processed: queueIndex,
      total: processQueue.length,
      currentFile: currentFile,
    });

    setLogs((prev) => [...prev, {
      message: `Processing [${queueIndex + 1}/${processQueue.length}]: ${currentFile}`,
      type: 'info'
    }]);

    if (window.electron?.pipeline) {
      window.electron.pipeline.run([currentFile]);
    }
  }, [processQueue, queueIndex, isProcessing]);

  const handleStopProcess = () => {
    setProcessQueue([]);
    setQueueIndex(0);
    setIsProcessing(false);
    setProcessProgress({ processed: 0, total: 0, currentFile: '' });
    setFileStep({ step: 0, totalSteps: 5, stepName: '' });
    setLogs((prev) => [...prev, { message: 'Processing stopped by user.', type: 'warning' }]);
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
                disabled={selectedFiles.size === 0}
                className={`
                  flex items-center space-x-1 px-4 py-1.5 rounded text-sm font-medium transition-colors
                  ${
                    selectedFiles.size > 0
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
                />
              </div>
            )}
          </div>

          {/* Logs Overlay / Bottom Panel */}
          <StatusBar
            logs={logs}
            clearLogs={clearLogs}
            isProcessing={isProcessing}
            processed={queueIndex}
            total={processQueue.length}
            currentFile={processProgress.currentFile}
            fileStep={fileStep}
            onStop={handleStopProcess}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
