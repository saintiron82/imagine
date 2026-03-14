/**
 * ArchivingPage — Dual-mode tab:
 *   Folders mode: Disk/WebDAV folder browsing + queue creation (archiving)
 *   Archive mode: Browse DB-registered files (processed archive)
 */

import { useState, useCallback } from 'react';
import { FolderOpen, Archive, Package, FolderInput } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { scanFolder, registerPaths } from '../api/admin';
import FolderInfoBar from '../components/FolderInfoBar';
import FileGrid from '../components/FileGrid';

export default function ArchivingPage({
  currentPath,
  selectedFiles,
  setSelectedFiles,
  selectedPaths,
  onProcessFolder,
  onFindSimilar,
  isProcessing,
  reloadSignal,
  appMode,
  queueReloadSignal,
  onShowToast,
}) {
  const { t } = useLocale();
  const [mode, setMode] = useState('folders'); // 'folders' | 'archive'
  const [archiving, setArchiving] = useState(false);

  // Archive selected files: register them to the job queue
  const handleArchiveSelected = useCallback(async () => {
    if (selectedFiles.size === 0) return;
    setArchiving(true);
    try {
      const paths = Array.from(selectedFiles);
      if (isElectron && window.electron?.queue) {
        await window.electron.queue.registerPaths(paths, 'normal');
      } else {
        await registerPaths(paths);
      }
      onShowToast?.(t('archiving.toast_queued', { count: paths.length }), 'success');
      setSelectedFiles(new Set());
    } catch (e) {
      console.error('Failed to archive selected files:', e);
      onShowToast?.(`Error: ${e.message}`, 'error');
    } finally {
      setArchiving(false);
    }
  }, [selectedFiles, setSelectedFiles, onShowToast, t]);

  // Archive entire folder: scan and register all files in current path
  const handleArchiveFolder = useCallback(async () => {
    if (!currentPath) return;
    setArchiving(true);
    try {
      if (isElectron && window.electron?.queue) {
        await window.electron.queue.scanFolder(currentPath, 'normal');
      } else {
        await scanFolder(currentPath);
      }
      onShowToast?.(t('archiving.toast_queued', { count: '...' }), 'success');
    } catch (e) {
      console.error('Failed to archive folder:', e);
      onShowToast?.(`Error: ${e.message}`, 'error');
    } finally {
      setArchiving(false);
    }
  }, [currentPath, onShowToast, t]);

  return (
    <div className="flex flex-col h-full">
      {/* Mode toggle + Archiving actions */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        {/* Left: Mode toggle */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setMode('folders')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              mode === 'folders'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            <FolderOpen size={14} />
            {t('archiving.mode_folders')}
          </button>
          <button
            onClick={() => setMode('archive')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              mode === 'archive'
                ? 'bg-emerald-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            <Archive size={14} />
            {t('archiving.mode_archive')}
          </button>
        </div>

        {/* Right: Archiving action buttons (folders mode only) */}
        {mode === 'folders' && (
          <div className="flex items-center gap-2">
            {selectedFiles.size > 0 && (
              <button
                onClick={handleArchiveSelected}
                disabled={archiving || isProcessing}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Package size={14} />
                {t('archiving.action_archive_selected')} ({selectedFiles.size})
              </button>
            )}
            {currentPath && (
              <button
                onClick={handleArchiveFolder}
                disabled={archiving || isProcessing}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <FolderInput size={14} />
                {t('archiving.action_archive_folder')}
              </button>
            )}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {mode === 'folders' ? (
          <div className="h-full flex flex-col">
            <FolderInfoBar
              currentPath={currentPath}
              onProcessFolder={onProcessFolder}
              isProcessing={isProcessing}
              reloadSignal={reloadSignal}
              appMode={appMode}
            />
            <div className="flex-1 overflow-y-auto p-4 pb-16">
              <FileGrid
                currentPath={currentPath}
                selectedFiles={selectedFiles}
                setSelectedFiles={setSelectedFiles}
                selectedPaths={selectedPaths}
                onFindSimilar={onFindSimilar}
              />
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <Archive size={48} className="mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">{t('archiving.mode_archive')}</p>
              <p className="text-sm mt-1 text-gray-600">Coming soon</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
