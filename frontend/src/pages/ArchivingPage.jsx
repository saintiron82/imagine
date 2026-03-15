/**
 * ArchivingPage — Folder browsing + queue creation (archiving).
 *
 * Queue registration is delegated to App.jsx's handleProcessFolder / handleProcessFiles
 * which already handle appMode branching (standalone / server / client).
 */

import { useCallback } from 'react';
import { Package, X } from 'lucide-react';
import { useLocale } from '../i18n';
import FolderInfoBar from '../components/FolderInfoBar';
import FileGrid from '../components/FileGrid';

export default function ArchivingPage({
  currentPath,
  selectedFiles,
  setSelectedFiles,
  selectedPaths,
  onProcessFolder,
  onProcessFiles,
  onFindSimilar,
  isProcessing,
  reloadSignal,
  appMode,
  onShowToast,
}) {
  const { t } = useLocale();

  const handleEnqueueSelected = useCallback(() => {
    if (selectedFiles.size === 0) return;
    onProcessFiles?.(Array.from(selectedFiles));
  }, [selectedFiles, onProcessFiles]);

  return (
    <div className="h-full flex flex-col relative">
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

      {/* Floating selection action bar */}
      {selectedFiles.size > 0 && (
        <SelectionActionBar
          count={selectedFiles.size}
          onEnqueue={handleEnqueueSelected}
          onClear={() => setSelectedFiles(new Set())}
          isProcessing={isProcessing}
        />
      )}
    </div>
  );
}

function SelectionActionBar({ count, onEnqueue, onClear, isProcessing }) {
  const { t } = useLocale();
  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 flex items-center gap-3 px-4 py-2.5 rounded-lg bg-gray-800/95 border border-gray-600 shadow-2xl backdrop-blur-sm">
      <span className="text-sm text-gray-300 font-medium">
        {t('archiving.selected_count', { count })}
      </span>
      <button
        onClick={onEnqueue}
        disabled={isProcessing}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        <Package size={14} />
        {t('archiving.action_enqueue')}
      </button>
      <button
        onClick={onClear}
        className="p-1.5 rounded-md hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
        title={t('action.clear')}
      >
        <X size={14} />
      </button>
    </div>
  );
}
