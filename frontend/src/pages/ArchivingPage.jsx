/**
 * ArchivingPage — Dual-mode tab: Disk Folders (queue creation) + Archive (DB browsing)
 * Phase 1: Thin wrapper around existing ServerArchiveView (to be expanded in Phase 2)
 */

import { useState } from 'react';
import { FolderOpen, Archive } from 'lucide-react';
import { useLocale } from '../i18n';
import ServerArchiveView from '../components/ServerArchiveView';

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
}) {
  const { t } = useLocale();
  const [mode, setMode] = useState('folders'); // 'folders' | 'archive'

  return (
    <div className="flex flex-col h-full">
      {/* Mode toggle */}
      <div className="flex items-center gap-1 px-4 py-2 bg-gray-800 border-b border-gray-700">
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

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {mode === 'folders' ? (
          <ServerArchiveView
            currentPath={currentPath}
            selectedFiles={selectedFiles}
            setSelectedFiles={setSelectedFiles}
            selectedPaths={selectedPaths}
            onProcessFolder={onProcessFolder}
            onFindSimilar={onFindSimilar}
            isProcessing={isProcessing}
            reloadSignal={reloadSignal}
            appMode={appMode}
            queueReloadSignal={queueReloadSignal}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <Archive size={48} className="mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">{t('archiving.mode_archive')}</p>
              <p className="text-sm mt-1">Coming in Phase 2.5</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
