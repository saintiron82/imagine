/**
 * ArchivingPage — Dual-mode tab:
 *   Folders mode: Disk/WebDAV folder browsing + queue creation (archiving)
 *   Archive mode: Browse DB-registered files (processed archive)
 */

import { useState, useEffect, useCallback } from 'react';
import { FolderOpen, Archive, Package, FolderInput, Loader2, Filter, CheckCircle2, AlertCircle } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { scanFolder, registerPaths } from '../api/admin';
import { getArchiveFolders, getArchiveFiles, getThumbnailUrl } from '../services/bridge';
import { isLocalMode } from '../services/bridge';
import FolderInfoBar from '../components/FolderInfoBar';
import FileGrid from '../components/FileGrid';

// Phase status badge component
function PhaseBadge({ phase }) {
  const allDone = phase.mc && phase.vv && phase.mv;
  const noneDone = !phase.mc && !phase.vv && !phase.mv;

  if (allDone) {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-green-400 font-medium">
        <CheckCircle2 size={10} />
      </span>
    );
  }
  if (noneDone) {
    return (
      <span className="flex items-center gap-0.5 text-[10px] text-gray-500 font-medium">
        <AlertCircle size={10} />
      </span>
    );
  }
  return (
    <span className="flex items-center gap-0.5 text-[10px] font-mono">
      <span className={phase.mc ? 'text-green-400' : 'text-gray-600'}>MC</span>
      <span className={phase.vv ? 'text-blue-400' : 'text-gray-600'}>VV</span>
      <span className={phase.mv ? 'text-purple-400' : 'text-gray-600'}>MV</span>
    </span>
  );
}

// Thumbnail helper
function toFileUrl(p) {
  if (!p || typeof p !== 'string') return '';
  let resolved = p;
  if (!/^[A-Za-z]:/.test(p) && !/^\//.test(p)) {
    const root = window.electron?.projectRoot || '';
    resolved = root ? `${root}/${p}` : p;
  }
  const normalized = resolved.replace(/\\/g, '/');
  const encoded = normalized.split('/').map(s => encodeURIComponent(s)).join('/');
  return /^[A-Za-z]:/.test(normalized) ? `file:///${encoded}` : `file://${encoded}`;
}

const IMAGE_PREVIEW_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg'];

// Archive file card
function ArchiveCard({ file, onShowMeta }) {
  const ext = file.file_name.substring(file.file_name.lastIndexOf('.')).toLowerCase();
  const canPreviewNatively = IMAGE_PREVIEW_EXTS.includes(ext);

  const thumbnailSrc = isLocalMode()
    ? (file.thumbnail_url ? toFileUrl(file.thumbnail_url) : canPreviewNatively ? toFileUrl(file.file_path) : null)
    : getThumbnailUrl(file.thumbnail_url, file.id);

  return (
    <div
      className="group bg-gray-800 border border-gray-700 rounded-lg overflow-hidden hover:border-emerald-500/50 hover:shadow-lg cursor-pointer h-full flex flex-col transition-[border-color,box-shadow] duration-150"
      onClick={() => onShowMeta?.(file)}
    >
      {/* Thumbnail */}
      <div className="aspect-square bg-gray-900 overflow-hidden relative shrink-0">
        {thumbnailSrc ? (
          <img
            src={thumbnailSrc}
            alt={file.file_name}
            className="w-full h-full object-cover"
            onError={(e) => { e.target.style.display = 'none'; }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-600">
            <div className="text-3xl font-bold uppercase">{ext.replace('.', '')}</div>
          </div>
        )}

        {/* Phase badge (top-right) */}
        <div className="absolute top-1.5 right-1.5">
          <PhaseBadge phase={file.phase} />
        </div>

        {/* image_type badge (top-left) */}
        {file.image_type && (
          <span className="absolute top-1.5 left-1.5 bg-purple-900/80 text-purple-300 text-[10px] font-bold px-1.5 py-0.5 rounded uppercase">
            {file.image_type}
          </span>
        )}
      </div>

      {/* Info */}
      <div className="p-2 flex-1 min-h-0 overflow-hidden">
        <div className="text-xs font-medium text-white truncate">{file.file_name}</div>
        {file.folder_path && (
          <div className="text-[10px] text-gray-500 truncate mt-0.5">{file.folder_path}</div>
        )}
        {file.ai_tags && file.ai_tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {file.ai_tags.slice(0, 3).map((tag, i) => (
              <span key={i} className="text-[9px] bg-gray-700 text-gray-300 px-1 py-0.5 rounded">
                {tag}
              </span>
            ))}
            {file.ai_tags.length > 3 && (
              <span className="text-[9px] text-gray-500">+{file.ai_tags.length - 3}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}


// Archive browse mode content
function ArchiveBrowseMode({ onShowToast }) {
  const { t } = useLocale();
  const [folders, setFolders] = useState([]);
  const [imageTypes, setImageTypes] = useState([]);
  const [selectedFolder, setSelectedFolder] = useState('[all]');
  const [selectedType, setSelectedType] = useState('');
  const [files, setFiles] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 60;

  // Load folders on mount
  useEffect(() => {
    (async () => {
      try {
        const result = await getArchiveFolders();
        if (result?.success !== false) {
          setFolders(result.folders || []);
          // Extract unique image types from folders
          const typeSet = {};
          (result.folders || []).forEach(f => {
            Object.entries(f.image_types || {}).forEach(([type, count]) => {
              typeSet[type] = (typeSet[type] || 0) + count;
            });
          });
          setImageTypes(Object.entries(typeSet).sort((a, b) => b[1] - a[1]));
        }
      } catch (e) {
        console.error('Failed to load archive folders:', e);
      }
    })();
  }, []);

  // Load files when folder or type changes
  useEffect(() => {
    setPage(0);
    loadFiles(0);
  }, [selectedFolder, selectedType]);

  const loadFiles = useCallback(async (pageNum) => {
    setLoading(true);
    try {
      const params = {
        limit: PAGE_SIZE,
        offset: (pageNum ?? page) * PAGE_SIZE,
      };
      if (selectedFolder && selectedFolder !== '[all]') {
        params.folder_path = selectedFolder;
      }
      if (selectedType) {
        params.image_type = selectedType;
      }
      const result = await getArchiveFiles(params);
      if (result?.success !== false) {
        setFiles(result.files || []);
        setTotal(result.total || 0);
      }
    } catch (e) {
      console.error('Failed to load archive files:', e);
    } finally {
      setLoading(false);
    }
  }, [selectedFolder, selectedType, page]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  const handlePageChange = (newPage) => {
    setPage(newPage);
    loadFiles(newPage);
  };

  // Calculate total stats
  const totalFiles = folders.reduce((sum, f) => sum + f.total, 0);
  const totalComplete = folders.reduce((sum, f) => sum + Math.min(f.mc, f.vv, f.mv), 0);

  if (totalFiles === 0 && !loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <Archive size={48} className="mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium">{t('archiving.no_files')}</p>
          <p className="text-sm mt-1 text-gray-600">{t('archiving.no_files_desc')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="w-56 border-r border-gray-700 bg-gray-850 flex flex-col overflow-hidden shrink-0">
        {/* Stats header */}
        <div className="px-3 py-2 border-b border-gray-700">
          <div className="text-[11px] text-gray-400 font-medium">
            {totalFiles} files / {totalComplete} complete
          </div>
        </div>

        {/* Folder list */}
        <div className="flex-1 overflow-y-auto">
          <div className="px-2 py-1">
            <div className="text-[10px] text-gray-500 uppercase font-semibold px-1 py-1 mt-1">
              Folders
            </div>
            <button
              onClick={() => setSelectedFolder('[all]')}
              className={`w-full text-left px-2 py-1.5 rounded text-xs transition-colors ${
                selectedFolder === '[all]'
                  ? 'bg-emerald-900/40 text-emerald-300'
                  : 'text-gray-400 hover:bg-gray-700 hover:text-white'
              }`}
            >
              {t('archiving.all_folders')} ({totalFiles})
            </button>
            {folders.map((folder) => {
              const done = Math.min(folder.mc, folder.vv, folder.mv);
              const isComplete = done === folder.total;
              return (
                <button
                  key={folder.folder_path}
                  onClick={() => setSelectedFolder(folder.folder_path)}
                  className={`w-full text-left px-2 py-1.5 rounded text-xs transition-colors group ${
                    selectedFolder === folder.folder_path
                      ? 'bg-emerald-900/40 text-emerald-300'
                      : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                  }`}
                  title={folder.folder_path}
                >
                  <div className="flex items-center justify-between">
                    <span className="truncate flex-1">{folder.folder_path}</span>
                    <span className="text-[10px] text-gray-500 ml-1 shrink-0">{folder.total}</span>
                  </div>
                  {/* Mini progress */}
                  <div className="mt-0.5 flex items-center gap-1">
                    <div className="flex-1 h-0.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${isComplete ? 'bg-green-500' : 'bg-yellow-500'}`}
                        style={{ width: `${folder.total > 0 ? (done / folder.total) * 100 : 0}%` }}
                      />
                    </div>
                    <span className="text-[9px] text-gray-600">{done}/{folder.total}</span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Image type filter */}
          {imageTypes.length > 0 && (
            <div className="px-2 py-1 border-t border-gray-700/50">
              <div className="text-[10px] text-gray-500 uppercase font-semibold px-1 py-1">
                <Filter size={9} className="inline mr-1" />
                Type
              </div>
              <button
                onClick={() => setSelectedType('')}
                className={`w-full text-left px-2 py-1 rounded text-xs transition-colors ${
                  !selectedType
                    ? 'bg-emerald-900/40 text-emerald-300'
                    : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                }`}
              >
                {t('archiving.all_types')}
              </button>
              {imageTypes.map(([type, count]) => (
                <button
                  key={type}
                  onClick={() => setSelectedType(type)}
                  className={`w-full text-left px-2 py-1 rounded text-xs transition-colors ${
                    selectedType === type
                      ? 'bg-emerald-900/40 text-emerald-300'
                      : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <span className="truncate">{type}</span>
                  <span className="text-[10px] text-gray-500 ml-1">({count})</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700/50">
          <div className="text-sm text-gray-300">
            {selectedFolder === '[all]' ? t('archiving.all_folders') : selectedFolder}
            {selectedType && <span className="text-gray-500 ml-2">/ {selectedType}</span>}
            <span className="text-gray-500 ml-2">({total})</span>
          </div>
          {totalPages > 1 && (
            <div className="flex items-center gap-2 text-xs">
              <button
                onClick={() => handlePageChange(Math.max(0, page - 1))}
                disabled={page === 0}
                className="px-2 py-0.5 rounded bg-gray-700 text-gray-300 disabled:opacity-30 hover:bg-gray-600"
              >
                ←
              </button>
              <span className="text-gray-400">{page + 1} / {totalPages}</span>
              <button
                onClick={() => handlePageChange(Math.min(totalPages - 1, page + 1))}
                disabled={page >= totalPages - 1}
                className="px-2 py-0.5 rounded bg-gray-700 text-gray-300 disabled:opacity-30 hover:bg-gray-600"
              >
                →
              </button>
            </div>
          )}
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center h-full text-gray-500">
              <Loader2 size={24} className="animate-spin mr-2" />
              {t('archiving.loading')}
            </div>
          ) : files.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-500">
              <p className="text-sm">{t('archiving.no_files')}</p>
            </div>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3">
              {files.map((file) => (
                <ArchiveCard key={file.id} file={file} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


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
          <ArchiveBrowseMode onShowToast={onShowToast} />
        )}
      </div>
    </div>
  );
}
