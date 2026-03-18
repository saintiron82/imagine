/**
 * EnqueueFolderModal — Queue registration popup with progress feedback.
 *
 * Shows loading state during scan, then result (jobs created / errors).
 * Blocks UI until registration completes.
 */

import { useState, useEffect, useRef } from 'react';
import { FolderInput, X, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { useLocale } from '../i18n';

export default function EnqueueFolderModal({ folderPath, folderName, onConfirm, onClose }) {
  const { t } = useLocale();
  const [requestName, setRequestName] = useState(folderName || '');
  const [includeSubfolders, setIncludeSubfolders] = useState(true);
  const [status, setStatus] = useState('idle'); // 'idle' | 'scanning' | 'done' | 'error'
  const [result, setResult] = useState(null); // { jobs_created, discovered, skipped, error }
  const inputRef = useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => inputRef.current?.select(), 50);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape' && status !== 'scanning') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose, status]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (status === 'scanning') return;

    const name = requestName.trim() || folderName;
    setStatus('scanning');
    setResult(null);

    try {
      const res = await onConfirm(folderPath, name, includeSubfolders);
      setResult(res || {});
      setStatus(res?.success === false ? 'error' : 'done');
    } catch (err) {
      setResult({ error: err.message || String(err) });
      setStatus('error');
    }
  };

  const isScanning = status === 'scanning';
  const isDone = status === 'done';
  const isError = status === 'error';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={isScanning ? undefined : onClose}>
      <div
        className="bg-gray-800 border border-gray-600 rounded-lg shadow-2xl w-96 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <FolderInput size={16} className="text-blue-400" />
            <span className="text-sm font-medium text-white">{t('queue.add_to_queue')}</span>
          </div>
          {!isScanning && (
            <button onClick={onClose} className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white">
              <X size={16} />
            </button>
          )}
        </div>

        {/* Body */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Folder path */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('queue.folder_path')}</label>
            <div className="text-xs text-gray-300 bg-gray-900/50 px-3 py-2 rounded border border-gray-700 truncate" title={folderPath}>
              {folderPath}
            </div>
          </div>

          {/* Work request name */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('queue.request_name')}</label>
            <input
              ref={inputRef}
              type="text"
              value={requestName}
              onChange={(e) => setRequestName(e.target.value)}
              placeholder={folderName}
              disabled={isScanning || isDone}
              className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-600 rounded text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
            />
          </div>

          {/* Include subfolders */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={includeSubfolders}
              onChange={(e) => setIncludeSubfolders(e.target.checked)}
              disabled={isScanning || isDone}
              className="rounded border-gray-600 bg-gray-900 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-300">{t('queue.include_subfolders')}</span>
          </label>

          {/* Scanning progress */}
          {isScanning && (
            <div className="flex items-center gap-3 px-3 py-3 rounded bg-blue-900/20 border border-blue-800/30">
              <Loader2 size={18} className="text-blue-400 animate-spin flex-shrink-0" />
              <div>
                <div className="text-sm text-blue-300 font-medium">{t('queue.scanning') || 'Scanning...'}</div>
                <div className="text-xs text-gray-500">{t('queue.scanning_desc') || 'Listing files and registering jobs'}</div>
              </div>
            </div>
          )}

          {/* Success result */}
          {isDone && result && (
            <div className="flex items-start gap-3 px-3 py-3 rounded bg-green-900/20 border border-green-800/30">
              <CheckCircle size={18} className="text-green-400 flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-sm text-green-300 font-medium">{t('queue.registered') || 'Registered'}</div>
                <div className="text-xs text-gray-400 space-y-0.5 mt-1">
                  {result.discovered != null && <div>{t('queue.files_found') || 'Files found'}: <span className="text-gray-300">{result.discovered}</span></div>}
                  {result.jobs_created != null && <div>{t('queue.jobs_created') || 'Jobs created'}: <span className="text-green-400 font-bold">{result.jobs_created}</span></div>}
                  {result.skipped > 0 && <div>{t('queue.skipped') || 'Skipped'}: <span className="text-gray-500">{result.skipped}</span></div>}
                </div>
              </div>
            </div>
          )}

          {/* Error result */}
          {isError && result && (
            <div className="flex items-start gap-3 px-3 py-3 rounded bg-red-900/20 border border-red-800/30">
              <AlertCircle size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-sm text-red-300 font-medium">{t('queue.error') || 'Error'}</div>
                <div className="text-xs text-red-400/80 mt-1 break-all">{result.error || 'Unknown error'}</div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-1">
            {(isDone || isError) ? (
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-white bg-gray-600 hover:bg-gray-500 rounded transition-colors"
              >
                {t('action.close') || 'Close'}
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={onClose}
                  disabled={isScanning}
                  className="px-4 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors disabled:opacity-30"
                >
                  {t('action.cancel')}
                </button>
                <button
                  type="submit"
                  disabled={isScanning}
                  className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-500 rounded transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {isScanning && <Loader2 size={14} className="animate-spin" />}
                  {t('archiving.action_enqueue')}
                </button>
              </>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
