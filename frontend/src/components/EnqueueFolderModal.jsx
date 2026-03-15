/**
 * EnqueueFolderModal — Quick popup for adding a folder to the processing queue.
 *
 * Opens from Sidebar right-click → "큐에 추가".
 * Pre-fills work request name with the folder name.
 * On confirm → calls onConfirm(folderPath, requestName).
 */

import { useState, useEffect, useRef } from 'react';
import { FolderInput, X } from 'lucide-react';
import { useLocale } from '../i18n';

export default function EnqueueFolderModal({ folderPath, folderName, onConfirm, onClose }) {
  const { t } = useLocale();
  const [requestName, setRequestName] = useState(folderName || '');
  const [includeSubfolders, setIncludeSubfolders] = useState(true);
  const inputRef = useRef(null);

  useEffect(() => {
    // Auto-focus and select text on open
    const timer = setTimeout(() => inputRef.current?.select(), 50);
    return () => clearTimeout(timer);
  }, []);

  // Close on Escape
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const name = requestName.trim() || folderName;
    onConfirm(folderPath, name, includeSubfolders);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
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
          <button onClick={onClose} className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white">
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Folder path display */}
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
              className="w-full px-3 py-2 text-sm bg-gray-900 border border-gray-600 rounded text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
            />
          </div>

          {/* Include subfolders checkbox */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={includeSubfolders}
              onChange={(e) => setIncludeSubfolders(e.target.checked)}
              className="rounded border-gray-600 bg-gray-900 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-300">{t('queue.include_subfolders')}</span>
          </label>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors"
            >
              {t('action.cancel')}
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-500 rounded transition-colors"
            >
              {t('archiving.action_enqueue')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
