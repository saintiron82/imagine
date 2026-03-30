/**
 * DiscoverPanel — server-side folder browsing + scan trigger.
 * Extracted from AdminPage.jsx.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import { browseFolders, scanFolder } from '../../api/admin';
import {
  Folder, ArrowUp, Play, Loader2,
} from 'lucide-react';


export default function DiscoverPanel() {
  const { t } = useLocale();
  const [currentPath, setCurrentPath] = useState('/');
  const [pathInput, setPathInput] = useState('/');
  const [browseData, setBrowseData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scanning, setScanning] = useState(null); // folder name being scanned
  const [scanResult, setScanResult] = useState(null);
  const [priority, setPriority] = useState(0);
  const [error, setError] = useState('');

  const browse = useCallback(async (path) => {
    setLoading(true);
    setError('');
    setScanResult(null);
    try {
      const data = await browseFolders(path);
      setBrowseData(data);
      setCurrentPath(data.current);
      setPathInput(data.current);
    } catch (e) {
      setError(e.detail || e.message || 'Browse failed');
      setBrowseData(null);
    }
    setLoading(false);
  }, []);

  useEffect(() => { browse('/'); }, [browse]);

  const handleBrowse = () => browse(pathInput);
  const handleKeyDown = (e) => { if (e.key === 'Enter') handleBrowse(); };

  const handleNavigate = (dirName) => {
    const next = currentPath.endsWith('/') ? currentPath + dirName : currentPath + '/' + dirName;
    browse(next);
  };

  const handleGoUp = () => {
    if (browseData?.parent) browse(browseData.parent);
  };

  const handleScan = async (folderPath) => {
    setScanning(folderPath);
    setScanResult(null);
    try {
      const data = await scanFolder(folderPath, priority);
      setScanResult(data);
    } catch (e) {
      setScanResult({ success: false, error: e.detail || e.message });
    }
    setScanning(null);
  };

  const handleScanCurrent = () => handleScan(currentPath);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">{t('admin.discover_title')}</h2>

      {/* Path input */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="/mnt/nas/assets"
            className="flex-1 px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none font-mono"
          />
          <button
            onClick={handleBrowse}
            disabled={loading}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded-lg text-sm text-white"
          >
            {t('admin.discover_browse')}
          </button>
          <button
            onClick={handleScanCurrent}
            disabled={!!scanning || loading}
            className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 rounded-lg text-sm text-white"
          >
            {scanning === currentPath ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {t('admin.discover_scan_all')}
          </button>
        </div>

        {/* Priority */}
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span>{t('admin.discover_priority')}:</span>
          <input
            type="number"
            min={0}
            max={10}
            value={priority}
            onChange={(e) => setPriority(parseInt(e.target.value) || 0)}
            className="w-16 px-2 py-1 bg-gray-900 border border-gray-600 rounded text-sm text-white"
          />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 mb-4 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Scan result */}
      {scanResult && (
        <div className={`p-3 mb-4 rounded-lg text-xs border ${
          scanResult.success
            ? 'bg-green-900/30 border-green-800 text-green-400'
            : 'bg-red-900/30 border-red-800 text-red-400'
        }`}>
          {scanResult.success
            ? t('admin.discover_scan_result', {
                discovered: scanResult.discovered,
                jobs: scanResult.jobs_created,
                skipped: scanResult.skipped,
              })
            : scanResult.error}
        </div>
      )}

      {/* Directory listing */}
      {loading ? (
        <div className="text-gray-400 text-sm flex items-center gap-2">
          <Loader2 size={14} className="animate-spin" /> {t('status.loading')}
        </div>
      ) : browseData && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          {/* Current path header */}
          <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-700 bg-gray-800/50">
            <button
              onClick={handleGoUp}
              disabled={!browseData.parent}
              className="p-1 rounded hover:bg-gray-600 disabled:opacity-30 text-gray-400"
              title="Parent directory"
            >
              <ArrowUp size={16} />
            </button>
            <code className="text-xs text-gray-300 font-mono flex-1 truncate">{currentPath}</code>
            {browseData.files_count > 0 && (
              <span className="text-xs text-gray-500">
                {browseData.files_count} {t('admin.discover_files_here')}
              </span>
            )}
          </div>

          {/* Directories */}
          {browseData.dirs.length === 0 && browseData.files_count === 0 ? (
            <div className="text-center text-gray-500 py-8 text-sm">{t('admin.discover_empty')}</div>
          ) : (
            <div className="divide-y divide-gray-700/50">
              {browseData.dirs.map((dir) => {
                const fullPath = currentPath.endsWith('/')
                  ? currentPath + dir.name
                  : currentPath + '/' + dir.name;
                return (
                  <div
                    key={dir.name}
                    className="flex items-center gap-3 px-4 py-2.5 hover:bg-gray-700/30 group"
                  >
                    <Folder size={16} className="text-yellow-500 flex-shrink-0" />
                    <button
                      onClick={() => handleNavigate(dir.name)}
                      className="flex-1 text-left text-sm text-gray-200 hover:text-white truncate"
                    >
                      {dir.name}/
                    </button>
                    <span className="text-xs text-gray-500 flex-shrink-0">
                      {dir.files_count > 0 ? `${dir.files_count} files` : ''}
                    </span>
                    <button
                      onClick={() => handleScan(fullPath)}
                      disabled={!!scanning}
                      className="flex items-center gap-1 px-2.5 py-1 bg-gray-700 hover:bg-blue-600 rounded text-xs text-gray-300 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-30"
                    >
                      {scanning === fullPath ? (
                        <Loader2 size={12} className="animate-spin" />
                      ) : (
                        <Play size={12} />
                      )}
                      Scan
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
