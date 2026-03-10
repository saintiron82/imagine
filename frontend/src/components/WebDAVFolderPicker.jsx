import React, { useState, useEffect } from 'react';
import { X, Loader2, Folder, ChevronRight, Globe, Check } from 'lucide-react';
import { useLocale } from '../i18n';

/**
 * Modal for browsing and selecting folders from a connected WebDAV NAS source.
 * Used from Sidebar's "Add Folder" dropdown when user selects a NAS source.
 *
 * Props:
 *  - source: { id, name, url, remote_path, ... } — the WebDAV source to browse
 *  - onSelect(webdavPath): called with 'webdav://source-id/selected/path'
 *  - onClose(): close modal
 */
const WebDAVFolderPicker = ({ source, onSelect, onClose }) => {
    const { t } = useLocale();
    const [folders, setFolders] = useState([]);
    const [loading, setLoading] = useState(true);
    const [currentPath, setCurrentPath] = useState(source.remote_path || '/');
    const [browseHistory, setBrowseHistory] = useState([source.remote_path || '/']);

    const browseFolders = async (browsePath) => {
        setLoading(true);
        try {
            const webdavPath = `webdav://${source.id}${browsePath}`;
            const result = await window.electron?.webdav?.browseFolders({ webdavPath });
            if (result?.success) {
                setFolders(result.folders || []);
                setCurrentPath(browsePath);
            }
        } catch (e) {
            console.error('Failed to browse WebDAV folders:', e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        browseFolders(source.remote_path || '/');
    }, [source.id]);

    const handleFolderOpen = (folderPath) => {
        setBrowseHistory(prev => [...prev, folderPath]);
        browseFolders(folderPath);
    };

    const handleBack = () => {
        if (browseHistory.length <= 1) return;
        const newHistory = browseHistory.slice(0, -1);
        setBrowseHistory(newHistory);
        browseFolders(newHistory[newHistory.length - 1]);
    };

    const handleSelectCurrent = () => {
        const webdavPath = `webdav://${source.id}${currentPath}`;
        onSelect(webdavPath);
    };

    const handleSelectFolder = (folderPath) => {
        const webdavPath = `webdav://${source.id}${folderPath}`;
        onSelect(webdavPath);
    };

    return (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
            <div
                className="bg-gray-800 rounded-lg border border-gray-600 shadow-2xl w-full max-w-md mx-4"
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <div className="flex items-center gap-2">
                        <Globe size={18} className="text-blue-400" />
                        <h2 className="text-sm font-bold text-gray-200">
                            {source.name || source.url}
                        </h2>
                    </div>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
                        <X size={18} />
                    </button>
                </div>

                {/* Current path + back */}
                <div className="bg-gray-900/50 px-4 py-2 border-b border-gray-700 flex items-center justify-between">
                    <div className="text-[11px] text-gray-500 font-mono truncate flex-1">
                        {currentPath}
                    </div>
                    {browseHistory.length > 1 && (
                        <button
                            onClick={handleBack}
                            className="text-[10px] text-blue-400 hover:text-blue-300 ml-2 flex-shrink-0"
                        >
                            ../{browseHistory[browseHistory.length - 2]?.split('/').filter(Boolean).pop() || '/'}
                        </button>
                    )}
                </div>

                {/* Folder list */}
                <div className="max-h-64 overflow-y-auto custom-scrollbar">
                    {loading ? (
                        <div className="flex justify-center py-8">
                            <Loader2 size={20} className="animate-spin text-blue-500" />
                        </div>
                    ) : folders.length === 0 ? (
                        <div className="text-xs text-gray-500 text-center py-8">
                            {t('webdav.no_subfolders')}
                        </div>
                    ) : (
                        folders.map((folder, idx) => (
                            <div
                                key={idx}
                                className="flex items-center gap-2 px-4 py-2 hover:bg-gray-700 group"
                            >
                                <Folder size={14} className="text-yellow-500 flex-shrink-0" />
                                <span
                                    className="flex-1 text-xs text-gray-300 truncate cursor-pointer hover:text-white"
                                    onClick={() => handleFolderOpen(folder.path)}
                                >
                                    {folder.name}
                                </span>
                                <button
                                    onClick={() => handleSelectFolder(folder.path)}
                                    className="text-gray-600 hover:text-green-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                                    title={t('webdav.select_folder')}
                                >
                                    <Check size={14} />
                                </button>
                                <button
                                    onClick={() => handleFolderOpen(folder.path)}
                                    className="text-gray-600 hover:text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                                >
                                    <ChevronRight size={14} />
                                </button>
                            </div>
                        ))
                    )}
                </div>

                {/* Footer */}
                <div className="flex justify-between items-center p-4 border-t border-gray-700">
                    <button
                        onClick={onClose}
                        className="px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200"
                    >
                        {t('action.cancel')}
                    </button>
                    <button
                        onClick={handleSelectCurrent}
                        className="px-4 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded flex items-center gap-1"
                    >
                        <Check size={12} />
                        {t('webdav.use_current_folder')}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default WebDAVFolderPicker;
