import React, { useState } from 'react';
import { X, Loader2, CheckCircle, AlertCircle, Globe, ShieldOff, Folder, ChevronRight, ChevronDown } from 'lucide-react';
import { useLocale } from '../i18n';

/**
 * Modal dialog for adding a new WebDAV connection.
 * After successful test, shows folder browser for remote_path selection.
 */
const WebDAVConnectDialog = ({ onClose, onAdded }) => {
    const { t } = useLocale();
    const [form, setForm] = useState({
        name: '',
        url: '',
        remote_path: '/',
        username: '',
        password: '',
        verify_ssl: true,
    });
    const [testing, setTesting] = useState(false);
    const [testResult, setTestResult] = useState(null);
    const [adding, setAdding] = useState(false);

    // Folder browser state
    const [showFolders, setShowFolders] = useState(false);
    const [folders, setFolders] = useState([]);
    const [loadingFolders, setLoadingFolders] = useState(false);
    const [currentBrowsePath, setCurrentBrowsePath] = useState('/');
    const [browseHistory, setBrowseHistory] = useState(['/']);

    const updateField = (field, value) => {
        setForm(prev => ({ ...prev, [field]: value }));
        if (field !== 'remote_path') {
            setTestResult(null);
            setShowFolders(false);
        }
    };

    const isFormValid = form.url && form.username && form.password;

    const handleTest = async () => {
        if (!isFormValid) return;
        setTesting(true);
        setTestResult(null);
        setShowFolders(false);
        try {
            const result = await window.electron?.webdav?.testConnection({
                url: form.url,
                username: form.username,
                password: form.password,
                remote_path: form.remote_path || '/',
                verify_ssl: form.verify_ssl,
            });
            setTestResult(result);

            // On success, auto-browse root folders
            if (result?.success) {
                setShowFolders(true);
                browseFolders('/');
            }
        } catch (e) {
            setTestResult({ success: false, message: e.message });
        } finally {
            setTesting(false);
        }
    };

    const browseFolders = async (browsePath) => {
        setLoadingFolders(true);
        try {
            const result = await window.electron?.webdav?.browseFolders({
                url: form.url,
                username: form.username,
                password: form.password,
                remote_path: '/',
                verify_ssl: form.verify_ssl,
                path: browsePath === '/' ? null : browsePath,
            });
            if (result?.success) {
                setFolders(result.folders || []);
                setCurrentBrowsePath(browsePath);
            }
        } catch (e) {
            console.error('Failed to browse folders:', e);
        } finally {
            setLoadingFolders(false);
        }
    };

    const handleFolderClick = (folderPath) => {
        setBrowseHistory(prev => [...prev, folderPath]);
        browseFolders(folderPath);
    };

    const handleFolderSelect = (folderPath) => {
        updateField('remote_path', folderPath);
    };

    const handleBrowseBack = () => {
        if (browseHistory.length <= 1) return;
        const newHistory = browseHistory.slice(0, -1);
        setBrowseHistory(newHistory);
        browseFolders(newHistory[newHistory.length - 1]);
    };

    const handleAdd = async () => {
        if (!testResult?.success) return;
        setAdding(true);
        try {
            const result = await window.electron?.webdav?.addSource(form);
            if (result?.success) {
                onAdded?.(result.source);
                onClose();
            }
        } catch (e) {
            console.error('Failed to add WebDAV source:', e);
        } finally {
            setAdding(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
            <div
                className="bg-gray-800 rounded-lg border border-gray-600 shadow-2xl w-full max-w-md mx-4 max-h-[90vh] overflow-y-auto custom-scrollbar"
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-700">
                    <div className="flex items-center gap-2">
                        <Globe size={18} className="text-blue-400" />
                        <h2 className="text-sm font-bold text-gray-200">{t('webdav.connect_title')}</h2>
                    </div>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
                        <X size={18} />
                    </button>
                </div>

                {/* Form */}
                <div className="p-4 space-y-3">
                    {/* Display Name */}
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">{t('webdav.name')}</label>
                        <input
                            type="text"
                            value={form.name}
                            onChange={e => updateField('name', e.target.value)}
                            placeholder="My NAS"
                            className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                        />
                    </div>

                    {/* URL */}
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">{t('webdav.url')} *</label>
                        <input
                            type="text"
                            value={form.url}
                            onChange={e => updateField('url', e.target.value)}
                            placeholder="https://192.168.1.100:5006"
                            className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                        />
                    </div>

                    {/* Username & Password */}
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="block text-xs text-gray-400 mb-1">{t('webdav.username')} *</label>
                            <input
                                type="text"
                                value={form.username}
                                onChange={e => updateField('username', e.target.value)}
                                className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                            />
                        </div>
                        <div>
                            <label className="block text-xs text-gray-400 mb-1">{t('webdav.password')} *</label>
                            <input
                                type="password"
                                value={form.password}
                                onChange={e => updateField('password', e.target.value)}
                                className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:outline-none"
                            />
                        </div>
                    </div>

                    {/* SSL Toggle */}
                    <div className="flex items-center justify-between p-2 bg-gray-900/50 rounded">
                        <div className="flex items-center gap-2">
                            <ShieldOff size={14} className={form.verify_ssl ? 'text-gray-500' : 'text-yellow-400'} />
                            <span className="text-xs text-gray-400">{t('webdav.verify_ssl')}</span>
                        </div>
                        <button
                            onClick={() => updateField('verify_ssl', !form.verify_ssl)}
                            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                                form.verify_ssl ? 'bg-blue-600' : 'bg-gray-600'
                            }`}
                        >
                            <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                                form.verify_ssl ? 'translate-x-4.5' : 'translate-x-0.5'
                            }`} />
                        </button>
                    </div>

                    {!form.verify_ssl && (
                        <div className="text-[10px] text-yellow-500 flex items-center gap-1">
                            <AlertCircle size={10} />
                            {t('webdav.ssl_warning')}
                        </div>
                    )}

                    {/* Test Result */}
                    {testResult && (
                        <div className={`p-2 rounded text-xs flex items-center gap-2 ${
                            testResult.success
                                ? 'bg-green-900/30 border border-green-700 text-green-400'
                                : 'bg-red-900/30 border border-red-700 text-red-400'
                        }`}>
                            {testResult.success
                                ? <CheckCircle size={14} />
                                : <AlertCircle size={14} />
                            }
                            {testResult.success
                                ? t('webdav.test_success', { count: testResult.file_count || 0 })
                                : `${t('webdav.test_fail')}: ${testResult.message}`
                            }
                        </div>
                    )}

                    {/* Folder Browser (shown after successful test) */}
                    {showFolders && (
                        <div className="border border-gray-600 rounded overflow-hidden">
                            <div className="bg-gray-900 px-3 py-2 flex items-center justify-between border-b border-gray-700">
                                <div className="flex items-center gap-2">
                                    <Folder size={12} className="text-yellow-400" />
                                    <span className="text-xs text-gray-300 font-medium">{t('webdav.select_folder')}</span>
                                </div>
                                {browseHistory.length > 1 && (
                                    <button
                                        onClick={handleBrowseBack}
                                        className="text-[10px] text-blue-400 hover:text-blue-300"
                                    >
                                        ../{browseHistory[browseHistory.length - 2]?.split('/').filter(Boolean).pop() || '/'}
                                    </button>
                                )}
                            </div>

                            {/* Current path display */}
                            <div className="bg-gray-900/50 px-3 py-1.5 border-b border-gray-700">
                                <div className="text-[10px] text-gray-500 font-mono truncate">
                                    {currentBrowsePath}
                                </div>
                            </div>

                            {/* Folder list */}
                            <div className="max-h-40 overflow-y-auto custom-scrollbar bg-gray-850">
                                {loadingFolders ? (
                                    <div className="flex justify-center py-4">
                                        <Loader2 size={16} className="animate-spin text-blue-500" />
                                    </div>
                                ) : folders.length === 0 ? (
                                    <div className="text-[10px] text-gray-500 text-center py-4">
                                        {t('webdav.no_subfolders')}
                                    </div>
                                ) : (
                                    folders.map((folder, idx) => (
                                        <div
                                            key={idx}
                                            className={`flex items-center gap-2 px-3 py-1.5 hover:bg-gray-700 cursor-pointer text-xs ${
                                                form.remote_path === folder.path ? 'bg-blue-900/30 text-blue-300' : 'text-gray-300'
                                            }`}
                                        >
                                            <Folder size={12} className="text-yellow-500 flex-shrink-0" />
                                            <span
                                                className="flex-1 truncate"
                                                onClick={() => handleFolderSelect(folder.path)}
                                            >
                                                {folder.name}
                                            </span>
                                            <button
                                                onClick={() => handleFolderClick(folder.path)}
                                                className="text-gray-500 hover:text-gray-300 flex-shrink-0"
                                            >
                                                <ChevronRight size={12} />
                                            </button>
                                        </div>
                                    ))
                                )}
                            </div>

                            {/* Use root button */}
                            <div className="bg-gray-900/50 px-3 py-1.5 border-t border-gray-700">
                                <button
                                    onClick={() => handleFolderSelect(currentBrowsePath)}
                                    className="text-[10px] text-blue-400 hover:text-blue-300"
                                >
                                    {t('webdav.use_current_folder')}
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Selected remote path display */}
                    {testResult?.success && (
                        <div className="flex items-center gap-2 p-2 bg-gray-900/50 rounded">
                            <Folder size={12} className="text-yellow-400" />
                            <span className="text-xs text-gray-400">{t('webdav.remote_path')}:</span>
                            <span className="text-xs text-gray-200 font-mono">{form.remote_path}</span>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex justify-end gap-2 p-4 border-t border-gray-700">
                    <button
                        onClick={handleTest}
                        disabled={!isFormValid || testing}
                        className="px-4 py-2 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded disabled:opacity-40 flex items-center gap-1"
                    >
                        {testing ? <Loader2 size={12} className="animate-spin" /> : <Globe size={12} />}
                        {testing ? t('webdav.testing') : t('webdav.test_connection')}
                    </button>
                    <button
                        onClick={handleAdd}
                        disabled={!testResult?.success || adding}
                        className="px-4 py-2 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded disabled:opacity-40 flex items-center gap-1"
                    >
                        {adding && <Loader2 size={12} className="animate-spin" />}
                        {t('webdav.add')}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default WebDAVConnectDialog;
