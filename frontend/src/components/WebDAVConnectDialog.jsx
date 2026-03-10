import React, { useState } from 'react';
import { X, Loader2, CheckCircle, AlertCircle, Globe, ShieldOff } from 'lucide-react';
import { useLocale } from '../i18n';

/**
 * Modal dialog for adding a new WebDAV connection.
 *
 * Props:
 *  - onClose: close the dialog
 *  - onAdded(source): callback after successfully adding a source
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

    const updateField = (field, value) => {
        setForm(prev => ({ ...prev, [field]: value }));
        setTestResult(null); // Reset test on any change
    };

    const isFormValid = form.url && form.username && form.password;

    const handleTest = async () => {
        if (!isFormValid) return;
        setTesting(true);
        setTestResult(null);
        try {
            const result = await window.electron?.webdav?.testConnection({
                url: form.url,
                username: form.username,
                password: form.password,
                remote_path: form.remote_path || '/',
                verify_ssl: form.verify_ssl,
            });
            setTestResult(result);
        } catch (e) {
            setTestResult({ success: false, message: e.message });
        } finally {
            setTesting(false);
        }
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
                className="bg-gray-800 rounded-lg border border-gray-600 shadow-2xl w-full max-w-md mx-4"
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

                    {/* Remote Path */}
                    <div>
                        <label className="block text-xs text-gray-400 mb-1">{t('webdav.remote_path')}</label>
                        <input
                            type="text"
                            value={form.remote_path}
                            onChange={e => updateField('remote_path', e.target.value)}
                            placeholder="/"
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
