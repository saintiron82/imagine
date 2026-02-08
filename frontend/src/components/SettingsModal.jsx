import React, { useState, useEffect } from 'react';
import { Loader2, CheckCircle, AlertTriangle, Terminal, X, Download } from 'lucide-react';
import { useLocale } from '../i18n';

const SettingsModal = ({ onClose }) => {
    const { t } = useLocale();
    const [status, setStatus] = useState(null); // { dependencies: {...}, dependencies_ok: bool, model_status: str }
    const [loading, setLoading] = useState(true);
    const [installing, setInstalling] = useState(false);
    const [logs, setLogs] = useState([]);

    useEffect(() => {
        checkStatus();

        // Subscribe to logs
        window.electron?.pipeline?.onInstallLog((data) => {
            setLogs(prev => [...prev, data]);
            if (data.done) {
                setInstalling(false);
                checkStatus(); // Re-check after install
            }
        });

        return () => window.electron?.pipeline?.offInstallLog();
    }, []);

    const checkStatus = async () => {
        setLoading(true);
        try {
            const res = await window.electron?.pipeline?.checkEnv();
            setStatus(res);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const handleInstall = () => {
        setInstalling(true);
        setLogs([]);
        window.electron?.pipeline?.installEnv();
    };

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 animate-fadeIn" onClick={onClose}>
            <div className="bg-gray-800 rounded-lg max-w-2xl w-full mx-4 shadow-2xl border border-gray-700 overflow-hidden" onClick={e => e.stopPropagation()}>
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-900/50">
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        <Terminal size={20} />
                        {t('settings.title')}
                    </h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-white"><X size={24} /></button>
                </div>

                {/* Content */}
                <div className="p-6">
                    {loading ? (
                        <div className="flex justify-center p-8"><Loader2 className="animate-spin text-blue-500" size={32} /></div>
                    ) : (
                        <div className="space-y-6">
                            {/* Status Card */}
                            <div className={`p-4 rounded-lg border flex items-center gap-4 ${status?.dependencies_ok ? 'bg-green-900/20 border-green-800' : 'bg-amber-900/20 border-amber-800'}`}>
                                {status?.dependencies_ok ? (
                                    <CheckCircle className="text-green-500" size={32} />
                                ) : (
                                    <AlertTriangle className="text-amber-500" size={32} />
                                )}
                                <div>
                                    <div className={`font-bold text-lg ${status?.dependencies_ok ? 'text-green-400' : 'text-amber-400'}`}>
                                        {status?.dependencies_ok ? t('settings.system_ok') : t('settings.deps_missing')}
                                    </div>
                                    <div className="text-gray-400 text-sm">
                                        {status?.dependencies_ok
                                            ? t('settings.all_ready')
                                            : t('settings.some_missing')}
                                    </div>
                                </div>
                            </div>

                            {/* Details Table */}
                            <div className="bg-gray-900/50 rounded border border-gray-700 p-3">
                                <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">{t('label.component_status')}</h3>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    {status?.dependencies && Object.entries(status.dependencies).map(([pkg, ok]) => (
                                        <div key={pkg} className="flex items-center justify-between p-2 rounded bg-gray-800">
                                            <span className="text-gray-300 font-mono">{pkg}</span>
                                            {ok ? <span className="text-green-500 text-xs font-bold">{t('status.installed')}</span> : <span className="text-red-500 text-xs font-bold">{t('status.missing')}</span>}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Install Button */}
                            {!installing && !status?.dependencies_ok && (
                                <button
                                    onClick={handleInstall}
                                    className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white rounded font-bold shadow-lg flex items-center justify-center gap-2 transition-all"
                                >
                                    <Download size={20} />
                                    {t('action.install_all')}
                                </button>
                            )}

                            {/* Force Install Option */}
                            {!installing && status?.dependencies_ok && (
                                <div className="text-right">
                                    <button
                                        onClick={handleInstall}
                                        className="text-xs text-gray-500 hover:text-gray-300 underline"
                                    >
                                        {t('action.reinstall_verify')}
                                    </button>
                                </div>
                            )}

                            {/* Logs Terminal */}
                            {installing && (
                                <div className="bg-black rounded border border-gray-700 p-4 h-48 overflow-y-auto font-mono text-xs custom-scrollbar">
                                    {logs.map((log, i) => (
                                        <div key={i} className={`mb-1 ${log.type === 'error' ? 'text-red-400' : log.type === 'warning' ? 'text-amber-400' : log.type === 'success' ? 'text-green-400' : 'text-gray-300'}`}>
                                            <span className="text-gray-600">[{new Date().toLocaleTimeString()}]</span> {log.message}
                                        </div>
                                    ))}
                                    <div ref={el => el?.scrollIntoView()} />
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SettingsModal;
