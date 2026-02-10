import React, { useState, useEffect } from 'react';
import { Loader2, CheckCircle, AlertTriangle, Terminal, X, Download } from 'lucide-react';
import { useLocale } from '../i18n';
import RegisteredFoldersPanel from './RegisteredFoldersPanel';

const SettingsModal = ({ onClose }) => {
    const { t } = useLocale();
    const [status, setStatus] = useState(null); // { dependencies: {...}, dependencies_ok: bool, model_status: str }
    const [loading, setLoading] = useState(true);
    const [installing, setInstalling] = useState(false);
    const [logs, setLogs] = useState([]);
    const [aiMode, setAiMode] = useState(null); // { auto_detect: bool, override: string | null }
    const [tierChanging, setTierChanging] = useState(false);
    const [batchMode, setBatchMode] = useState(null); // { enabled: bool }
    const [batchChanging, setBatchChanging] = useState(false);

    useEffect(() => {
        checkStatus();
        loadAiMode();
        loadBatchMode();

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

    const loadAiMode = async () => {
        try {
            const result = await window.electron?.pipeline?.getConfig();
            if (result?.success) {
                const config = result.config;
                setAiMode(config.ai_mode || { auto_detect: true, override: null });
            }
        } catch (e) {
            console.error('Failed to load AI mode:', e);
        }
    };

    const loadBatchMode = async () => {
        try {
            const result = await window.electron?.pipeline?.getConfig();
            if (result?.success) {
                const config = result.config;
                // Default to enabled if not explicitly set
                setBatchMode({ enabled: config.batch_processing?.enabled !== false });
            }
        } catch (e) {
            console.error('Failed to load batch mode:', e);
            setBatchMode({ enabled: true }); // Default to enabled
        }
    };

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

    const handleTierChange = async (value) => {
        setTierChanging(true);
        try {
            // Update config
            if (value === 'auto') {
                await window.electron?.pipeline?.updateConfig('ai_mode.auto_detect', true);
                await window.electron?.pipeline?.updateConfig('ai_mode.override', null);
            } else {
                await window.electron?.pipeline?.updateConfig('ai_mode.auto_detect', false);
                await window.electron?.pipeline?.updateConfig('ai_mode.override', value);
            }

            // Reload config
            await loadAiMode();

            // Show restart notification
            alert('AI Tier changed successfully. Please restart the application for changes to take effect.');
        } catch (e) {
            console.error('Failed to update tier:', e);
            alert('Failed to update AI Tier: ' + e.message);
        } finally {
            setTierChanging(false);
        }
    };

    const handleBatchModeChange = async (enabled) => {
        setBatchChanging(true);
        try {
            // Update config
            await window.electron?.pipeline?.updateConfig('batch_processing.enabled', enabled);

            // Reload config
            await loadBatchMode();

            // Show notification
            alert(t('settings.batch_restart_required'));
        } catch (e) {
            console.error('Failed to update batch mode:', e);
            alert('Failed to update batch mode: ' + e.message);
        } finally {
            setBatchChanging(false);
        }
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

                            {/* AI Model Tier Selection */}
                            {aiMode && (
                                <div className="bg-gray-900/50 rounded border border-gray-700 p-4">
                                    <h3 className="text-sm font-bold text-gray-400 mb-3">AI Model Tier</h3>
                                    <div className="space-y-3">
                                        <select
                                            value={aiMode.auto_detect ? 'auto' : (aiMode.override || 'auto')}
                                            onChange={(e) => handleTierChange(e.target.value)}
                                            disabled={tierChanging}
                                            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white text-sm focus:border-blue-500 focus:outline-none disabled:opacity-50"
                                        >
                                            <option value="auto">Auto-Detect (Recommended)</option>
                                            <option value="standard">Standard (~6GB VRAM)</option>
                                            <option value="pro">Pro (8-16GB VRAM)</option>
                                            <option value="ultra">Ultra (20GB+ VRAM)</option>
                                        </select>

                                        {/* Current Mode Display */}
                                        <div className="text-xs text-gray-500 space-y-1">
                                            <div>
                                                <span className="font-bold">Current Mode:</span>{' '}
                                                <span className="text-gray-400 font-mono">
                                                    {aiMode.auto_detect ? 'AUTO-DETECT' : (aiMode.override || 'AUTO').toUpperCase()}
                                                </span>
                                            </div>
                                            {!aiMode.auto_detect && aiMode.override && (
                                                <div className="text-amber-500">
                                                    ⚠️ Manual override active. Auto-detection disabled.
                                                </div>
                                            )}
                                        </div>

                                        {/* Tier Descriptions */}
                                        <div className="text-xs text-gray-500 bg-gray-800/50 rounded p-2 space-y-1">
                                            <div><span className="font-bold">Standard:</span> Moondream2, SigLIP-base (fastest, ~6GB)</div>
                                            <div><span className="font-bold">Pro:</span> Qwen3-VL-4B, SigLIP-so400m (balanced, 8-16GB)</div>
                                            <div><span className="font-bold">Ultra:</span> Qwen3-VL-8B, SigLIP-giant (highest quality, 20GB+)</div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Batch Processing Mode */}
                            {batchMode && (
                                <div className="bg-gray-900/50 rounded border border-gray-700 p-4">
                                    <h3 className="text-sm font-bold text-gray-400 mb-3">{t('settings.batch_mode_title')}</h3>
                                    <div className="space-y-3">
                                        <div className="flex items-center justify-between">
                                            <div className="flex-1">
                                                <div className="text-sm text-gray-300">{t('settings.batch_mode_desc')}</div>
                                                <div className="text-xs text-gray-500 mt-1">
                                                    <span className="font-bold">Status:</span>{' '}
                                                    <span className={batchMode.enabled ? 'text-green-400' : 'text-gray-400'}>
                                                        {batchMode.enabled ? t('settings.batch_enabled') : t('settings.batch_disabled')}
                                                    </span>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => handleBatchModeChange(!batchMode.enabled)}
                                                disabled={batchChanging}
                                                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 ${
                                                    batchMode.enabled ? 'bg-blue-600' : 'bg-gray-600'
                                                }`}
                                            >
                                                <span
                                                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                                                        batchMode.enabled ? 'translate-x-6' : 'translate-x-1'
                                                    }`}
                                                />
                                            </button>
                                        </div>

                                        {/* Description */}
                                        <div className="text-xs text-gray-500 bg-gray-800/50 rounded p-2">
                                            <div><span className="font-bold">Enabled:</span> Adaptive batch sizing (1→2→3→5→8→10...) for optimal speed</div>
                                            <div className="mt-1"><span className="font-bold">Disabled:</span> Sequential processing (batch_size=1)</div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Registered Folders */}
                            <RegisteredFoldersPanel />

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
