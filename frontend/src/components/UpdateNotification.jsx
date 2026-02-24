import React, { useState, useEffect, useCallback } from 'react';
import { Download, RefreshCw, X, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';
import { useLocale } from '../i18n';

const UpdateNotification = () => {
    const { t } = useLocale();
    const [state, setState] = useState('idle');
    const [updateInfo, setUpdateInfo] = useState(null);
    const [progress, setProgress] = useState({ percent: 0 });
    const [errorMsg, setErrorMsg] = useState('');

    useEffect(() => {
        const updater = window.electron?.updater;
        if (!updater) return;

        updater.onChecking(() => setState('checking'));
        updater.onAvailable((info) => {
            setUpdateInfo(info);
            setState('available');
        });
        updater.onNotAvailable(() => setState('idle'));
        updater.onProgress((p) => {
            setProgress(p);
            setState('downloading');
        });
        updater.onDownloaded((info) => {
            setUpdateInfo(info);
            setState('downloaded');
        });
        updater.onError((err) => {
            setErrorMsg(err.message);
            setState('error');
        });

        return () => {
            updater.offChecking();
            updater.offAvailable();
            updater.offNotAvailable();
            updater.offProgress();
            updater.offDownloaded();
            updater.offError();
        };
    }, []);

    const handleDownload = useCallback(() => {
        window.electron?.updater?.download();
    }, []);

    const handleInstall = useCallback(() => {
        window.electron?.updater?.quitAndInstall();
    }, []);

    const handleDismiss = useCallback(() => {
        setState('idle');
    }, []);

    const handleRetry = useCallback(async () => {
        setState('checking');
        await window.electron?.updater?.check();
    }, []);

    if (state === 'idle' || state === 'checking' || !window.electron?.updater) return null;

    return (
        <div className="fixed top-4 right-4 z-50 w-80 rounded-lg shadow-2xl border backdrop-blur-sm animate-in slide-in-from-top-2 duration-300"
            style={{
                backgroundColor: state === 'downloaded' ? 'rgba(20, 83, 45, 0.95)' :
                    state === 'error' ? 'rgba(127, 29, 29, 0.95)' : 'rgba(30, 58, 138, 0.95)',
                borderColor: state === 'downloaded' ? 'rgb(34, 197, 94)' :
                    state === 'error' ? 'rgb(239, 68, 68)' : 'rgb(59, 130, 246)',
            }}
        >
            <div className="p-3.5">
                {/* Header */}
                <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                        {state === 'available' && <Download size={16} className="text-blue-300 flex-shrink-0" />}
                        {state === 'downloading' && <Loader2 size={16} className="text-blue-300 flex-shrink-0 animate-spin" />}
                        {state === 'downloaded' && <CheckCircle size={16} className="text-green-300 flex-shrink-0" />}
                        {state === 'error' && <AlertTriangle size={16} className="text-red-300 flex-shrink-0" />}
                        <span className="text-sm font-medium text-white">
                            {state === 'available' && t('update.available', { version: updateInfo?.version })}
                            {state === 'downloading' && t('update.downloading')}
                            {state === 'downloaded' && t('update.downloaded')}
                            {state === 'error' && t('update.error')}
                        </span>
                    </div>
                    <button onClick={handleDismiss} className="text-white/50 hover:text-white/80 transition-colors p-0.5">
                        <X size={14} />
                    </button>
                </div>

                {/* Description */}
                <p className="text-xs text-white/70 mb-3">
                    {state === 'available' && t('update.available_desc')}
                    {state === 'downloading' && t('update.download_percent', { percent: progress.percent || 0 })}
                    {state === 'downloaded' && t('update.downloaded_desc')}
                    {state === 'error' && errorMsg}
                </p>

                {/* Progress bar (downloading) */}
                {state === 'downloading' && (
                    <div className="w-full h-1.5 bg-blue-900/50 rounded-full mb-3 overflow-hidden">
                        <div
                            className="h-full bg-blue-400 rounded-full transition-all duration-300"
                            style={{ width: `${progress.percent || 0}%` }}
                        />
                    </div>
                )}

                {/* Actions */}
                <div className="flex gap-2">
                    {state === 'available' && (
                        <>
                            <button onClick={handleDownload}
                                className="flex-1 px-3 py-1.5 text-xs font-medium bg-blue-500 hover:bg-blue-400 text-white rounded-md transition-colors">
                                {t('update.download')}
                            </button>
                            <button onClick={handleDismiss}
                                className="px-3 py-1.5 text-xs text-white/60 hover:text-white/90 transition-colors">
                                {t('update.later')}
                            </button>
                        </>
                    )}
                    {state === 'downloaded' && (
                        <>
                            <button onClick={handleInstall}
                                className="flex-1 px-3 py-1.5 text-xs font-medium bg-green-600 hover:bg-green-500 text-white rounded-md transition-colors">
                                {t('update.restart_now')}
                            </button>
                            <button onClick={handleDismiss}
                                className="px-3 py-1.5 text-xs text-white/60 hover:text-white/90 transition-colors">
                                {t('update.restart_later')}
                            </button>
                        </>
                    )}
                    {state === 'error' && (
                        <>
                            <button onClick={handleRetry}
                                className="flex-1 px-3 py-1.5 text-xs font-medium bg-red-600 hover:bg-red-500 text-white rounded-md transition-colors flex items-center justify-center gap-1">
                                <RefreshCw size={12} />
                                {t('update.error_retry')}
                            </button>
                            <button onClick={handleDismiss}
                                className="px-3 py-1.5 text-xs text-white/60 hover:text-white/90 transition-colors">
                                {t('update.later')}
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default UpdateNotification;
