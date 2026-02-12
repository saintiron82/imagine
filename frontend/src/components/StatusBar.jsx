import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Terminal, X, Loader2, Square, Cpu } from 'lucide-react';
import { useLocale } from '../i18n';

function formatEta(ms) {
    const sec = Math.ceil(ms / 1000);
    if (sec < 60) return `${sec}s`;
    const min = Math.floor(sec / 60);
    const remSec = sec % 60;
    if (min < 60) return `${min}m ${remSec}s`;
    const hr = Math.floor(min / 60);
    const remMin = min % 60;
    return `${hr}h ${remMin}m`;
}

const StatusBar = ({ logs, clearLogs, isProcessing, isDiscovering = false, discoverProgress = '', processed = 0, total = 0, skipped = 0, currentFile = '', etaMs = null, phaseIdx = 0, phaseName = '', phaseCurrent = 0, phaseTotal = 0, fileStep = {}, onStop }) => {
    const { t } = useLocale();
    const [isOpen, setIsOpen] = useState(false);
    const [aiTier, setAiTier] = useState(null);
    const endRef = useRef(null);

    // Load AI Tier info
    useEffect(() => {
        const loadTierInfo = async () => {
            try {
                const result = await window.electron?.pipeline?.getConfig();
                if (result?.success) {
                    const config = result.config;
                    const aiMode = config.ai_mode || {};
                    setAiTier({
                        auto: aiMode.auto_detect !== false,
                        override: aiMode.override || null,
                    });
                }
            } catch (e) {
                console.error('Failed to load AI tier:', e);
            }
        };
        loadTierInfo();
    }, []);

    // Auto scroll to bottom
    useEffect(() => {
        if (endRef.current && isOpen) {
            endRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs, isOpen]);

    // Count errors (memoized to avoid re-scanning on every render)
    const errorCount = useMemo(() => logs.filter(l => l.type === 'error').length, [logs]);
    const latestLog = logs.length > 0 ? logs[logs.length - 1] : null;
    const phasePct = phaseTotal > 0 ? Math.round((phaseCurrent / phaseTotal) * 100) : 0;

    return (
        <div className={`fixed bottom-0 left-0 right-0 bg-gray-900 border-t border-gray-700 transition-all ${isOpen ? 'h-64' : 'h-8'}`}>
            {/* Header / Collapsed View */}
            <div
                className="h-8 bg-blue-900 flex items-center px-4 justify-between cursor-pointer hover:bg-blue-800 text-xs"
                onClick={() => setIsOpen(!isOpen)}
            >
                <div className="flex items-center space-x-2 min-w-0 flex-1">
                    <Terminal size={14} className="flex-shrink-0" />
                    <span className="font-bold flex-shrink-0">{t('label.output')}</span>
                    {errorCount > 0 && (
                        <span className="bg-red-500 text-white px-1 rounded flex-shrink-0">{t('status.errors', { count: errorCount })}</span>
                    )}
                    {!isOpen && latestLog && !isProcessing && (
                        <span className="text-gray-300 truncate max-w-lg border-l border-blue-700 pl-2">
                            {latestLog.message}
                        </span>
                    )}
                </div>

                {/* Phase-based progress */}
                {isProcessing && (
                    <div className="flex items-center space-x-3 flex-shrink-0 mx-4" onClick={(e) => e.stopPropagation()}>
                        <Loader2 className="animate-spin text-blue-400" size={14} />

                        {/* Phase indicator: 4 dots showing which phase is active */}
                        <div className="flex items-center space-x-1">
                            {['P', 'V', 'E', 'S'].map((label, i) => (
                                <span key={i} className={`w-4 h-4 rounded text-[9px] font-bold flex items-center justify-center ${
                                    i < phaseIdx ? 'bg-green-600 text-white' :
                                    i === phaseIdx ? 'bg-blue-500 text-white animate-pulse' :
                                    'bg-gray-700 text-gray-500'
                                }`}>{label}</span>
                            ))}
                        </div>

                        {/* Current phase progress bar */}
                        <div className="flex items-center space-x-1.5">
                            <span className="text-blue-300 font-medium w-20 text-right">
                                {phaseName} {phaseCurrent}/{phaseTotal}
                            </span>
                            <div className="w-28 bg-gray-700 rounded-full h-2 overflow-hidden">
                                <div
                                    className="h-full bg-blue-400 transition-all duration-300 rounded-full"
                                    style={{ width: `${phasePct}%` }}
                                />
                            </div>
                        </div>

                        {/* Stored count */}
                        <span className="text-gray-500 text-[11px]">
                            {processed > 0 && `${processed} stored`}
                            {skipped > 0 && ` ${skipped} skipped`}
                        </span>

                        {/* Current file name */}
                        <span className="text-gray-400 truncate max-w-[120px]">{currentFile?.split(/[/\\]/).pop()}</span>

                        {etaMs != null && (
                            <span className="text-gray-500 font-mono text-[11px]">{formatEta(etaMs)}</span>
                        )}

                        <button
                            onClick={onStop}
                            className="p-0.5 rounded hover:bg-red-900/50 text-red-400 hover:text-red-300 transition-colors"
                            title={t('action.stop_processing')}
                        >
                            <Square size={12} />
                        </button>
                    </div>
                )}

                {/* Discover progress */}
                {isDiscovering && !isProcessing && (
                    <div className="flex items-center space-x-2 flex-shrink-0 mx-4" onClick={(e) => e.stopPropagation()}>
                        <Loader2 className="animate-spin text-green-400" size={14} />
                        <span className="text-green-300 text-xs font-medium">{t('status.discovering')}</span>
                        <span className="text-gray-400 text-xs truncate max-w-[250px]">{discoverProgress}</span>
                    </div>
                )}

                {/* AI Tier Display */}
                {aiTier && !isProcessing && !isDiscovering && (
                    <div className="flex items-center gap-1.5 text-xs text-gray-400 border-l border-blue-700 pl-3 mr-3">
                        <Cpu size={12} className="flex-shrink-0" />
                        <span className="font-mono">
                            {aiTier.auto ? 'AUTO' : (aiTier.override || 'AUTO').toUpperCase()}
                        </span>
                    </div>
                )}

                <div className="flex-shrink-0">
                    {isOpen ? <X size={14} /> : t('label.show_logs')}
                </div>
            </div>

            {/* Expanded Log View */}
            {isOpen && (
                <div className="h-56 overflow-y-auto p-2 font-mono text-xs bg-black text-gray-300">
                    {logs.length === 0 && <div className="text-gray-600 italic">{t('msg.no_logs')}</div>}
                    {logs.map((log, i) => (
                        <div key={i} className={`mb-1 break-words ${log.type === 'error' ? 'text-red-400' :
                                log.type === 'success' ? 'text-green-400' :
                                log.type === 'warning' ? 'text-yellow-400' : 'text-gray-300'
                            }`}>
                            <span className="opacity-50 mr-2">[{log.timestamp || ''}]</span>
                            {log.message}
                        </div>
                    ))}
                    <div ref={endRef} />
                </div>
            )}
        </div>
    );
};

export default StatusBar;
