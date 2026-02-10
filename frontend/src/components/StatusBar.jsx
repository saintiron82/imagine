import React, { useEffect, useRef, useState } from 'react';
import { Terminal, X, Loader2, Square, Cpu } from 'lucide-react';
import { useLocale } from '../i18n';

const StatusBar = ({ logs, clearLogs, isProcessing, isDiscovering = false, discoverProgress = '', processed = 0, total = 0, currentFile = '', fileStep = {}, onStop }) => {
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

    // Count errors
    const errorCount = logs.filter(l => l.type === 'error').length;
    const latestLog = logs.length > 0 ? logs[logs.length - 1] : null;
    const queuePct = total > 0 ? Math.round((processed / total) * 100) : 0;
    const stepPct = fileStep.totalSteps > 0 ? Math.round((fileStep.step / fileStep.totalSteps) * 100) : 0;

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

                {/* Dual progress bars */}
                {isProcessing && (
                    <div className="flex items-center space-x-3 flex-shrink-0 mx-4" onClick={(e) => e.stopPropagation()}>
                        <Loader2 className="animate-spin text-blue-400" size={14} />

                        {/* File step progress (inner) */}
                        <div className="flex items-center space-x-1.5">
                            <span className="text-gray-400 w-16 text-right truncate">{fileStep.stepName || '...'}</span>
                            <div className="w-20 bg-gray-700 rounded-full h-1.5 overflow-hidden">
                                <div
                                    className="h-full bg-green-400 transition-all duration-300 rounded-full"
                                    style={{ width: `${stepPct}%` }}
                                />
                            </div>
                        </div>

                        {/* Queue progress (outer) */}
                        <div className="flex items-center space-x-1.5">
                            <span className="text-blue-300 font-medium">{processed}/{total}</span>
                            <div className="w-24 bg-gray-700 rounded-full h-1.5 overflow-hidden">
                                <div
                                    className="h-full bg-blue-400 transition-all duration-300 rounded-full"
                                    style={{ width: `${queuePct}%` }}
                                />
                            </div>
                        </div>

                        {/* Current file name */}
                        <span className="text-gray-400 truncate max-w-[150px]">{currentFile?.split(/[/\\]/).pop()}</span>

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
                            <span className="opacity-50 mr-2">[{new Date().toLocaleTimeString()}]</span>
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
