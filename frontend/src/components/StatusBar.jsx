import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Terminal, X, Loader2, Square, Cpu, Layers } from 'lucide-react';
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

/** Compact phase progress pill: label + count + mini bar */
function PhasePill({ label, count, total, isActive, color }) {
    const pct = total > 0 ? Math.min(100, Math.round((count / total) * 100)) : 0;
    const isDone = count >= total && total > 0;

    return (
        <div className={`flex items-center space-x-1 px-1.5 py-0.5 rounded ${
            isActive ? 'bg-gray-800 ring-1 ring-blue-500/50' : ''
        }`}>
            <span className={`text-[9px] font-bold whitespace-nowrap ${
                isDone ? 'text-green-400' :
                isActive ? 'text-blue-300' :
                count > 0 ? 'text-gray-300' : 'text-gray-600'
            }`}>{label}</span>
            <div className="w-12 bg-gray-700 rounded-full h-1.5 overflow-hidden">
                <div
                    className={`h-full rounded-full transition-all duration-300 ${
                        isDone ? 'bg-green-500' :
                        isActive ? `${color} animate-pulse` : color
                    }`}
                    style={{ width: `${pct}%` }}
                />
            </div>
            <span className={`text-[10px] font-mono w-8 text-right ${
                isDone ? 'text-green-400' :
                isActive ? 'text-blue-300' :
                count > 0 ? 'text-gray-400' : 'text-gray-600'
            }`}>{count}</span>
        </div>
    );
}

const StatusBar = ({
    logs, clearLogs, isProcessing, isDiscovering = false, discoverProgress = {},
    processed = 0, total = 0, skipped = 0, currentFile = '', etaMs = null,
    cumParse = 0, cumMC = 0, cumVV = 0, cumMV = 0,
    activePhase = 0, phaseSubCount = 0, phaseSubTotal = 0,
    batchInfo = '', fileStep = {}, onStop
}) => {
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

    // Phase data for pills
    const phases = [
        { label: t('status.phase.parse'), count: cumParse, color: 'bg-cyan-400' },
        { label: 'MC', count: cumMC, color: 'bg-blue-400' },
        { label: 'VV', count: cumVV, color: 'bg-purple-400' },
        { label: 'MV', count: cumMV, color: 'bg-green-400' },
    ];

    // Overall progress: stored / (total - skipped)
    const effectiveTotal = total - skipped;
    const overallPct = effectiveTotal > 0 ? Math.round((processed / effectiveTotal) * 100) : 0;

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

                {/* 4-Phase independent progress */}
                {isProcessing && (
                    <div className="flex items-center space-x-2 flex-shrink-0 mx-4" onClick={(e) => e.stopPropagation()}>
                        <Loader2 className="animate-spin text-blue-400" size={14} />

                        {/* 4 phase pills with independent progress */}
                        <div className="flex items-center space-x-0.5">
                            {phases.map((p, i) => (
                                <PhasePill
                                    key={p.label}
                                    label={p.label}
                                    count={p.count}
                                    total={total}
                                    isActive={i === activePhase}
                                    color={p.color}
                                />
                            ))}
                        </div>

                        {/* Overall stored / total */}
                        <div className="flex items-center space-x-1.5 border-l border-blue-700 pl-2">
                            <span className="text-green-300 font-mono font-bold text-[11px]">
                                {processed}/{effectiveTotal}
                            </span>
                            {skipped > 0 && (
                                <span className="text-gray-500 text-[10px]">+{skipped}skip</span>
                            )}
                        </div>

                        {/* Batch size indicator */}
                        {batchInfo && (() => {
                            const parts = batchInfo.split(':');
                            const batchNum = parts[0];
                            const batchTag = parts[1] || '';
                            return (
                                <div className="flex items-center gap-1 bg-yellow-900/40 border border-yellow-600/50 px-1.5 py-0.5 rounded" title={t('status.batch_size')}>
                                    <Layers size={11} className="text-yellow-400 flex-shrink-0" />
                                    <span className="text-yellow-300 font-mono font-bold text-[11px]">{batchNum}</span>
                                    {batchTag && <span className="text-yellow-500 font-mono text-[9px]">{batchTag}</span>}
                                </div>
                            );
                        })()}

                        {/* Current file name */}
                        <span className="text-gray-400 truncate max-w-[100px]">{currentFile?.split(/[/\\]/).pop()}</span>

                        {etaMs != null && (
                            <span className="text-blue-300 font-mono text-[11px]">~{formatEta(etaMs)}</span>
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

                {/* Discover progress — reuses same 4-phase pills as pipeline */}
                {isDiscovering && !isProcessing && (() => {
                    const dp = discoverProgress || {};
                    const hasPhaseData = dp.cumParse > 0 || dp.cumMC > 0 || dp.cumVV > 0 || dp.cumMV > 0;
                    const dPhases = [
                        { label: t('status.phase.parse'), count: dp.cumParse || 0, color: 'bg-cyan-400' },
                        { label: 'MC', count: dp.cumMC || 0, color: 'bg-blue-400' },
                        { label: 'VV', count: dp.cumVV || 0, color: 'bg-purple-400' },
                        { label: 'MV', count: dp.cumMV || 0, color: 'bg-green-400' },
                    ];
                    const dTotal = dp.total || 0;
                    const dProcessed = dp.processed || 0;
                    const dSkipped = dp.skipped || 0;
                    const dEffective = dTotal - dSkipped;

                    return (
                        <div className="flex items-center space-x-2 flex-shrink-0 mx-4" onClick={(e) => e.stopPropagation()}>
                            <Loader2 className="animate-spin text-green-400" size={14} />
                            {hasPhaseData ? (
                                <>
                                    <div className="flex items-center space-x-0.5">
                                        {dPhases.map((p, i) => (
                                            <PhasePill
                                                key={p.label}
                                                label={p.label}
                                                count={p.count}
                                                total={dTotal}
                                                isActive={i === (dp.activePhase || 0)}
                                                color={p.color}
                                            />
                                        ))}
                                    </div>
                                    <div className="flex items-center space-x-1.5 border-l border-green-700 pl-2">
                                        <span className="text-green-300 font-mono font-bold text-[11px]">
                                            {dProcessed}/{dEffective > 0 ? dEffective : dTotal}
                                        </span>
                                        {dSkipped > 0 && (
                                            <span className="text-gray-500 text-[10px]">+{dSkipped}skip</span>
                                        )}
                                    </div>
                                    {dp.batchInfo && (() => {
                                        const parts = dp.batchInfo.split(':');
                                        return (
                                            <div className="flex items-center gap-1 bg-yellow-900/40 border border-yellow-600/50 px-1.5 py-0.5 rounded" title={t('status.batch_size')}>
                                                <Layers size={11} className="text-yellow-400 flex-shrink-0" />
                                                <span className="text-yellow-300 font-mono font-bold text-[11px]">{parts[0]}</span>
                                                {parts[1] && <span className="text-yellow-500 font-mono text-[9px]">{parts[1]}</span>}
                                            </div>
                                        );
                                    })()}
                                    <span className="text-gray-400 truncate max-w-[100px]">{dp.currentFile?.split(/[/\\]/).pop()}</span>
                                    {dp.etaMs != null && (
                                        <span className="text-green-300 font-mono text-[11px]">~{formatEta(dp.etaMs)}</span>
                                    )}
                                </>
                            ) : (
                                <span className="text-green-300 text-xs font-medium">
                                    {dTotal > 0
                                        ? `${t('status.discovering')} (${dTotal} files)`
                                        : t('status.discovering')}
                                </span>
                            )}
                        </div>
                    );
                })()}

                {/* AI Tier Display */}
                {aiTier && !isProcessing && !isDiscovering && (
                    <div className="flex items-center gap-1.5 text-xs text-gray-400 border-l border-blue-700 pl-3 mr-3">
                        <Cpu size={12} className="flex-shrink-0" />
                        <span className="font-mono">
                            {aiTier.auto ? 'AUTO' : (aiTier.override || 'AUTO').toUpperCase()}
                        </span>
                    </div>
                )}

                <span className="text-[9px] text-gray-600 font-mono mr-2">v6.5.1.20260220_26</span>
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
