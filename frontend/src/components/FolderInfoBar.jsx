import React, { useState, useEffect } from 'react';
import { Play, MoreHorizontal } from 'lucide-react';
import { useLocale } from '../i18n';

/** Mini progress bar */
function PhaseBar({ label, count, total, color }) {
    const pct = total > 0 ? Math.min(100, Math.round((count / total) * 100)) : 0;
    const isDone = count >= total && total > 0;
    return (
        <div className="flex items-center gap-1">
            <span className={`text-[10px] font-bold w-6 ${isDone ? 'text-green-400' : 'text-gray-400'}`}>{label}</span>
            <div className="w-16 bg-gray-700 rounded-full h-1.5 overflow-hidden">
                <div
                    className={`h-full rounded-full transition-all duration-300 ${isDone ? 'bg-green-500' : color}`}
                    style={{ width: `${pct}%` }}
                />
            </div>
            <span className={`text-[10px] font-mono ${isDone ? 'text-green-400' : 'text-gray-400'}`}>
                {count}/{total}
            </span>
        </div>
    );
}

const FolderInfoBar = ({ currentPath, onProcessFolder, isProcessing }) => {
    const { t } = useLocale();
    const [stats, setStats] = useState(null);
    const [menuOpen, setMenuOpen] = useState(false);

    useEffect(() => {
        if (!currentPath) { setStats(null); return; }
        let cancelled = false;

        const load = async () => {
            try {
                const result = await window.electron?.pipeline?.getFolderPhaseStats(currentPath);
                if (cancelled) return;
                if (result?.success && result.folders?.length > 0) {
                    const totals = result.folders.reduce((acc, f) => ({
                        total: acc.total + f.total,
                        mc: acc.mc + f.mc,
                        vv: acc.vv + f.vv,
                        mv: acc.mv + f.mv,
                    }), { total: 0, mc: 0, vv: 0, mv: 0 });
                    setStats(totals);
                } else {
                    setStats(null);
                }
            } catch (e) {
                console.error('Failed to load folder stats:', e);
                if (!cancelled) setStats(null);
            }
        };
        load();
        return () => { cancelled = true; };
    }, [currentPath, isProcessing]);

    if (!currentPath) return null;

    const folderName = currentPath.split(/[/\\]/).pop() || currentPath;

    return (
        <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700 flex-shrink-0">
            <div className="flex items-center gap-4 min-w-0">
                <span className="text-sm font-medium text-gray-200 truncate max-w-[200px]">{folderName}</span>

                {stats && stats.total > 0 ? (
                    <div className="flex items-center gap-3">
                        <PhaseBar label="MC" count={stats.mc} total={stats.total} color="bg-blue-400" />
                        <PhaseBar label="VV" count={stats.vv} total={stats.total} color="bg-purple-400" />
                        <PhaseBar label="MV" count={stats.mv} total={stats.total} color="bg-green-400" />
                    </div>
                ) : (
                    <span className="text-xs text-gray-500">{t('status.not_processed')}</span>
                )}
            </div>

            {/* ... menu */}
            <div className="relative flex-shrink-0">
                <button
                    onClick={() => setMenuOpen(!menuOpen)}
                    className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
                >
                    <MoreHorizontal size={16} />
                </button>
                {menuOpen && (
                    <>
                        <div className="fixed inset-0 z-10" onClick={() => setMenuOpen(false)} />
                        <div className="absolute right-0 top-8 z-20 bg-gray-700 border border-gray-600 rounded shadow-lg py-1 min-w-[160px]">
                            <button
                                onClick={() => {
                                    setMenuOpen(false);
                                    onProcessFolder?.(currentPath);
                                }}
                                disabled={isProcessing}
                                className={`w-full text-left px-3 py-1.5 text-sm flex items-center gap-2 ${
                                    isProcessing
                                        ? 'text-gray-500 cursor-not-allowed'
                                        : 'text-gray-200 hover:bg-gray-600'
                                }`}
                            >
                                <Play size={12} fill="currentColor" />
                                {t('action.process_folder')}
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default FolderInfoBar;
