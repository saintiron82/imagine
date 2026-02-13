import React from 'react';
import { AlertTriangle, Folder, Play, X } from 'lucide-react';
import { useLocale } from '../i18n';

const ResumeDialog = ({ stats, onResume, onDismiss }) => {
    const { t } = useLocale();

    if (!stats || stats.total_incomplete === 0) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
            <div className="bg-gray-800 border border-gray-600 rounded-lg shadow-2xl w-[480px] max-h-[70vh] flex flex-col">
                {/* Header */}
                <div className="flex items-center gap-3 px-6 py-4 border-b border-gray-700">
                    <AlertTriangle className="text-yellow-400 flex-shrink-0" size={22} />
                    <h2 className="text-white font-bold text-lg">{t('resume.title')}</h2>
                    <button
                        onClick={onDismiss}
                        className="ml-auto text-gray-500 hover:text-gray-300 transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Description */}
                <div className="px-6 py-3">
                    <p className="text-gray-300 text-sm">
                        {t('resume.description', {
                            total: stats.total_files,
                            incomplete: stats.total_incomplete,
                        })}
                    </p>
                </div>

                {/* Folder list */}
                <div className="flex-1 overflow-y-auto px-6 pb-3">
                    <div className="space-y-1.5">
                        {stats.folders.map((folder) => {
                            const name = folder.storage_root.split(/[/\\]/).pop() || folder.storage_root;
                            const pct = folder.total > 0
                                ? Math.round((folder.done / folder.total) * 100)
                                : 0;
                            return (
                                <div
                                    key={folder.storage_root}
                                    className="flex items-center gap-2 px-3 py-2 bg-gray-900/60 rounded"
                                >
                                    <Folder size={14} className="text-yellow-500 flex-shrink-0" />
                                    <span className="text-gray-200 text-sm truncate flex-1" title={folder.storage_root}>
                                        {name}
                                    </span>
                                    {/* Mini progress bar */}
                                    <div className="w-20 bg-gray-700 rounded-full h-1.5 overflow-hidden flex-shrink-0">
                                        <div
                                            className="h-full bg-blue-500 rounded-full"
                                            style={{ width: `${pct}%` }}
                                        />
                                    </div>
                                    <span className="text-gray-400 text-xs font-mono w-16 text-right flex-shrink-0">
                                        {t('resume.folder_status', { done: folder.done, total: folder.total })}
                                    </span>
                                    <span className="text-orange-400 text-xs font-mono flex-shrink-0">
                                        {t('resume.incomplete_count', { count: folder.incomplete })}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Actions */}
                <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-700">
                    <button
                        onClick={onDismiss}
                        className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700"
                    >
                        {t('resume.action_dismiss')}
                    </button>
                    <button
                        onClick={onResume}
                        className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded shadow-lg shadow-blue-900/40 transition-colors"
                    >
                        <Play size={14} fill="currentColor" />
                        {t('resume.action_resume')}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ResumeDialog;
