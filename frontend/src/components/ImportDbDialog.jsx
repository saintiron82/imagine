import React, { useState } from 'react';
import { Database, Upload, FolderOpen, ArrowRight, Check, AlertTriangle, X, Loader2, Trash2 } from 'lucide-react';
import { useLocale } from '../i18n';

const STEP_SELECT = 0;
const STEP_PREVIEW = 1;
const STEP_RESULT = 2;

const ImportDbDialog = ({ onClose, onProcessNew }) => {
    const { t } = useLocale();
    const [step, setStep] = useState(STEP_SELECT);
    const [archivePath, setArchivePath] = useState('');
    const [folderPath, setFolderPath] = useState('');
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [deleteMissing, setDeleteMissing] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const archiveName = archivePath ? archivePath.split(/[/\\]/).pop() : '';
    const folderName = folderPath ? folderPath.split(/[/\\]/).pop() : '';

    const handleSelectArchive = async () => {
        try {
            const selected = await window.electron?.db?.selectArchive();
            if (selected) {
                setArchivePath(selected);
                setError('');
            }
        } catch (e) {
            console.error('Archive selection failed:', e);
        }
    };

    const handleSelectFolder = async () => {
        try {
            const selected = await window.electron?.pipeline?.openFolderDialog();
            if (selected) {
                setFolderPath(selected);
                setError('');
            }
        } catch (e) {
            console.error('Folder selection failed:', e);
        }
    };

    const handlePreview = async () => {
        if (!archivePath || !folderPath) return;
        setLoading(true);
        setError('');
        try {
            const data = await window.electron?.db?.relinkPreview(archivePath, folderPath);
            if (data?.success) {
                setPreview(data);
                setStep(STEP_PREVIEW);
            } else {
                setError(data?.error || 'Preview failed');
            }
        } catch (e) {
            setError(e.message || 'Preview failed');
        }
        setLoading(false);
    };

    const handleApply = async () => {
        if (!archivePath || !folderPath) return;
        setLoading(true);
        setError('');
        try {
            const data = await window.electron?.db?.relinkApply(archivePath, folderPath, deleteMissing);
            if (data?.success) {
                setResult(data);
                setStep(STEP_RESULT);
            } else {
                setError(data?.error || 'Apply failed');
            }
        } catch (e) {
            setError(e.message || 'Apply failed');
        }
        setLoading(false);
    };

    const handleProcessNewFiles = () => {
        if (result?.new_files > 0 && folderPath) {
            onProcessNew?.(folderPath);
        }
        onClose();
    };

    const stepLabels = [
        t('import.step_select'),
        t('import.step_preview'),
        t('import.step_result'),
    ];

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
            <div className="bg-gray-800 border border-gray-600 rounded-lg shadow-2xl w-[520px] max-h-[80vh] flex flex-col">
                {/* Header */}
                <div className="flex items-center gap-3 px-6 py-4 border-b border-gray-700">
                    <Database className="text-blue-400 flex-shrink-0" size={20} />
                    <h2 className="text-white font-bold text-lg">{t('import.title')}</h2>
                    <button
                        onClick={onClose}
                        className="ml-auto text-gray-500 hover:text-gray-300 transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Step Indicator */}
                <div className="flex items-center gap-2 px-6 py-3 border-b border-gray-700/50">
                    {stepLabels.map((label, i) => (
                        <React.Fragment key={i}>
                            <div className={`flex items-center gap-1.5 text-xs font-medium ${
                                i === step ? 'text-blue-400' :
                                i < step ? 'text-green-400' : 'text-gray-500'
                            }`}>
                                <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                                    i === step ? 'bg-blue-500/20 text-blue-400 ring-1 ring-blue-500/40' :
                                    i < step ? 'bg-green-500/20 text-green-400' : 'bg-gray-700 text-gray-500'
                                }`}>
                                    {i < step ? <Check size={10} /> : i + 1}
                                </span>
                                {label}
                            </div>
                            {i < stepLabels.length - 1 && (
                                <ArrowRight size={12} className="text-gray-600 flex-shrink-0" />
                            )}
                        </React.Fragment>
                    ))}
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto px-6 py-4">
                    {/* Error */}
                    {error && (
                        <div className="flex items-center gap-2 mb-3 px-3 py-2 bg-red-900/30 border border-red-700/50 rounded text-red-300 text-sm">
                            <AlertTriangle size={14} className="flex-shrink-0" />
                            {error}
                        </div>
                    )}

                    {/* Step 1: Select Files */}
                    {step === STEP_SELECT && (
                        <div className="space-y-4">
                            {/* Archive selector */}
                            <div>
                                <label className="text-xs text-gray-400 mb-1.5 block">
                                    {t('import.select_archive')}
                                </label>
                                <button
                                    onClick={handleSelectArchive}
                                    className="w-full flex items-center gap-3 px-4 py-3 bg-gray-900/60 border border-gray-600 rounded hover:border-blue-500/50 transition-colors text-left"
                                >
                                    <Upload size={16} className="text-gray-400 flex-shrink-0" />
                                    {archivePath ? (
                                        <span className="text-gray-200 text-sm truncate">
                                            {t('import.archive_selected', { name: archiveName })}
                                        </span>
                                    ) : (
                                        <span className="text-gray-500 text-sm">
                                            {t('import.select_archive')}
                                        </span>
                                    )}
                                </button>
                            </div>

                            {/* Folder selector */}
                            <div>
                                <label className="text-xs text-gray-400 mb-1.5 block">
                                    {t('import.select_folder')}
                                </label>
                                <button
                                    onClick={handleSelectFolder}
                                    className="w-full flex items-center gap-3 px-4 py-3 bg-gray-900/60 border border-gray-600 rounded hover:border-blue-500/50 transition-colors text-left"
                                >
                                    <FolderOpen size={16} className="text-gray-400 flex-shrink-0" />
                                    {folderPath ? (
                                        <span className="text-gray-200 text-sm truncate">
                                            {t('import.folder_selected', { path: folderName })}
                                        </span>
                                    ) : (
                                        <span className="text-gray-500 text-sm">
                                            {t('import.select_folder')}
                                        </span>
                                    )}
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Step 2: Preview */}
                    {step === STEP_PREVIEW && preview && (
                        <div className="space-y-3">
                            {/* Tier info */}
                            <div className={`flex items-center gap-2 px-3 py-2 rounded text-sm ${
                                preview.tier_match
                                    ? 'bg-green-900/20 border border-green-700/40 text-green-300'
                                    : 'bg-yellow-900/20 border border-yellow-700/40 text-yellow-300'
                            }`}>
                                {preview.tier_match ? (
                                    <Check size={14} className="flex-shrink-0" />
                                ) : (
                                    <AlertTriangle size={14} className="flex-shrink-0" />
                                )}
                                {preview.tier_match
                                    ? t('import.tier_match', { source: preview.tier_source, target: preview.tier_target })
                                    : t('import.tier_mismatch', { source: preview.tier_source, target: preview.tier_target })
                                }
                            </div>

                            {/* Match stats */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between px-3 py-2 bg-gray-900/60 rounded">
                                    <span className="text-gray-300 text-sm">{t('import.matched', { count: '' })}</span>
                                    <span className="text-green-400 font-mono text-sm font-bold">{preview.matched.toLocaleString()}</span>
                                </div>
                                <div className="flex items-center justify-between px-3 py-2 bg-gray-900/60 rounded">
                                    <span className="text-gray-300 text-sm">{t('import.new_files', { count: '' })}</span>
                                    <span className="text-blue-400 font-mono text-sm font-bold">{preview.new_files.toLocaleString()}</span>
                                </div>
                                <div className="flex items-center justify-between px-3 py-2 bg-gray-900/60 rounded">
                                    <span className="text-gray-300 text-sm">{t('import.missing_files', { count: '' })}</span>
                                    <span className="text-orange-400 font-mono text-sm font-bold">{preview.missing.toLocaleString()}</span>
                                </div>
                            </div>

                            {/* Delete missing checkbox */}
                            {preview.missing > 0 && (
                                <label className="flex items-center gap-2 px-3 py-2 bg-gray-900/40 rounded cursor-pointer hover:bg-gray-900/60 transition-colors">
                                    <input
                                        type="checkbox"
                                        checked={deleteMissing}
                                        onChange={(e) => setDeleteMissing(e.target.checked)}
                                        className="rounded border-gray-600 bg-gray-700 text-red-500 focus:ring-red-500/30"
                                    />
                                    <Trash2 size={13} className="text-red-400" />
                                    <span className="text-gray-300 text-sm">{t('import.delete_missing')}</span>
                                </label>
                            )}
                        </div>
                    )}

                    {/* Step 3: Result */}
                    {step === STEP_RESULT && result && (
                        <div className="space-y-3">
                            {/* Success */}
                            <div className="flex items-center gap-2 px-3 py-2 bg-green-900/20 border border-green-700/40 rounded text-green-300 text-sm">
                                <Check size={14} className="flex-shrink-0" />
                                {t('import.result_success', { count: result.matched })}
                            </div>

                            {/* Tier note */}
                            {!result.tier_match && (
                                <div className="flex items-center gap-2 px-3 py-2 bg-yellow-900/20 border border-yellow-700/40 rounded text-yellow-300 text-sm">
                                    <AlertTriangle size={14} className="flex-shrink-0" />
                                    {t('import.result_tier_note')}
                                </div>
                            )}

                            {/* New files prompt */}
                            {result.new_files > 0 && (
                                <div className="px-3 py-3 bg-blue-900/20 border border-blue-700/40 rounded">
                                    <p className="text-blue-300 text-sm">
                                        {t('import.result_new_prompt', { count: result.new_files })}
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Actions */}
                <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-700">
                    {step === STEP_SELECT && (
                        <>
                            <button
                                onClick={onClose}
                                className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700"
                            >
                                {t('import.cancel_btn')}
                            </button>
                            <button
                                onClick={handlePreview}
                                disabled={!archivePath || !folderPath || loading}
                                className={`flex items-center gap-2 px-5 py-2 text-sm font-medium rounded transition-colors ${
                                    archivePath && folderPath && !loading
                                        ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/40'
                                        : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                                }`}
                            >
                                {loading && <Loader2 size={14} className="animate-spin" />}
                                {t('import.preview_btn')}
                            </button>
                        </>
                    )}

                    {step === STEP_PREVIEW && (
                        <>
                            <button
                                onClick={() => { setStep(STEP_SELECT); setPreview(null); setError(''); }}
                                className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700"
                            >
                                {t('import.cancel_btn')}
                            </button>
                            <button
                                onClick={handleApply}
                                disabled={loading}
                                className={`flex items-center gap-2 px-5 py-2 text-sm font-medium rounded transition-colors ${
                                    !loading
                                        ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/40'
                                        : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                                }`}
                            >
                                {loading && <Loader2 size={14} className="animate-spin" />}
                                {t('import.apply_btn')}
                            </button>
                        </>
                    )}

                    {step === STEP_RESULT && (
                        <>
                            <button
                                onClick={onClose}
                                className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700"
                            >
                                {t('import.close_btn')}
                            </button>
                            {result?.new_files > 0 && (
                                <button
                                    onClick={handleProcessNewFiles}
                                    className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded shadow-lg shadow-blue-900/40 transition-colors"
                                >
                                    {t('resume.action_resume')}
                                </button>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ImportDbDialog;
