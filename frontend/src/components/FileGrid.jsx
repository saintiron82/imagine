import React, { useState, useEffect, useRef } from 'react';
import { CheckCircle, File, Loader2, Info, X, FolderOpen, ExternalLink } from 'lucide-react';
import { useLocale } from '../i18n';

const SUPPORTED_EXTS = ['.psd', '.png', '.jpg', '.jpeg'];
const IMAGE_PREVIEW_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.webp'];

// Global thumbnail path cache (stores disk paths, not base64 - memory efficient)
const thumbnailPathCache = new Map();

// --- Priority Queue System ---
const thumbnailQueue = [];          // [filePath, filePath, ...]
let isQueueRunning = false;
const queueListeners = new Set();   // Set<() => void> - notify on chunk completion
const inFlightPaths = new Set();    // Currently being processed by generateThumbnailsBatch

function prioritizeFolder(filePaths) {
    // Skip cached and in-flight (currently being processed)
    const needed = filePaths.filter(fp => !thumbnailPathCache.has(fp) && !inFlightPaths.has(fp));
    if (needed.length === 0) return;

    // Remove these files from queue (prevent duplicates)
    const needSet = new Set(needed);
    const remaining = thumbnailQueue.filter(fp => !needSet.has(fp));

    // New folder = front, rest = back
    thumbnailQueue.length = 0;
    thumbnailQueue.push(...needed, ...remaining);

    runQueue();
}

async function runQueue() {
    if (isQueueRunning) return;
    isQueueRunning = true;

    const BATCH_SIZE = 4;
    while (thumbnailQueue.length > 0) {
        const chunk = thumbnailQueue.splice(0, BATCH_SIZE);
        chunk.forEach(fp => inFlightPaths.add(fp));
        try {
            const results = await window.electron?.pipeline?.generateThumbnailsBatch(chunk);
            if (results) {
                for (const [fp, thumbPath] of Object.entries(results)) {
                    if (thumbPath) thumbnailPathCache.set(fp, thumbPath);
                }
                queueListeners.forEach(cb => cb());
            }
        } catch (err) {
            console.error('Thumbnail queue error:', err);
        } finally {
            chunk.forEach(fp => inFlightPaths.delete(fp));
        }
    }
    isQueueRunning = false;
}

// Category options (value keys for DB storage, labels via i18n)
const CATEGORY_OPTIONS = [
    { value: '', key: 'category.uncategorized' },
    { value: 'Characters', key: 'category.characters' },
    { value: 'Backgrounds', key: 'category.backgrounds' },
    { value: 'UI Elements', key: 'category.ui_elements' },
    { value: 'Concepts', key: 'category.concepts' },
    { value: 'References', key: 'category.references' },
    { value: 'Archive', key: 'category.archive' },
];

// Metadata Modal Component
const MetadataModal = ({ metadata, onClose }) => {
    const { t } = useLocale();
    const [lang, setLang] = useState('kr'); // 'original', 'kr', 'en'
    const [editedData, setEditedData] = useState({
        user_note: metadata.user_note || '',
        user_tags: metadata.user_tags || [],
        user_category: metadata.user_category || '',
        user_rating: metadata.user_rating || 0
    });
    const [newTag, setNewTag] = useState('');
    const saveTimer = useRef(null);
    const isInitialMount = useRef(true);

    if (!metadata) return null;

    // Auto-save: debounce 500ms after editedData changes
    useEffect(() => {
        // Skip auto-save on initial mount
        if (isInitialMount.current) {
            isInitialMount.current = false;
            return;
        }

        if (saveTimer.current) clearTimeout(saveTimer.current);
        saveTimer.current = setTimeout(async () => {
            try {
                await window.electron.metadata.updateUserData(
                    metadata.file_path, editedData
                );
                // Sync back to metadata object for persistence
                Object.assign(metadata, editedData);
            } catch (err) {
                console.error('Auto-save failed:', err);
            }
        }, 500);

        return () => {
            if (saveTimer.current) clearTimeout(saveTimer.current);
        };
    }, [editedData]);

    const handleAddTag = () => {
        if (newTag.trim() && !editedData.user_tags.includes(newTag.trim())) {
            setEditedData({
                ...editedData,
                user_tags: [...editedData.user_tags, newTag.trim()]
            });
            setNewTag('');
        }
    };

    const handleRemoveTag = (tag) => {
        setEditedData({
            ...editedData,
            user_tags: editedData.user_tags.filter(t => t !== tag)
        });
    };

    // Language Toggle Component
    const LangToggle = () => (
        <div className="flex bg-gray-700/50 rounded-lg p-1 gap-1">
            {[
                { label: t('lang.original'), value: 'original' },
                { label: t('lang.korean'), value: 'kr' },
                { label: t('lang.english'), value: 'en' }
            ].map((opt) => (
                <button
                    key={opt.value}
                    onClick={() => setLang(opt.value)}
                    className={`px-3 py-1 rounded text-xs font-bold transition-all ${lang === opt.value
                        ? 'bg-blue-600 text-white shadow-md'
                        : 'text-gray-400 hover:text-white hover:bg-gray-600'
                        }`}
                >
                    {opt.label}
                </button>
            ))}
        </div>
    );

    // Helpers
    const getTags = () => {
        if (lang === 'kr' && metadata.translated_tags) return metadata.translated_tags;
        if (lang === 'en' && metadata.translated_tags_en) return metadata.translated_tags_en;
        return metadata.semantic_tags || '';
    };

    const getText = (original, index) => {
        if (lang === 'kr' && metadata.translated_text?.[index]) return metadata.translated_text[index];
        if (lang === 'en' && metadata.translated_text_en?.[index]) return metadata.translated_text_en[index];
        return original || '';
    };

    const getLayerTree = () => {
        if (lang === 'kr' && metadata.translated_layer_tree) return metadata.translated_layer_tree;
        if (lang === 'en' && metadata.translated_layer_tree_en) return metadata.translated_layer_tree_en;
        return metadata.layer_tree;
    };

    const isTranslationAvailable = () => {
        if (lang === 'kr') return !!metadata.translated_tags; // Check tags as proxy
        if (lang === 'en') return !!metadata.translated_tags_en;
        return true;
    };

    const tags = getTags();

    return (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 animate-fadeIn" onClick={onClose}>
            <div className="bg-gray-800 rounded-lg max-w-[95vw] w-full h-[90vh] mx-4 overflow-hidden flex flex-col shadow-2xl border border-gray-700" onClick={e => e.stopPropagation()}>
                <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-900/50 shrink-0">
                    <div className="flex items-center gap-4 overflow-hidden">
                        <h2 className="text-xl font-bold text-white truncate max-w-[600px]">{metadata.file_name}</h2>
                        <LangToggle />
                        {lang !== 'original' && !isTranslationAvailable() && (
                            <span className="flex items-center gap-1 text-[10px] text-amber-400 bg-amber-400/10 px-2 py-1 rounded border border-amber-400/20 animate-pulse">
                                <Info size={12} />
                                {t('msg.no_translation')}
                            </span>
                        )}
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors"><X size={24} /></button>
                </div>

                <div className="p-6 overflow-hidden flex-1 grid grid-cols-12 gap-6">
                    {/* Left Column: Basic Info & Tags & Text Content (5 columns) */}
                    <div className="col-span-5 flex flex-col gap-6 overflow-hidden h-full">
                        <div className="grid grid-cols-2 gap-3 shrink-0">
                            {[
                                { label: t('label.format'), value: metadata.format },
                                { label: t('label.resolution'), value: metadata.resolution?.join(' x ') },
                                { label: t('label.layers'), value: metadata.layer_count },
                                { label: t('label.size'), value: `${(metadata.file_size / 1024 / 1024).toFixed(2)} MB` }
                            ].map((item, i) => (
                                <div key={i} className="bg-gray-900/50 p-3 rounded border border-gray-700/50">
                                    <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">{item.label}</div>
                                    <div className="text-white font-mono text-sm font-semibold">{item.value}</div>
                                </div>
                            ))}
                        </div>

                        {/* Path Information + File Actions */}
                        {metadata.file_path && (
                            <div className="shrink-0 bg-gray-900/50 border border-gray-700/50 rounded-lg p-3">
                                <div className="text-gray-400 text-[10px] uppercase tracking-wider mb-2">{t('label.path_info')}</div>
                                <div className="space-y-1.5">
                                    <div className="flex items-start gap-2">
                                        <span className="text-gray-500 text-[10px] uppercase shrink-0 w-14">{t('label.full_path')}</span>
                                        <span className="text-gray-300 text-[11px] font-mono break-all select-all leading-relaxed">{metadata.file_path}</span>
                                    </div>
                                    {metadata.folder_path && (
                                        <div className="flex items-start gap-2">
                                            <span className="text-gray-500 text-[10px] uppercase shrink-0 w-14">{t('label.folder')}</span>
                                            <span className="text-gray-300 text-[11px] font-mono">{metadata.folder_path}</span>
                                        </div>
                                    )}
                                </div>
                                <div className="flex gap-2 mt-2 pt-2 border-t border-gray-700/30">
                                    <button onClick={() => window.electron?.fs?.showInFolder(metadata.file_path)}
                                        className="flex items-center gap-1.5 px-2.5 py-1 bg-gray-700/50 hover:bg-gray-600 rounded text-[11px] text-gray-400 hover:text-white transition-colors">
                                        <FolderOpen size={12} /> {t('action.show_in_folder')}
                                    </button>
                                    <button onClick={() => window.electron?.fs?.openFile(metadata.file_path)}
                                        className="flex items-center gap-1.5 px-2.5 py-1 bg-gray-700/50 hover:bg-gray-600 rounded text-[11px] text-gray-400 hover:text-white transition-colors">
                                        <ExternalLink size={12} /> {t('action.open_file')}
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* User Metadata Section - Always Editable, Auto-Save */}
                        <div className="shrink-0 bg-blue-900/10 border border-blue-700/30 rounded-lg p-4">
                            <div className="text-blue-300 text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-2">
                                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-[10px]">USER</span>
                                {t('label.my_notes_tags')}
                            </div>

                            {/* Notes - Always textarea */}
                            <div className="mb-4">
                                <label className="text-gray-400 text-[10px] uppercase tracking-wider mb-1 block">{t('label.notes')}</label>
                                <textarea
                                    value={editedData.user_note}
                                    onChange={(e) => setEditedData({...editedData, user_note: e.target.value})}
                                    className="w-full bg-black/30 border border-gray-600 rounded p-2 text-sm text-white resize-none focus:border-blue-500 focus:outline-none transition-colors"
                                    rows={3}
                                    placeholder={t('placeholder.notes')}
                                />
                            </div>

                            {/* Custom Tags - Always editable */}
                            <div className="mb-4">
                                <label className="text-gray-400 text-[10px] uppercase tracking-wider mb-1 block">{t('label.custom_tags')}</label>
                                <div className="flex flex-wrap gap-1.5 mb-2">
                                    {editedData.user_tags.map((tag, i) => (
                                        <span key={i} className="bg-blue-700/30 text-blue-200 border border-blue-600/30 px-2 py-0.5 rounded text-xs flex items-center gap-1">
                                            {tag}
                                            <button onClick={() => handleRemoveTag(tag)} className="hover:text-red-400 ml-1">&times;</button>
                                        </span>
                                    ))}
                                </div>
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={newTag}
                                        onChange={(e) => setNewTag(e.target.value)}
                                        onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
                                        className="flex-1 bg-black/30 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:border-blue-500 focus:outline-none transition-colors"
                                        placeholder={t('placeholder.tag_input')}
                                    />
                                    <button onClick={handleAddTag}
                                            className="px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs transition-colors">
                                        {t('action.add')}
                                    </button>
                                </div>
                            </div>

                            {/* Category - Always select */}
                            <div className="mb-4">
                                <label className="text-gray-400 text-[10px] uppercase tracking-wider mb-1 block">{t('label.category')}</label>
                                <select
                                    value={editedData.user_category}
                                    onChange={(e) => setEditedData({...editedData, user_category: e.target.value})}
                                    className="w-full bg-black/30 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:border-blue-500 focus:outline-none transition-colors">
                                    {CATEGORY_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>{t(opt.key)}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Rating - Always clickable */}
                            <div>
                                <label className="text-gray-400 text-[10px] uppercase tracking-wider mb-1 block">{t('label.rating')}</label>
                                <div className="flex gap-1">
                                    {[1, 2, 3, 4, 5].map(star => (
                                        <button
                                            key={star}
                                            onClick={() => setEditedData({...editedData, user_rating: star === editedData.user_rating ? 0 : star})}
                                            className={`text-2xl hover:text-yellow-300 cursor-pointer transition-colors ${star <= editedData.user_rating ? 'text-yellow-400' : 'text-gray-600'}`}>
                                            &#9733;
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* AI Vision Analysis */}
                        {(metadata.mc_caption || metadata.ai_tags || metadata.dominant_color || metadata.ai_style) && (
                            <div className="shrink-0 bg-purple-900/10 border border-purple-700/30 rounded-lg p-3">
                                <div className="text-purple-300 text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-2">
                                    <span className="bg-purple-600 text-white px-2 py-0.5 rounded text-[10px]">AI</span>
                                    {t('label.vision_analysis')}
                                </div>

                                {metadata.mc_caption && (
                                    <div className="mb-3">
                                        <div className="text-gray-400 text-[10px] uppercase tracking-wider mb-1">{t('label.caption')}</div>
                                        <div className="text-purple-200 text-xs leading-relaxed bg-black/20 p-2 rounded">
                                            {metadata.mc_caption}
                                        </div>
                                    </div>
                                )}

                                {metadata.ai_tags && metadata.ai_tags.length > 0 && (
                                    <div className="mb-3">
                                        <div className="text-gray-400 text-[10px] uppercase tracking-wider mb-1">{t('label.tags')}</div>
                                        <div className="flex flex-wrap gap-1.5">
                                            {metadata.ai_tags.map((tag, i) => (
                                                <span key={i} className="bg-purple-700/30 text-purple-200 border border-purple-600/30 px-2 py-0.5 rounded text-xs">
                                                    {tag}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                <div className="grid grid-cols-2 gap-2">
                                    {metadata.dominant_color && (
                                        <div>
                                            <div className="text-gray-400 text-[10px] uppercase tracking-wider mb-1">{t('label.color')}</div>
                                            <div className="flex items-center gap-2">
                                                <div
                                                    className="w-6 h-6 rounded border border-gray-600 shadow-inner"
                                                    style={{ backgroundColor: metadata.dominant_color }}
                                                />
                                                <span className="text-purple-200 text-xs font-mono">{metadata.dominant_color}</span>
                                            </div>
                                        </div>
                                    )}

                                    {metadata.ai_style && (
                                        <div>
                                            <div className="text-gray-400 text-[10px] uppercase tracking-wider mb-1">{t('label.style')}</div>
                                            <div className="text-purple-200 text-xs bg-black/20 px-2 py-1 rounded">
                                                {metadata.ai_style}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {metadata.ocr_text && (
                                    <div className="mt-3">
                                        <div className="text-gray-400 text-[10px] uppercase tracking-wider mb-1">{t('label.ocr_text')}</div>
                                        <div className="text-purple-200 text-xs bg-black/20 p-2 rounded max-h-20 overflow-y-auto custom-scrollbar">
                                            {metadata.ocr_text}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Semantic Tags */}
                        <div className="shrink-0">
                            <div className="flex items-center gap-2 mb-2">
                                <div className="text-gray-400 text-xs font-bold uppercase tracking-wider">{t('label.semantic_tags')}</div>
                                <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${lang === 'original' ? 'bg-gray-700 text-gray-300' : lang === 'kr' ? 'bg-green-900 text-green-300' : 'bg-blue-900 text-blue-300'}`}>
                                    {lang === 'original' ? t('lang.badge_raw') : lang === 'kr' ? t('lang.badge_korean') : t('lang.badge_english')}
                                </span>
                            </div>
                            <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto pr-2 custom-scrollbar">
                                {tags ? tags.split(' ').filter(Boolean).map((tag, i) => (
                                    <span key={i} className={`px-2 py-1 rounded text-xs transition-colors ${lang === 'original' ? 'bg-gray-700 text-gray-200' :
                                        lang === 'kr' ? 'bg-green-900/40 text-green-300 border border-green-700/30' :
                                            'bg-blue-900/40 text-blue-300 border border-blue-700/30'
                                        }`}>
                                        {tag}
                                    </span>
                                )) : (
                                    <span className="text-gray-500 text-xs italic">{t('msg.no_tags')}</span>
                                )}
                            </div>
                        </div>

                        {metadata.used_fonts?.length > 0 && (
                            <div className="shrink-0">
                                <div className="text-gray-400 text-xs font-bold uppercase tracking-wider mb-2">{t('label.used_fonts')}</div>
                                <div className="flex flex-wrap gap-2 max-h-24 overflow-y-auto">
                                    {metadata.used_fonts.map((font, i) => (
                                        <span key={i} className="bg-purple-900/30 text-purple-300 border border-purple-700/30 px-2 py-1 rounded text-xs">{font}</span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Text Content (Left) */}
                        <div className="flex-1 flex flex-col min-h-0">
                            <div className="text-gray-400 text-xs font-bold uppercase tracking-wider mb-2 flex justify-between items-center">
                                <span>{t('label.extracted_text')}</span>
                                <span className="text-gray-600 text-[10px]">{t('msg.lines', { count: metadata.text_content?.length || 0 })}</span>
                            </div>
                            <div className="bg-gray-900 p-4 rounded-lg text-sm text-gray-300 overflow-y-auto border border-black/20 shadow-inner flex-1 custom-scrollbar">
                                {metadata.text_content?.length > 0 ? (
                                    <div className="space-y-4">
                                        {metadata.text_content.map((text, i) => (
                                            <div key={i} className="border-b border-gray-800 pb-3 last:border-0 last:pb-0 group">
                                                <div className={`leading-relaxed whitespace-pre-wrap transition-colors ${lang !== 'original' ? 'text-blue-200' : 'text-gray-300'}`}>
                                                    {getText(text, i)}
                                                </div>
                                                {lang !== 'original' && getText(text, i) !== text && (
                                                    <div className="mt-1 text-xs text-gray-600 select-none group-hover:text-gray-500 transition-colors">
                                                        {t('msg.original_prefix')} {text.substring(0, 50)}{text.length > 50 ? '...' : ''}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="h-full flex items-center justify-center text-gray-600 italic">
                                        {t('msg.no_text')}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Layer Tree (7 columns - Right) */}
                    <div className="col-span-7 flex flex-col gap-6 overflow-hidden h-full">
                        {metadata.layer_tree && (
                            <div className="flex-1 flex flex-col min-h-0">
                                <div className="text-gray-400 text-xs font-bold uppercase tracking-wider mb-2">{t('label.layer_tree')}</div>
                                <div className="bg-gray-900 p-4 rounded text-xs font-mono text-gray-400 overflow-auto border border-black/20 shadow-inner flex-1 custom-scrollbar">
                                    <pre>{JSON.stringify(getLayerTree(), null, 2)}</pre>
                                </div>
                            </div>
                        )}

                        {!metadata.layer_tree && (
                            <div className="flex-1 flex items-center justify-center text-gray-600 italic bg-gray-900/30 rounded border border-gray-800 border-dashed">
                                {t('msg.no_layer_info')}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

// FileCard Component
const FileCard = React.forwardRef(({ file, isSelected, onMouseDown, thumbnail, loading, onShowMeta, hasMetadata }, ref) => {
    const { t } = useLocale();
    const canPreviewNatively = IMAGE_PREVIEW_EXTS.includes(file.extension);

    const toFileUrl = (p) => {
        const encoded = p.replace(/\\/g, '/').split('/').map(s => encodeURIComponent(s)).join('/');
        return `file:///${encoded}`;
    };

    const handleMetaClick = (e) => {
        e.stopPropagation();
        onShowMeta(file.path);
    };

    return (
        <div
            ref={ref}
            data-path={file.path}
            className={`relative group border rounded-lg p-2 cursor-pointer transition-all select-none ${isSelected ? 'bg-blue-900 border-blue-500 ring-2 ring-blue-500/50' : 'bg-gray-800 border-gray-700 hover:bg-gray-700'
                }`}
            onMouseDown={onMouseDown}
        >
            {/* Metadata Status Indicator */}
            <div className="absolute top-2 left-2 z-10 pointer-events-none">
                {!hasMetadata && (
                    <div className="w-3 h-3 rounded-full bg-red-500 shadow-sm border border-black/20" title={t('status.not_processed')}></div>
                )}
            </div>

            {/* Selection checkbox */}
            <div className={`absolute top-2 right-2 z-10 ${isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
                <CheckCircle className={isSelected ? "text-blue-500 fill-current" : "text-gray-400"} size={20} />
            </div>

            {/* META button (only if metadata exists) */}
            {hasMetadata && (
                <button
                    className="absolute bottom-2 right-2 z-20 opacity-0 group-hover:opacity-100 bg-gray-900/90 hover:bg-blue-600 text-[10px] text-white px-2 py-1 rounded transition-all shadow-lg font-bold"
                    onClick={handleMetaClick}
                >
                    META
                </button>
            )}

            <div className="aspect-square bg-gray-900 rounded mb-2 overflow-hidden flex items-center justify-center pointer-events-none">
                {canPreviewNatively ? (
                    <img src={toFileUrl(file.path)} alt={file.name} className="w-full h-full object-cover" onError={e => { e.target.style.display = 'none'; }} />
                ) : loading ? (
                    <Loader2 size={32} className="animate-spin text-gray-500" />
                ) : thumbnail ? (
                    <img src={toFileUrl(thumbnail)} alt={file.name} className="w-full h-full object-cover" onError={e => { e.target.style.display = 'none'; }} />
                ) : (
                    <div className="flex flex-col items-center text-gray-500"><File size={48} /><span className="text-xs font-bold mt-1">PSD</span></div>
                )}
            </div>
            <div className="text-xs text-gray-300 truncate font-medium pointer-events-none">{file.name}</div>
            <div className="flex items-center gap-1.5 mt-1 pointer-events-none">
                <span className="text-[10px] text-gray-500 uppercase">{file.extension.replace('.', '')}</span>
                {file.score != null && (
                    <span className="text-[10px] text-blue-400 ml-auto">{(file.score * 100).toFixed(0)}%</span>
                )}
                {file.user_rating > 0 && (
                    <span className="text-[10px] text-yellow-400">{'&#9733;'.repeat ? '\u2605'.repeat(file.user_rating) : ''}</span>
                )}
                {file.user_category && (
                    <span className="text-[9px] bg-gray-700 text-gray-300 px-1 rounded">{file.user_category}</span>
                )}
            </div>
        </div>
    );
});

// Main FileGrid Component
const FileGrid = ({ currentPath, selectedFiles, setSelectedFiles }) => {
    const { t } = useLocale();
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(false);
    const [thumbnails, setThumbnails] = useState({});
    const [loadingThumbnails, setLoadingThumbnails] = useState(true);
    const [metadataStatus, setMetadataStatus] = useState({});
    const [dragStart, setDragStart] = useState(null);
    const [dragEnd, setDragEnd] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [metadata, setMetadata] = useState(null);
    const containerRef = useRef(null);
    const cardRefs = useRef({});
    const lastClickedIndex = useRef(null);
    const pendingClick = useRef(null);

    // Load files
    useEffect(() => {
        if (!currentPath) return;
        setLoading(true);
        setThumbnails({});
        setMetadataStatus({});
        window.electron?.fs?.listDir(currentPath)
            .then(items => {
                const supportedFiles = items.filter(i => !i.isDirectory && SUPPORTED_EXTS.includes(i.extension));
                setFiles(supportedFiles);
                // Check metadata existence for all files
                if (window.electron?.pipeline?.checkMetadataExists) {
                    window.electron.pipeline.checkMetadataExists(supportedFiles.map(f => f.path))
                        .then(setMetadataStatus)
                        .catch(console.error);
                }
            })
            .catch(console.error)
            .finally(() => setLoading(false));
    }, [currentPath]);

    // Re-check metadata periodically
    useEffect(() => {
        if (files.length === 0) return;
        const interval = setInterval(() => {
            if (window.electron?.pipeline?.checkMetadataExists) {
                window.electron.pipeline.checkMetadataExists(files.map(f => f.path))
                    .then(status => {
                        setMetadataStatus(prev => {
                            const isDifferent = Object.keys(status).some(k => status[k] !== prev[k]);
                            return isDifferent ? status : prev;
                        });
                    })
                    .catch(console.error);
            }
        }, 3000);
        return () => clearInterval(interval);
    }, [files]);

    // Load thumbnails: 2-stage (check disk first, generate missing via queue)
    useEffect(() => {
        const psdFiles = files.filter(f => f.extension === '.psd');
        if (psdFiles.length === 0) {
            setLoadingThumbnails(false);
            return;
        }

        // Stage 1: Show in-memory cached paths immediately
        const cached = {};
        psdFiles.forEach(f => {
            if (thumbnailPathCache.has(f.path)) cached[f.path] = thumbnailPathCache.get(f.path);
        });
        setThumbnails(cached);

        const unknownPaths = psdFiles.filter(f => !thumbnailPathCache.has(f.path)).map(f => f.path);
        if (unknownPaths.length === 0) {
            setLoadingThumbnails(false);
            return;
        }

        setLoadingThumbnails(true);

        // Stage 2: Check disk thumbnails (no Python needed), then generate missing
        const checkAndGenerate = async () => {
            const existResults = await window.electron?.pipeline?.checkThumbnailsExist(unknownPaths);
            if (existResults) {
                const nowCached = {};
                const stillMissing = [];
                for (const [fp, thumbPath] of Object.entries(existResults)) {
                    if (thumbPath) {
                        thumbnailPathCache.set(fp, thumbPath);
                        nowCached[fp] = thumbPath;
                    } else {
                        stillMissing.push(fp);
                    }
                }
                if (Object.keys(nowCached).length > 0) {
                    setThumbnails(prev => ({ ...prev, ...nowCached }));
                }
                if (stillMissing.length === 0) {
                    setLoadingThumbnails(false);
                    return;
                }
                // Generate only missing thumbnails via queue
                prioritizeFolder(stillMissing);
            } else {
                // Fallback: generate all via queue
                prioritizeFolder(unknownPaths);
            }
        };
        checkAndGenerate().catch(console.error);

        // Listen for chunk completions and merge newly cached thumbnail paths
        const onChunkDone = () => {
            setThumbnails(prev => {
                const merged = { ...prev };
                let changed = false;
                psdFiles.forEach(f => {
                    if (!merged[f.path] && thumbnailPathCache.has(f.path)) {
                        merged[f.path] = thumbnailPathCache.get(f.path);
                        changed = true;
                    }
                });
                return changed ? merged : prev;
            });
            const allDone = psdFiles.every(f => thumbnailPathCache.has(f.path));
            if (allDone) setLoadingThumbnails(false);
        };

        queueListeners.add(onChunkDone);
        return () => { queueListeners.delete(onChunkDone); };
    }, [files]);

    const handleShowMeta = async (filePath) => {
        const meta = await window.electron?.pipeline?.readMetadata(filePath);
        if (meta) {
            setMetadata(meta);
        } else {
            alert(t('msg.no_metadata'));
        }
    };

    const displayedFiles = files;

    const getSelectionRect = () => {
        if (!dragStart || !dragEnd) return null;
        return {
            left: Math.min(dragStart.x, dragEnd.x),
            top: Math.min(dragStart.y, dragEnd.y),
            width: Math.abs(dragEnd.x - dragStart.x),
            height: Math.abs(dragEnd.y - dragStart.y),
        };
    };

    const intersects = (rect, selRect) => {
        return !(rect.right < selRect.left || rect.left > selRect.left + selRect.width ||
            rect.bottom < selRect.top || rect.top > selRect.top + selRect.height);
    };

    const handleCardMouseDown = (e, file, index) => {
        // Prevent click if clicking META button
        if (e.target.tagName === 'BUTTON') return;

        pendingClick.current = { file, index, startX: e.clientX, startY: e.clientY };
        const rect = containerRef.current.getBoundingClientRect();
        const pos = { x: e.clientX - rect.left, y: e.clientY - rect.top + containerRef.current.scrollTop };
        setDragStart(pos);
        setDragEnd(pos);

        if (e.shiftKey && lastClickedIndex.current !== null) {
            const start = Math.min(lastClickedIndex.current, index);
            const end = Math.max(lastClickedIndex.current, index);
            setSelectedFiles(new Set([...selectedFiles, ...files.slice(start, end + 1).map(f => f.path)]));
            pendingClick.current = null;
        } else if (e.ctrlKey || e.metaKey) {
            const newSel = new Set(selectedFiles);
            newSel.has(file.path) ? newSel.delete(file.path) : newSel.add(file.path);
            setSelectedFiles(newSel);
            lastClickedIndex.current = index;
            pendingClick.current = null;
        }
    };

    const handleMouseDown = (e) => {
        const rect = containerRef.current.getBoundingClientRect();
        const pos = { x: e.clientX - rect.left, y: e.clientY - rect.top + containerRef.current.scrollTop };
        setDragStart(pos);
        setDragEnd(pos);

        const isCard = e.target.closest('[data-path]');
        if (!isCard && !e.ctrlKey && !e.shiftKey) {
            setSelectedFiles(new Set());
        }
    };

    const handleMouseMove = (e) => {
        if (!dragStart) return;
        const rect = containerRef.current.getBoundingClientRect();
        const pos = { x: e.clientX - rect.left, y: e.clientY - rect.top + containerRef.current.scrollTop };
        const dist = Math.sqrt(Math.pow(pos.x - dragStart.x, 2) + Math.pow(pos.y - dragStart.y, 2));

        if (dist > 5) {
            setIsDragging(true);
            pendingClick.current = null;
        }
        setDragEnd(pos);
    };

    const handleMouseUp = () => {
        if (pendingClick.current) {
            const { file, index } = pendingClick.current;
            setSelectedFiles(new Set([file.path]));
            lastClickedIndex.current = index;
            pendingClick.current = null;
        }

        if (isDragging && dragStart && dragEnd) {
            const selRect = getSelectionRect();
            if (selRect && selRect.width > 5 && selRect.height > 5) {
                const containerRect = containerRef.current.getBoundingClientRect();
                const scrollTop = containerRef.current.scrollTop;
                const newSelection = new Set(selectedFiles);
                files.forEach((file) => {
                    const el = cardRefs.current[file.path];
                    if (el) {
                        const elRect = el.getBoundingClientRect();
                        const relRect = {
                            left: elRect.left - containerRect.left,
                            top: elRect.top - containerRect.top + scrollTop,
                            right: elRect.right - containerRect.left,
                            bottom: elRect.bottom - containerRect.top + scrollTop,
                        };
                        if (intersects(relRect, selRect)) newSelection.add(file.path);
                    }
                });
                setSelectedFiles(newSelection);
            }
        }
        setIsDragging(false);
        setDragStart(null);
        setDragEnd(null);
    };

    const handleSelectAll = () => {
        setSelectedFiles(selectedFiles.size === files.length ? new Set() : new Set(files.map(f => f.path)));
    };

    if (!currentPath) return <div className="flex h-full items-center justify-center text-gray-500">{t('msg.select_folder')}</div>;
    if (loading) return <div className="text-gray-400 p-4">{t('status.loading')}</div>;
    if (files.length === 0) return <div className="text-gray-500 p-4">{t('msg.no_images')}</div>;

    const selRect = getSelectionRect();

    return (
        <>
            <div
                ref={containerRef}
                className="relative h-full overflow-y-auto select-none"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
            >
                <div className="flex items-center justify-between mb-4 sticky top-0 bg-gray-900 z-20 py-2 gap-4">
                    <h2 className="text-lg font-semibold text-gray-300 whitespace-nowrap">
                        {t('label.files', { count: files.length })}
                        {loadingThumbnails && <span className="text-xs text-blue-400 ml-2">{t('status.loading_thumbnails')}</span>}
                    </h2>
                    <button onClick={handleSelectAll} className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded text-gray-300 whitespace-nowrap">
                        {selectedFiles.size === displayedFiles.length ? t('action.deselect_all') : t('action.select_all')}
                    </button>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 grid-container pb-4">
                    {displayedFiles.map((file, idx) => (
                        <FileCard
                            key={file.path}
                            ref={el => { cardRefs.current[file.path] = el; }}
                            file={file}
                            isSelected={selectedFiles.has(file.path)}
                            onMouseDown={(e) => handleCardMouseDown(e, file, idx)}
                            thumbnail={thumbnails[file.path]}
                            loading={loadingThumbnails && file.extension === '.psd' && !thumbnails[file.path]}
                            onShowMeta={handleShowMeta}
                            hasMetadata={metadataStatus[file.path]}
                        />
                    ))}
                </div>
                {isDragging && selRect && selRect.width > 5 && selRect.height > 5 && (
                    <div className="absolute border-2 border-blue-500 bg-blue-500/20 pointer-events-none z-30"
                        style={{ left: selRect.left, top: selRect.top, width: selRect.width, height: selRect.height }} />
                )}
            </div>

            {metadata && <MetadataModal metadata={metadata} onClose={() => setMetadata(null)} />}
        </>
    );
};

export default FileGrid;
