import React, { useState, useEffect, useRef } from 'react';
import { Search, X, Loader2, SlidersHorizontal, Star, Info, Settings } from 'lucide-react';
import SettingsModal from './SettingsModal';
import { useLocale } from '../i18n';

const IMAGE_PREVIEW_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.webp'];

// Thumbnail cache shared across renders
const searchThumbnailCache = new Map();

// Category options (value keys for DB storage, labels via i18n)
const CATEGORY_OPTIONS = [
    { value: '', key: 'filter.all' },
    { value: 'Characters', key: 'category.characters' },
    { value: 'Backgrounds', key: 'category.backgrounds' },
    { value: 'UI Elements', key: 'category.ui_elements' },
    { value: 'Concepts', key: 'category.concepts' },
    { value: 'References', key: 'category.references' },
    { value: 'Archive', key: 'category.archive' },
];

// Category options for metadata modal (includes Uncategorized)
const META_CATEGORY_OPTIONS = [
    { value: '', key: 'category.uncategorized' },
    { value: 'Characters', key: 'category.characters' },
    { value: 'Backgrounds', key: 'category.backgrounds' },
    { value: 'UI Elements', key: 'category.ui_elements' },
    { value: 'Concepts', key: 'category.concepts' },
    { value: 'References', key: 'category.references' },
    { value: 'Archive', key: 'category.archive' },
];

// Metadata Modal (reused from FileGrid - imported inline to avoid circular deps)
const MetadataModal = ({ metadata, onClose }) => {
    const { t } = useLocale();
    const [lang, setLang] = useState('kr');
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

    useEffect(() => {
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
        if (lang === 'kr') return !!metadata.translated_tags;
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

                        <div className="shrink-0 bg-blue-900/10 border border-blue-700/30 rounded-lg p-4">
                            <div className="text-blue-300 text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-2">
                                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-[10px]">USER</span>
                                {t('label.my_notes_tags')}
                            </div>

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

                            <div className="mb-4">
                                <label className="text-gray-400 text-[10px] uppercase tracking-wider mb-1 block">{t('label.category')}</label>
                                <select
                                    value={editedData.user_category}
                                    onChange={(e) => setEditedData({...editedData, user_category: e.target.value})}
                                    className="w-full bg-black/30 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:border-blue-500 focus:outline-none transition-colors">
                                    {META_CATEGORY_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>{t(opt.key)}</option>
                                    ))}
                                </select>
                            </div>

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

                        {(metadata.ai_caption || metadata.ai_tags || metadata.dominant_color || metadata.ai_style) && (
                            <div className="shrink-0 bg-purple-900/10 border border-purple-700/30 rounded-lg p-3">
                                <div className="text-purple-300 text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-2">
                                    <span className="bg-purple-600 text-white px-2 py-0.5 rounded text-[10px]">AI</span>
                                    {t('label.vision_analysis')}
                                </div>

                                {metadata.ai_caption && (
                                    <div className="mb-3">
                                        <div className="text-gray-400 text-[10px] uppercase tracking-wider mb-1">{t('label.caption')}</div>
                                        <div className="text-purple-200 text-xs leading-relaxed bg-black/20 p-2 rounded">
                                            {metadata.ai_caption}
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

// Search Result Card Component
const SearchResultCard = ({ result, onShowMeta }) => {
    const fileName = result.path.split(/[/\\]/).pop();
    const ext = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();
    const canPreviewNatively = IMAGE_PREVIEW_EXTS.includes(ext);

    const toFileUrl = (p) => {
        // Resolve relative paths (e.g. "output/thumbnails/...") to absolute using projectRoot
        let resolved = p;
        if (!/^[A-Za-z]:/.test(p) && !/^\//.test(p)) {
            const root = window.electron?.projectRoot || '';
            resolved = root ? `${root}/${p}` : p;
        }
        const encoded = resolved.replace(/\\/g, '/').split('/').map(s => encodeURIComponent(s)).join('/');
        return `file:///${encoded}`;
    };

    // Prefer DB thumbnail (always generated during Process), fallback to native preview
    const thumbnailSrc = result.thumbnail_path
        ? toFileUrl(result.thumbnail_path)
        : canPreviewNatively
            ? toFileUrl(result.path)
            : null;

    return (
        <div
            className="group bg-gray-800 border border-gray-700 rounded-lg overflow-hidden hover:border-blue-500/50 transition-all hover:shadow-lg hover:shadow-blue-900/10 cursor-pointer"
            onClick={() => onShowMeta(result.path)}
        >
            {/* Thumbnail */}
            <div className="aspect-square bg-gray-900 overflow-hidden relative">
                {thumbnailSrc ? (
                    <img
                        src={thumbnailSrc}
                        alt={fileName}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                        onError={(e) => { e.target.style.display = 'none'; }}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-600">
                        <div className="text-center">
                            <div className="text-3xl font-bold uppercase">{ext.replace('.', '')}</div>
                        </div>
                    </div>
                )}

                {/* 2-axis score badges (Vector=blue, Text=green) */}
                <div className="absolute top-2 right-2 flex gap-1">
                    {result.vector_score != null && (
                        <span className="bg-blue-900/80 text-blue-300 text-[10px] font-bold px-1.5 py-0.5 rounded">
                            V {(result.vector_score * 100).toFixed(0)}%
                        </span>
                    )}
                    {result.text_score != null && (
                        <span className="bg-green-900/80 text-green-300 text-[10px] font-bold px-1.5 py-0.5 rounded">
                            T {(result.text_score * 100).toFixed(0)}%
                        </span>
                    )}
                </div>
                {/* v3 P0: image_type badge */}
                {result.image_type && (
                    <span className="absolute top-2 left-2 bg-purple-900/80 text-purple-300 text-[10px] font-bold px-1.5 py-0.5 rounded uppercase">
                        {result.image_type}
                    </span>
                )}
            </div>

            {/* Info */}
            <div className="p-3">
                <div className="text-sm font-medium text-white truncate">{fileName}</div>

                {result.ai_caption && (
                    <div className="text-xs text-gray-400 mt-1 line-clamp-2 leading-relaxed">
                        {result.ai_caption}
                    </div>
                )}

                {result.ai_tags && result.ai_tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                        {result.ai_tags.slice(0, 3).map((tag, i) => (
                            <span
                                key={i}
                                className="px-1.5 py-0.5 bg-blue-900/30 text-blue-400 text-[10px] rounded"
                            >
                                {tag}
                            </span>
                        ))}
                        {result.ai_tags.length > 3 && (
                            <span className="text-[10px] text-gray-500">+{result.ai_tags.length - 3}</span>
                        )}
                    </div>
                )}

                <div className="flex items-center gap-2 mt-2 text-[10px] text-gray-500">
                    <span className="uppercase font-medium">{result.format || ext.replace('.', '')}</span>
                    {result.width && result.height && (
                        <span>{result.width}x{result.height}</span>
                    )}
                    {result.layer_count > 0 && (
                        <span>{result.layer_count}L</span>
                    )}
                    <div className="flex items-center gap-1 ml-auto">
                        {result.user_rating > 0 && (
                            <span className="text-yellow-400">{'\u2605'.repeat(result.user_rating)}</span>
                        )}
                        {result.user_category && (
                            <span className="bg-gray-700 text-gray-300 px-1 rounded">{result.user_category}</span>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default function SearchPanel() {
    const { t } = useLocale();
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [error, setError] = useState(null);
    const [showFilters, setShowFilters] = useState(false);
    const [activeFilters, setActiveFilters] = useState({});
    const [threshold, setThreshold] = useState(0.15);
    const [metadata, setMetadata] = useState(null);
    const [showSettings, setShowSettings] = useState(false);
    const [dbStats, setDbStats] = useState(null);
    const inputRef = useRef(null);

    // Auto-focus search input on mount
    useEffect(() => {
        if (inputRef.current) {
            inputRef.current.focus();
        }
    }, []);

    // Load DB stats on mount
    useEffect(() => {
        if (window.electron?.pipeline?.getDbStats) {
            window.electron.pipeline.getDbStats()
                .then(stats => { if (stats.success) setDbStats(stats); })
                .catch(() => {});
        }
    }, []);

    const handleSearch = async () => {
        if (!query.trim()) return;

        setIsSearching(true);
        setError(null);

        try {
            const filters = {};
            if (activeFilters.format) filters.format = activeFilters.format;
            if (activeFilters.user_category) filters.user_category = activeFilters.user_category;
            if (activeFilters.min_rating) filters.min_rating = activeFilters.min_rating;
            if (activeFilters.image_type) filters.image_type = activeFilters.image_type;
            if (activeFilters.art_style) filters.art_style = activeFilters.art_style;

            const response = await window.electron.pipeline.searchVector({
                query,
                limit: 20,
                mode: 'triaxis',
                threshold,
                filters: Object.keys(filters).length > 0 ? filters : null,
            });

            if (response.success) {
                setResults(response.results);
            } else {
                setError(response.error || 'Search failed');
                setResults([]);
            }
        } catch (err) {
            setError(err.message);
            setResults([]);
        } finally {
            setIsSearching(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    };

    const clearSearch = () => {
        setResults([]);
        setQuery('');
        setError(null);
    };

    const handleShowMeta = async (filePath) => {
        const meta = await window.electron?.pipeline?.readMetadata(filePath);
        if (meta) {
            setMetadata(meta);
        }
    };

    const hasActiveFilters = Object.values(activeFilters).some(v => v);
    const hasResults = results.length > 0;
    const isEmptyState = !hasResults && !isSearching && !error;

    return (
        <div className="flex flex-col h-full bg-gray-900">
            {/* Search Header Area */}
            <div className={`flex flex-col items-center transition-all duration-300 ${isEmptyState ? 'pt-[20vh]' : 'pt-6'} px-6 pb-4 shrink-0`}>
                {/* Title (only in empty state) */}
                {isEmptyState && (
                    <div className="mb-6 text-center">
                        <h2 className="text-2xl font-bold text-gray-300 mb-2">{t('search.title')}</h2>
                        <p className="text-sm text-gray-500">{t('search.subtitle')}</p>
                        {dbStats && dbStats.total_files > 0 && (
                            <div className="mt-3 flex items-center justify-center gap-3 text-xs text-gray-500">
                                <span className="bg-gray-800 px-3 py-1.5 rounded-full border border-gray-700">
                                    <span className="text-blue-400 font-bold">{dbStats.total_files}</span> {t('msg.images_archived', { count: '' }).trim()}
                                </span>
                                {dbStats.format_distribution && Object.keys(dbStats.format_distribution).length > 0 && (
                                    <span className="bg-gray-800 px-3 py-1.5 rounded-full border border-gray-700">
                                        {Object.entries(dbStats.format_distribution).map(([fmt, cnt]) => (
                                            `${fmt} ${cnt}`
                                        )).join(' / ')}
                                    </span>
                                )}
                            </div>
                        )}
                    </div>
                )}

                {/* Search Bar */}
                <div className="w-full max-w-2xl">
                    <div className="flex items-center space-x-2">
                        <div className="flex-1 relative">
                            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                            <input
                                ref={inputRef}
                                type="text"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder={t('placeholder.search')}
                                className="w-full pl-10 pr-10 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 text-base"
                            />
                            {query && (
                                <button
                                    onClick={clearSearch}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                                >
                                    <X size={16} />
                                </button>
                            )}
                        </div>
                        <button
                            onClick={handleSearch}
                            disabled={!query.trim() || isSearching}
                            className="px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg flex items-center space-x-2 transition-colors"
                        >
                            {isSearching ? (
                                <Loader2 size={18} className="animate-spin" />
                            ) : (
                                <Search size={18} />
                            )}
                        </button>
                        {/* Filter Toggle */}
                        <button
                            onClick={() => setShowFilters(!showFilters)}
                            className={`p-3 rounded-lg transition-colors ${
                                showFilters || hasActiveFilters
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-800 hover:bg-gray-700 text-gray-400 border border-gray-600'
                            }`}
                            title={t('search.filters_title')}
                        >
                            <SlidersHorizontal size={18} />
                        </button>
                        {/* Settings Button */}
                        <button
                            onClick={() => setShowSettings(true)}
                            className="p-3 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 border border-gray-600 transition-colors"
                            title={t('search.settings_title')}
                        >
                            <Settings size={18} />
                        </button>
                    </div>

                    {/* Filter Bar */}
                    {showFilters && (
                        <div className="flex items-center gap-3 mt-3 px-2 py-2.5 bg-gray-800/50 rounded-lg border border-gray-700/50 flex-wrap">
                            {/* Format Filter */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase">{t('filter.format')}</span>
                                <select
                                    value={activeFilters.format || ''}
                                    onChange={(e) => setActiveFilters({...activeFilters, format: e.target.value || undefined})}
                                    className="bg-gray-700 text-white text-xs px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                                >
                                    <option value="">{t('filter.all')}</option>
                                    <option value="PSD">PSD</option>
                                    <option value="PNG">PNG</option>
                                    <option value="JPG">JPG</option>
                                </select>
                            </div>

                            {/* Category Filter */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase">{t('filter.category')}</span>
                                <select
                                    value={activeFilters.user_category || ''}
                                    onChange={(e) => setActiveFilters({...activeFilters, user_category: e.target.value || undefined})}
                                    className="bg-gray-700 text-white text-xs px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                                >
                                    {CATEGORY_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>{t(opt.key)}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Image Type Filter (v3 P0) */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase">TYPE</span>
                                <select
                                    value={activeFilters.image_type || ''}
                                    onChange={(e) => setActiveFilters({...activeFilters, image_type: e.target.value || undefined})}
                                    className="bg-gray-700 text-white text-xs px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                                >
                                    <option value="">{t('filter.all')}</option>
                                    <option value="character">Character</option>
                                    <option value="background">Background</option>
                                    <option value="ui_element">UI</option>
                                    <option value="item">Item</option>
                                    <option value="icon">Icon</option>
                                    <option value="texture">Texture</option>
                                    <option value="effect">Effect</option>
                                    <option value="logo">Logo</option>
                                    <option value="photo">Photo</option>
                                    <option value="illustration">Illustration</option>
                                </select>
                            </div>

                            {/* Art Style Filter (v3 P0) */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase">STYLE</span>
                                <select
                                    value={activeFilters.art_style || ''}
                                    onChange={(e) => setActiveFilters({...activeFilters, art_style: e.target.value || undefined})}
                                    className="bg-gray-700 text-white text-xs px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                                >
                                    <option value="">{t('filter.all')}</option>
                                    <option value="realistic">Realistic</option>
                                    <option value="anime">Anime</option>
                                    <option value="pixel">Pixel</option>
                                    <option value="painterly">Painterly</option>
                                    <option value="cartoon">Cartoon</option>
                                    <option value="3d_render">3D Render</option>
                                    <option value="flat_design">Flat Design</option>
                                    <option value="sketch">Sketch</option>
                                </select>
                            </div>

                            {/* Rating Filter */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase">{t('label.min_rating')}</span>
                                <div className="flex gap-0.5">
                                    {[1, 2, 3, 4, 5].map(star => (
                                        <button
                                            key={star}
                                            onClick={() => setActiveFilters({
                                                ...activeFilters,
                                                min_rating: activeFilters.min_rating === star ? undefined : star
                                            })}
                                            className={`text-sm transition-colors ${
                                                star <= (activeFilters.min_rating || 0) ? 'text-yellow-400' : 'text-gray-600 hover:text-gray-400'
                                            }`}
                                        >
                                            &#9733;
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Similarity Threshold */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase whitespace-nowrap">{t('label.threshold')}</span>
                                <input
                                    type="range"
                                    min="0"
                                    max="0.5"
                                    step="0.05"
                                    value={threshold}
                                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                    className="w-20 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                />
                                <span className="text-xs text-blue-400 font-mono w-8">{(threshold * 100).toFixed(0)}%</span>
                            </div>

                            {/* Clear Filters */}
                            {hasActiveFilters && (
                                <button
                                    onClick={() => { setActiveFilters({}); setThreshold(0.15); }}
                                    className="text-[10px] text-red-400 hover:text-red-300 px-2 py-1 rounded bg-red-900/20 border border-red-800/30"
                                >
                                    {t('action.clear')}
                                </button>
                            )}
                        </div>
                    )}

                    {error && (
                        <div className="mt-3 px-3 py-2 bg-red-900/30 border border-red-700 rounded-lg text-red-400 text-sm">
                            {error}
                        </div>
                    )}
                </div>
            </div>

            {/* Results Area */}
            <div className="flex-1 overflow-y-auto px-6 pb-6">
                {isSearching ? (
                    <div className="flex items-center justify-center h-40 text-blue-400 gap-2">
                        <Loader2 className="animate-spin" size={20} />
                        <span>{t('status.searching')}</span>
                    </div>
                ) : hasResults ? (
                    <>
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-medium text-gray-400">
                                {t('status.results_found', { count: results.length })}
                            </h3>
                            <button
                                onClick={clearSearch}
                                className="text-xs text-gray-500 hover:text-gray-300"
                            >
                                {t('action.clear_results')}
                            </button>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-4">
                            {results.map((result, index) => (
                                <SearchResultCard
                                    key={index}
                                    result={result}
                                    onShowMeta={handleShowMeta}
                                />
                            ))}
                        </div>
                    </>
                ) : null}
            </div>

            {/* Metadata Modal */}
            {metadata && <MetadataModal metadata={metadata} onClose={() => setMetadata(null)} />}
            {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
        </div>
    );
}
