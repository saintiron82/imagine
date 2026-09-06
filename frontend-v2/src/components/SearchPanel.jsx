import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Search, X, Loader2, SlidersHorizontal, Star, Info, Settings, FolderOpen, ExternalLink, Archive, Sparkles } from 'lucide-react';
import ImageSearchInput from './ImageSearchInput';
import SearchHistorySidebar from './SearchHistorySidebar';
import { useLocale } from '../i18n';
import { useResponsiveColumns } from '../hooks/useResponsiveColumns';
import { searchImages, getDbStats as bridgeGetDbStats, getFileDetail, updateUserMeta, getThumbnailUrl, getActiveDomainConfig, postSearchFeedback } from '../services/bridge';
import { pageState } from '../lib/paging';
import { isElectron, getServerUrl } from '../api/client';

const FETCH_LIMIT = 100;   // Backend returns up to this many results in one call
const DISPLAY_PAGE = 20;   // Show this many per "page" in the UI


// Fallback filter options (used when domain config unavailable)
const FALLBACK_IMAGE_TYPES = ['character', 'background', 'ui_element', 'item', 'icon', 'texture', 'effect', 'logo', 'photo', 'illustration'];
const FALLBACK_ART_STYLES = ['realistic', 'anime', 'pixel', 'painterly', 'cartoon', '3d_render', 'flat_design', 'sketch'];

// Confidence badge styling — one of 'high' | 'medium' | 'low' | 'empty'
function confidenceBadgeClass(level) {
    return {
        high:   'bg-emerald-900/30 text-emerald-300 border border-emerald-700/40 px-2 py-0.5 rounded text-xs',
        medium: 'bg-blue-900/30 text-blue-300 border border-blue-700/40 px-2 py-0.5 rounded text-xs',
        low:    'bg-amber-900/30 text-amber-300 border border-amber-700/40 px-2 py-0.5 rounded text-xs',
        empty:  'bg-gray-900/40 text-gray-400 border border-gray-700/40 px-2 py-0.5 rounded text-xs',
    }[level] || 'text-gray-400 text-xs';
}

function dominantAxis(result) {
    const v = Number(result.vector_score ?? 0);
    const m = Number(result.text_vec_score ?? 0);
    const f = Number(result.text_score ?? 0) > 0 ? 0.4 : 0;
    const scores = [['vv', v], ['mv', m], ['fts', f]];
    scores.sort((a, b) => b[1] - a[1]);
    if (scores[0][1] <= 0) return null;
    return scores[0][0];
}

function axisBadgeClass(axis) {
    return {
        vv:  'bg-blue-900/30 text-blue-300 border border-blue-700/40 px-1.5 py-0.5 rounded text-[10px]',
        mv:  'bg-purple-900/30 text-purple-300 border border-purple-700/40 px-1.5 py-0.5 rounded text-[10px]',
        fts: 'bg-amber-900/30 text-amber-300 border border-amber-700/40 px-1.5 py-0.5 rounded text-[10px]',
    }[axis] || '';
}

// Filter label mapping for chips display
const FILTER_LABELS = {
    format: 'filter.format',
    user_category: 'filter.category',
    image_type: 'filter.type_label',
    art_style: 'filter.style_label',
    min_rating: 'filter.rating_label',
};

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

// Metadata Modal
const MetadataModal = ({ metadata, onClose, onNavigateToFolder }) => {
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

    // metadata 부재 가드는 호출부({metadata && <MetadataModal/>})가 담당한다 —
    // 훅 사이 조건부 return 은 훅 규칙 위반이라 여기서 두지 않는다.

    useEffect(() => {
        if (isInitialMount.current) {
            isInitialMount.current = false;
            return;
        }

        if (saveTimer.current) clearTimeout(saveTimer.current);
        saveTimer.current = setTimeout(async () => {
            try {
                await updateUserMeta(metadata.id, editedData);
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

    const getSpatialObjects = () => {
        if (Array.isArray(metadata.spatial_objects) && metadata.spatial_objects.length > 0) {
            return metadata.spatial_objects;
        }
        const structured = typeof metadata.structured_meta === 'string'
            ? (() => { try { return JSON.parse(metadata.structured_meta); } catch { return {}; } })()
            : metadata.structured_meta;
        return Array.isArray(structured?.objects) ? structured.objects : [];
    };

    const tags = getTags();
    const spatialObjects = getSpatialObjects();

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

                        {/* Path Information */}
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
                                    {metadata.relative_path && (
                                        <div className="flex items-start gap-2">
                                            <span className="text-gray-500 text-[10px] uppercase shrink-0 w-14">{t('label.relative_path')}</span>
                                            <span className="text-gray-300 text-[11px] font-mono">{metadata.relative_path}</span>
                                        </div>
                                    )}
                                </div>
                                {/* File Actions */}
                                <div className="flex flex-wrap gap-2 mt-2 pt-2 border-t border-gray-700/30">
                                    {isElectron && (
                                        <>
                                            <button onClick={() => window.electron?.fs?.showInFolder(metadata.file_path)}
                                                className="flex items-center gap-1.5 px-2.5 py-1 bg-gray-700/50 hover:bg-gray-600 rounded text-[11px] text-gray-400 hover:text-white transition-colors">
                                                <FolderOpen size={12} /> {t('action.show_in_folder')}
                                            </button>
                                            <button onClick={() => window.electron?.fs?.openFile(metadata.file_path)}
                                                className="flex items-center gap-1.5 px-2.5 py-1 bg-gray-700/50 hover:bg-gray-600 rounded text-[11px] text-gray-400 hover:text-white transition-colors">
                                                <ExternalLink size={12} /> {t('action.open_file')}
                                            </button>
                                        </>
                                    )}
                                    {onNavigateToFolder && (
                                        <button onClick={() => {
                                            const folderPath = metadata.file_path.replace(/[/\\][^/\\]+$/, '');
                                            onNavigateToFolder(folderPath);
                                            onClose();
                                        }}
                                            className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-900/30 hover:bg-blue-800/50 rounded text-[11px] text-blue-400 hover:text-blue-300 transition-colors">
                                            <Archive size={12} /> {t('action.browse_in_archive')}
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}

                        <div className="shrink-0 bg-blue-900/10 border border-blue-700/30 rounded-lg p-4">
                            <div className="text-blue-300 text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-2">
                                <span className="bg-blue-600 text-white px-2 py-0.5 rounded text-[10px]">USER</span>
                                {t('label.my_notes_tags')}
                            </div>

                            <div className="mb-4">
                                <label className="text-gray-400 text-[10px] uppercase tracking-wider mb-1 block">{t('label.notes')}</label>
                                <textarea
                                    value={editedData.user_note}
                                    onChange={(e) => setEditedData({ ...editedData, user_note: e.target.value })}
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
                                    onChange={(e) => setEditedData({ ...editedData, user_category: e.target.value })}
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
                                            onClick={() => setEditedData({ ...editedData, user_rating: star === editedData.user_rating ? 0 : star })}
                                            className={`text-2xl hover:text-yellow-300 cursor-pointer transition-colors ${star <= editedData.user_rating ? 'text-yellow-400' : 'text-gray-600'}`}>
                                            &#9733;
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>

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

                                {spatialObjects.length > 0 && (
                                    <div className="mb-3">
                                        <div className="text-gray-400 text-[10px] uppercase tracking-wider mb-1">SPATIAL OBJECTS</div>
                                        <div className="flex flex-wrap gap-1.5">
                                            {spatialObjects.slice(0, 12).map((obj, i) => {
                                                const label = [obj.ko_name || obj.name, obj.primary_location]
                                                    .filter(Boolean).join(' · ');
                                                return (
                                                    <span key={i} className="bg-emerald-800/30 text-emerald-200 border border-emerald-600/30 px-2 py-0.5 rounded text-xs">
                                                        {label}
                                                    </span>
                                                );
                                            })}
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

// Search Result Card Component (memoized to avoid re-renders on scroll/parent update)
const SearchResultCard = React.memo(({ result, onShowMeta, onContextMenu, onNavigateToFolder, onFindSimilar, onMarkIrrelevant }) => {
    const { t } = useLocale();
    const dbPath = result.db_path || result.path;
    const localPath = result.resolved_path || result.path;
    const fileName = (localPath || dbPath).split(/[/\\]/).pop();
    const ext = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();

    // DB thumbnail via server API URL with JWT
    const thumbnailSrc = getThumbnailUrl(result.thumbnail_path, result.id);

    return (
        <div
            className="group bg-gray-800 border border-gray-700 rounded-lg overflow-hidden hover:border-blue-500/50 hover:shadow-lg hover:shadow-blue-900/10 cursor-pointer h-full flex flex-col transition-[border-color,box-shadow] duration-150"
            onClick={() => onShowMeta(result.id || dbPath)}
            onContextMenu={onContextMenu}
        >
            {/* Thumbnail */}
            <div className="aspect-square bg-gray-900 overflow-hidden relative shrink-0">
                {thumbnailSrc ? (
                    <img
                        src={thumbnailSrc}
                        alt={fileName}
                        className="w-full h-full object-cover"
                        onError={(e) => { e.target.style.display = 'none'; }}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-600">
                        <div className="text-center">
                            <div className="text-3xl font-bold uppercase">{ext.replace('.', '')}</div>
                        </div>
                    </div>
                )}

                {/* Top: Combined RRF score (left) + axis scores (right) */}
                <div className="absolute top-2 left-2 right-2 flex items-start justify-between">
                    {result.combined_score > 0 ? (
                        <span className="bg-yellow-900/80 text-yellow-300 text-[10px] font-bold px-1.5 py-0.5 rounded">
                            ★ {(result.combined_score / 0.01639 * 100).toFixed(0)}%
                        </span>
                    ) : <span />}
                    <div className="flex gap-1">
                        {result.vector_score != null && (
                            <span className="bg-blue-900/80 text-blue-300 text-[10px] font-bold px-1.5 py-0.5 rounded">
                                {t('badge.vv')} {(result.vector_score * 100).toFixed(0)}
                            </span>
                        )}
                        {result.text_vec_score != null && (
                            <span className="bg-purple-900/80 text-purple-300 text-[10px] font-bold px-1.5 py-0.5 rounded">
                                {t('badge.mv')} {(result.text_vec_score * 100).toFixed(0)}
                            </span>
                        )}
                        {result.text_score != null && (
                            <span className="bg-green-900/80 text-green-300 text-[10px] font-bold px-1.5 py-0.5 rounded">
                                {t('badge.mc')} {(result.text_score * 100).toFixed(0)}
                            </span>
                        )}
                    </div>
                </div>
                {/* Bottom: image_type badge + dominant-axis badge */}
                <div className="absolute bottom-2 left-2 flex items-center gap-1">
                    {/* Progressive availability: file is searchable before MC
                        completes (FTS/VV axes) — label it so empty caption/tags
                        read as "in progress", not as a bug */}
                    {!result.mc_caption && (
                        <span className="bg-amber-900/80 text-amber-300 text-[10px] font-bold px-1.5 py-0.5 rounded">
                            {t('search.mc_pending')}
                        </span>
                    )}
                    {result.image_type && (
                        <span className="bg-purple-900/80 text-purple-300 text-[10px] font-bold px-1.5 py-0.5 rounded uppercase">
                            {result.image_type}
                        </span>
                    )}
                    {dominantAxis(result) && (
                        <span className={axisBadgeClass(dominantAxis(result))}>
                            {t(`search.axis_${dominantAxis(result)}`)}
                        </span>
                    )}
                </div>
            </div>

            {/* Info — compact 2-line layout */}
            <div className="px-2 py-1.5 overflow-hidden">
                {/* Line 1: filename + folder + action buttons */}
                <div className="flex items-center gap-1">
                    <div className="text-[11px] font-medium text-white truncate flex-1">{fileName}</div>
                    {onMarkIrrelevant && (
                        <button
                            onClick={(e) => { e.stopPropagation(); onMarkIrrelevant(result.id); }}
                            className="text-xs text-gray-400 hover:text-red-300 px-2 py-1 rounded border border-gray-700/40 shrink-0"
                            title={t('search.mark_irrelevant')}
                            aria-label={t('search.mark_irrelevant')}
                        >
                            <X size={12} />
                        </button>
                    )}
                    <div className="flex gap-0.5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                            onClick={(e) => { e.stopPropagation(); onFindSimilar?.(result.id); }}
                            className="p-0.5 hover:bg-blue-600/40 rounded text-gray-400 hover:text-blue-300 transition-colors"
                            title={t('action.find_similar_visual')}
                        >
                            <Sparkles size={12} />
                        </button>
                        {isElectron && (
                            <>
                                <button
                                    onClick={(e) => { e.stopPropagation(); window.electron?.fs?.showInFolder(localPath); }}
                                    className="p-0.5 hover:bg-gray-600 rounded text-gray-400 hover:text-white transition-colors"
                                    title={t('action.show_in_folder')}
                                >
                                    <FolderOpen size={12} />
                                </button>
                                <button
                                    onClick={(e) => { e.stopPropagation(); window.electron?.fs?.openFile(localPath); }}
                                    className="p-0.5 hover:bg-gray-600 rounded text-gray-400 hover:text-white transition-colors"
                                    title={t('action.open_file')}
                                >
                                    <ExternalLink size={12} />
                                </button>
                            </>
                        )}
                    </div>
                </div>
                {/* Line 2: folder path + caption (single truncated line) */}
                <div className="text-[9px] text-gray-500 truncate mt-0.5" title={result.mc_caption || ''}>
                    {result.folder_path && (
                        <span
                            className={onNavigateToFolder ? 'hover:text-blue-400 cursor-pointer' : ''}
                            onClick={onNavigateToFolder ? (e) => {
                                e.stopPropagation();
                                const filePath = localPath || dbPath;
                                const folderPath = filePath.replace(/[/\\][^/\\]+$/, '');
                                onNavigateToFolder(folderPath);
                            } : undefined}
                        >
                            {result.folder_path}
                        </span>
                    )}
                    {result.folder_path && result.mc_caption && <span className="mx-1 text-gray-700">|</span>}
                    {result.mc_caption && <span className="text-gray-500">{result.mc_caption}</span>}
                </div>

                {/* Line 3: tags + file info (inline) */}
                <div className="flex items-center gap-1 mt-0.5 overflow-hidden">
                    {result.ai_tags && result.ai_tags.slice(0, 3).map((tag, i) => (
                        <span key={i} className="px-1 py-0 bg-blue-900/30 text-blue-400 text-[8px] rounded shrink-0">{tag}</span>
                    ))}
                    {result.ai_tags && result.ai_tags.length > 3 && (
                        <span className="text-[8px] text-gray-600 shrink-0">+{result.ai_tags.length - 3}</span>
                    )}
                    <span className="text-[8px] text-gray-600 ml-auto shrink-0 uppercase">{result.format || ext.replace('.', '')}</span>
                    {result.width && result.height && (
                        <span className="text-[8px] text-gray-600 shrink-0">{result.width}x{result.height}</span>
                    )}
                    {result.layer_count > 0 && (
                        <span className="text-[8px] text-gray-600 shrink-0">{result.layer_count}L</span>
                    )}
                    {result.user_rating > 0 && (
                        <span className="text-yellow-400 text-[8px] shrink-0">{'\u2605'.repeat(result.user_rating)}</span>
                    )}
                </div>
            </div>
        </div>
    );
});

// Isolated search input — manages its own typing state so keystrokes
// never re-render the parent SearchPanel (fixes backspace flicker).
const EFFORT_LEVELS = [
    { id: 'low', label: 'Low', color: 'text-green-400 border-green-500/50' },
    { id: 'medium', label: 'Med', color: 'text-blue-400 border-blue-500/50' },
    { id: 'high', label: 'High', color: 'text-purple-400 border-purple-500/50' },
];

const SearchInput = React.memo(({ onSearch, onClear, hasImages, isSearching, showFilters, hasActiveFilters, onToggleFilters, onOpenSettings, inputRef, resetSignal }) => {
    const { t } = useLocale();
    const [localQuery, setLocalQuery] = useState('');
    const [useCodex, setUseCodex] = useState(() => localStorage.getItem('search_use_codex') !== 'false');
    const [effort, setEffort] = useState(() => localStorage.getItem('search_effort') || 'low');

    useEffect(() => { if (resetSignal > 0) setLocalQuery(''); }, [resetSignal]);
    useEffect(() => { if (inputRef.current) inputRef.current.focus(); }, []);

    const toggleCodex = () => { const next = !useCodex; setUseCodex(next); localStorage.setItem('search_use_codex', String(next)); };
    const cycleEffort = () => { const idx = EFFORT_LEVELS.findIndex(l => l.id === effort); const next = EFFORT_LEVELS[(idx + 1) % EFFORT_LEVELS.length].id; setEffort(next); localStorage.setItem('search_effort', next); };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && (localQuery.trim() || hasImages)) {
            onSearch(localQuery, { useCodex, effort });
        }
    };
    const handleSearchClick = () => {
        if (localQuery.trim() || hasImages) onSearch(localQuery, { useCodex, effort });
    };
    const handleClear = () => { setLocalQuery(''); onClear(); };

    return (
        <div className="flex items-center space-x-2">
            <div className="flex-1 relative">
                <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input
                    ref={inputRef}
                    type="text"
                    value={localQuery}
                    onChange={(e) => setLocalQuery(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={t('placeholder.search')}
                    className="w-full pl-10 pr-10 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 text-base"
                />
                {(localQuery || hasImages) && (
                    <button onClick={handleClear} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
                        <X size={16} />
                    </button>
                )}
            </div>
            <button
                onClick={handleSearchClick}
                disabled={(!localQuery.trim() && !hasImages) || isSearching}
                className="px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors shrink-0"
            >
                {isSearching ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
            </button>
            {/* Codex Toggle */}
            <button
                onClick={toggleCodex}
                className={`px-2.5 py-3 rounded-lg border text-[10px] font-bold transition-colors ${
                    useCodex
                        ? 'text-emerald-400 border-emerald-500/50 bg-emerald-900/20 hover:bg-emerald-900/30'
                        : 'text-gray-500 border-gray-600 bg-gray-800 hover:bg-gray-700'
                }`}
                title={useCodex ? t('search.codex_on') : t('search.codex_off')}
            >
                Codex
            </button>
            {/* Effort Level Toggle */}
            {(() => {
                const level = EFFORT_LEVELS.find(l => l.id === effort) || EFFORT_LEVELS[0];
                return (
                    <button
                        onClick={cycleEffort}
                        className={`px-2.5 py-3 rounded-lg border text-[10px] font-bold transition-colors ${level.color} bg-gray-800 hover:bg-gray-700`}
                        title={t('search.effort')}
                    >
                        {level.label}
                    </button>
                );
            })()}
            {/* Filter Toggle */}
            <button
                onClick={onToggleFilters}
                className={`p-3 rounded-lg transition-colors ${showFilters || hasActiveFilters
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 hover:bg-gray-700 text-gray-400 border border-gray-600'
                    }`}
                title={t('search.filters_title')}
            >
                <SlidersHorizontal size={18} />
            </button>
            {/* Settings Button */}
            <button
                onClick={onOpenSettings}
                className="p-3 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 border border-gray-600 transition-colors"
                title={t('search.settings_title')}
            >
                <Settings size={18} />
            </button>
        </div>
    );
});

const SEARCH_GAP = 16;

// Virtualized search results grid (memoized — only re-renders when its own props change)
const SearchResults = React.memo(({ results, isSearching, searchStage, hasResults, onShowMeta, onClear, noMoreResults, onLoadMore, onContextMenu, onNavigateToFolder, onFindSimilar, onMarkIrrelevant, activeFilters, onRemoveFilter, searchScope, onClearScope, refineStack, refineInput, onRefineInputChange, onRefineCommit, onRefineRemove, totalCount, confidence }) => {
    const { t } = useLocale();
    const scrollRef = useRef(null);

    const { columnCount, cardWidth, rowHeight } = useResponsiveColumns(scrollRef, {
        breakpoints: [2, 500, 3, 768, 4, 1024, 5, 1280, 6],
        gap: SEARCH_GAP,
        cardAspectTotal: 1.45, // compact cards (image + 2-line info)
    });

    const rowCount = Math.ceil(results.length / columnCount);
    // +1 row for load-more button area
    const totalRows = hasResults ? rowCount + 1 : 0;

    const rowVirtualizer = useVirtualizer({
        count: totalRows,
        getScrollElement: () => scrollRef.current,
        estimateSize: (index) => index < rowCount ? rowHeight : 64,
        overscan: 5,
    });

    useEffect(() => {
        rowVirtualizer.measure();
    }, [rowHeight, columnCount]);

    // Reset scroll on new search results
    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = 0;
    }, [results.length > 0 && results[0]?.path]);

    const hasAnyResults = hasResults || (totalCount > 0);
    if (!hasAnyResults && !isSearching) {
        if (confidence === 'empty') {
            return (
                <div className="flex-1 px-6 pb-6">
                    <div className="flex items-center gap-2 mt-2">
                        <span className={confidenceBadgeClass('empty')}>
                            {t('search.confidence_empty')}
                        </span>
                    </div>
                    <p className="text-sm text-gray-400 mt-2">
                        {t('search.empty_explainer')}
                    </p>
                </div>
            );
        }
        return <div className="flex-1" />;
    }

    return (
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 pb-6 relative">
            {/* Searching overlay (keeps results visible underneath) */}
            {isSearching && (
                <div className="sticky top-0 z-10 flex justify-center py-3">
                    <div className="flex items-center gap-2 text-blue-400 bg-gray-900/90 px-4 py-2 rounded-full border border-blue-800/50 backdrop-blur-sm">
                        <Loader2 className="animate-spin" size={16} />
                        <span className="text-sm transition-opacity duration-300">
                            {searchStage ? t(`search.stage.${searchStage}`) : t('status.searching')}
                        </span>
                    </div>
                </div>
            )}

            {/* Scope bar — AI-extracted search scope */}
            {hasAnyResults && searchScope && (
                <div className="flex items-center gap-2 mb-2 px-3 py-1.5 bg-amber-900/20 border border-amber-700/30 rounded-lg text-[11px]">
                    <FolderOpen size={12} className="text-amber-400 flex-shrink-0" />
                    {searchScope.folder && (
                        <span className="text-amber-300 font-medium">{searchScope.folder}</span>
                    )}
                    {searchScope.image_type && (
                        <span className="text-amber-300/70">· {searchScope.image_type}</span>
                    )}
                    {searchScope.file_count != null && (
                        <span className="text-gray-500">({t('scope.files_count', { count: searchScope.file_count })})</span>
                    )}
                    <span className="text-gray-500">{t('scope.searching_in')}</span>
                    <button
                        onClick={onClearScope}
                        className="text-amber-500 hover:text-amber-300 ml-auto text-[10px]"
                    >
                        {t('scope.clear')}
                    </button>
                </div>
            )}

            {/* Refine stack — committed levels (read-only chips) + new input chain */}
            {hasAnyResults && (
                <div className="flex flex-col gap-1.5 mb-2">
                    {/* Committed refine levels */}
                    {refineStack.map((level, i) => (
                        <div key={i} className="flex items-center gap-2">
                            <span className="text-[9px] text-gray-500 shrink-0">{t('scope.searching_in')}</span>
                            <div className="flex-1 max-w-xs px-3 py-1 bg-cyan-900/20 border border-cyan-700/30 rounded-lg text-[11px] text-cyan-300 flex items-center justify-between">
                                <span className="truncate">{level.query}</span>
                                <button onClick={() => onRefineRemove(i)} className="text-cyan-500/60 hover:text-cyan-300 ml-2 shrink-0">
                                    <X size={11} />
                                </button>
                            </div>
                        </div>
                    ))}
                    {/* New refine input */}
                    <div className="flex items-center gap-2">
                        <span className="text-[9px] text-gray-500 shrink-0">{t('scope.searching_in')}</span>
                        <div className="relative flex-1 max-w-xs">
                            <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-500" />
                            <input type="text" value={refineInput}
                                onChange={(e) => onRefineInputChange(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); onRefineCommit(); } }}
                                placeholder={t('search.refine_placeholder')}
                                className="w-full pl-7 pr-8 py-1.5 bg-gray-800/60 border border-gray-700/50 rounded-lg text-white text-[11px] placeholder-gray-600 focus:outline-none focus:border-blue-500/50" />
                            {refineInput && (
                                <button onClick={() => onRefineInputChange('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
                                    <X size={12} /></button>
                            )}
                        </div>
                        {refineInput && <span className="text-[10px] text-gray-500">Enter↵</span>}
                    </div>
                </div>
            )}

            {hasAnyResults && (
                <div className="flex items-center justify-between mb-4 gap-2 flex-wrap">
                    <div className="flex items-center gap-2 flex-wrap">
                        <h3 className="text-sm font-medium text-gray-400">
                            {t('status.results_found', { count: results.length })}
                        </h3>
                        {confidence && (
                            <span className={confidenceBadgeClass(confidence)}>
                                {t(`search.confidence_${confidence}`)}
                            </span>
                        )}
                        {/* Active filter chips — always visible */}
                        {activeFilters && Object.entries(activeFilters).map(([key, value]) => {
                            if (!value) return null;
                            const label = FILTER_LABELS[key];
                            const display = key === 'min_rating' ? `${'★'.repeat(value)}` : value;
                            return (
                                <span key={key} className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-900/30 border border-blue-700/40 rounded-full text-[10px] text-blue-300">
                                    <span className="text-blue-400/60 uppercase">{label ? t(label) : key}:</span>
                                    <span>{display}</span>
                                    <button
                                        onClick={() => onRemoveFilter(key)}
                                        className="text-blue-500/60 hover:text-blue-300 ml-0.5"
                                    >
                                        <X size={10} />
                                    </button>
                                </span>
                            );
                        })}
                    </div>
                    <button onClick={onClear} className="text-xs text-gray-500 hover:text-gray-300">
                        {t('action.clear_results')}
                    </button>
                </div>
            )}

            {hasResults && (
                <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, width: '100%', position: 'relative' }}>
                    {rowVirtualizer.getVirtualItems().map(virtualRow => {
                        // Last virtual row = load more button
                        if (virtualRow.index >= rowCount) {
                            return (
                                <div
                                    key="load-more"
                                    style={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        width: '100%',
                                        height: `${virtualRow.size}px`,
                                        transform: `translateY(${virtualRow.start}px)`,
                                    }}
                                >
                                    <div className="flex justify-center py-3">
                                        {noMoreResults ? (
                                            <span className="text-xs text-gray-600">{t('status.no_more_results')}</span>
                                        ) : (
                                            <button
                                                onClick={onLoadMore}
                                                className="px-6 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white rounded-lg border border-gray-600 hover:border-gray-500 transition-all flex items-center gap-2 text-sm"
                                            >
                                                {t('action.load_more')}
                                            </button>
                                        )}
                                    </div>
                                </div>
                            );
                        }

                        const rowStartIdx = virtualRow.index * columnCount;
                        const rowResults = results.slice(rowStartIdx, rowStartIdx + columnCount);
                        return (
                            <div
                                key={virtualRow.key}
                                style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: `${virtualRow.size}px`,
                                    transform: `translateY(${virtualRow.start}px)`,
                                }}
                            >
                                <div className="flex" style={{ gap: `${SEARCH_GAP}px` }}>
                                    {rowResults.map((result, colIdx) => (
                                        <div key={result.db_path || result.path || colIdx} style={{ width: `${cardWidth}px`, height: `${rowHeight - SEARCH_GAP}px`, flexShrink: 0 }}>
                                            <SearchResultCard
                                                result={result}
                                                onShowMeta={onShowMeta}
                                                onContextMenu={(e) => onContextMenu(e, result)}
                                                onNavigateToFolder={onNavigateToFolder}
                                                onFindSimilar={onFindSimilar}
                                                onMarkIrrelevant={onMarkIrrelevant}
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
});

// Search Result Card, SearchInput, SearchResults...

function SearchPanel({ onScanFolder, isBusy, initialSearch, onSearchConsumed, reloadSignal = 0, onOpenSettings, onNavigateToFolder }) {
    const { t } = useLocale();
    // query stores the last *submitted* search text (not live typing)
    const [query, setQuery] = useState('');
    const [queryFileId, setQueryFileId] = useState(null);
    const [searchMode, setSearchMode] = useState(null); // 'structure', 'vector', or null (inferred)
    const [queryImages, setQueryImages] = useState([]); // base64 string array
    const [imageSearchMode, setImageSearchMode] = useState('and'); // 'and' | 'or'
    const [results, setResults] = useState([]); // Currently displayed slice
    const [confidence, setConfidence] = useState(null); // 'high'|'medium'|'low'|'empty' from backend
    const allResultsRef = useRef([]); // Full results from backend (up to FETCH_LIMIT)
    const [isSearching, setIsSearching] = useState(false);
    const [searchStage, setSearchStage] = useState(null); // 'decompose'|'visual'|'semantic'|'keyword'|'ranking'
    const [searchHistory, setSearchHistory] = useState(() => {
        try { return JSON.parse(localStorage.getItem('search_history_v2') || '[]'); }
        catch { return []; }
    });
    const [error, setError] = useState(null);
    const [showFilters, setShowFilters] = useState(false);
    const [activeFilters, setActiveFilters] = useState({});
    const [threshold, setThreshold] = useState(0);
    const [searchScope, setSearchScope] = useState(null); // {folder, image_type, format, file_count, ...}
    const [domainConfig, setDomainConfig] = useState(null); // active domain image_types/art_styles
    // Refine stack: [{query, resultIds}] — each level is a single query within previous results
    const [refineStack, setRefineStack] = useState([]);
    const [refineInput, setRefineInput] = useState('');
    const [contextMenu, setContextMenu] = useState(null); // { x, y, result }
    const [metadata, setMetadata] = useState(null);
    // showSettings removed — settings now in dedicated tab
    const [dbStats, setDbStats] = useState(null);
    const [currentLimit, setCurrentLimit] = useState(20);
    const [noMoreResults, setNoMoreResults] = useState(false);
    const [resetSignal, setResetSignal] = useState(0);
    const inputRef = useRef(null);

    // Listen for search progress events (Electron IPC → window.electron.pipeline.onSearchProgress)
    useEffect(() => {
        const subscribe = window.electron?.pipeline?.onSearchProgress;
        if (!subscribe) return;
        const cleanup = subscribe((stage) => {
            setSearchStage(stage);
        });
        return () => cleanup?.();
    }, []);

    // Server mode: estimated stage progression (no IPC events available)
    useEffect(() => {
        if (!isSearching) return;
        if (window.electron?.pipeline?.onSearchProgress) return;
        const stages = ['decompose', 'visual', 'semantic', 'keyword', 'ranking'];
        const delays = [0, 800, 1800, 2600, 3200]; // estimated ms
        const timers = stages.map((stage, i) =>
            setTimeout(() => setSearchStage(stage), delays[i])
        );
        return () => timers.forEach(clearTimeout);
    }, [isSearching]);

    // Load DB stats on mount and when pipeline/discover refresh signal changes.
    useEffect(() => {
        bridgeGetDbStats()
            .then(stats => { if (stats.success !== false) setDbStats(stats); })
            .catch(() => { });
    }, [reloadSignal]);

    // Load active domain config for dynamic filter options
    useEffect(() => {
        getActiveDomainConfig()
            .then(data => { if (data?.domain) setDomainConfig(data.domain); })
            .catch(() => {});
    }, []);

    // Handle initial search trigger (initialSearch prop)
    useEffect(() => {
        if (initialSearch) {
            const { queryFileId, mode } = initialSearch;
            setQuery('');
            setQueryImages([]);
            setQueryFileId(queryFileId);
            setSearchMode(mode);
            allResultsRef.current = [];
            setResults([]);
            setConfidence(null);
            setNoMoreResults(false);
            setIsSearching(true); setSearchStage('decompose');
            setError(null);
            // Reset filters for a fresh "find similar"
            setActiveFilters({});
            setThreshold(0);

            const performSearch = async () => {
                try {
                    const searchOptions = {
                        limit: FETCH_LIMIT,
                        threshold: 0,
                        filters: null,
                        queryFileId,
                        mode
                    };
                    const response = await searchImages(searchOptions);
                    if (response.success) {
                        const all = response.results || [];
                        allResultsRef.current = all;
                        const page = pageState(all, DISPLAY_PAGE);
                        setResults(page.visible);
                        setConfidence(response.confidence || null);
                        setCurrentLimit(DISPLAY_PAGE);
                        setNoMoreResults(page.noMore);
                    } else {
                        setError(response.error || 'Search failed');
                        allResultsRef.current = [];
                        setResults([]);
                        setConfidence(null);
                    }
                } catch (err) {
                    setError(err.message);
                } finally {
                    setIsSearching(false); setSearchStage(null);
                    if (onSearchConsumed) onSearchConsumed();
                }
            };
            performSearch();
        }
    }, [initialSearch]);

    // Removed: last_search_query auto-restore (session cache handles this now)

    const handleCardContextMenu = useCallback((e, result) => {
        e.preventDefault();
        setContextMenu({ x: e.clientX, y: e.clientY, result });
    }, []);

    // Mark a result irrelevant for the current query: post feedback then
    // locally remove it so the user sees immediate effect. The backend will
    // soft-demote on subsequent searches via Phase D demotion.
    const handleMarkIrrelevant = useCallback(async (fileId) => {
        const currentQuery = query;
        if (!currentQuery || !fileId) return;
        // Optimistically remove from both the displayed slice and the full result set
        setResults(prev => prev.filter(r => r.id !== fileId));
        allResultsRef.current = allResultsRef.current.filter(r => r.id !== fileId);
        try {
            await postSearchFeedback(currentQuery, fileId);
        } catch (err) {
            // Non-fatal: log but don't restore (user intent was clear)
            console.warn('postSearchFeedback failed:', err);
        }
    }, [query]);

    const triggerStructureSearch = (fileId, mode) => {
        setQuery('');
        setQueryImages([]);
        setQueryFileId(fileId);
        setSearchMode(mode);
        allResultsRef.current = [];
        setResults([]);
        setConfidence(null);
        setNoMoreResults(false);
        setIsSearching(true); setSearchStage('decompose');
        setError(null);
        setActiveFilters({});
        setThreshold(0);

        const performSearch = async () => {
            try {
                const searchOptions = {
                    limit: FETCH_LIMIT,
                    threshold: 0,
                    filters: null,
                    queryFileId: fileId,
                    mode
                };
                const response = await searchImages(searchOptions);
                if (response.success) {
                    const all = response.results || [];
                    allResultsRef.current = all;
                    const page = pageState(all, DISPLAY_PAGE);
                    setResults(page.visible);
                    setConfidence(response.confidence || null);
                    setCurrentLimit(DISPLAY_PAGE);
                    setNoMoreResults(page.noMore);
                } else {
                    setError(response.error || 'Search failed');
                    allResultsRef.current = [];
                    setResults([]);
                    setConfidence(null);
                }
            } catch (err) {
                setError(err.message);
            } finally {
                setIsSearching(false); setSearchStage(null);
            }
        };
        performSearch();
    };

    // Ref to capture latest state for stable callbacks
    const searchStateRef = useRef();
    const lastSearchConfigRef = useRef({ useCodex: true, effort: 'low' }); // Track last search config for Load More
    searchStateRef.current = { query, queryFileId, searchMode, queryImages, imageSearchMode, activeFilters, threshold, currentLimit, results };

    const handleSearch = useCallback(async (searchQuery, { useCodex = true, effort = 'low' } = {}) => {
        const { queryImages, imageSearchMode, activeFilters, threshold } = searchStateRef.current;
        const hasText = searchQuery && searchQuery.trim().length > 0;
        const hasImages = queryImages.length > 0;
        if (!hasText && !hasImages) return;

        setQuery(searchQuery || '');
        setQueryFileId(null);
        setSearchMode(null);
        setIsSearching(true); setSearchStage('decompose');
        setError(null);
        setRefineStack([]);
        setRefineInput('');

        lastSearchConfigRef.current = { useCodex, effort };

        const t0 = performance.now();
        try {
            const filters = {};
            if (activeFilters.format) filters.format = activeFilters.format;
            if (activeFilters.user_category) filters.user_category = activeFilters.user_category;
            if (activeFilters.min_rating) filters.min_rating = activeFilters.min_rating;
            if (activeFilters.image_type) filters.image_type = activeFilters.image_type;
            if (activeFilters.art_style) filters.art_style = activeFilters.art_style;

            const baseOpts = {
                limit: FETCH_LIMIT, threshold, mode: 'triaxis',
                filters: Object.keys(filters).length > 0 ? filters : null,
                use_codex: useCodex, effort,
            };

            let response;
            if (hasText && !hasImages) {
                response = await searchImages({ ...baseOpts, query: searchQuery });
            } else if (hasImages && !hasText) {
                const opts = { ...baseOpts, mode: 'vector' };
                if (queryImages.length === 1) opts.queryImage = queryImages[0];
                else { opts.queryImages = queryImages; opts.imageSearchMode = imageSearchMode; }
                response = await searchImages(opts);
            } else {
                const opts = { ...baseOpts, query: searchQuery };
                if (queryImages.length === 1) opts.queryImage = queryImages[0];
                else { opts.queryImages = queryImages; opts.imageSearchMode = imageSearchMode; }
                response = await searchImages(opts);
            }

            const elapsed = (performance.now() - t0).toFixed(0);
            console.log(`[Search] ✅ ${response.results?.length || 0} results in ${elapsed}ms`);

            if (response.success) {
                // Sort by combined_score descending so display order matches ★ badge
                const all = (response.results || []).sort((a, b) => (b.combined_score || 0) - (a.combined_score || 0));
                allResultsRef.current = all;
                const page = pageState(all, DISPLAY_PAGE);
                setResults(page.visible);
                setConfidence(response.confidence || null);
                setCurrentLimit(DISPLAY_PAGE);
                setNoMoreResults(page.noMore);

                // Update scope from backend decomposition
                const scope = response.scope || null;
                const hasScope = scope && (scope.folder || scope.image_type || scope.format);
                setSearchScope(hasScope ? scope : null);

                // Update history
                const cacheKey = searchQuery?.trim();
                if (cacheKey) {
                    setSearchHistory(prev => {
                        const filtered = prev.filter(h => h.query !== cacheKey);
                        const { activeFilters, threshold } = searchStateRef.current;
                        const next = [{
                            query: cacheKey,
                            resultCount: allResultsRef.current.length,
                            timestamp: Date.now(),
                            filters: { ...activeFilters },
                            threshold,
                        }, ...filtered].slice(0, 30);
                        try { localStorage.setItem('search_history_v2', JSON.stringify(next)); } catch {}
                        return next;
                    });
                }
            } else {
                setError(response.error || 'Search failed');
                setResults([]);
                setConfidence(null);
            }
        } catch (err) {
            setError(err.message);
            setResults([]);
            setConfidence(null);
        } finally {
            setIsSearching(false); setSearchStage(null);
        }
    }, []);

    const handleLoadMore = useCallback(() => {
        const { currentLimit } = searchStateRef.current;
        const nextLimit = currentLimit + DISPLAY_PAGE;
        const page = pageState(allResultsRef.current, nextLimit);
        setResults(page.visible);
        setCurrentLimit(nextLimit);
        setNoMoreResults(page.noMore);
    }, []);

    const handleRefineCommit = useCallback(async () => {
        if (!refineInput.trim()) return;
        const currentResults = searchStateRef.current.results;
        const fileIds = currentResults.map(r => r.id).filter(Boolean);
        if (!fileIds.length) return;

        setIsSearching(true); setSearchStage('decompose');
        try {
            const cfg = lastSearchConfigRef.current;
            const response = await searchImages({
                query: refineInput, file_ids: fileIds, limit: FETCH_LIMIT,
                mode: 'triaxis', use_codex: cfg.useCodex, effort: cfg.effort,
            });
            if (response.success) {
                // (response.results || []) — a success:true body with results
                // missing would otherwise throw on .sort and kill the panel.
                const sorted = (response.results || []).sort((a, b) => (b.combined_score || 0) - (a.combined_score || 0));
                // allResultsRef is the pool handleLoadMore slices. Leaving it on
                // the pre-refine results is the same defect 21ef8ed fixed for the
                // structure/initial paths; it is only masked here by noMore=true.
                allResultsRef.current = sorted;
                setResults(sorted);
                setConfidence(response.confidence || null);
                setCurrentLimit(sorted.length);
                setNoMoreResults(true);
                setRefineStack(prev => [...prev, { query: refineInput, resultIds: fileIds }]);
                setRefineInput('');
            }
        } catch (err) {
            console.error('[Refine] search failed:', err);
        } finally {
            setIsSearching(false); setSearchStage(null);
        }
    }, [refineInput]);

    const handleRefineRemove = useCallback(async (index) => {
        setRefineInput('');
        if (index === 0) {
            setRefineStack([]);
            const all = allResultsRef.current;
            setResults(all.slice(0, DISPLAY_PAGE));
            setCurrentLimit(DISPLAY_PAGE);
            setNoMoreResults(DISPLAY_PAGE >= all.length);
            return;
        }
        const kept = refineStack.slice(0, index);
        setRefineStack(kept);
        const prevLevel = kept[kept.length - 1];
        const cfg = lastSearchConfigRef.current;
        setIsSearching(true); setSearchStage('decompose');
        try {
            const response = await searchImages({
                query: prevLevel.query, file_ids: prevLevel.resultIds, limit: FETCH_LIMIT,
                mode: 'triaxis', use_codex: cfg.useCodex, effort: cfg.effort,
            });
            if (response.success) {
                const sorted = (response.results || []).sort((a, b) => (b.combined_score || 0) - (a.combined_score || 0));
                allResultsRef.current = sorted;   // keep the load-more pool in step
                setResults(sorted);
                setConfidence(response.confidence || null);
                setCurrentLimit(sorted.length);
                setNoMoreResults(true);
            }
        } catch (err) {
            console.error('[Refine] re-search failed:', err);
        } finally {
            setIsSearching(false); setSearchStage(null);
        }
    }, [refineStack]);

    const clearSearch = useCallback(() => {
        setResults([]);
        setConfidence(null);
        allResultsRef.current = [];
        setQuery('');
        setQueryImages([]);
        setRefineStack([]);
        setRefineInput('');
        setError(null);
        setCurrentLimit(DISPLAY_PAGE);
        setNoMoreResults(false);
        setResetSignal(c => c + 1);
    }, []);

    const toggleFilters = useCallback(() => {
        setShowFilters(prev => !prev);
    }, []);

    const openSettings = useCallback(() => {
        onOpenSettings?.();
    }, [onOpenSettings]);

    const handleShowMeta = useCallback(async (filePathOrId) => {
        try {
            const result = await getFileDetail(filePathOrId);
            const meta = result?.file || result;
            if (meta) setMetadata(meta);
        } catch (e) {
            console.error('Failed to load file detail:', filePathOrId, e);
        }
    }, []);

    const hasActiveFilters = Object.values(activeFilters).some(v => v);
    const hasResults = results.length > 0;
    const isEmptyState = !hasResults && !isSearching && !error;
    const buildStatus = dbStats?.build_status || null;
    const rebuildRequired = !!buildStatus?.needs_rebuild;

    const handleHistorySelect = useCallback((item) => {
        // Support both old format (string) and new format (object with filters)
        if (typeof item === 'string') {
            handleSearch(item);
            return;
        }
        if (item.filters) setActiveFilters(item.filters);
        if (item.threshold != null) setThreshold(item.threshold);
        handleSearch(item.query);
    }, [handleSearch]);

    const handleHistoryDelete = useCallback((q) => {
        setSearchHistory(prev => {
            const next = prev.filter(h => h.query !== q);
            try { localStorage.setItem('search_history_v2', JSON.stringify(next)); } catch {}
            return next;
        });
    }, []);

    const handleHistoryClearAll = useCallback(() => {
        setSearchHistory([]);
        try { localStorage.removeItem('search_history_v2'); } catch {}
    }, []);

    return (
        <div className="flex h-full bg-gray-900">
            {/* Sidebar — Search History */}
            {searchHistory.length > 0 && (
                <div className="w-48 flex-shrink-0 border-r border-gray-800/50 bg-gray-900/50">
                    <SearchHistorySidebar
                        history={searchHistory}
                        activeQuery={query}
                        onSelect={handleHistorySelect}
                        onDelete={handleHistoryDelete}
                        onClearAll={handleHistoryClearAll}
                    />
                </div>
            )}

            {/* Main content */}
            <div className="flex flex-col flex-1 min-w-0">
            {/* Search Header Area */}
            <div className={`flex flex-col items-center ${isEmptyState ? 'pt-[20vh]' : 'pt-6'} px-6 pb-4 shrink-0`}>
                {/* Title (only in empty state) */}
                {isEmptyState && (
                    <div className="mb-6 text-center">
                        <h2 className="text-2xl font-bold text-gray-300 mb-2">{t('search.title')}</h2>
                        <p className="text-sm text-gray-500">{t('search.subtitle')}</p>
                        {dbStats && dbStats.total_files > 0 && (
                            <div className="mt-3 flex items-center justify-center gap-3 text-xs text-gray-500">
                                <span className="bg-gray-800 px-3 py-1.5 rounded-full border border-gray-700">
                                    <span className="text-blue-400 font-bold">{dbStats.fully_archived ?? 0}</span>
                                    {' / '}
                                    <span className="text-gray-400">{dbStats.total_files}</span>
                                    {' '}{t('msg.images_archived', { count: '' }).trim()}
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

                {/* Unified Search Bar: Text + Images */}
                <div className="w-full max-w-2xl">
                    {/* Text Input Row (isolated — typing only re-renders SearchInput) */}
                    <SearchInput
                        onSearch={handleSearch}
                        onClear={clearSearch}
                        hasImages={queryImages.length > 0}
                        isSearching={isSearching}
                        showFilters={showFilters}
                        hasActiveFilters={hasActiveFilters}
                        onToggleFilters={toggleFilters}
                        onOpenSettings={openSettings}
                        inputRef={inputRef}
                        resetSignal={resetSignal}
                    />

                    {/* Image Input Area (always visible) */}
                    <ImageSearchInput
                        queryImages={queryImages}
                        onImagesChange={setQueryImages}
                    />

                    {/* AND/OR Toggle (visible when 2+ images) */}
                    {queryImages.length >= 2 && (
                        <div className="flex items-center gap-2 mt-2 ml-1">
                            <div className="flex bg-gray-800 rounded-lg p-0.5 border border-gray-700">
                                <button
                                    onClick={() => setImageSearchMode('and')}
                                    className={`px-3 py-1 rounded text-xs font-bold transition-all ${imageSearchMode === 'and'
                                        ? 'bg-blue-600 text-white shadow-md'
                                        : 'text-gray-400 hover:text-white hover:bg-gray-700'
                                        }`}
                                >
                                    {t('label.search_mode_and')}
                                </button>
                                <button
                                    onClick={() => setImageSearchMode('or')}
                                    className={`px-3 py-1 rounded text-xs font-bold transition-all ${imageSearchMode === 'or'
                                        ? 'bg-orange-600 text-white shadow-md'
                                        : 'text-gray-400 hover:text-white hover:bg-gray-700'
                                        }`}
                                >
                                    {t('label.search_mode_or')}
                                </button>
                            </div>
                            <span className="text-[10px] text-gray-500">
                                {imageSearchMode === 'and' ? t('label.search_mode_and_desc') : t('label.search_mode_or_desc')}
                            </span>
                        </div>
                    )}

                    {/* Filter Bar */}
                    {showFilters && (
                        <div className="flex items-center gap-3 mt-3 px-2 py-2.5 bg-gray-800/50 rounded-lg border border-gray-700/50 flex-wrap">
                            {/* Format Filter */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase">{t('filter.format')}</span>
                                <select
                                    value={activeFilters.format || ''}
                                    onChange={(e) => setActiveFilters({ ...activeFilters, format: e.target.value || undefined })}
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
                                    onChange={(e) => setActiveFilters({ ...activeFilters, user_category: e.target.value || undefined })}
                                    className="bg-gray-700 text-white text-xs px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                                >
                                    {CATEGORY_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>{t(opt.key)}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Image Type Filter (dynamic from domain YAML) */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase">{t('filter.type_label')}</span>
                                <select
                                    value={activeFilters.image_type || ''}
                                    onChange={(e) => setActiveFilters({ ...activeFilters, image_type: e.target.value || undefined })}
                                    className="bg-gray-700 text-white text-xs px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                                >
                                    <option value="">{t('filter.all')}</option>
                                    {(domainConfig?.image_types?.map(t => t.id) || FALLBACK_IMAGE_TYPES).map(type => (
                                        <option key={type} value={type}>{type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Art Style Filter (dynamic from domain YAML) */}
                            <div className="flex items-center gap-1.5">
                                <span className="text-[10px] text-gray-500 uppercase">{t('filter.style_label')}</span>
                                <select
                                    value={activeFilters.art_style || ''}
                                    onChange={(e) => setActiveFilters({ ...activeFilters, art_style: e.target.value || undefined })}
                                    className="bg-gray-700 text-white text-xs px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                                >
                                    <option value="">{t('filter.all')}</option>
                                    {(domainConfig?.common_hints?.art_style || FALLBACK_ART_STYLES).map(style => (
                                        <option key={style} value={style}>{style.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</option>
                                    ))}
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
                                            className={`text-sm transition-colors ${star <= (activeFilters.min_rating || 0) ? 'text-yellow-400' : 'text-gray-600 hover:text-gray-400'
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
                                    onClick={() => { setActiveFilters({}); setThreshold(0); }}
                                    className="text-[10px] text-red-400 hover:text-red-300 px-2 py-1 rounded bg-red-900/20 border border-red-800/30"
                                >
                                    {t('action.clear')}
                                </button>
                            )}
                        </div>
                    )}

                    {/* Rebuild banner moved to Admin tab */}

                    {error && (
                        <div className="mt-3 px-3 py-2 bg-red-900/30 border border-red-700 rounded-lg text-red-400 text-sm">
                            {error}
                        </div>
                    )}
                </div>
            </div>

            {/* Results Area */}
            <SearchResults
                results={results}
                isSearching={isSearching}
                searchStage={searchStage}
                hasResults={hasResults}
                onShowMeta={handleShowMeta}
                onContextMenu={handleCardContextMenu}
                onClear={clearSearch}
                noMoreResults={noMoreResults}
                onLoadMore={handleLoadMore}
                onNavigateToFolder={onNavigateToFolder}
                onFindSimilar={(fileId) => triggerStructureSearch(fileId, 'vector')}
                onMarkIrrelevant={handleMarkIrrelevant}
                activeFilters={activeFilters}
                onRemoveFilter={(key) => setActiveFilters(prev => ({ ...prev, [key]: undefined }))}
                searchScope={searchScope}
                onClearScope={() => { setSearchScope(null); handleSearch(query); }}
                refineStack={refineStack}
                refineInput={refineInput}
                onRefineInputChange={setRefineInput}
                onRefineCommit={handleRefineCommit}
                onRefineRemove={handleRefineRemove}
                totalCount={allResultsRef.current.length}
                confidence={confidence}
            />

            {/* Search history moved to sidebar (SearchHistorySidebar) */}

            {/* Metadata Modal */}
            {metadata && <MetadataModal metadata={metadata} onClose={() => setMetadata(null)} onNavigateToFolder={onNavigateToFolder} />}

            {/* Context Menu */}
            {contextMenu && (
                <div className="fixed inset-0 z-50" onClick={() => setContextMenu(null)} onContextMenu={(e) => { e.preventDefault(); setContextMenu(null); }}>
                    <div
                        className="absolute bg-gray-800 border border-gray-600 rounded-lg shadow-2xl py-1 min-w-[200px] animate-fadeIn"
                        style={{ top: Math.min(contextMenu.y, window.innerHeight - 150), left: Math.min(contextMenu.x, window.innerWidth - 200) }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="px-3 py-2 border-b border-gray-700 bg-gray-900/50">
                            <div className="text-xs font-bold text-white truncate max-w-[180px]">
                                {(contextMenu.result.resolved_path || contextMenu.result.path).split(/[/\\]/).pop()}
                            </div>
                        </div>
                        <button
                            onClick={() => {
                                window.electron?.fs?.showInFolder(contextMenu.result.resolved_path || contextMenu.result.path);
                                setContextMenu(null);
                            }}
                            className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white flex items-center gap-2"
                        >
                            <FolderOpen size={14} /> {t('action.show_in_folder')}
                        </button>
                        <div className="h-px bg-gray-700 my-1" />
                        <button
                            onClick={() => { triggerStructureSearch(contextMenu.result.id, 'vector'); setContextMenu(null); }}
                            className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white flex items-center gap-2"
                        >
                            <Search size={14} /> {t('action.find_similar_visual')}
                        </button>
                        <button
                            onClick={() => { triggerStructureSearch(contextMenu.result.id, 'structure'); setContextMenu(null); }}
                            className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white flex items-center gap-2"
                        >
                            <div className="rotate-90"><Search size={14} /></div> {t('action.find_similar_structure')}
                        </button>
                    </div>
                </div>
            )}

            </div>{/* end Main content */}
        </div>
    );
}

export default React.memo(SearchPanel);
