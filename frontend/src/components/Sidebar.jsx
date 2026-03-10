import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { ChevronRight, ChevronDown, Folder, FolderOpen, FolderPlus, Trash2, CheckSquare, Globe, Plus, Monitor } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { browseFolders } from '../api/admin';
import WebDAVFolderPicker from './WebDAVFolderPicker';
import WebDAVConnectDialog from './WebDAVConnectDialog';

/** Aggregate phase stats for entries whose storage_root starts with prefix */
function aggregateStats(phaseStats, folderPath) {
    if (!phaseStats || phaseStats.length === 0) return null;
    // NFC normalize to match DB (macOS returns NFD paths)
    const nfc = (s) => s.normalize('NFC');
    const fp = nfc(folderPath);
    const prefix = fp.endsWith('/') ? fp : fp + '/';
    const matched = phaseStats.filter(s =>
        nfc(s.storage_root) === fp || nfc(s.storage_root).startsWith(prefix)
    );
    if (matched.length === 0) return null;
    return matched.reduce((acc, f) => ({
        total: acc.total + f.total,
        mc: acc.mc + f.mc,
        vv: acc.vv + f.vv,
        mv: acc.mv + f.mv,
        missing_relative_path_count: acc.missing_relative_path_count + (f.missing_relative_path_count || 0),
        missing_structure_count: acc.missing_structure_count + (f.missing_structure_count || 0),
        rebuild_needed: acc.rebuild_needed || !!f.rebuild_needed,
        fts_version_mismatch: acc.fts_version_mismatch || !!f.fts_version_mismatch,
    }), {
        total: 0,
        mc: 0,
        vv: 0,
        mv: 0,
        missing_relative_path_count: 0,
        missing_structure_count: 0,
        rebuild_needed: false,
        fts_version_mismatch: false,
    });
}

const isWebDAVPath = (p) => p && p.startsWith('webdav://');

const TreeNode = ({ path, name, onSelect, currentPath, level = 0, selectedPaths = new Set(), onFolderToggle, isRoot = false, isWebDAV = false, onRemoveRoot, phaseStats }) => {
    const { t } = useLocale();
    const [isOpen, setIsOpen] = useState(isRoot); // Roots start open
    const [children, setChildren] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const isCtrlSelected = selectedPaths.has(path);
    const isCurrentPath = currentPath === path;
    const webdav = isWebDAV || isWebDAVPath(path);

    // Aggregate stats for this node's subtree
    const myStats = useMemo(() => aggregateStats(phaseStats, path), [phaseStats, path]);

    // Auto-load children for root nodes
    useEffect(() => {
        if (isRoot && isOpen && children.length === 0) {
            loadChildren();
        }
    }, [isRoot, isOpen]);

    const loadChildren = async () => {
        setIsLoading(true);
        try {
            if (webdav) {
                // WebDAV: browse remote folders via PROPFIND
                const result = await window.electron?.webdav?.browseFolders({ webdavPath: path });
                if (result?.success) {
                    setChildren((result.folders || []).map(f => ({
                        name: f.name,
                        path: f.path.startsWith('webdav://') ? f.path : path.replace(/\/$/, '') + '/' + f.name,
                        isDirectory: true,
                    })));
                }
            } else if (isElectron && window.electron?.fs) {
                const items = await window.electron.fs.listDir(path);
                const folders = items.filter(item => item.isDirectory);
                setChildren(folders);
            } else {
                // Web mode: browse server filesystem via API
                const data = await browseFolders(path);
                const folders = (data.dirs || []).map(d => ({
                    name: d.name,
                    path: d.path,
                    isDirectory: true,
                }));
                setChildren(folders);
            }
        } catch (err) {
            console.error("Failed to load folders", err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleToggle = async (e) => {
        e.stopPropagation();
        if (!isOpen && children.length === 0) {
            await loadChildren();
        }
        setIsOpen(!isOpen);
    };

    const handleClick = (e) => {
        if (e.ctrlKey || e.metaKey) {
            // Ctrl+click: toggle multi-select
            onFolderToggle?.(path);
        } else {
            // Normal click: single folder select
            onSelect(path);
        }
    };

    const highlightClass = isCtrlSelected
        ? 'bg-purple-900/50 border-l-4 border-purple-500'
        : (isCurrentPath ? 'bg-blue-900 border-l-4 border-blue-500' : '');

    return (
        <div className="select-none">
            <div
                className={`flex items-center py-1 px-2 cursor-pointer hover:bg-gray-700 transition-colors group ${highlightClass}`}
                style={{ paddingLeft: `${level * 16 + 8}px` }}
                onClick={handleClick}
            >
                <div onClick={handleToggle} className="p-1 mr-1 text-gray-400 hover:text-white">
                    {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </div>

                {isCtrlSelected && (
                    <CheckSquare size={14} className="text-purple-400 mr-1.5 flex-shrink-0" />
                )}

                {webdav && isRoot ? (
                    <Globe size={16} className="mr-2 text-blue-400" />
                ) : isOpen ? (
                    <FolderOpen size={16} className={`mr-2 ${isRoot ? 'text-blue-400' : 'text-yellow-500'}`} />
                ) : (
                    <Folder size={16} className={`mr-2 ${isRoot ? 'text-blue-400' : 'text-yellow-500'}`} />
                )}

                <span className={`text-sm truncate flex-1 ${isRoot ? 'text-white font-medium' : 'text-gray-200'}`}>{name}</span>

                {/* Phase completion indicator */}
                {myStats && myStats.total > 0 && (() => {
                    if (myStats.rebuild_needed) {
                        const parts = [];
                        if (myStats.fts_version_mismatch) parts.push('FTS');
                        if (myStats.missing_structure_count > 0) parts.push(`SV ${myStats.missing_structure_count}`);
                        if (myStats.missing_relative_path_count > 0) parts.push(`PATH ${myStats.missing_relative_path_count}`);
                        const detail = parts.length > 0 ? ` (${parts.join(', ')})` : '';
                        return (
                            <span className="flex items-center flex-shrink-0 mr-1" title={`${t('status.rebuild_needed_badge')}${detail}`}>
                                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                            </span>
                        );
                    }

                    const minPhase = Math.min(myStats.mc, myStats.vv, myStats.mv);
                    const maxPhase = Math.max(myStats.mc, myStats.vv, myStats.mv);
                    const ratio = minPhase / myStats.total;
                    // fully done (or within rounding: 98%+)
                    if (ratio >= 0.98) {
                        return (
                            <span className="flex items-center flex-shrink-0 mr-1" title={`${myStats.total}/${myStats.total} complete`}>
                                <span className="w-2 h-2 rounded-full bg-green-500" />
                            </span>
                        );
                    }
                    // partially processed — any phase has started
                    if (maxPhase > 0) {
                        const incomplete = myStats.total - minPhase;
                        return (
                            <span className="flex items-center flex-shrink-0 mr-1" title={`${incomplete}/${myStats.total} remaining`}>
                                <span className="w-2 h-2 rounded-full bg-orange-400 animate-pulse" />
                            </span>
                        );
                    }
                    // never processed — no indicator
                    return null;
                })()}

                {isRoot && onRemoveRoot && (
                    <button
                        onClick={(e) => { e.stopPropagation(); onRemoveRoot(path); }}
                        className="text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity p-0.5"
                        title={t('action.remove_folder')}
                    >
                        <Trash2 size={12} />
                    </button>
                )}
            </div>

            {isOpen && (
                <div>
                    {children.map((child) => (
                        <TreeNode
                            key={child.path}
                            path={child.path}
                            name={child.name}
                            onSelect={onSelect}
                            currentPath={currentPath}
                            level={level + 1}
                            isWebDAV={webdav}
                            selectedPaths={selectedPaths}
                            onFolderToggle={onFolderToggle}
                            phaseStats={phaseStats}
                        />
                    ))}
                    {children.length === 0 && !isLoading && (
                        <div className="text-xs text-gray-500 py-1" style={{ paddingLeft: `${(level + 1) * 16 + 28}px` }}>
                            {t('msg.no_subfolders')}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

const Sidebar = ({ currentPath, onFolderSelect, selectedPaths = new Set(), onFolderToggle, reloadSignal = 0 }) => {
    const { t } = useLocale();
    const [roots, setRoots] = useState([]);
    const [loading, setLoading] = useState(true);
    const [rootPhaseStats, setRootPhaseStats] = useState({});

    // WebDAV integration state
    const [webdavSources, setWebdavSources] = useState([]);
    const [showAddMenu, setShowAddMenu] = useState(false);
    const [showNASPicker, setShowNASPicker] = useState(null); // source object or null
    const [showNASConnect, setShowNASConnect] = useState(false);
    const addMenuRef = useRef(null);

    // Close dropdown on outside click
    useEffect(() => {
        if (!showAddMenu) return;
        const handleClickOutside = (e) => {
            if (addMenuRef.current && !addMenuRef.current.contains(e.target)) {
                setShowAddMenu(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [showAddMenu]);

    const loadWebDAVSources = useCallback(async () => {
        try {
            const result = await window.electron?.webdav?.getSources();
            if (result?.success) {
                setWebdavSources(result.sources || []);
            }
        } catch (e) {
            // WebDAV not available — ignore
        }
    }, []);

    const loadRoots = useCallback(async () => {
        setLoading(true);
        try {
            const result = await window.electron?.pipeline?.getRegisteredFolders();
            if (result?.success) {
                const validRoots = (result.folders || []).filter(f => f.exists).map(f => ({
                    path: f.path,
                    name: isWebDAVPath(f.path)
                        ? _webdavDisplayName(f.path, webdavSources)
                        : (f.path.split(/[/\\]/).pop() || f.path),
                    isWebDAV: !!f.isWebDAV,
                }));
                setRoots(validRoots);

                // Load phase stats for local roots only (WebDAV folders don't have local phase stats)
                const statsMap = {};
                await Promise.all(validRoots.filter(r => !r.isWebDAV).map(async (root) => {
                    try {
                        const statsResult = await window.electron?.pipeline?.getFolderPhaseStats(root.path);
                        if (statsResult?.success) {
                            statsMap[root.path] = statsResult.folders || [];
                        }
                    } catch (e) {
                        console.error('Failed to load phase stats for', root.path, e);
                    }
                }));
                setRootPhaseStats(statsMap);
            }
        } catch (e) {
            console.error('Failed to load registered folders:', e);
        } finally {
            setLoading(false);
        }
    }, [webdavSources]);

    useEffect(() => {
        loadWebDAVSources();
    }, [loadWebDAVSources]);

    useEffect(() => {
        loadRoots();
    }, [loadRoots, reloadSignal]);

    const handleAddFolder = async () => {
        // If WebDAV sources exist, show dropdown menu
        if (webdavSources.length > 0) {
            setShowAddMenu(prev => !prev);
            return;
        }
        // No NAS → directly open Finder dialog
        await addLocalFolder();
    };

    const addLocalFolder = async () => {
        setShowAddMenu(false);
        try {
            const result = await window.electron?.pipeline?.addRegisteredFolder();
            if (result?.success && result.folders) {
                await loadRoots();
                if (result.added?.length > 0) {
                    onFolderSelect(result.added[0]);
                }
            }
        } catch (e) {
            console.error('Failed to add folder:', e);
        }
    };

    const handleNASFolderSelect = async (webdavPath) => {
        setShowNASPicker(null);
        setShowAddMenu(false);
        try {
            const result = await window.electron?.pipeline?.addRegisteredWebDAVFolder(webdavPath);
            if (result?.success) {
                await loadRoots();
                onFolderSelect(webdavPath);
            }
        } catch (e) {
            console.error('Failed to add WebDAV folder:', e);
        }
    };

    const handleNASConnected = () => {
        setShowNASConnect(false);
        loadWebDAVSources();
    };

    const handleRemoveRoot = async (folderPath) => {
        try {
            const result = await window.electron?.pipeline?.removeRegisteredFolder(folderPath);
            if (result?.success) {
                await loadRoots();
            }
        } catch (e) {
            console.error('Failed to remove folder:', e);
        }
    };

    return (
        <div>
            {/* Add Folder Button */}
            <div className="mb-2 relative" ref={addMenuRef}>
                <button
                    onClick={handleAddFolder}
                    className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded flex items-center justify-center space-x-2"
                >
                    <FolderPlus size={16} />
                    <span>{t('action.add_folder')}</span>
                </button>

                {/* Dropdown menu (visible when WebDAV sources exist) */}
                {showAddMenu && (
                    <div className="absolute top-full left-0 right-0 mt-1 bg-gray-800 border border-gray-600 rounded-lg shadow-xl z-30 overflow-hidden">
                        {/* Local folder option */}
                        <button
                            onClick={addLocalFolder}
                            className="w-full px-3 py-2 text-xs text-gray-200 hover:bg-gray-700 flex items-center gap-2 text-left"
                        >
                            <Monitor size={14} className="text-yellow-500" />
                            {t('sidebar.add_local_folder')}
                        </button>

                        {/* Divider + NAS sources */}
                        <div className="border-t border-gray-700 mx-2" />
                        <div className="px-3 py-1">
                            <span className="text-[10px] text-gray-500 uppercase tracking-wider">NAS</span>
                        </div>

                        {webdavSources.map(source => (
                            <button
                                key={source.id}
                                onClick={() => { setShowAddMenu(false); setShowNASPicker(source); }}
                                className="w-full px-3 py-2 text-xs text-gray-200 hover:bg-gray-700 flex items-center gap-2 text-left"
                            >
                                <Globe size={14} className="text-blue-400" />
                                <div className="flex-1 min-w-0">
                                    <div className="truncate">{source.name || source.url}</div>
                                    <div className="text-[10px] text-gray-500 truncate">{source.url}</div>
                                </div>
                            </button>
                        ))}

                        {/* Add new NAS */}
                        <div className="border-t border-gray-700 mx-2" />
                        <button
                            onClick={() => { setShowAddMenu(false); setShowNASConnect(true); }}
                            className="w-full px-3 py-2 text-xs text-gray-400 hover:bg-gray-700 hover:text-gray-200 flex items-center gap-2 text-left"
                        >
                            <Plus size={14} />
                            {t('sidebar.add_nas_connection')}
                        </button>
                    </div>
                )}
            </div>

            {/* Ctrl+click multi-select info */}
            {selectedPaths.size > 0 && (
                <div className="text-xs text-purple-400 mb-2 px-2">
                    {t('status.folders_selected', { count: selectedPaths.size })}
                </div>
            )}

            {/* Registered folder roots */}
            {loading ? (
                <div className="text-xs text-gray-500 p-4 text-center">{t('status.loading')}</div>
            ) : roots.length === 0 ? (
                <div className="text-xs text-gray-500 p-4 text-center">{t('msg.no_registered_folders')}</div>
            ) : (
                <div>
                    {roots.map((root) => (
                        <TreeNode
                            key={root.path}
                            path={root.path}
                            name={root.name}
                            onSelect={onFolderSelect}
                            currentPath={currentPath}
                            isRoot={true}
                            isWebDAV={root.isWebDAV}
                            onRemoveRoot={handleRemoveRoot}
                            selectedPaths={selectedPaths}
                            onFolderToggle={onFolderToggle}
                            phaseStats={rootPhaseStats[root.path] || null}
                        />
                    ))}
                </div>
            )}

            {/* NAS Folder Picker Modal */}
            {showNASPicker && (
                <WebDAVFolderPicker
                    source={showNASPicker}
                    onSelect={handleNASFolderSelect}
                    onClose={() => setShowNASPicker(null)}
                />
            )}

            {/* NAS Connect Dialog */}
            {showNASConnect && (
                <WebDAVConnectDialog
                    onClose={() => setShowNASConnect(false)}
                    onAdded={handleNASConnected}
                />
            )}
        </div>
    );
};

/** Build display name for a webdav:// path */
function _webdavDisplayName(webdavPath, sources) {
    const withoutScheme = webdavPath.replace('webdav://', '');
    const slashIdx = withoutScheme.indexOf('/');
    const sourceId = slashIdx === -1 ? withoutScheme : withoutScheme.substring(0, slashIdx);
    const subPath = slashIdx === -1 ? '/' : withoutScheme.substring(slashIdx);

    const source = sources.find(s => s.id === sourceId);
    const sourceName = source?.name || source?.url || sourceId;
    const folderName = subPath === '/' ? '' : subPath.split('/').filter(Boolean).pop();

    return folderName ? `${folderName} (${sourceName})` : sourceName;
}

export default Sidebar;
