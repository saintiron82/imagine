import React, { useState, useEffect, useCallback } from 'react';
import { ChevronRight, ChevronDown, Folder, FolderOpen, FolderPlus, Trash2, CheckSquare } from 'lucide-react';
import { useLocale } from '../i18n';

/** Mini phase progress bar for folder stats */
function FolderPhaseBar({ label, count, total, color }) {
    const pct = total > 0 ? Math.min(100, Math.round((count / total) * 100)) : 0;
    const isDone = count >= total && total > 0;
    return (
        <div className="flex items-center space-x-0.5">
            <span className={`text-[8px] font-bold w-5 ${isDone ? 'text-green-400' : 'text-gray-500'}`}>{label}</span>
            <div className="w-10 bg-gray-700 rounded-full h-1 overflow-hidden">
                <div
                    className={`h-full rounded-full ${isDone ? 'bg-green-500' : color}`}
                    style={{ width: `${pct}%` }}
                />
            </div>
            <span className={`text-[8px] font-mono w-6 text-right ${isDone ? 'text-green-400' : 'text-gray-500'}`}>{count}</span>
        </div>
    );
}

const TreeNode = ({ path, name, onSelect, currentPath, level = 0, selectedPaths = new Set(), onFolderToggle, isRoot = false, onRemoveRoot }) => {
    const { t } = useLocale();
    const [isOpen, setIsOpen] = useState(isRoot); // Roots start open
    const [children, setChildren] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [phaseStats, setPhaseStats] = useState(null);
    const isCtrlSelected = selectedPaths.has(path);
    const isCurrentPath = currentPath === path;

    // Load phase stats for root nodes when opened
    useEffect(() => {
        if (isRoot && isOpen) {
            const loadStats = async () => {
                try {
                    const result = await window.electron?.pipeline?.getFolderPhaseStats(path);
                    if (result?.success) {
                        setPhaseStats(result.folders || []);
                    }
                } catch (e) {
                    console.error('Failed to load folder phase stats:', e);
                }
            };
            loadStats();
        }
    }, [isRoot, isOpen, path]);

    // Auto-load children for root nodes
    useEffect(() => {
        if (isRoot && isOpen && children.length === 0) {
            loadChildren();
        }
    }, [isRoot, isOpen]);

    const loadChildren = async () => {
        setIsLoading(true);
        try {
            if (window.electron && window.electron.fs) {
                const items = await window.electron.fs.listDir(path);
                const folders = items.filter(item => item.isDirectory);
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

                {isOpen ? (
                    <FolderOpen size={16} className={`mr-2 ${isRoot ? 'text-blue-400' : 'text-yellow-500'}`} />
                ) : (
                    <Folder size={16} className={`mr-2 ${isRoot ? 'text-blue-400' : 'text-yellow-500'}`} />
                )}

                <span className={`text-sm truncate flex-1 ${isRoot ? 'text-white font-medium' : 'text-gray-200'}`}>{name}</span>

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

            {/* Phase stats progress bars for root folders */}
            {isRoot && isOpen && phaseStats && phaseStats.length > 0 && (() => {
                const totals = phaseStats.reduce((acc, f) => ({
                    total: acc.total + f.total,
                    mc: acc.mc + f.mc,
                    vv: acc.vv + f.vv,
                    mv: acc.mv + f.mv,
                }), { total: 0, mc: 0, vv: 0, mv: 0 });
                return (
                    <div className="flex items-center gap-1.5 px-3 py-1 bg-gray-800/50 border-b border-gray-700/50"
                         style={{ paddingLeft: `${level * 16 + 32}px` }}>
                        <span className="text-[9px] text-gray-500 font-mono mr-0.5">{totals.total}</span>
                        <FolderPhaseBar label="MC" count={totals.mc} total={totals.total} color="bg-blue-400" />
                        <FolderPhaseBar label="VV" count={totals.vv} total={totals.total} color="bg-purple-400" />
                        <FolderPhaseBar label="MV" count={totals.mv} total={totals.total} color="bg-green-400" />
                    </div>
                );
            })()}

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
                            selectedPaths={selectedPaths}
                            onFolderToggle={onFolderToggle}
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

const Sidebar = ({ currentPath, onFolderSelect, selectedPaths = new Set(), onFolderToggle }) => {
    const { t } = useLocale();
    const [roots, setRoots] = useState([]);
    const [loading, setLoading] = useState(true);

    const loadRoots = useCallback(async () => {
        setLoading(true);
        try {
            const result = await window.electron?.pipeline?.getRegisteredFolders();
            if (result?.success) {
                setRoots((result.folders || []).filter(f => f.exists).map(f => ({
                    path: f.path,
                    name: f.path.split(/[/\\]/).pop() || f.path,
                })));
            }
        } catch (e) {
            console.error('Failed to load registered folders:', e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadRoots();
    }, [loadRoots]);

    const handleAddFolder = async () => {
        try {
            const result = await window.electron?.pipeline?.addRegisteredFolder();
            if (result?.success && result.folders) {
                setRoots(result.folders.filter(f => f.exists).map(f => ({
                    path: f.path,
                    name: f.path.split(/[/\\]/).pop() || f.path,
                })));
                // Auto-select the first newly added folder
                if (result.added?.length > 0) {
                    onFolderSelect(result.added[0]);
                }
            }
        } catch (e) {
            console.error('Failed to add folder:', e);
        }
    };

    const handleRemoveRoot = async (folderPath) => {
        try {
            const result = await window.electron?.pipeline?.removeRegisteredFolder(folderPath);
            if (result?.success) {
                setRoots((result.folders || []).filter(f => f.exists).map(f => ({
                    path: f.path,
                    name: f.path.split(/[/\\]/).pop() || f.path,
                })));
            }
        } catch (e) {
            console.error('Failed to remove folder:', e);
        }
    };

    return (
        <div>
            {/* Add Folder Button */}
            <div className="mb-2">
                <button
                    onClick={handleAddFolder}
                    className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded flex items-center justify-center space-x-2"
                >
                    <FolderPlus size={16} />
                    <span>{t('action.add_folder')}</span>
                </button>
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
                            onRemoveRoot={handleRemoveRoot}
                            selectedPaths={selectedPaths}
                            onFolderToggle={onFolderToggle}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

export default Sidebar;
