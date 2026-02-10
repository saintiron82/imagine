import React, { useState, useEffect, useCallback } from 'react';
import { ChevronRight, ChevronDown, Folder, FolderOpen, FolderPlus, Trash2, CheckSquare, Square } from 'lucide-react';
import { useLocale } from '../i18n';

const TreeNode = ({ path, name, onSelect, currentPath, level = 0, multiSelectMode = false, selectedPaths = new Set(), onFolderToggle, isRoot = false, onRemoveRoot }) => {
    const { t } = useLocale();
    const [isOpen, setIsOpen] = useState(isRoot); // Roots start open
    const [children, setChildren] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const isSelected = multiSelectMode ? selectedPaths.has(path) : currentPath === path;

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

    const handleClick = () => {
        if (multiSelectMode) {
            onFolderToggle?.(path);
        } else {
            onSelect(path);
        }
    };

    const highlightClass = multiSelectMode
        ? (isSelected ? 'bg-purple-900/50 border-l-4 border-purple-500' : '')
        : (isSelected ? 'bg-blue-900 border-l-4 border-blue-500' : '');

    return (
        <div className="select-none">
            <div
                className={`flex items-center py-1 px-2 cursor-pointer hover:bg-gray-700 transition-colors group ${highlightClass}`}
                style={{ paddingLeft: `${level * 16 + 8}px` }}
                onClick={handleClick}
            >
                <div onClick={handleToggle} className="p-1 mr-1 text-gray-400 hover:text-white">
                    {children.length > 0 || isOpen ? (
                        isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />
                    ) : (
                        <ChevronRight size={14} className="opacity-0" />
                    )}
                </div>

                {multiSelectMode && (
                    isSelected
                        ? <CheckSquare size={14} className="text-purple-400 mr-1.5 flex-shrink-0" />
                        : <Square size={14} className="text-gray-500 mr-1.5 flex-shrink-0" />
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
                            multiSelectMode={multiSelectMode}
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

const Sidebar = ({ currentPath, onFolderSelect, multiSelectMode = false, selectedPaths = new Set(), onMultiSelectToggle, onFolderToggle }) => {
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
            {/* Add Folder + Multi-Select Buttons */}
            <div className="flex gap-1 mb-2">
                <button
                    onClick={handleAddFolder}
                    className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded flex items-center justify-center space-x-2"
                >
                    <FolderPlus size={16} />
                    <span>{t('action.add_folder')}</span>
                </button>
                <button
                    onClick={() => onMultiSelectToggle?.(!multiSelectMode)}
                    className={`px-3 py-2 text-sm font-medium rounded transition-colors ${
                        multiSelectMode
                            ? 'bg-purple-600 hover:bg-purple-500 text-white'
                            : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                    }`}
                    title={multiSelectMode ? t('action.exit_multi_select') : t('action.multi_select')}
                >
                    {multiSelectMode ? <CheckSquare size={16} /> : <Square size={16} />}
                </button>
            </div>

            {/* Multi-select info */}
            {multiSelectMode && selectedPaths.size > 0 && (
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
                            multiSelectMode={multiSelectMode}
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
