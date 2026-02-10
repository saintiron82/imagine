import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Folder, FolderOpen, CheckSquare, Square } from 'lucide-react';
import { useLocale } from '../i18n';

const TreeNode = ({ path, name, onSelect, currentPath, level = 0, multiSelectMode = false, selectedPaths = new Set(), onFolderToggle }) => {
    const { t } = useLocale();
    const [isOpen, setIsOpen] = useState(false);
    const [children, setChildren] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const isSelected = multiSelectMode ? selectedPaths.has(path) : currentPath === path;

    // Load children when opened
    const handleToggle = async (e) => {
        e.stopPropagation();

        if (!isOpen) {
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
                className={`flex items-center py-1 px-2 cursor-pointer hover:bg-gray-700 transition-colors ${highlightClass}`}
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
                    <FolderOpen size={16} className="text-yellow-500 mr-2" />
                ) : (
                    <Folder size={16} className="text-yellow-500 mr-2" />
                )}

                <span className="text-sm truncate text-gray-200">{name}</span>
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
    const [root, setRoot] = useState(null);

    useEffect(() => {
        if (window.electron && window.electron.fs) {
            const home = window.electron.fs.getHomeDir();
            setRoot({
                path: home,
                name: t('msg.home'),
            });
        }
    }, [t]);

    const handleBrowse = async () => {
        if (window.electron && window.electron.pipeline) {
            const folderPath = await window.electron.pipeline.openFolderDialog();
            if (folderPath) {
                onFolderSelect(folderPath);
            }
        }
    };

    if (!root) return <div className="p-4 text-gray-500">{t('status.loading')}</div>;

    return (
        <div>
            {/* Browse + Multi-Select Buttons */}
            <div className="flex gap-1 mb-2">
                <button
                    onClick={handleBrowse}
                    className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded flex items-center justify-center space-x-2"
                >
                    <FolderOpen size={16} />
                    <span>{t('action.browse_folder')}</span>
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

            <div className="text-xs text-gray-500 mb-2 px-2">{t('msg.navigate_below')}</div>

            <TreeNode
                path={root.path}
                name={root.name}
                onSelect={onFolderSelect}
                currentPath={currentPath}
                multiSelectMode={multiSelectMode}
                selectedPaths={selectedPaths}
                onFolderToggle={onFolderToggle}
            />
        </div>
    );
};

export default Sidebar;
