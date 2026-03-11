import React from 'react';

const SettingRow = ({ label, description, badge, children }) => {
    return (
        <div className="flex items-center justify-between py-3 border-b border-gray-700/50 last:border-b-0">
            <div className="flex-1 min-w-0 mr-4">
                <div className="text-sm text-gray-200 font-medium flex items-center gap-2">
                    {label}
                    {badge && (
                        <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-amber-900/50 text-amber-400">
                            {badge}
                        </span>
                    )}
                </div>
                {description && (
                    <div className="text-xs text-gray-500 mt-0.5">{description}</div>
                )}
            </div>
            <div className="flex-shrink-0">
                {children}
            </div>
        </div>
    );
};

export default SettingRow;
