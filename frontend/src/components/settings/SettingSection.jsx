import React from 'react';

const SettingSection = ({ title, icon: Icon, children }) => {
    return (
        <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-5 mb-6">
            <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wide mb-4 flex items-center gap-2">
                {Icon && <Icon size={16} />}
                {title}
            </h3>
            <div className="space-y-0">
                {children}
            </div>
        </div>
    );
};

export default SettingSection;
