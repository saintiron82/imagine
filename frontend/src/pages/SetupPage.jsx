import React, { useState } from 'react';
import { Shield, Cpu, ArrowRight } from 'lucide-react';
import { useLocale } from '../i18n';

const SetupPage = ({ onComplete }) => {
    const { t } = useLocale();
    const [selectedMode, setSelectedMode] = useState(null);

    const handleConfirm = () => {
        if (!selectedMode) return;
        onComplete(selectedMode);
    };

    return (
        <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
            <div className="max-w-2xl w-full px-8">
                {/* Title */}
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold mb-2">Imagine</h1>
                    <p className="text-gray-400 text-sm">{t('setup.subtitle')}</p>
                </div>

                {/* Mode Cards */}
                <div className="grid grid-cols-2 gap-6 mb-8">
                    {/* Server Mode Card */}
                    <button
                        onClick={() => setSelectedMode('server')}
                        className={`p-6 rounded-xl border-2 text-left transition-all ${
                            selectedMode === 'server'
                                ? 'border-blue-500 bg-blue-900/20 shadow-lg shadow-blue-900/30'
                                : 'border-gray-700 bg-gray-800/50 hover:border-gray-500'
                        }`}
                    >
                        <div className="flex items-center gap-3 mb-4">
                            <div className={`p-2.5 rounded-lg ${
                                selectedMode === 'server' ? 'bg-blue-600' : 'bg-gray-700'
                            }`}>
                                <Shield size={24} />
                            </div>
                            <h2 className="text-lg font-bold">{t('setup.server_title')}</h2>
                        </div>
                        <p className="text-sm text-gray-400 leading-relaxed">
                            {t('setup.server_desc')}
                        </p>
                        <ul className="mt-4 space-y-1.5 text-xs text-gray-500">
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-blue-400" />
                                {t('setup.server_feature1')}
                            </li>
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-blue-400" />
                                {t('setup.server_feature2')}
                            </li>
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-blue-400" />
                                {t('setup.server_feature3')}
                            </li>
                        </ul>
                    </button>

                    {/* Client Mode Card */}
                    <button
                        onClick={() => setSelectedMode('client')}
                        className={`p-6 rounded-xl border-2 text-left transition-all ${
                            selectedMode === 'client'
                                ? 'border-emerald-500 bg-emerald-900/20 shadow-lg shadow-emerald-900/30'
                                : 'border-gray-700 bg-gray-800/50 hover:border-gray-500'
                        }`}
                    >
                        <div className="flex items-center gap-3 mb-4">
                            <div className={`p-2.5 rounded-lg ${
                                selectedMode === 'client' ? 'bg-emerald-600' : 'bg-gray-700'
                            }`}>
                                <Cpu size={24} />
                            </div>
                            <h2 className="text-lg font-bold">{t('setup.client_title')}</h2>
                        </div>
                        <p className="text-sm text-gray-400 leading-relaxed">
                            {t('setup.client_desc')}
                        </p>
                        <ul className="mt-4 space-y-1.5 text-xs text-gray-500">
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                                {t('setup.client_feature1')}
                            </li>
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                                {t('setup.client_feature2')}
                            </li>
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                                {t('setup.client_feature3')}
                            </li>
                        </ul>
                    </button>
                </div>

                {/* Confirm Button */}
                <div className="flex justify-between items-center">
                    <p className="text-xs text-gray-600">{t('setup.changeable_later')}</p>
                    <button
                        onClick={handleConfirm}
                        disabled={!selectedMode}
                        className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                            selectedMode
                                ? 'bg-blue-600 hover:bg-blue-500 text-white'
                                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        }`}
                    >
                        {t('setup.start')}
                        <ArrowRight size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SetupPage;
