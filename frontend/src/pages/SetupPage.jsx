import React, { useState } from 'react';
import { Shield, Cpu, ArrowRight, Wifi } from 'lucide-react';
import { useLocale } from '../i18n';

const SetupPage = ({ onComplete }) => {
    const { t } = useLocale();
    const [selectedMode, setSelectedMode] = useState(null); // 'server' | 'client'
    const [serverUrl, setServerUrl] = useState(
        localStorage.getItem('imagine-server-url') || 'http://'
    );
    const [error, setError] = useState('');

    const handleConfirm = async () => {
        if (!selectedMode) return;

        if (selectedMode === 'client') {
            // Validate server URL
            try {
                const url = new URL(serverUrl);
                if (!url.hostname) throw new Error('Invalid hostname');
            } catch {
                setError(t('setup.invalid_url'));
                return;
            }
        }

        // No config.yaml persistence — mode is session-only.
        // Each app launch shows this page fresh.
        onComplete(selectedMode, selectedMode === 'client' ? serverUrl : null);
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
                        onClick={() => { setSelectedMode('server'); setError(''); }}
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
                        onClick={() => { setSelectedMode('client'); setError(''); }}
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

                {/* Server URL input (only for client mode) */}
                {selectedMode === 'client' && (
                    <div className="mb-6">
                        <label className="block text-sm text-gray-400 mb-2">
                            <Wifi size={14} className="inline mr-1.5" />
                            {t('setup.server_url')}
                        </label>
                        <input
                            type="text"
                            value={serverUrl}
                            onChange={(e) => setServerUrl(e.target.value)}
                            placeholder="http://192.168.1.10:8000"
                            className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-emerald-500 focus:outline-none font-mono text-sm"
                        />
                    </div>
                )}

                {/* Error */}
                {error && (
                    <div className="mb-4 px-4 py-2 bg-red-900/30 border border-red-700 rounded-lg text-red-400 text-sm">
                        {error}
                    </div>
                )}

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
