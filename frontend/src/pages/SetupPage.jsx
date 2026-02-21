import React, { useState, useEffect } from 'react';
import { Shield, Cpu, ArrowRight, Wifi, Radio, Clock, X } from 'lucide-react';
import { useLocale } from '../i18n';
import { useMdnsDiscovery } from '../hooks/useMdnsDiscovery';
import {
  getServerHistory,
  removeServerFromHistory,
  formatRelativeTime,
} from '../utils/serverHistory';

const SetupPage = ({ onComplete }) => {
    const { t } = useLocale();
    const [selectedMode, setSelectedMode] = useState(null);
    const [serverUrl, setServerUrl] = useState(
        localStorage.getItem('imagine-server-url') || 'http://'
    );
    const [error, setError] = useState('');

    // mDNS discovery (active only when client mode selected)
    const { servers: mdnsServers, browsing } = useMdnsDiscovery(selectedMode === 'client');

    // Recent server history
    const [history, setHistory] = useState([]);
    useEffect(() => {
        if (selectedMode === 'client') {
            setHistory(getServerHistory());
        }
    }, [selectedMode]);

    const handleSelectMdns = (server) => {
        const addr = server.addresses?.[0] || server.host;
        setServerUrl(`http://${addr}:${server.port}`);
        setError('');
    };

    const handleSelectHistory = (url) => {
        setServerUrl(url);
        setError('');
    };

    const handleRemoveHistory = (e, url) => {
        e.stopPropagation();
        removeServerFromHistory(url);
        setHistory(getServerHistory());
    };

    const handleConfirm = async () => {
        if (!selectedMode) return;

        if (selectedMode === 'client') {
            try {
                const url = new URL(serverUrl);
                if (!url.hostname) throw new Error('Invalid hostname');
            } catch {
                setError(t('setup.invalid_url'));
                return;
            }
        }

        onComplete(selectedMode, selectedMode === 'client' ? serverUrl : null);
    };

    // Filter history to exclude mDNS-discovered URLs
    const mdnsUrls = new Set(
        mdnsServers.map((s) => `http://${s.addresses?.[0]}:${s.port}`)
    );
    const filteredHistory = history.filter((h) => !mdnsUrls.has(h.url));

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

                {/* Server URL section (only for client mode) */}
                {selectedMode === 'client' && (
                    <div className="mb-6 space-y-3">
                        {/* mDNS Discovered Servers */}
                        {(mdnsServers.length > 0 || browsing) && (
                            <div>
                                <div className="flex items-center gap-1.5 mb-2">
                                    <Radio size={12} className={browsing ? 'text-green-400 animate-pulse' : 'text-gray-500'} />
                                    <span className="text-xs text-gray-400">
                                        {t('auth.discovered_servers')}
                                    </span>
                                </div>
                                {mdnsServers.length > 0 ? (
                                    <div className="space-y-1">
                                        {mdnsServers.map((s) => {
                                            const addr = s.addresses?.[0] || s.host;
                                            const sUrl = `http://${addr}:${s.port}`;
                                            return (
                                                <button
                                                    key={s.name}
                                                    onClick={() => handleSelectMdns(s)}
                                                    className={`w-full flex items-center justify-between px-4 py-2.5 rounded-lg text-left transition-colors ${
                                                        serverUrl === sUrl
                                                            ? 'bg-emerald-900/30 border border-emerald-600'
                                                            : 'bg-gray-800 border border-gray-700 hover:border-gray-500'
                                                    }`}
                                                >
                                                    <div className="flex items-center gap-2">
                                                        <span className="w-2 h-2 rounded-full bg-green-400" />
                                                        <span className="text-white font-medium text-sm">
                                                            {s.serverName || s.name}
                                                        </span>
                                                    </div>
                                                    <span className="text-xs text-gray-500 font-mono">
                                                        {addr}:{s.port}
                                                    </span>
                                                </button>
                                            );
                                        })}
                                    </div>
                                ) : (
                                    <p className="text-xs text-gray-600 italic pl-5">{t('auth.discovering')}</p>
                                )}
                            </div>
                        )}

                        {/* Recent Servers */}
                        {filteredHistory.length > 0 && (
                            <div>
                                <div className="flex items-center gap-1.5 mb-2">
                                    <Clock size={12} className="text-gray-500" />
                                    <span className="text-xs text-gray-400">
                                        {t('auth.recent_servers')}
                                    </span>
                                </div>
                                <div className="space-y-1">
                                    {filteredHistory.slice(0, 3).map((h) => (
                                        <button
                                            key={h.url}
                                            onClick={() => handleSelectHistory(h.url)}
                                            className={`w-full flex items-center justify-between px-4 py-2.5 rounded-lg text-left transition-colors group ${
                                                serverUrl === h.url
                                                    ? 'bg-emerald-900/30 border border-emerald-600'
                                                    : 'bg-gray-800 border border-gray-700 hover:border-gray-500'
                                            }`}
                                        >
                                            <div className="min-w-0 flex-1">
                                                <span className="text-white text-sm truncate block">
                                                    {h.name || h.url.replace(/^https?:\/\//, '')}
                                                </span>
                                                <span className="text-[10px] text-gray-500">
                                                    {h.lastUsername && `${h.lastUsername} · `}
                                                    {formatRelativeTime(h.lastConnected, t)}
                                                </span>
                                            </div>
                                            <button
                                                onClick={(e) => handleRemoveHistory(e, h.url)}
                                                className="p-1 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                                            >
                                                <X size={12} />
                                            </button>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Manual URL Input */}
                        <div>
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
