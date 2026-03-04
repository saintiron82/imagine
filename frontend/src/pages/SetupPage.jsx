import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Shield, Cpu, ArrowRight, ArrowLeft, Download, CheckCircle, AlertCircle, Loader2, SkipForward, Zap, Star, Rocket } from 'lucide-react';
import { useLocale } from '../i18n';

const TIERS = [
    { id: 'standard', icon: Zap,    color: 'emerald' },
    { id: 'pro',      icon: Star,   color: 'blue' },
    { id: 'ultra',    icon: Rocket, color: 'purple' },
];

const SetupPage = ({ onComplete }) => {
    const { t } = useLocale();
    const [selectedMode, setSelectedMode] = useState(null);
    const [selectedTier, setSelectedTier] = useState(null);

    // Environment check state
    const [phase, setPhase] = useState('select'); // 'select' | 'tier' | 'checking' | 'install' | 'installing' | 'done'
    const [envStatus, setEnvStatus] = useState(null);
    const [installLogs, setInstallLogs] = useState([]);
    const [installDone, setInstallDone] = useState(false);
    const [installSuccess, setInstallSuccess] = useState(false);
    const logEndRef = useRef(null);

    // Auto-scroll install logs
    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [installLogs]);

    // Listen for install-log events
    useEffect(() => {
        const pipeline = window.electron?.pipeline;
        if (!pipeline) return;

        const handler = (data) => {
            setInstallLogs(prev => [...prev, data]);
            if (data.done) {
                setInstallDone(true);
                setInstallSuccess(data.type === 'success');
                setPhase('done');
            }
        };

        pipeline.onInstallLog?.(handler);
        return () => pipeline.offInstallLog?.();
    }, []);

    // Mode selection → tier selection
    const handleModeConfirm = useCallback(() => {
        if (!selectedMode) return;
        if (!window.electron?.pipeline?.checkEnv) {
            onComplete(selectedMode);
            return;
        }
        setPhase('tier');
    }, [selectedMode, onComplete]);

    // Tier selection → save + env check
    const handleTierConfirm = useCallback(async () => {
        if (!selectedTier) return;

        // Save tier to user-settings.yaml
        try {
            await window.electron.pipeline.updateConfig('ai_mode.override', selectedTier);
            await window.electron.pipeline.updateConfig('ai_mode.auto_detect', false);
        } catch (e) {
            console.error('Failed to save tier:', e);
        }

        // Check environment
        setPhase('checking');
        try {
            const status = await window.electron.pipeline.checkEnv();
            setEnvStatus(status);
            const modelsOk = status.visual_model_cached && status.dependencies_ok;
            if (modelsOk) {
                onComplete(selectedMode);
            } else {
                setPhase('install');
            }
        } catch (e) {
            console.error('check-env failed:', e);
            onComplete(selectedMode);
        }
    }, [selectedTier, selectedMode, onComplete]);

    const handleInstall = useCallback(() => {
        setPhase('installing');
        setInstallLogs([]);
        window.electron?.pipeline?.installEnv();
    }, []);

    const handleSkip = useCallback(() => {
        onComplete(selectedMode);
    }, [selectedMode, onComplete]);

    const handleFinish = useCallback(() => {
        onComplete(selectedMode);
    }, [selectedMode, onComplete]);

    // Phase: Environment check in progress
    if (phase === 'checking') {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="text-center">
                    <Loader2 size={40} className="animate-spin text-blue-400 mx-auto mb-4" />
                    <p className="text-gray-300">{t('setup.checking_env')}</p>
                </div>
            </div>
        );
    }

    // Phase: Models missing — offer install
    if (phase === 'install') {
        const missing = [];
        if (!envStatus?.visual_model_cached) missing.push('SigLIP2 (VV)');
        if (!envStatus?.dependencies_ok) missing.push(t('setup.dependencies'));

        return (
            <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="max-w-lg w-full px-8">
                    <div className="text-center mb-8">
                        <AlertCircle size={48} className="text-yellow-400 mx-auto mb-4" />
                        <h2 className="text-xl font-bold mb-2">{t('setup.models_required')}</h2>
                        <p className="text-sm text-gray-400">{t('setup.models_required_desc')}</p>
                    </div>

                    {missing.length > 0 && (
                        <div className="bg-gray-800 rounded-lg p-4 mb-6 border border-gray-700">
                            <p className="text-xs text-gray-500 mb-2">{t('setup.missing_items')}</p>
                            {missing.map((item, i) => (
                                <div key={i} className="flex items-center gap-2 text-sm text-yellow-300">
                                    <AlertCircle size={14} />
                                    {item}
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="flex gap-3">
                        <button onClick={handleInstall}
                            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors">
                            <Download size={18} />
                            {t('setup.download_models')}
                        </button>
                        <button onClick={handleSkip}
                            className="flex items-center gap-2 px-4 py-3 rounded-lg font-medium bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors">
                            <SkipForward size={16} />
                            {t('setup.skip')}
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    // Phase: Installing models
    if (phase === 'installing' || phase === 'done') {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="max-w-2xl w-full px-8">
                    <div className="text-center mb-6">
                        {phase === 'installing' ? (
                            <>
                                <Loader2 size={40} className="animate-spin text-blue-400 mx-auto mb-4" />
                                <h2 className="text-xl font-bold mb-1">{t('setup.downloading')}</h2>
                                <p className="text-xs text-gray-500">{t('setup.downloading_desc')}</p>
                            </>
                        ) : installSuccess ? (
                            <>
                                <CheckCircle size={40} className="text-green-400 mx-auto mb-4" />
                                <h2 className="text-xl font-bold mb-1">{t('setup.install_complete')}</h2>
                            </>
                        ) : (
                            <>
                                <AlertCircle size={40} className="text-red-400 mx-auto mb-4" />
                                <h2 className="text-xl font-bold mb-1">{t('setup.install_failed')}</h2>
                                <p className="text-xs text-gray-500">{t('setup.install_failed_desc')}</p>
                            </>
                        )}
                    </div>

                    {/* Install logs */}
                    <div className="bg-gray-800 rounded-lg border border-gray-700 mb-6">
                        <div className="h-64 overflow-y-auto p-3 font-mono text-xs space-y-0.5">
                            {installLogs.map((log, i) => (
                                <div key={i} className={
                                    log.type === 'error' ? 'text-red-400' :
                                    log.type === 'success' ? 'text-green-400' :
                                    log.type === 'warning' ? 'text-yellow-400' :
                                    'text-gray-400'
                                }>
                                    {log.message}
                                </div>
                            ))}
                            <div ref={logEndRef} />
                        </div>
                    </div>

                    {phase === 'done' && (
                        <div className="text-center">
                            <button onClick={handleFinish}
                                className="flex items-center gap-2 mx-auto px-6 py-3 rounded-lg font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors">
                                {t('setup.start')}
                                <ArrowRight size={16} />
                            </button>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    // Phase: Tier selection
    if (phase === 'tier') {
        const colorMap = {
            emerald: { active: 'border-emerald-500 bg-emerald-900/20 shadow-lg shadow-emerald-900/30', icon: 'bg-emerald-600', dot: 'bg-emerald-400' },
            blue:    { active: 'border-blue-500 bg-blue-900/20 shadow-lg shadow-blue-900/30', icon: 'bg-blue-600', dot: 'bg-blue-400' },
            purple:  { active: 'border-purple-500 bg-purple-900/20 shadow-lg shadow-purple-900/30', icon: 'bg-purple-600', dot: 'bg-purple-400' },
        };

        return (
            <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="max-w-3xl w-full px-8">
                    <div className="text-center mb-10">
                        <h1 className="text-2xl font-bold mb-2">{t('setup.select_tier')}</h1>
                        <p className="text-gray-400 text-sm">{t('setup.select_tier_desc')}</p>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mb-8">
                        {TIERS.map(({ id, icon: Icon, color }) => {
                            const isSelected = selectedTier === id;
                            const cm = colorMap[color];
                            return (
                                <button key={id}
                                    onClick={() => setSelectedTier(id)}
                                    className={`p-5 rounded-xl border-2 text-left transition-all ${
                                        isSelected ? cm.active : 'border-gray-700 bg-gray-800/50 hover:border-gray-500'
                                    }`}
                                >
                                    <div className="flex items-center gap-3 mb-3">
                                        <div className={`p-2 rounded-lg ${isSelected ? cm.icon : 'bg-gray-700'}`}>
                                            <Icon size={20} />
                                        </div>
                                        <h2 className="text-base font-bold">{t(`setup.tier_${id}_title`)}</h2>
                                    </div>
                                    <p className="text-xs text-gray-400 leading-relaxed mb-3">
                                        {t(`setup.tier_${id}_desc`)}
                                    </p>
                                    <div className="space-y-1.5 text-xs text-gray-500">
                                        <div className="flex items-center gap-1.5">
                                            <span className={`w-1 h-1 rounded-full ${cm.dot}`} />
                                            {t(`setup.tier_${id}_vram`)}
                                        </div>
                                        <div className="flex items-center gap-1.5">
                                            <span className={`w-1 h-1 rounded-full ${cm.dot}`} />
                                            VLM: {t(`setup.tier_${id}_vlm`)}
                                        </div>
                                    </div>
                                </button>
                            );
                        })}
                    </div>

                    <div className="flex justify-between items-center">
                        <button onClick={() => setPhase('select')}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white transition-colors">
                            <ArrowLeft size={16} />
                            {t('action.back') || 'Back'}
                        </button>
                        <div className="flex items-center gap-4">
                            <p className="text-xs text-gray-600">{t('setup.changeable_later')}</p>
                            <button
                                onClick={handleTierConfirm}
                                disabled={!selectedTier}
                                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                                    selectedTier
                                        ? 'bg-blue-600 hover:bg-blue-500 text-white'
                                        : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                                }`}
                            >
                                {t('setup.next')}
                                <ArrowRight size={16} />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // Phase: Mode selection (default)
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
                        onClick={handleModeConfirm}
                        disabled={!selectedMode}
                        className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                            selectedMode
                                ? 'bg-blue-600 hover:bg-blue-500 text-white'
                                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        }`}
                    >
                        {t('setup.next')}
                        <ArrowRight size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SetupPage;
