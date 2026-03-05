import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Shield, Cpu, ArrowRight, ArrowLeft, Download, CheckCircle, AlertCircle, Loader2, SkipForward, Zap, Star, Rocket, Users, Globe, Lock, User, X, Languages, Play } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron, setServerUrl as setClientServerUrl } from '../api/client';
import { getServerInfo, initServer, register as apiRegister } from '../api/auth';
import { useMdnsDiscovery } from '../hooks/useMdnsDiscovery';
import {
  getServerHistory,
  addServerToHistory,
} from '../utils/serverHistory';

const TIERS = [
    { id: 'standard', icon: Zap,    color: 'emerald' },
    { id: 'pro',      icon: Star,   color: 'blue' },
    { id: 'ultra',    icon: Rocket, color: 'purple' },
];

/**
 * SetupPage — Group-based setup wizard.
 *
 * Phases:
 *   'select'     — Choose "Create Group" or "Join Group"
 *   'create'     — Enter group name, server password, admin account
 *   'join'       — Enter server URL, server password, user account
 *   'tier'       — (Create only) Select AI tier
 *   'checking'   — (Create only) Environment check
 *   'install'    — (Create only) Models missing, offer install
 *   'installing' — (Create only) Installing models
 *   'done'       — (Create only) Installation complete
 */
const SetupPage = ({ onComplete }) => {
    const { t, locale, setLocale, availableLocales } = useLocale();
    const [phase, setPhase] = useState('select');
    const [selectedTier, setSelectedTier] = useState(null);

    // Create group form
    const [groupName, setGroupName] = useState('');
    const [serverPassword, setServerPassword] = useState('');
    const [adminUsername, setAdminUsername] = useState('');
    const [adminPassword, setAdminPassword] = useState('');

    // Join group form
    const [joinServerUrl, setJoinServerUrl] = useState('');
    const [joinServerPassword, setJoinServerPassword] = useState('');
    const [joinUsername, setJoinUsername] = useState('');
    const [joinPassword, setJoinPassword] = useState('');
    const [joinGroupName, setJoinGroupName] = useState('');
    const [joinServerChecked, setJoinServerChecked] = useState(false);
    const [joinServerChecking, setJoinServerChecking] = useState(false);
    const [joinError, setJoinError] = useState('');
    const [joinSubmitting, setJoinSubmitting] = useState(false);

    // Create group state
    const [createError, setCreateError] = useState('');
    const [createSubmitting, setCreateSubmitting] = useState(false);

    // Existing group detection
    const [existingGroup, setExistingGroup] = useState(null); // { group_name }
    const [existingLoading, setExistingLoading] = useState(false);
    const [resumeError, setResumeError] = useState('');
    const [resumeSubmitting, setResumeSubmitting] = useState(false);

    // Environment check state
    const [envStatus, setEnvStatus] = useState(null);
    const [installLogs, setInstallLogs] = useState([]);
    const [installDone, setInstallDone] = useState(false);
    const [installSuccess, setInstallSuccess] = useState(false);
    const logEndRef = useRef(null);

    // mDNS discovery (for join mode)
    const { servers: mdnsServers, browsing } = useMdnsDiscovery();

    // Detect existing group on mount (Electron only)
    useEffect(() => {
        if (!isElectron) return;
        let cancelled = false;

        const detect = async () => {
            setExistingLoading(true);
            try {
                // Try starting server to check if group exists
                const port = 8000;
                try {
                    const status = await window.electron?.server?.getStatus();
                    if (!status?.running) {
                        await window.electron?.server?.start({ port });
                        await new Promise(r => setTimeout(r, 1500));
                    }
                } catch { /* server start may fail, that's ok */ }

                // Check if server has an initialized group
                const info = await getServerInfo(`http://localhost:${port}`);
                if (!cancelled && info.ok && info.initialized && info.group_name) {
                    setExistingGroup({ group_name: info.group_name });
                }
            } catch { /* no existing group */ }
            if (!cancelled) setExistingLoading(false);
        };

        detect();
        return () => { cancelled = true; };
    }, []);

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

    // ── Resume existing group: start server + go to login ──
    const handleResumeGroup = useCallback(async () => {
        setResumeError('');
        setResumeSubmitting(true);
        try {
            const port = 8000;
            // Ensure server is running
            const status = await window.electron?.server?.getStatus();
            if (!status?.running) {
                await window.electron?.server?.start({ port });
                await new Promise(r => setTimeout(r, 1500));
            }
            const baseUrl = `http://localhost:${port}`;
            setClientServerUrl(baseUrl);
            // Complete as server mode — LoginPage will handle authentication
            onComplete('server');
        } catch (e) {
            setResumeError(e.message || 'Failed to start server');
            setResumeSubmitting(false);
        }
    }, [onComplete]);

    // ── Join: Check server ──
    const handleCheckJoinServer = useCallback(async (url) => {
        const targetUrl = (url || joinServerUrl).trim();
        if (!targetUrl) return;
        setJoinServerChecking(true);
        setJoinError('');
        setJoinGroupName('');
        setJoinServerChecked(false);

        try {
            const info = await getServerInfo(targetUrl);
            if (info.ok) {
                setJoinGroupName(info.group_name || '');
                setJoinServerChecked(true);
                if (!info.initialized) {
                    setJoinError(t('group.server_not_initialized'));
                    setJoinServerChecked(false);
                }
            } else {
                setJoinError(info.error || 'Connection failed');
            }
        } catch (e) {
            setJoinError(e.message);
        }
        setJoinServerChecking(false);
    }, [joinServerUrl, t]);

    // ── Join: Select mDNS server ──
    const handleSelectMdns = useCallback((server) => {
        const addr = server.addresses?.[0] || server.host;
        const url = `http://${addr}:${server.port}`;
        setJoinServerUrl(url);
        handleCheckJoinServer(url);
    }, [handleCheckJoinServer]);

    // ── Join: Submit ──
    const handleJoinSubmit = useCallback(async (e) => {
        e.preventDefault();
        setJoinError('');
        setJoinSubmitting(true);

        try {
            const trimmedUrl = joinServerUrl.trim();
            setClientServerUrl(trimmedUrl);

            // Register via API
            await apiRegister({
                server_password: joinServerPassword,
                username: joinUsername,
                password: joinPassword,
            });

            // Save to history
            addServerToHistory({
                url: trimmedUrl,
                name: joinGroupName,
                version: '',
                lastUsername: joinUsername,
            });

            // Complete: client mode
            onComplete('client');
        } catch (e) {
            setJoinError(e.detail || e.message || 'Registration failed');
        }
        setJoinSubmitting(false);
    }, [joinServerUrl, joinServerPassword, joinUsername, joinPassword, joinGroupName, onComplete]);

    // ── Create: Tier confirm → save + env check ──
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
                await handleCreateFinalize();
            } else {
                setPhase('install');
            }
        } catch (e) {
            console.error('check-env failed:', e);
            await handleCreateFinalize();
        }
    }, [selectedTier]);

    // ── Create: Start server + init + auto-login ──
    const handleCreateFinalize = useCallback(async () => {
        setCreateError('');
        setCreateSubmitting(true);

        try {
            // Start embedded FastAPI server
            const port = 8000;
            await window.electron.server.start({ port });

            // Wait a moment for server to be ready
            await new Promise(r => setTimeout(r, 1500));

            const baseUrl = `http://localhost:${port}`;
            setClientServerUrl(baseUrl);

            // Pre-check: verify server is not already initialized
            const info = await getServerInfo(baseUrl);
            if (info.ok && info.initialized) {
                throw new Error(t('validation.server_already_initialized'));
            }

            // Initialize server (creates group + admin)
            await initServer(baseUrl, {
                group_name: groupName,
                server_password: serverPassword,
                admin_username: adminUsername,
                admin_password: adminPassword,
            });

            // Complete: server mode
            onComplete('server');
        } catch (e) {
            setCreateError(e.message || 'Server initialization failed');
            setCreateSubmitting(false);
            setPhase('create');  // Return to form so error is visible
        }
    }, [groupName, serverPassword, adminUsername, adminPassword, onComplete]);

    const handleInstall = useCallback(() => {
        setPhase('installing');
        setInstallLogs([]);
        window.electron?.pipeline?.installEnv();
    }, []);

    const handleSkip = useCallback(async () => {
        await handleCreateFinalize();
    }, [handleCreateFinalize]);

    const handleFinish = useCallback(async () => {
        await handleCreateFinalize();
    }, [handleCreateFinalize]);

    // ── Phase: Checking environment ──
    if (phase === 'checking') {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="text-center max-w-md">
                    {createError ? (
                        <>
                            <AlertCircle size={40} className="text-red-400 mx-auto mb-4" />
                            <p className="text-gray-300 mb-3">{t('setup.checking_env')}</p>
                            <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400 mb-4">
                                {createError}
                            </div>
                            <button onClick={() => { setCreateError(''); setPhase('create'); }}
                                className="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white bg-gray-800 hover:bg-gray-700 transition-colors">
                                <ArrowLeft size={14} className="inline mr-1" />
                                {t('action.back') || 'Back'}
                            </button>
                        </>
                    ) : (
                        <>
                            <Loader2 size={40} className="animate-spin text-blue-400 mx-auto mb-4" />
                            <p className="text-gray-300">{t('setup.checking_env')}</p>
                        </>
                    )}
                </div>
            </div>
        );
    }

    // ── Phase: Models missing ──
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

    // ── Phase: Installing models ──
    if (phase === 'installing' || (phase === 'done' && installDone)) {
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
                                disabled={createSubmitting}
                                className="flex items-center gap-2 mx-auto px-6 py-3 rounded-lg font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors disabled:opacity-50">
                                {createSubmitting ? <Loader2 size={16} className="animate-spin" /> : null}
                                {t('setup.start')}
                                <ArrowRight size={16} />
                            </button>
                            {createError && (
                                <p className="text-xs text-red-400 mt-2">{createError}</p>
                            )}
                        </div>
                    )}
                </div>
            </div>
        );
    }

    // ── Phase: Tier selection (create only) ──
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
                        <button onClick={() => setPhase('create')}
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

    // ── Phase: Create Group form ──
    if (phase === 'create') {
        const checks = {
            groupName: groupName.trim().length > 0,
            serverPassword: serverPassword.trim().length >= 1,
            adminUsername: adminUsername.trim().length >= 2,
            adminPassword: adminPassword.length >= 4,
        };
        const canProceed = checks.groupName && checks.serverPassword && checks.adminUsername && checks.adminPassword;
        // Show hints after user has interacted (typed something)
        const touched = {
            groupName: groupName.length > 0,
            serverPassword: serverPassword.length > 0,
            adminUsername: adminUsername.length > 0,
            adminPassword: adminPassword.length > 0,
        };

        const handleCreateNext = async () => {
            if (!canProceed) return;
            setCreateError('');
            setCreateSubmitting(true);

            // Pre-check: if server is already running, verify it's not already initialized
            try {
                const info = await getServerInfo('http://localhost:8000');
                if (info.ok && info.initialized) {
                    setCreateError(t('validation.server_already_initialized'));
                    setCreateSubmitting(false);
                    return;
                }
            } catch {
                // Server not running yet — that's expected for fresh create
            }
            setCreateSubmitting(false);

            if (window.electron?.pipeline?.checkEnv) {
                setPhase('tier');
            } else {
                handleCreateFinalize();
            }
        };

        return (
            <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="max-w-md w-full px-8">
                    <div className="text-center mb-8">
                        <div className="p-3 rounded-full bg-blue-600/20 w-fit mx-auto mb-4">
                            <Users size={32} className="text-blue-400" />
                        </div>
                        <h1 className="text-2xl font-bold mb-2">{t('group.create')}</h1>
                        <p className="text-sm text-gray-400">{t('group.create_desc')}</p>
                    </div>

                    <div className="space-y-4">
                        {/* Group Name */}
                        <div>
                            <label className="block text-xs text-gray-400 mb-1">{t('group.name')}</label>
                            <input
                                type="text"
                                value={groupName}
                                onChange={(e) => setGroupName(e.target.value)}
                                placeholder={t('group.name_placeholder')}
                                className={`w-full px-3 py-2 bg-gray-800 border rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none ${
                                    touched.groupName && !checks.groupName ? 'border-red-500/60' : checks.groupName ? 'border-green-600/40' : 'border-gray-600'
                                }`}
                                maxLength={100}
                            />
                            {touched.groupName && !checks.groupName && (
                                <p className="text-[10px] text-red-400 mt-1">{t('validation.group_name_required')}</p>
                            )}
                        </div>

                        {/* Server Password */}
                        <div>
                            <label className="block text-xs text-gray-400 mb-1">{t('group.server_password')}</label>
                            <input
                                type="password"
                                value={serverPassword}
                                onChange={(e) => setServerPassword(e.target.value)}
                                placeholder={t('group.server_password_placeholder')}
                                className={`w-full px-3 py-2 bg-gray-800 border rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none ${
                                    touched.serverPassword && !checks.serverPassword ? 'border-red-500/60' : checks.serverPassword ? 'border-green-600/40' : 'border-gray-600'
                                }`}
                                maxLength={128}
                            />
                            {touched.serverPassword && !checks.serverPassword ? (
                                <p className="text-[10px] text-red-400 mt-1">{t('validation.server_password_required')}</p>
                            ) : (
                                <p className="text-[10px] text-gray-600 mt-1">{t('group.server_password_hint')}</p>
                            )}
                        </div>

                        <div className="border-t border-gray-700 pt-4">
                            <p className="text-xs text-gray-500 mb-3">{t('group.admin_account')}</p>

                            {/* Admin Username */}
                            <div className="mb-3">
                                <label className="block text-xs text-gray-400 mb-1">{t('auth.username')}</label>
                                <input
                                    type="text"
                                    value={adminUsername}
                                    onChange={(e) => setAdminUsername(e.target.value)}
                                    placeholder={t('auth.username_placeholder')}
                                    className={`w-full px-3 py-2 bg-gray-800 border rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none ${
                                        touched.adminUsername && !checks.adminUsername ? 'border-red-500/60' : checks.adminUsername ? 'border-green-600/40' : 'border-gray-600'
                                    }`}
                                    maxLength={50}
                                />
                                {touched.adminUsername && !checks.adminUsername && (
                                    <p className="text-[10px] text-red-400 mt-1">{t('validation.username_min')}</p>
                                )}
                            </div>

                            {/* Admin Password */}
                            <div>
                                <label className="block text-xs text-gray-400 mb-1">{t('auth.password')}</label>
                                <input
                                    type="password"
                                    value={adminPassword}
                                    onChange={(e) => setAdminPassword(e.target.value)}
                                    placeholder="••••••••"
                                    className={`w-full px-3 py-2 bg-gray-800 border rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none ${
                                        touched.adminPassword && !checks.adminPassword ? 'border-red-500/60' : checks.adminPassword ? 'border-green-600/40' : 'border-gray-600'
                                    }`}
                                    maxLength={128}
                                />
                                {touched.adminPassword && !checks.adminPassword && (
                                    <p className="text-[10px] text-red-400 mt-1">{t('validation.password_min_4')}</p>
                                )}
                            </div>
                        </div>

                        {/* Checklist summary */}
                        {!canProceed && (touched.groupName || touched.serverPassword || touched.adminUsername || touched.adminPassword) && (
                            <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px]">
                                <span className={checks.groupName ? 'text-green-500' : 'text-gray-600'}>
                                    {checks.groupName ? '✓' : '○'} {t('group.name')}
                                </span>
                                <span className={checks.serverPassword ? 'text-green-500' : 'text-gray-600'}>
                                    {checks.serverPassword ? '✓' : '○'} {t('group.server_password')}
                                </span>
                                <span className={checks.adminUsername ? 'text-green-500' : 'text-gray-600'}>
                                    {checks.adminUsername ? '✓' : '○'} {t('auth.username')}
                                </span>
                                <span className={checks.adminPassword ? 'text-green-500' : 'text-gray-600'}>
                                    {checks.adminPassword ? '✓' : '○'} {t('auth.password')}
                                </span>
                            </div>
                        )}

                        {createError && (
                            <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400">
                                {createError}
                            </div>
                        )}

                        <div className="flex justify-between items-center pt-2">
                            <button onClick={() => setPhase('select')}
                                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white transition-colors">
                                <ArrowLeft size={16} />
                                {t('action.back') || 'Back'}
                            </button>
                            <button
                                onClick={handleCreateNext}
                                disabled={!canProceed || createSubmitting}
                                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                                    canProceed && !createSubmitting
                                        ? 'bg-blue-600 hover:bg-blue-500 text-white'
                                        : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                                }`}
                            >
                                {createSubmitting ? <Loader2 size={16} className="animate-spin" /> : null}
                                {t('setup.next')}
                                <ArrowRight size={16} />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // ── Phase: Join Group form ──
    if (phase === 'join') {
        const canSubmit = joinServerChecked && joinServerPassword.trim()
            && joinUsername.trim().length >= 2 && joinPassword.trim().length >= 6;

        return (
            <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
                <div className="max-w-md w-full px-8">
                    <div className="text-center mb-8">
                        <div className="p-3 rounded-full bg-emerald-600/20 w-fit mx-auto mb-4">
                            <Globe size={32} className="text-emerald-400" />
                        </div>
                        <h1 className="text-2xl font-bold mb-2">{t('group.join')}</h1>
                        <p className="text-sm text-gray-400">{t('group.join_desc')}</p>
                    </div>

                    {/* mDNS Discovered Servers */}
                    {isElectron && (mdnsServers.length > 0 || browsing) && (
                        <div className="mb-4">
                            <p className="text-[10px] text-gray-500 uppercase mb-1.5">{t('auth.discovered_servers')}</p>
                            {mdnsServers.length > 0 ? (
                                <div className="space-y-1">
                                    {mdnsServers.map((s) => {
                                        const addr = s.addresses?.[0] || s.host;
                                        const sUrl = `http://${addr}:${s.port}`;
                                        const isSelected = joinServerUrl === sUrl;
                                        return (
                                            <button key={s.name} type="button"
                                                onClick={() => handleSelectMdns(s)}
                                                className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-left text-sm transition-colors ${
                                                    isSelected
                                                        ? 'bg-emerald-900/40 border border-emerald-600'
                                                        : 'bg-gray-800/50 border border-gray-700 hover:border-gray-500'
                                                }`}
                                            >
                                                <div className="flex items-center gap-2">
                                                    <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                                    <span className="text-white font-medium">{s.serverName || s.name}</span>
                                                </div>
                                                <span className="text-xs text-gray-500 font-mono">{addr}:{s.port}</span>
                                            </button>
                                        );
                                    })}
                                </div>
                            ) : browsing ? (
                                <p className="text-xs text-gray-600 italic">{t('auth.discovering')}</p>
                            ) : null}
                        </div>
                    )}

                    <form onSubmit={handleJoinSubmit} className="space-y-4">
                        {/* Server URL */}
                        <div>
                            <label className="block text-xs text-gray-400 mb-1">{t('auth.server_url')}</label>
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={joinServerUrl}
                                    onChange={(e) => {
                                        setJoinServerUrl(e.target.value);
                                        setJoinServerChecked(false);
                                        setJoinGroupName('');
                                    }}
                                    placeholder="http://192.168.1.10:8000"
                                    className="flex-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none font-mono"
                                />
                                <button type="button"
                                    onClick={() => handleCheckJoinServer()}
                                    disabled={!joinServerUrl.trim() || joinServerChecking}
                                    className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-xs text-gray-300 hover:bg-gray-600 disabled:opacity-50 transition-colors"
                                >
                                    {joinServerChecking ? '...' : t('auth.check')}
                                </button>
                            </div>
                            {joinServerChecked && joinGroupName && (
                                <div className="flex items-center gap-2 mt-2 px-3 py-1.5 bg-emerald-900/20 border border-emerald-800/40 rounded-lg">
                                    <Users size={12} className="text-emerald-400" />
                                    <span className="text-sm text-emerald-300 font-medium">{joinGroupName}</span>
                                    <CheckCircle size={12} className="text-emerald-400 ml-auto" />
                                </div>
                            )}
                        </div>

                        {/* Server Password */}
                        <div>
                            <label className="block text-xs text-gray-400 mb-1">{t('group.server_password')}</label>
                            <input
                                type="password"
                                value={joinServerPassword}
                                onChange={(e) => setJoinServerPassword(e.target.value)}
                                placeholder={t('auth.server_password_placeholder')}
                                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                            />
                        </div>

                        <div className="border-t border-gray-700 pt-4">
                            <p className="text-xs text-gray-500 mb-3">{t('group.your_account')}</p>

                            {/* Username */}
                            <div className="mb-3">
                                <label className="block text-xs text-gray-400 mb-1">{t('auth.username')}</label>
                                <input
                                    type="text"
                                    value={joinUsername}
                                    onChange={(e) => setJoinUsername(e.target.value)}
                                    placeholder={t('auth.username_placeholder')}
                                    className={`w-full px-3 py-2 bg-gray-800 border rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none ${
                                        joinUsername.length > 0 && joinUsername.trim().length < 2 ? 'border-red-500/50' : 'border-gray-600'
                                    }`}
                                />
                                {joinUsername.length > 0 && joinUsername.trim().length < 2 && (
                                    <p className="text-[10px] text-red-400 mt-1">{t('validation.username_min')}</p>
                                )}
                            </div>

                            {/* Password */}
                            <div>
                                <label className="block text-xs text-gray-400 mb-1">{t('auth.password')}</label>
                                <input
                                    type="password"
                                    value={joinPassword}
                                    onChange={(e) => setJoinPassword(e.target.value)}
                                    placeholder="••••••••"
                                    className={`w-full px-3 py-2 bg-gray-800 border rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none ${
                                        joinPassword.length > 0 && joinPassword.length < 6 ? 'border-red-500/50' : 'border-gray-600'
                                    }`}
                                />
                                {joinPassword.length > 0 && joinPassword.length < 6 && (
                                    <p className="text-[10px] text-red-400 mt-1">{t('validation.password_min')}</p>
                                )}
                            </div>
                        </div>

                        {joinError && (
                            <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400">
                                {joinError}
                            </div>
                        )}

                        {!joinServerChecked && joinServerUrl.trim() && !joinServerChecking && (
                            <p className="text-[10px] text-yellow-400">{t('validation.check_server_first')}</p>
                        )}

                        <div className="flex justify-between items-center pt-2">
                            <button type="button" onClick={() => setPhase('select')}
                                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white transition-colors">
                                <ArrowLeft size={16} />
                                {t('action.back') || 'Back'}
                            </button>
                            <button type="submit"
                                disabled={!canSubmit || joinSubmitting}
                                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                                    canSubmit && !joinSubmitting
                                        ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                                        : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                                }`}
                            >
                                {joinSubmitting ? <Loader2 size={16} className="animate-spin" /> : null}
                                {t('group.join_btn')}
                                <ArrowRight size={16} />
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        );
    }

    // ── Locale label map ──
    const localeLabels = { 'en-US': 'EN', 'ko-KR': '한국어' };

    const handleQuit = () => {
        if (isElectron && window.electron?.app?.quit) {
            window.electron.app.quit();
        } else if (isElectron) {
            window.close();
        }
    };

    // ── Phase: Select mode (default) ──
    return (
        <div className="relative flex items-center justify-center h-screen bg-gray-900 text-white">
            {/* Top bar: drag region + language + close */}
            <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-2 z-10"
                 style={{ WebkitAppRegion: 'drag' }}>
                {/* Left spacer (macOS traffic lights area) */}
                <div className="w-20" />

                {/* Right controls */}
                <div className="flex items-center gap-2" style={{ WebkitAppRegion: 'no-drag' }}>
                    {/* Language switcher */}
                    <div className="flex items-center gap-1 bg-gray-800/60 rounded-lg px-1.5 py-1">
                        <Languages size={13} className="text-gray-500 mr-0.5" />
                        {availableLocales.map((loc) => (
                            <button key={loc}
                                onClick={() => setLocale(loc)}
                                className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                                    locale === loc
                                        ? 'bg-gray-600 text-white'
                                        : 'text-gray-400 hover:text-white'
                                }`}
                            >
                                {localeLabels[loc] || loc}
                            </button>
                        ))}
                    </div>

                    {/* Quit button (Electron only) */}
                    {isElectron && (
                        <button onClick={handleQuit}
                            className="p-1.5 rounded-lg text-gray-500 hover:text-red-400 hover:bg-gray-800/60 transition-colors"
                            title={t('action.quit') || 'Quit'}>
                            <X size={16} />
                        </button>
                    )}
                </div>
            </div>

            <div className="max-w-3xl w-full px-8">
                {/* Title */}
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold mb-2">Imagine</h1>
                    <p className="text-gray-400 text-sm">{t('group.setup_subtitle')}</p>
                </div>

                {/* Existing group: prominent resume card */}
                {existingGroup && (
                    <div className="mb-6">
                        <button
                            onClick={handleResumeGroup}
                            disabled={resumeSubmitting}
                            className="w-full p-5 rounded-xl border-2 border-amber-600/60 bg-amber-900/10 hover:border-amber-500 hover:bg-amber-900/20 text-left transition-all"
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="p-2.5 rounded-lg bg-amber-600">
                                        <Play size={22} className="text-white" />
                                    </div>
                                    <div>
                                        <h2 className="text-lg font-bold">{t('group.resume')}</h2>
                                        <p className="text-sm text-amber-300/70">{existingGroup.group_name}</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 text-amber-400">
                                    {resumeSubmitting ? <Loader2 size={18} className="animate-spin" /> : <ArrowRight size={18} />}
                                </div>
                            </div>
                        </button>
                        {resumeError && (
                            <p className="text-xs text-red-400 mt-2 px-2">{resumeError}</p>
                        )}
                    </div>
                )}

                {existingLoading && (
                    <div className="flex items-center justify-center gap-2 mb-6 text-gray-500 text-xs">
                        <Loader2 size={14} className="animate-spin" />
                        {t('status.loading')}
                    </div>
                )}

                {/* Mode Cards */}
                <div className="grid grid-cols-2 gap-6 mb-8">
                    {/* Create Group */}
                    <button
                        onClick={() => setPhase('create')}
                        className="p-6 rounded-xl border-2 border-gray-700 bg-gray-800/50 hover:border-blue-500 hover:bg-blue-900/10 text-left transition-all group"
                    >
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2.5 rounded-lg bg-gray-700 group-hover:bg-blue-600 transition-colors">
                                <Shield size={24} />
                            </div>
                            <h2 className="text-lg font-bold">{t('group.create')}</h2>
                        </div>
                        <p className="text-sm text-gray-400 leading-relaxed">
                            {t('group.create_desc')}
                        </p>
                        <ul className="mt-4 space-y-1.5 text-xs text-gray-500">
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-blue-400" />
                                {t('group.create_feature1')}
                            </li>
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-blue-400" />
                                {t('group.create_feature2')}
                            </li>
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-blue-400" />
                                {t('group.create_feature3')}
                            </li>
                        </ul>
                    </button>

                    {/* Join Group */}
                    <button
                        onClick={() => setPhase('join')}
                        className="p-6 rounded-xl border-2 border-gray-700 bg-gray-800/50 hover:border-emerald-500 hover:bg-emerald-900/10 text-left transition-all group"
                    >
                        <div className="flex items-center gap-3 mb-4">
                            <div className="p-2.5 rounded-lg bg-gray-700 group-hover:bg-emerald-600 transition-colors">
                                <Globe size={24} />
                            </div>
                            <h2 className="text-lg font-bold">{t('group.join')}</h2>
                        </div>
                        <p className="text-sm text-gray-400 leading-relaxed">
                            {t('group.join_desc')}
                        </p>
                        <ul className="mt-4 space-y-1.5 text-xs text-gray-500">
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                                {t('group.join_feature1')}
                            </li>
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                                {t('group.join_feature2')}
                            </li>
                            <li className="flex items-center gap-1.5">
                                <span className="w-1 h-1 rounded-full bg-emerald-400" />
                                {t('group.join_feature3')}
                            </li>
                        </ul>
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SetupPage;
