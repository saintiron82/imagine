import React, { useState, useEffect } from 'react';
import { Settings, Cpu, Server, Database, Wrench, Globe, Palette, LayoutGrid } from 'lucide-react';
import { useLocale } from '../i18n';
import { useAuth } from '../contexts/AuthContext';
import { isElectron } from '../api/client';
import SettingSection from '../components/settings/SettingSection';
import SettingRow from '../components/settings/SettingRow';
import ToggleSwitch from '../components/settings/ToggleSwitch';
import EnvironmentSetupSection from '../components/settings/EnvironmentSetupSection';
import RegisteredFoldersPanel from '../components/RegisteredFoldersPanel';

const SettingsPage = ({ onScanFolder, isBusy }) => {
    const { t, locale, setLocale, availableLocales } = useLocale();
    const { isAdmin } = useAuth();

    // Config state
    const [config, setConfig] = useState(null);
    const [loading, setLoading] = useState(true);
    const [restartNeeded, setRestartNeeded] = useState(new Set());

    // Individual setting states
    const [aiMode, setAiMode] = useState({ auto_detect: true, override: null });
    const [batchEnabled, setBatchEnabled] = useState(true);
    const [mvEnabled, setMvEnabled] = useState(true);
    const [autoScan, setAutoScan] = useState(true);
    const [parseAhead, setParseAhead] = useState(true);
    const [autoProcessing, setAutoProcessing] = useState(true);
    const [restBetween, setRestBetween] = useState(30);
    const [webdavAutoProcess, setWebdavAutoProcess] = useState(true);
    const [gridDensity, setGridDensity] = useState('default');

    useEffect(() => {
        loadConfig();
    }, []);

    const loadConfig = async () => {
        setLoading(true);
        try {
            const result = isElectron
                ? await window.electron?.pipeline?.getConfig()
                : null;

            if (result?.success) {
                const c = result.config;
                setConfig(c);
                setAiMode(c.ai_mode || { auto_detect: true, override: null });
                setBatchEnabled(c.batch_processing?.enabled !== false);
                setMvEnabled(c.embedding?.text?.enabled !== false);
                setAutoScan(c.registered_folders?.auto_scan !== false);
                setParseAhead(c.server?.parse_ahead?.enabled !== false);
                setAutoProcessing(c.server?.auto_processing?.enabled !== false);
                setRestBetween(c.server?.auto_processing?.rest_after_batch_s ?? 30);
                setWebdavAutoProcess(c.webdav?.auto_process !== false);
                setGridDensity(c.ui?.grid_density || 'default');
            }
        } catch (e) {
            console.error('Failed to load config:', e);
        } finally {
            setLoading(false);
        }
    };

    const updateSetting = async (key, value) => {
        try {
            await window.electron?.pipeline?.updateConfig(key, value);
        } catch (e) {
            console.error(`Failed to update ${key}:`, e);
        }
    };

    const markRestart = (key) => {
        setRestartNeeded(prev => new Set([...prev, key]));
    };

    // Handlers
    const handleTierChange = async (value) => {
        if (value === 'auto') {
            await updateSetting('ai_mode.auto_detect', true);
            await updateSetting('ai_mode.override', null);
            setAiMode({ auto_detect: true, override: null });
        } else {
            await updateSetting('ai_mode.auto_detect', false);
            await updateSetting('ai_mode.override', value);
            setAiMode({ auto_detect: false, override: value });
        }
        markRestart('ai_tier');
    };

    const handleToggle = async (key, value, setter, needsRestart = false) => {
        setter(value);
        await updateSetting(key, value);
        if (needsRestart) markRestart(key);
    };

    const handleRestBetween = async (value) => {
        const num = Math.max(0, parseInt(value) || 0);
        setRestBetween(num);
        await updateSetting('server.auto_processing.rest_after_batch_s', num);
    };

    const handleGridDensity = async (value) => {
        setGridDensity(value);
        await updateSetting('ui.grid_density', value);
    };

    const handleLocaleChange = (newLocale) => {
        setLocale(newLocale);
        updateSetting('ui.locale', newLocale);
    };

    if (loading) {
        return (
            <div className="flex-1 flex items-center justify-center">
                <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full" />
            </div>
        );
    }

    const restartBadge = (key) => restartNeeded.has(key) ? t('settings.restart_required') : null;

    return (
        <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
            <div className="max-w-2xl mx-auto">
                {/* Page Title */}
                <h1 className="text-xl font-bold text-white flex items-center gap-2 mb-6">
                    <Settings size={22} />
                    {t('tab.settings')}
                </h1>

                {/* Section 1: General */}
                <SettingSection title={t('settings.section_general')} icon={Globe}>
                    {/* Language */}
                    <SettingRow
                        label={t('settings.language')}
                        description={t('settings.language_desc')}
                    >
                        <select
                            value={locale}
                            onChange={(e) => handleLocaleChange(e.target.value)}
                            className="bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-white text-sm focus:border-blue-500 focus:outline-none"
                        >
                            {availableLocales.map(loc => (
                                <option key={loc} value={loc}>{loc === 'en-US' ? 'English' : '한국어'}</option>
                            ))}
                        </select>
                    </SettingRow>

                    {/* Theme */}
                    <SettingRow
                        label={t('settings.theme')}
                        description={t('settings.theme_desc')}
                    >
                        <select
                            value="dark"
                            disabled
                            className="bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-gray-400 text-sm cursor-not-allowed opacity-50"
                        >
                            <option value="dark">{t('settings.theme_dark')}</option>
                            <option value="light">{t('settings.theme_light')}</option>
                        </select>
                    </SettingRow>

                    {/* Grid Density */}
                    <SettingRow
                        label={t('settings.grid_density')}
                        description={t('settings.grid_density_desc')}
                    >
                        <div className="flex rounded-lg overflow-hidden border border-gray-600">
                            {['compact', 'default', 'large'].map((d) => (
                                <button
                                    key={d}
                                    onClick={() => handleGridDensity(d)}
                                    className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                                        gridDensity === d
                                            ? 'bg-blue-600 text-white'
                                            : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'
                                    }`}
                                >
                                    {t(`settings.grid_${d}`)}
                                </button>
                            ))}
                        </div>
                    </SettingRow>
                </SettingSection>

                {/* Section 2: AI Pipeline (Electron only) */}
                {isElectron && (
                    <SettingSection title={t('settings.section_pipeline')} icon={Cpu}>
                        {/* AI Model Tier */}
                        <SettingRow
                            label={t('settings.ai_tier')}
                            description={t('settings.ai_tier_desc')}
                            badge={restartBadge('ai_tier')}
                        >
                            <select
                                value={aiMode.auto_detect ? 'auto' : (aiMode.override || 'auto')}
                                onChange={(e) => handleTierChange(e.target.value)}
                                className="bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-white text-sm focus:border-blue-500 focus:outline-none"
                            >
                                <option value="auto">Auto-Detect</option>
                                <option value="standard">Standard (~6GB)</option>
                                <option value="pro">Pro (8-16GB)</option>
                                <option value="ultra">Ultra (20GB+)</option>
                            </select>
                        </SettingRow>

                        {/* Batch Processing */}
                        <SettingRow
                            label={t('settings.batch_mode_title')}
                            description={t('settings.batch_mode_desc')}
                            badge={restartBadge('batch_processing.enabled')}
                        >
                            <ToggleSwitch
                                checked={batchEnabled}
                                onChange={(v) => handleToggle('batch_processing.enabled', v, setBatchEnabled, true)}
                            />
                        </SettingRow>

                        {/* MV Generation */}
                        <SettingRow
                            label={t('settings.mv_generation')}
                            description={t('settings.mv_generation_desc')}
                        >
                            <ToggleSwitch
                                checked={mvEnabled}
                                onChange={(v) => handleToggle('embedding.text.enabled', v, setMvEnabled)}
                            />
                        </SettingRow>

                        {/* Auto-scan */}
                        <SettingRow
                            label={t('settings.auto_scan')}
                            description={t('settings.auto_scan_desc')}
                        >
                            <ToggleSwitch
                                checked={autoScan}
                                onChange={(v) => handleToggle('registered_folders.auto_scan', v, setAutoScan)}
                            />
                        </SettingRow>
                    </SettingSection>
                )}

                {/* Section 3: Server & Worker (Electron + admin only) */}
                {isElectron && isAdmin && (
                    <SettingSection title={t('settings.section_server')} icon={Server}>
                        {/* Parse Ahead */}
                        <SettingRow
                            label={t('settings.parse_ahead')}
                            description={t('settings.parse_ahead_desc')}
                        >
                            <ToggleSwitch
                                checked={parseAhead}
                                onChange={(v) => handleToggle('server.parse_ahead.enabled', v, setParseAhead)}
                            />
                        </SettingRow>

                        {/* Auto Processing */}
                        <SettingRow
                            label={t('settings.auto_processing')}
                            description={t('settings.auto_processing_desc')}
                        >
                            <ToggleSwitch
                                checked={autoProcessing}
                                onChange={(v) => handleToggle('server.auto_processing.enabled', v, setAutoProcessing)}
                            />
                        </SettingRow>

                        {/* Rest Between Batches */}
                        <SettingRow
                            label={t('settings.rest_between')}
                            description={t('settings.rest_between_desc')}
                        >
                            <div className="flex items-center gap-2">
                                <input
                                    type="number"
                                    min="0"
                                    max="300"
                                    value={restBetween}
                                    onChange={(e) => handleRestBetween(e.target.value)}
                                    className="w-20 bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-white text-sm text-center focus:border-blue-500 focus:outline-none"
                                />
                                <span className="text-xs text-gray-500">sec</span>
                            </div>
                        </SettingRow>
                    </SettingSection>
                )}

                {/* Section 4: Data Sources (Electron only) */}
                {isElectron && (
                    <SettingSection title={t('settings.section_data')} icon={Database}>
                        {/* Registered Folders */}
                        <div className="mb-4">
                            <RegisteredFoldersPanel onScanFolder={onScanFolder} isBusy={isBusy} />
                        </div>

                        {/* WebDAV Auto-Process */}
                        <SettingRow
                            label={t('settings.webdav_auto')}
                            description={t('settings.webdav_auto_desc')}
                        >
                            <ToggleSwitch
                                checked={webdavAutoProcess}
                                onChange={(v) => handleToggle('webdav.auto_process', v, setWebdavAutoProcess)}
                            />
                        </SettingRow>
                    </SettingSection>
                )}

                {/* Section 5: Environment Setup (Electron only) */}
                {isElectron && (
                    <SettingSection title={t('settings.section_env')} icon={Wrench}>
                        <EnvironmentSetupSection />
                    </SettingSection>
                )}
            </div>
        </div>
    );
};

export default SettingsPage;
