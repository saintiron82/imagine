/**
 * ConnectionInfoPanel — Phase 3 of secure external worker access.
 *
 * Shows the connection candidates that this server publishes, plus a
 * Linux worker install command generator. Every candidate is labeled
 * with a security level so the admin cannot mistake "공인 IP 포트 직접
 * 공개" for a safe path.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocale } from '../../i18n';
import {
  getConnectionInfo,
  createHeadlessWorkerCommand,
  getRelayStatus,
} from '../../api/admin';

const SECURITY_TONE = {
  'safe-local': 'text-emerald-400 border-emerald-700/40 bg-emerald-900/20',
  'lan-only': 'text-blue-300 border-blue-700/40 bg-blue-900/20',
  'recommended-external': 'text-indigo-300 border-indigo-700/40 bg-indigo-900/20',
  'public-not-verified': 'text-orange-300 border-orange-700/40 bg-orange-900/20',
};

const MODE_LABEL_KEY = {
  direct_local: 'connection.mode_direct_local',
  direct_lan: 'connection.mode_direct_lan',
  manual_external: 'connection.mode_manual_external',
  relay_session: 'connection.mode_relay_session',
};

const SECURITY_LABEL_KEY = {
  'safe-local': 'connection.security_safe',
  'lan-only': 'connection.security_lan',
  'recommended-external': 'connection.security_recommended',
  'public-not-verified': 'connection.security_public_warning',
};

function ModeRow({ mode, t }) {
  const tone = SECURITY_TONE[mode.security] || 'text-gray-300 border-gray-700/40 bg-gray-900/20';
  return (
    <div className={`flex items-start justify-between gap-4 rounded border px-3 py-2 ${tone}`}>
      <div className="min-w-0">
        <p className="text-sm font-medium">{t(MODE_LABEL_KEY[mode.mode] || mode.mode)}</p>
        <p className="text-xs font-mono break-all opacity-80">
          {mode.url || t('connection.url_unset')}
        </p>
      </div>
      <div className="text-right shrink-0">
        <p className="text-[11px] uppercase tracking-wide">
          {t(SECURITY_LABEL_KEY[mode.security] || mode.security)}
        </p>
        <p className="text-[11px] mt-0.5 opacity-70">
          {mode.available ? t('connection.available_yes') : t('connection.available_no')}
        </p>
      </div>
    </div>
  );
}

export default function ConnectionInfoPanel() {
  const { t } = useLocale();

  const [info, setInfo] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [relayStatus, setRelayStatus] = useState(null);

  const [workerName, setWorkerName] = useState('cloud-worker-1');
  const [launcher, setLauncher] = useState('cloud');
  const [expiresMinutes, setExpiresMinutes] = useState(1440);
  const [selectedMode, setSelectedMode] = useState('direct_lan');
  const [overrideUrl, setOverrideUrl] = useState('');
  const [command, setCommand] = useState(null);
  const [commandError, setCommandError] = useState(null);
  const [issuing, setIssuing] = useState(false);

  const loadInfo = useCallback(async () => {
    try {
      const resp = await getConnectionInfo();
      const data = resp.data ?? resp;
      setInfo(data);
      setLoadError(null);
      if (data?.connect_modes) {
        const lan = data.connect_modes.find(m => m.mode === 'direct_lan' && m.available);
        if (lan) setSelectedMode('direct_lan');
      }
    } catch (err) {
      setLoadError(err?.response?.data?.detail || err.message || 'load failed');
    }
  }, []);

  const loadRelayStatus = useCallback(async () => {
    try {
      const resp = await getRelayStatus();
      setRelayStatus(resp.data ?? resp);
    } catch {
      setRelayStatus(null);
    }
  }, []);

  useEffect(() => {
    loadInfo();
    loadRelayStatus();
  }, [loadInfo, loadRelayStatus]);

  const issueCommand = async () => {
    setIssuing(true);
    setCommandError(null);
    setCommand(null);
    try {
      const resp = await createHeadlessWorkerCommand({
        worker_name: workerName.trim() || 'headless-worker',
        launcher,
        expires_minutes: Number(expiresMinutes) || 1440,
        server_url: overrideUrl.trim() || null,
        connect_mode: selectedMode,
      });
      setCommand(resp.data ?? resp);
    } catch (err) {
      setCommandError(err?.response?.data?.detail || err.message || 'failed');
    } finally {
      setIssuing(false);
    }
  };

  const copyCommand = async () => {
    if (!command?.linux_command) return;
    try {
      await navigator.clipboard.writeText(command.linux_command);
    } catch {
      // clipboard may be unavailable — leave the text visible for manual copy
    }
  };

  const modeOptions = useMemo(() => (
    [
      { value: 'direct_lan', label: t('connection.mode_direct_lan') },
      { value: 'manual_external', label: t('connection.mode_manual_external') },
      { value: 'relay_session', label: t('connection.mode_relay_session') },
    ]
  ), [t]);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">{t('connection.title')}</h2>

      {loadError && (
        <p className="text-xs text-red-400">{loadError}</p>
      )}

      {info && (
        <section className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-3">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-medium text-white">{t('connection.current_address')}</h3>
            <span className="text-[11px] font-mono text-gray-500">{info.server_id}</span>
          </div>
          <p className="text-xs text-gray-400">
            <span className="font-mono text-gray-200">{info.request_origin}</span>
            {info.group_name ? ` · ${info.group_name}` : ''}
          </p>

          <div className="space-y-2">
            {(info.connect_modes || []).map(mode => (
              <ModeRow key={mode.mode} mode={mode} t={t} />
            ))}
          </div>

          <div className="space-y-1 text-[11px] text-gray-400 pt-1 border-t border-gray-700/60">
            <p>{t('connection.warn_public_port')}</p>
            <p>{t('connection.warn_relay_recommended')}</p>
            <p>{t('connection.warn_aws_cost')}</p>
            <p>{t('connection.warn_data_plane')}</p>
          </div>
        </section>
      )}

      {relayStatus && (
        <section className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-2">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-medium text-white">{t('connection.relay_status_title')}</h3>
            <span
              className={`text-[11px] uppercase tracking-wide px-2 py-0.5 rounded border ${
                relayStatus.online
                  ? 'text-emerald-300 bg-emerald-900/30 border-emerald-700/40'
                  : relayStatus.configured
                    ? 'text-amber-300 bg-amber-900/30 border-amber-700/40'
                    : 'text-gray-400 bg-gray-900/40 border-gray-700/40'
              }`}
            >
              {relayStatus.online
                ? t('connection.relay_status_online')
                : relayStatus.configured
                  ? t('connection.relay_status_configured_offline')
                  : t('connection.relay_status_unconfigured')}
            </span>
          </div>
          {relayStatus.endpoint ? (
            <p className="text-xs font-mono text-gray-200 break-all">{relayStatus.endpoint}</p>
          ) : (
            <p className="text-xs text-gray-500">{t('connection.relay_status_no_endpoint_hint')}</p>
          )}
          {!relayStatus.secret_configured && relayStatus.configured && (
            <p className="text-[11px] text-amber-300">{t('connection.relay_status_missing_secret')}</p>
          )}
        </section>
      )}

      <section className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-3">
        <h3 className="text-sm font-medium text-white">{t('connection.install_title')}</h3>
        <p className="text-xs text-gray-400">{t('connection.install_desc')}</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <label className="text-xs text-gray-300">
            {t('connection.worker_name')}
            <input
              className="mt-1 w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm font-mono text-white"
              value={workerName}
              onChange={e => setWorkerName(e.target.value)}
            />
          </label>
          <label className="text-xs text-gray-300">
            {t('connection.launcher')}
            <select
              className="mt-1 w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-white"
              value={launcher}
              onChange={e => setLauncher(e.target.value)}
            >
              <option value="cloud">cloud</option>
              <option value="cli">cli</option>
              <option value="service">service</option>
            </select>
          </label>
          <label className="text-xs text-gray-300">
            {t('connection.expires_minutes')}
            <input
              type="number"
              min={15}
              max={43200}
              className="mt-1 w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm font-mono text-white"
              value={expiresMinutes}
              onChange={e => setExpiresMinutes(e.target.value)}
            />
          </label>
          <label className="text-xs text-gray-300">
            {t('connection.connect_mode')}
            <select
              className="mt-1 w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-white"
              value={selectedMode}
              onChange={e => setSelectedMode(e.target.value)}
            >
              {modeOptions.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </label>
        </div>

        {selectedMode !== 'relay_session' && (
          <label className="block text-xs text-gray-300">
            {t('connection.server_url_override')}
            <input
              className="mt-1 w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm font-mono text-white"
              placeholder={info?.request_origin || 'https://...'}
              value={overrideUrl}
              onChange={e => setOverrideUrl(e.target.value)}
            />
          </label>
        )}

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={issueCommand}
            disabled={issuing}
            className="px-3 py-1.5 rounded text-xs font-medium bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white transition-colors"
          >
            {issuing ? t('connection.issuing') : t('connection.issue_command')}
          </button>
          {command?.linux_command && (
            <button
              type="button"
              onClick={copyCommand}
              className="px-3 py-1.5 rounded text-xs font-medium bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            >
              {t('connection.copy')}
            </button>
          )}
        </div>

        {commandError && (
          <p className="text-xs text-red-400">{commandError}</p>
        )}

        {command?.linux_command && (
          <pre className="bg-gray-950 border border-gray-800 rounded p-3 text-[11px] text-emerald-300 font-mono whitespace-pre-wrap break-all">
{command.linux_command}
          </pre>
        )}
      </section>
    </div>
  );
}
