/**
 * SubscriptionPanel — Admin subscription status panel.
 *
 * Shows: plan, status, user count, file count, expiry, days remaining.
 * Actions: Manage Subscription (Stripe Portal), Enter License Key (offline).
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../i18n';
import { getLicenseInfo, activateLicense, refreshLicense } from '../api/license';
import {
  Shield, Users, HardDrive, Calendar, RefreshCw,
  Key, Loader2, CheckCircle, AlertTriangle, Crown,
} from 'lucide-react';

const STATUS_COLORS = {
  active: 'text-emerald-400',
  grace_period: 'text-amber-400',
  expired: 'text-red-400',
  suspended: 'text-red-400',
  free: 'text-zinc-400',
};

const STATUS_BG = {
  active: 'bg-emerald-500/10 border-emerald-500/30',
  grace_period: 'bg-amber-500/10 border-amber-500/30',
  expired: 'bg-red-500/10 border-red-500/30',
  suspended: 'bg-red-500/10 border-red-500/30',
  free: 'bg-zinc-500/10 border-zinc-500/30',
};

function StatCard({ icon: Icon, label, value, sub, color = 'text-zinc-300' }) {
  return (
    <div className="bg-zinc-800/60 border border-zinc-700/50 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-1">
        <Icon size={14} className="text-zinc-500" />
        <span className="text-[11px] text-zinc-500">{label}</span>
      </div>
      <p className={`text-lg font-semibold ${color}`}>{value}</p>
      {sub && <p className="text-[10px] text-zinc-600 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function SubscriptionPanel() {
  const { t } = useLocale();
  const [license, setLicense] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [keyInput, setKeyInput] = useState('');
  const [keyError, setKeyError] = useState('');
  const [keySuccess, setKeySuccess] = useState(false);

  const fetchLicense = useCallback(async () => {
    try {
      const data = await getLicenseInfo();
      setLicense(data.license);
    } catch (e) {
      console.error('Failed to fetch license:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLicense();
  }, [fetchLicense]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refreshLicense();
      await fetchLicense();
    } catch (e) {
      console.error('License refresh failed:', e);
    } finally {
      setRefreshing(false);
    }
  };

  const handleActivateKey = async () => {
    setKeyError('');
    setKeySuccess(false);
    if (!keyInput.trim()) {
      setKeyError(t('subscription.key_required'));
      return;
    }
    try {
      await activateLicense(keyInput.trim());
      setKeySuccess(true);
      setKeyInput('');
      await fetchLicense();
    } catch (e) {
      setKeyError(e.detail || e.message || 'Invalid key');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 size={20} className="animate-spin text-zinc-500" />
      </div>
    );
  }

  if (!license) return null;

  const usersPercent = license.max_users > 0 ? Math.round((license.current_users / license.max_users) * 100) : 0;
  const filesPercent = license.max_files > 0 ? Math.round((license.current_files / license.max_files) * 100) : 0;

  return (
    <div className="space-y-4">
      {/* Status badge */}
      <div className={`flex items-center justify-between p-3 rounded-lg border ${STATUS_BG[license.status]}`}>
        <div className="flex items-center gap-2">
          <Crown size={16} className={STATUS_COLORS[license.status]} />
          <div>
            <p className={`text-sm font-medium ${STATUS_COLORS[license.status]}`}>
              {license.plan_id === 'free' ? t('subscription.free_plan') : license.plan_id}
            </p>
            <p className="text-[10px] text-zinc-500">
              {t(`subscription.status_${license.status}`)}
            </p>
          </div>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="text-zinc-500 hover:text-zinc-300 transition-colors p-1 disabled:opacity-50"
          title={t('subscription.refresh')}
        >
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2">
        <StatCard
          icon={Users}
          label={t('subscription.users')}
          value={`${license.current_users} / ${license.max_users}`}
          sub={`${usersPercent}%`}
          color={usersPercent >= 90 ? 'text-amber-400' : 'text-zinc-300'}
        />
        <StatCard
          icon={HardDrive}
          label={t('subscription.files')}
          value={license.current_files.toLocaleString()}
          sub={`/ ${license.max_files.toLocaleString()}`}
        />
        <StatCard
          icon={Calendar}
          label={t('subscription.expires')}
          value={license.expires_at ? new Date(license.expires_at).toLocaleDateString() : '\u2014'}
          sub={license.days_remaining !== null ? `${license.days_remaining}d` : ''}
          color={license.days_remaining !== null && license.days_remaining <= 7 ? 'text-amber-400' : 'text-zinc-300'}
        />
        <StatCard
          icon={Shield}
          label={t('subscription.features')}
          value={license.features?.length || 0}
          sub={license.features?.join(', ') || t('subscription.basic_only')}
        />
      </div>

      {/* Offline license key */}
      <div className="border-t border-zinc-700/40 pt-3">
        {!showKeyInput ? (
          <button
            onClick={() => setShowKeyInput(true)}
            className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <Key size={13} />
            {t('subscription.enter_license_key')}
          </button>
        ) : (
          <div className="space-y-2">
            <div className="flex gap-2">
              <input
                type="text"
                value={keyInput}
                onChange={e => setKeyInput(e.target.value)}
                placeholder={t('subscription.license_key_placeholder')}
                className="flex-1 px-3 py-2 bg-zinc-800/60 border border-zinc-700/60 rounded-lg
                  text-xs text-zinc-100 placeholder-zinc-500
                  focus:outline-none focus:border-blue-500/60"
              />
              <button
                onClick={handleActivateKey}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs rounded-lg transition-colors"
              >
                {t('subscription.activate')}
              </button>
            </div>
            {keyError && (
              <div className="flex items-center gap-1 text-xs text-red-400">
                <AlertTriangle size={12} />
                {keyError}
              </div>
            )}
            {keySuccess && (
              <div className="flex items-center gap-1 text-xs text-emerald-400">
                <CheckCircle size={12} />
                {t('subscription.key_activated')}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
