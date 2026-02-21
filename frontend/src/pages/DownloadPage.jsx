/**
 * DownloadPage — dedicated download page for desktop app installers.
 *
 * Accessible from LoginPage and AppDownloadBanner (web mode).
 * Shows platform-specific downloads, server URL, and setup instructions.
 */

import { useState, useEffect, useCallback } from 'react';
import { Download, Monitor, Apple, ArrowLeft, Copy, Check, Server, ChevronRight, ExternalLink } from 'lucide-react';
import { useLocale } from '../i18n';
import { getServerUrl } from '../api/client';

function detectPlatform() {
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes('mac') || ua.includes('darwin')) return 'mac';
  if (ua.includes('win')) return 'win';
  if (ua.includes('linux')) return 'linux';
  return 'unknown';
}

function PlatformIcon({ platform, size = 20 }) {
  if (platform === 'mac') return <Apple size={size} />;
  return <Monitor size={size} />;
}

const PLATFORM_INFO = {
  mac: { label: 'macOS', ext: '.dmg', color: 'from-gray-600 to-gray-700', iconBg: 'bg-gray-500/20' },
  win: { label: 'Windows', ext: '.exe', color: 'from-blue-600 to-blue-700', iconBg: 'bg-blue-500/20' },
  linux: { label: 'Linux', ext: '.AppImage', color: 'from-orange-600 to-orange-700', iconBg: 'bg-orange-500/20' },
};

export default function DownloadPage({ onBack }) {
  const { t } = useLocale();
  const [downloads, setDownloads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  const userPlatform = detectPlatform();
  const serverUrl = getServerUrl() || window.location.origin;

  useEffect(() => {
    const fetchDownloads = async () => {
      try {
        const base = getServerUrl() || '';
        const resp = await fetch(`${base}/api/v1/app/downloads`);
        if (resp.ok) {
          const data = await resp.json();
          setDownloads(data.files || []);
        }
      } catch { /* ignore */ }
      setLoading(false);
    };
    fetchDownloads();
  }, []);

  const handleCopyUrl = useCallback(() => {
    navigator.clipboard?.writeText(serverUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [serverUrl]);

  const handleDownload = (filename) => {
    const base = getServerUrl() || '';
    window.open(`${base}/api/v1/app/downloads/${encodeURIComponent(filename)}`, '_blank');
  };

  const recommended = downloads.find(f => f.platform === userPlatform);
  const others = downloads.filter(f => f !== recommended);

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="border-b border-gray-800 px-6 py-4 flex items-center gap-4">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <ArrowLeft size={16} />
          <span>{t('download.back')}</span>
        </button>
        <div className="w-px h-5 bg-gray-700" />
        <h1 className="text-lg font-bold text-white">Imagine</h1>
      </div>

      {/* Content */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-2xl">
          {/* Title */}
          <div className="text-center mb-10">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-emerald-600/10 border border-emerald-600/20 mb-4">
              <Download size={28} className="text-emerald-400" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">{t('download.page_title')}</h2>
            <p className="text-gray-400 text-sm max-w-md mx-auto">{t('download.page_desc')}</p>
          </div>

          {/* Downloads Grid */}
          {loading ? (
            <div className="text-center text-gray-500 py-12">{t('status.loading')}</div>
          ) : downloads.length > 0 ? (
            <div className="space-y-3 mb-8">
              {/* Recommended (user's platform) */}
              {recommended && (
                <button
                  onClick={() => handleDownload(recommended.name)}
                  className="w-full group relative overflow-hidden rounded-xl border border-emerald-600/30 bg-gradient-to-r from-emerald-900/30 to-emerald-800/20 hover:from-emerald-900/50 hover:to-emerald-800/30 transition-all p-5"
                >
                  <div className="flex items-center gap-4">
                    <div className={`flex items-center justify-center w-12 h-12 rounded-xl ${PLATFORM_INFO[recommended.platform]?.iconBg || 'bg-gray-500/20'}`}>
                      <PlatformIcon platform={recommended.platform} size={24} />
                    </div>
                    <div className="flex-1 text-left">
                      <div className="flex items-center gap-2">
                        <span className="text-white font-semibold text-lg">
                          {t('download.download_for', { platform: PLATFORM_INFO[recommended.platform]?.label || 'Desktop' })}
                        </span>
                        <span className="px-2 py-0.5 rounded-full bg-emerald-600/20 text-emerald-400 text-[10px] font-bold uppercase">
                          {t('download.recommended')}
                        </span>
                      </div>
                      <div className="text-gray-400 text-sm mt-0.5">
                        {recommended.name} ({recommended.size_display})
                      </div>
                    </div>
                    <Download size={20} className="text-emerald-400 group-hover:translate-y-0.5 transition-transform" />
                  </div>
                </button>
              )}

              {/* Other platforms */}
              {others.map(f => {
                const info = PLATFORM_INFO[f.platform] || PLATFORM_INFO.win;
                return (
                  <button
                    key={f.name}
                    onClick={() => handleDownload(f.name)}
                    className="w-full group rounded-xl border border-gray-700/50 bg-gray-800/30 hover:bg-gray-800/60 hover:border-gray-600/50 transition-all p-4"
                  >
                    <div className="flex items-center gap-4">
                      <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${info.iconBg}`}>
                        <PlatformIcon platform={f.platform} size={20} />
                      </div>
                      <div className="flex-1 text-left">
                        <span className="text-gray-200 font-medium">
                          {t('download.download_for', { platform: info.label })}
                        </span>
                        <div className="text-gray-500 text-xs mt-0.5">
                          {f.name} ({f.size_display})
                        </div>
                      </div>
                      <Download size={16} className="text-gray-500 group-hover:text-gray-300 transition-colors" />
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-12 mb-8">
              <div className="text-gray-500 text-sm mb-2">{t('download.no_builds')}</div>
              <div className="text-gray-600 text-xs">{t('download.no_builds_desc')}</div>
            </div>
          )}

          {/* Server URL + Setup Instructions */}
          <div className="rounded-xl border border-gray-700/50 bg-gray-800/30 p-5">
            <div className="flex items-center gap-2 mb-3">
              <Server size={16} className="text-blue-400" />
              <span className="text-sm font-medium text-gray-300">{t('download.setup_title')}</span>
            </div>

            {/* Server URL */}
            <div className="flex items-center gap-2 bg-gray-900/60 rounded-lg px-4 py-3 mb-4">
              <code className="flex-1 text-sm font-mono text-blue-300 truncate">{serverUrl}</code>
              <button
                onClick={handleCopyUrl}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-gray-700 hover:bg-gray-600 text-xs text-gray-300 hover:text-white transition-colors flex-shrink-0"
              >
                {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                <span>{copied ? t('admin.worker_token_copied') : t('download.copy_url')}</span>
              </button>
            </div>

            {/* Steps */}
            <div className="space-y-2.5">
              {['download.step_1', 'download.step_2', 'download.step_3'].map((key, i) => (
                <div key={key} className="flex items-start gap-3">
                  <div className="flex items-center justify-center w-5 h-5 rounded-full bg-blue-600/20 text-blue-400 text-xs font-bold flex-shrink-0 mt-0.5">
                    {i + 1}
                  </div>
                  <span className="text-sm text-gray-400">{t(key)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-center text-xs text-gray-600 py-4">
        Imagine Server
      </div>
    </div>
  );
}
