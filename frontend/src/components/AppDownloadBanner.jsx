import React, { useState, useEffect } from 'react';
import { Download, Monitor, X, Apple, ChevronDown } from 'lucide-react';
import { useLocale } from '../i18n';
import { getServerUrl } from '../api/client';

function detectUserPlatform() {
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes('mac') || ua.includes('darwin')) return 'mac';
  if (ua.includes('win')) return 'win';
  if (ua.includes('linux')) return 'linux';
  return 'unknown';
}

function PlatformIcon({ platform, size = 14 }) {
  if (platform === 'mac') return <Apple size={size} />;
  if (platform === 'win') return <Monitor size={size} />;
  // Linux — use Monitor as fallback
  return <Monitor size={size} />;
}

const PLATFORM_LABELS = { mac: 'macOS', win: 'Windows', linux: 'Linux' };

export default function AppDownloadBanner({ compact = false }) {
  const { t } = useLocale();
  const [downloads, setDownloads] = useState([]);
  const [available, setAvailable] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [loading, setLoading] = useState(true);

  const userPlatform = detectUserPlatform();
  const serverUrl = getServerUrl() || window.location.origin;

  useEffect(() => {
    const fetchDownloads = async () => {
      try {
        const base = getServerUrl() || '';
        const resp = await fetch(`${base}/api/v1/app/downloads`);
        if (resp.ok) {
          const data = await resp.json();
          setDownloads(data.files || []);
          setAvailable(data.available || false);
        }
      } catch {
        // Server may not have builds available
      }
      setLoading(false);
    };
    fetchDownloads();
  }, []);

  if (dismissed || loading || !available) return null;

  const recommended = downloads.find(f => f.platform === userPlatform);
  const others = downloads.filter(f => f !== recommended);

  const handleDownload = (filename) => {
    const base = getServerUrl() || '';
    window.open(`${base}/api/v1/app/downloads/${encodeURIComponent(filename)}`, '_blank');
  };

  // Compact mode: small inline button (for header bar)
  if (compact) {
    return (
      <button
        onClick={() => recommended && handleDownload(recommended.name)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded text-xs font-medium bg-emerald-700/60 text-emerald-300 hover:bg-emerald-600/60 transition-colors"
        title={t('download.get_app')}
      >
        <Download size={14} />
        <span>{t('download.get_app')}</span>
      </button>
    );
  }

  // Full banner mode
  return (
    <div className="bg-gradient-to-r from-emerald-900/40 to-blue-900/40 border-b border-emerald-700/30 px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-emerald-600/30 flex-shrink-0">
            <Download size={16} className="text-emerald-400" />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-medium text-emerald-200">
              {t('download.banner_title')}
            </div>
            <div className="text-xs text-gray-400 mt-0.5">
              {t('download.banner_desc')}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0 ml-4">
          {/* Recommended download for user's platform */}
          {recommended && (
            <button
              onClick={() => handleDownload(recommended.name)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors shadow-lg shadow-emerald-900/30"
            >
              <PlatformIcon platform={recommended.platform} />
              <span>{t('download.download_for', { platform: PLATFORM_LABELS[recommended.platform] || recommended.platform })}</span>
              <span className="text-emerald-200 text-xs">({recommended.size_display})</span>
            </button>
          )}

          {/* Other platforms dropdown */}
          {others.length > 0 && (
            <div className="relative">
              <button
                onClick={() => setShowAll(!showAll)}
                className="flex items-center gap-1 px-2 py-2 rounded-lg text-xs text-gray-400 hover:text-white hover:bg-gray-700/50 transition-colors"
                title={t('download.other_platforms')}
              >
                <ChevronDown size={14} className={`transition-transform ${showAll ? 'rotate-180' : ''}`} />
              </button>
              {showAll && (
                <>
                  <div className="fixed inset-0 z-40" onClick={() => setShowAll(false)} />
                  <div className="absolute right-0 top-full mt-1 z-50 bg-gray-800 border border-gray-600 rounded-lg shadow-xl py-1 min-w-[200px]">
                    {others.map(f => (
                      <button
                        key={f.name}
                        onClick={() => { handleDownload(f.name); setShowAll(false); }}
                        className="w-full flex items-center gap-2 px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                      >
                        <PlatformIcon platform={f.platform} />
                        <span className="flex-1 text-left">{PLATFORM_LABELS[f.platform] || f.platform}</span>
                        <span className="text-gray-500 text-xs">{f.size_display}</span>
                      </button>
                    ))}
                    <div className="border-t border-gray-700 mx-2 my-1" />
                    <div className="px-4 py-2 text-[10px] text-gray-500">
                      {t('download.server_url_hint')}: <span className="font-mono text-gray-400">{serverUrl}</span>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Dismiss */}
          <button
            onClick={() => setDismissed(true)}
            className="p-1 rounded text-gray-500 hover:text-gray-300 hover:bg-gray-700/50 transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Server URL hint for client configuration */}
      <div className="mt-2 flex items-center gap-2 text-[11px] text-gray-500">
        <span>{t('download.server_url_hint')}:</span>
        <code className="bg-gray-800/80 px-2 py-0.5 rounded font-mono text-gray-400">{serverUrl}</code>
      </div>
    </div>
  );
}
