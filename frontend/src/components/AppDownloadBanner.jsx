import React, { useState, useEffect, useCallback } from 'react';
import { Download, Monitor, X, Apple, ExternalLink, Copy, Check } from 'lucide-react';
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
  return <Monitor size={size} />;
}

const PLATFORM_LABELS = { mac: 'macOS', win: 'Windows', linux: 'Linux', unknown: 'Desktop' };

export default function AppDownloadBanner({ onShowDownload }) {
  const { t } = useLocale();
  const [downloads, setDownloads] = useState([]);
  const [dismissed, setDismissed] = useState(false);
  const [copied, setCopied] = useState(false);

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
        }
      } catch { /* ignore */ }
    };
    fetchDownloads();
  }, []);

  const handleCopyUrl = useCallback(() => {
    navigator.clipboard?.writeText(serverUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [serverUrl]);

  if (dismissed) return null;

  const recommended = downloads.find(f => f.platform === userPlatform);
  const others = downloads.filter(f => f !== recommended);
  const hasDownloads = downloads.length > 0;

  const handleDownload = (filename) => {
    const base = getServerUrl() || '';
    window.open(`${base}/api/v1/app/downloads/${encodeURIComponent(filename)}`, '_blank');
  };

  return (
    <div className="bg-gradient-to-r from-emerald-900/30 via-gray-900/50 to-blue-900/30 border-b border-emerald-700/20 px-4 py-2.5 flex-shrink-0">
      <div className="flex items-center gap-4">
        {/* Icon + Text */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="flex items-center justify-center w-7 h-7 rounded-md bg-emerald-600/20 flex-shrink-0">
            <Download size={14} className="text-emerald-400" />
          </div>
          <div className="min-w-0">
            <span className="text-xs font-medium text-emerald-300">{t('download.banner_title')}</span>
            <span className="text-[11px] text-gray-500 ml-2 hidden sm:inline">{t('download.banner_desc_short')}</span>
          </div>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Download buttons OR "no builds" notice */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {hasDownloads ? (
            <>
              {/* Primary download button */}
              {recommended && (
                <button
                  onClick={() => handleDownload(recommended.name)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium transition-colors"
                >
                  <PlatformIcon platform={recommended.platform} size={13} />
                  <span>{PLATFORM_LABELS[recommended.platform]}</span>
                  <Download size={12} />
                </button>
              )}
              {/* Other platform buttons (inline, not dropdown) */}
              {others.map(f => (
                <button
                  key={f.name}
                  onClick={() => handleDownload(f.name)}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-gray-700/60 hover:bg-gray-600/60 text-gray-300 hover:text-white text-xs transition-colors"
                >
                  <PlatformIcon platform={f.platform} size={12} />
                  <span>{PLATFORM_LABELS[f.platform]}</span>
                </button>
              ))}
            </>
          ) : (
            <span className="text-[11px] text-gray-500">{t('download.no_builds')}</span>
          )}

          {/* Server URL copy */}
          <div className="flex items-center gap-1 border-l border-gray-700 pl-2 ml-1">
            <code className="text-[11px] font-mono text-gray-400 max-w-[180px] truncate">{serverUrl}</code>
            <button
              onClick={handleCopyUrl}
              className="p-1 rounded text-gray-500 hover:text-emerald-400 transition-colors"
              title={t('download.copy_url')}
            >
              {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
            </button>
          </div>

          {/* View all downloads link */}
          {onShowDownload && (
            <button
              onClick={onShowDownload}
              className="text-[11px] text-emerald-500 hover:text-emerald-400 transition-colors whitespace-nowrap"
            >
              {t('download.view_all')}
            </button>
          )}

          {/* Dismiss */}
          <button
            onClick={() => setDismissed(true)}
            className="p-1 rounded text-gray-600 hover:text-gray-400 transition-colors"
          >
            <X size={12} />
          </button>
        </div>
      </div>
    </div>
  );
}
