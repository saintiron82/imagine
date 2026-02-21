import React, { useState } from 'react';
import { Copy, Check, Globe, Wifi, QrCode } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';
import { useLocale } from '../i18n';

const ServerInfoPanel = ({
  serverPort,
  serverLanUrl,
  serverLanAddresses = [],
  tunnelUrl,
  tunnelRunning,
  tunnelDownloading,
  onTunnelStart,
  onTunnelStop,
  onClose,
}) => {
  const { t } = useLocale();
  const [copied, setCopied] = useState(null);
  const [showQr, setShowQr] = useState(false);

  const copyUrl = (url, key) => {
    navigator.clipboard?.writeText(url);
    setCopied(key);
    setTimeout(() => setCopied(null), 2000);
  };

  const localUrl = `http://localhost:${serverPort}`;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-40" onClick={onClose} />

      {/* Panel */}
      <div className="absolute right-0 top-full mt-1 w-80 bg-gray-800 border border-gray-600 rounded-lg shadow-xl z-50 p-4">
        <h3 className="text-sm font-bold text-white mb-3">{t('server.info_title')}</h3>

        {/* Local URL */}
        <div className="mb-3">
          <label className="text-[10px] text-gray-500 uppercase tracking-wide">
            {t('server.local_url')}
          </label>
          <div className="flex items-center gap-2 mt-1">
            <code className="flex-1 text-xs font-mono text-gray-300 bg-gray-900 px-2 py-1.5 rounded truncate">
              {localUrl}
            </code>
            <button
              onClick={() => copyUrl(localUrl, 'local')}
              className="p-1 rounded hover:bg-gray-700 transition-colors"
            >
              {copied === 'local'
                ? <Check size={14} className="text-green-400" />
                : <Copy size={14} className="text-gray-400" />}
            </button>
          </div>
        </div>

        {/* LAN URL */}
        {serverLanUrl && (
          <div className="mb-3">
            <label className="text-[10px] text-gray-500 uppercase tracking-wide flex items-center gap-1">
              <Wifi size={10} />
              {t('server.lan_url')}
            </label>
            <div className="flex items-center gap-2 mt-1">
              <code className="flex-1 text-xs font-mono text-green-300 bg-gray-900 px-2 py-1.5 rounded truncate">
                {serverLanUrl}
              </code>
              <button
                onClick={() => copyUrl(serverLanUrl, 'lan')}
                className="p-1 rounded hover:bg-gray-700 transition-colors"
              >
                {copied === 'lan'
                  ? <Check size={14} className="text-green-400" />
                  : <Copy size={14} className="text-gray-400" />}
              </button>
              <button
                onClick={() => setShowQr(!showQr)}
                className="p-1 rounded hover:bg-gray-700 transition-colors"
              >
                <QrCode size={14} className={showQr ? 'text-green-400' : 'text-gray-400'} />
              </button>
            </div>
            {/* Multiple LAN addresses */}
            {serverLanAddresses.length > 1 && (
              <div className="mt-1 text-[10px] text-gray-500">
                {serverLanAddresses.map((a, i) => (
                  <span key={i} className="mr-2">{a.name}: {a.address}</span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* QR Code */}
        {showQr && serverLanUrl && (
          <div className="mb-3 flex justify-center p-3 bg-white rounded-lg">
            <QRCodeSVG value={serverLanUrl} size={160} />
          </div>
        )}

        {/* Tunnel Section */}
        <div className="border-t border-gray-700 pt-3 mt-3">
          <label className="text-[10px] text-gray-500 uppercase tracking-wide flex items-center gap-1">
            <Globe size={10} />
            {t('server.tunnel_title')}
          </label>
          {tunnelRunning && tunnelUrl ? (
            <div className="mt-1">
              <div className="flex items-center gap-2">
                <code className="flex-1 text-xs font-mono text-blue-300 bg-gray-900 px-2 py-1.5 rounded truncate">
                  {tunnelUrl}
                </code>
                <button
                  onClick={() => copyUrl(tunnelUrl, 'tunnel')}
                  className="p-1 rounded hover:bg-gray-700 transition-colors"
                >
                  {copied === 'tunnel'
                    ? <Check size={14} className="text-green-400" />
                    : <Copy size={14} className="text-gray-400" />}
                </button>
              </div>
              {/* QR for tunnel URL */}
              <div className="mt-2 flex justify-center p-2 bg-white rounded-lg">
                <QRCodeSVG value={tunnelUrl} size={120} />
              </div>
              <button
                onClick={onTunnelStop}
                className="mt-2 w-full px-3 py-1.5 text-xs text-red-400 border border-red-700 rounded hover:bg-red-900/30 transition-colors"
              >
                {t('server.tunnel_stop')}
              </button>
            </div>
          ) : (
            <div className="mt-1">
              <p className="text-[11px] text-gray-400 mb-2">{t('server.tunnel_desc')}</p>
              <button
                onClick={onTunnelStart}
                disabled={tunnelDownloading}
                className="w-full px-3 py-1.5 text-xs text-blue-300 border border-blue-600 rounded hover:bg-blue-900/30 transition-colors disabled:opacity-50 disabled:cursor-wait"
              >
                <Globe size={12} className="inline mr-1" />
                {tunnelDownloading ? t('server.tunnel_connecting') : t('server.tunnel_start')}
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ServerInfoPanel;
