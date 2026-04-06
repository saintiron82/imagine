/**
 * ToolsPanel — admin tools (parse data repair, etc.)
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useLocale } from '../../i18n';
import { startRepairParse, getRepairParseStatus } from '../../api/admin';

const POLL_INTERVAL = 5000;

export default function ToolsPanel() {
  const { t } = useLocale();
  const [status, setStatus] = useState(null); // null | { running, done, total, current_file, failed, skipped }
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const poll = useCallback(async () => {
    try {
      const resp = await getRepairParseStatus();
      const data = resp.data ?? resp;
      setStatus(data);
      if (!data.running) {
        stopPolling();
      }
    } catch {
      stopPolling();
    }
  }, [stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    timerRef.current = setInterval(poll, POLL_INTERVAL);
  }, [poll, stopPolling]);

  // Check status on mount (in case a repair is already running)
  useEffect(() => {
    poll();
    return stopPolling;
  }, [poll, stopPolling]);

  const handleStart = async () => {
    setError(null);
    try {
      const resp = await startRepairParse();
      const data = resp.data ?? resp;
      if (data.success) {
        // Start returned, begin polling for real status
        setStatus({ running: true, progress: { done: 0, failed: 0, total: data.total, current_file: '' } });
        startPolling();
      } else {
        setError(data.detail || 'Start failed');
      }
    } catch (err) {
      setError(err?.response?.data?.detail ?? err.message ?? 'Unknown error');
    }
  };

  // Normalize: status API returns {running, progress: {done, total, ...}}
  const prog = status?.progress ?? status ?? {};
  const isRunning = status?.running === true;
  const isFinished = status && !status.running && (prog.done > 0 || prog.failed > 0);
  const pct = prog.total > 0 ? ((prog.done / prog.total) * 100).toFixed(1) : '0.0';

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">{t('admin.tools_title')}</h2>

      {/* Parse Data Repair card */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-3">
        <div>
          <h3 className="text-sm font-medium text-white">{t('admin.tools_repair_parse_title')}</h3>
          <p className="text-xs text-gray-400 mt-1">{t('admin.tools_repair_parse_desc')}</p>
        </div>

        {error && (
          <p className="text-xs text-red-400">{error}</p>
        )}

        {isRunning ? (
          <div className="space-y-1">
            <p className="text-xs text-yellow-400 font-medium">{t('admin.tools_repair_parse_running')}</p>
            <div className="flex items-center gap-3">
              <p className="text-xs font-mono text-emerald-400">
                {prog.done}/{prog.total} ({pct}%)
              </p>
              {(prog.skipped ?? 0) > 0 && (
                <span className="text-xs text-gray-500">{prog.skipped} {t('admin.tools_repair_parse_skipped')}</span>
              )}
              {(prog.failed ?? 0) > 0 && (
                <span className="text-xs text-red-400">{prog.failed} {t('admin.tools_repair_parse_failed')}</span>
              )}
            </div>
            {/* Progress bar */}
            <div className="w-full bg-gray-700 rounded-full h-1.5">
              <div className="bg-emerald-500 h-1.5 rounded-full transition-all" style={{ width: `${Math.min(100, ((prog.done + (prog.skipped ?? 0) + (prog.failed ?? 0)) / Math.max(1, prog.total)) * 100)}%` }} />
            </div>
            {prog.current_file && (
              <p className="text-[10px] text-gray-500 truncate">{prog.current_file}</p>
            )}
          </div>
        ) : isFinished ? (
          <div className="space-y-2">
            <p className="text-xs font-medium text-emerald-400">
              {t('admin.tools_repair_parse_done')}
            </p>
            <div className="flex gap-3 text-xs font-mono">
              <span className="text-emerald-400">{prog.done} repaired</span>
              {(prog.skipped ?? 0) > 0 && <span className="text-gray-500">{prog.skipped} skipped</span>}
              {(prog.failed ?? 0) > 0 && <span className="text-red-400">{prog.failed} failed</span>}
            </div>
            <button
              onClick={handleStart}
              className="px-3 py-1.5 rounded text-xs font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors"
            >
              {t('admin.tools_repair_parse_start')}
            </button>
          </div>
        ) : (
          <button
            onClick={handleStart}
            className="px-3 py-1.5 rounded text-xs font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors"
          >
            {t('admin.tools_repair_parse_start')}
          </button>
        )}
      </div>
    </div>
  );
}
