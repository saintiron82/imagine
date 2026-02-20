import React, { useState, useEffect, useCallback } from 'react';
import { X, RotateCcw, Trash2, ChevronLeft, ChevronRight, AlertCircle } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { listJobs, cancelJob, retryFailedJobs, clearCompletedJobs } from '../api/admin';

const PAGE_SIZE = 20;

const STATUS_COLORS = {
  pending: 'bg-yellow-900/50 text-yellow-300 border-yellow-700/50',
  assigned: 'bg-blue-900/50 text-blue-300 border-blue-700/50',
  processing: 'bg-cyan-900/50 text-cyan-300 border-cyan-700/50',
  completed: 'bg-green-900/50 text-green-300 border-green-700/50',
  failed: 'bg-red-900/50 text-red-300 border-red-700/50',
  cancelled: 'bg-gray-800/50 text-gray-400 border-gray-600/50',
};

function PhaseIndicator({ phaseCompleted }) {
  const p = phaseCompleted?.parse;
  const v = phaseCompleted?.vision;
  const e = phaseCompleted?.embed;

  return (
    <div className="flex items-center gap-0.5">
      <span className={`w-2 h-2 rounded-full ${p ? 'bg-blue-400' : 'bg-gray-600'}`} title="Parse" />
      <span className={`w-2 h-2 rounded-full ${v ? 'bg-purple-400' : 'bg-gray-600'}`} title="Vision" />
      <span className={`w-2 h-2 rounded-full ${e ? 'bg-green-400' : 'bg-gray-600'}`} title="Embed" />
      <span className="text-[10px] text-gray-500 ml-1">
        {p && v && e ? 'Done' : p && v ? 'E' : p ? 'V' : ''}
      </span>
    </div>
  );
}

function timeAgo(dateStr) {
  if (!dateStr) return '';
  const date = new Date(dateStr + (dateStr.endsWith('Z') ? '' : 'Z'));
  const now = Date.now();
  const diff = Math.floor((now - date.getTime()) / 1000);
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

function fileName(filePath) {
  if (!filePath) return '';
  const parts = filePath.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1];
}

export default function QueueManagerPanel({ stats, onRefresh }) {
  const { t } = useLocale();
  const [filter, setFilter] = useState(null); // null = all
  const [jobs, setJobs] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);

  const fetchJobs = useCallback(async () => {
    setLoading(true);
    try {
      let result;
      if (isElectron && window.electron?.queue?.listJobs) {
        result = await window.electron.queue.listJobs({
          status: filter,
          limit: PAGE_SIZE,
          offset: page * PAGE_SIZE,
        });
      } else {
        result = await listJobs(filter, PAGE_SIZE, page * PAGE_SIZE);
      }
      if (result?.success !== false) {
        setJobs(result.jobs || []);
        setTotal(result.total || 0);
      }
    } catch { /* ignore */ }
    setLoading(false);
  }, [filter, page]);

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, [fetchJobs]);

  // Reset page when filter changes
  useEffect(() => { setPage(0); }, [filter]);

  const handleCancel = async (jobId) => {
    try {
      if (isElectron && window.electron?.queue?.cancelJob) {
        await window.electron.queue.cancelJob(jobId);
      } else {
        await cancelJob(jobId);
      }
      fetchJobs();
      onRefresh?.();
    } catch { /* ignore */ }
  };

  const handleRetryAll = async () => {
    try {
      if (isElectron && window.electron?.queue?.retryFailed) {
        await window.electron.queue.retryFailed();
      } else {
        await retryFailedJobs();
      }
      fetchJobs();
      onRefresh?.();
    } catch { /* ignore */ }
  };

  const handleClearCompleted = async () => {
    try {
      if (isElectron && window.electron?.queue?.clearCompleted) {
        await window.electron.queue.clearCompleted();
      } else {
        await clearCompletedJobs();
      }
      fetchJobs();
      onRefresh?.();
    } catch { /* ignore */ }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const from = total > 0 ? page * PAGE_SIZE + 1 : 0;
  const to = Math.min((page + 1) * PAGE_SIZE, total);

  const filters = [
    { key: null, label: t('queue.filter_all'), count: stats?.total },
    { key: 'pending', label: t('queue.filter_pending'), count: stats?.pending },
    { key: 'processing', label: t('queue.filter_processing'), count: (stats?.assigned || 0) + (stats?.processing || 0) },
    { key: 'completed', label: t('queue.filter_completed'), count: stats?.completed },
    { key: 'failed', label: t('queue.filter_failed'), count: stats?.failed },
  ];

  return (
    <div className="space-y-2 pt-2">
      {/* Filter tabs + bulk actions */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-1">
          {filters.map(({ key, label, count }) => (
            <button
              key={key ?? 'all'}
              onClick={() => setFilter(key === 'processing' ? null : key)}
              className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                filter === key || (key === null && filter === null)
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600/50 hover:text-gray-300'
              }`}
            >
              {label}
              {count != null && count > 0 && (
                <span className="ml-1 text-[10px] opacity-70">{count}</span>
              )}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          {(stats?.failed || 0) > 0 && (
            <button
              onClick={handleRetryAll}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-orange-900/40 text-orange-300 hover:bg-orange-800/50 transition-colors border border-orange-700/40"
            >
              <RotateCcw size={10} />
              {t('queue.action_retry_all')}
            </button>
          )}
          {(stats?.completed || 0) > 0 && (
            <button
              onClick={handleClearCompleted}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-gray-700/50 text-gray-400 hover:bg-gray-600/50 hover:text-gray-300 transition-colors"
            >
              <Trash2 size={10} />
              {t('queue.action_clear_completed')}
            </button>
          )}
        </div>
      </div>

      {/* Job list */}
      <div className="max-h-[300px] overflow-y-auto">
        {jobs.length === 0 && !loading ? (
          <div className="text-center py-6 text-xs text-gray-500">
            {t('queue.no_jobs')}
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-gray-800">
              <tr className="text-gray-500 text-[10px] uppercase">
                <th className="text-left py-1 px-2 font-medium">{t('queue.col_filename')}</th>
                <th className="text-center py-1 px-2 font-medium w-20">{t('queue.col_status')}</th>
                <th className="text-center py-1 px-2 font-medium w-16">{t('queue.col_phase')}</th>
                <th className="text-right py-1 px-2 font-medium w-12">{t('queue.col_time')}</th>
                <th className="text-center py-1 px-2 font-medium w-12">{t('queue.col_action')}</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.job_id} className="border-t border-gray-700/30 hover:bg-gray-700/20">
                  <td className="py-1 px-2 truncate max-w-[200px]" title={job.file_path}>
                    <span className="text-gray-300">{fileName(job.file_path)}</span>
                    {job.error_message && (
                      <div className="flex items-center gap-1 mt-0.5">
                        <AlertCircle size={9} className="text-red-400 flex-shrink-0" />
                        <span className="text-[10px] text-red-400/80 truncate">{job.error_message}</span>
                      </div>
                    )}
                  </td>
                  <td className="py-1 px-2 text-center">
                    <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium border ${STATUS_COLORS[job.status] || ''}`}>
                      {job.status}
                    </span>
                  </td>
                  <td className="py-1 px-2 text-center">
                    <PhaseIndicator phaseCompleted={job.phase_completed} />
                  </td>
                  <td className="py-1 px-2 text-right text-gray-500 text-[10px]">
                    {timeAgo(job.completed_at || job.started_at || job.created_at)}
                  </td>
                  <td className="py-1 px-2 text-center">
                    <div className="flex items-center justify-center gap-1">
                      {(job.status === 'pending' || job.status === 'assigned' || job.status === 'failed') && (
                        <button
                          onClick={() => handleCancel(job.job_id)}
                          className="p-0.5 rounded hover:bg-red-900/40 text-gray-500 hover:text-red-400 transition-colors"
                          title={t('queue.action_cancel')}
                        >
                          <X size={12} />
                        </button>
                      )}
                      {job.status === 'failed' && (
                        <button
                          onClick={() => handleCancel(job.job_id)}
                          className="p-0.5 rounded hover:bg-orange-900/40 text-gray-500 hover:text-orange-400 transition-colors"
                          title={t('queue.action_retry')}
                        >
                          <RotateCcw size={12} />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {total > PAGE_SIZE && (
        <div className="flex items-center justify-between pt-1">
          <span className="text-[10px] text-gray-500">
            {t('queue.showing', { from, to, total })}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="p-0.5 rounded hover:bg-gray-700 disabled:opacity-30 text-gray-400"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="text-[10px] text-gray-500 px-1">
              {page + 1}/{totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="p-0.5 rounded hover:bg-gray-700 disabled:opacity-30 text-gray-400"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
