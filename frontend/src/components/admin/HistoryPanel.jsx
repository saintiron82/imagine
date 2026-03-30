/**
 * HistoryPanel — work request history + job drill-down.
 * Extracted from AdminPage.jsx.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import { listHistorySessions, listHistoryJobs } from '../../api/admin';
import {
  ChevronRight, ChevronLeft, AlertCircle, Folder,
} from 'lucide-react';


const HISTORY_PAGE_SIZE = 20;

const HISTORY_STATUS_COLORS = {
  completed: 'bg-green-900/50 text-green-300 border-green-700/50',
  failed: 'bg-red-900/50 text-red-300 border-red-700/50',
  cancelled: 'bg-gray-800/50 text-gray-400 border-gray-600/50',
  pending: 'bg-yellow-900/50 text-yellow-300 border-yellow-700/50',
  assigned: 'bg-blue-900/50 text-blue-300 border-blue-700/50',
  processing: 'bg-cyan-900/50 text-cyan-300 border-cyan-700/50',
};

const WR_STATUS_COLORS = {
  queued: 'bg-yellow-900/50 text-yellow-300',
  processing: 'bg-cyan-900/50 text-cyan-300',
  completed: 'bg-green-900/50 text-green-300',
  paused: 'bg-gray-700/50 text-gray-400',
  cancelled: 'bg-red-900/50 text-red-300',
};

function formatDuration(startStr, endStr) {
  if (!startStr || !endStr) return '-';
  const start = new Date(startStr + (startStr.endsWith('Z') ? '' : 'Z'));
  const end = new Date(endStr + (endStr.endsWith('Z') ? '' : 'Z'));
  const sec = Math.floor((end - start) / 1000);
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return `${h}h ${m}m`;
}

function formatTime(dateStr) {
  if (!dateStr) return '-';
  const d = new Date(dateStr + (dateStr.endsWith('Z') ? '' : 'Z'));
  const pad = n => String(n).padStart(2, '0');
  return `${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}


export default function HistoryPanel() {
  const { t } = useLocale();
  const [sessions, setSessions] = useState([]);
  const [sessionTotal, setSessionTotal] = useState(0);
  const [sessionPage, setSessionPage] = useState(0);
  const [selectedSession, setSelectedSession] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [jobTotal, setJobTotal] = useState(0);
  const [jobPage, setJobPage] = useState(0);
  const [jobFilter, setJobFilter] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch sessions
  const fetchSessions = useCallback(async () => {
    setLoading(true);
    try {
      const res = await listHistorySessions(HISTORY_PAGE_SIZE, sessionPage * HISTORY_PAGE_SIZE);
      if (res?.success !== false) {
        setSessions(res.sessions || []);
        setSessionTotal(res.total || 0);
      }
    } catch { /* ignore */ }
    setLoading(false);
  }, [sessionPage]);

  useEffect(() => { fetchSessions(); }, [fetchSessions]);

  // Fetch jobs for selected session
  const fetchJobs = useCallback(async () => {
    if (selectedSession === null) return;
    setLoading(true);
    try {
      const res = await listHistoryJobs(
        selectedSession.id, jobFilter,
        HISTORY_PAGE_SIZE, jobPage * HISTORY_PAGE_SIZE
      );
      if (res?.success !== false) {
        setJobs(res.jobs || []);
        setJobTotal(res.total || 0);
      }
    } catch { /* ignore */ }
    setLoading(false);
  }, [selectedSession, jobFilter, jobPage]);

  useEffect(() => { fetchJobs(); }, [fetchJobs]);
  useEffect(() => { setJobPage(0); }, [jobFilter]);

  const sessionPages = Math.ceil(sessionTotal / HISTORY_PAGE_SIZE);
  const jobPages = Math.ceil(jobTotal / HISTORY_PAGE_SIZE);

  // ── Job detail view (drill-down) ──
  if (selectedSession) {
    const s = selectedSession;
    const jobFilters = [
      { key: null, label: t('history.filter_all') },
      { key: 'completed', label: t('history.filter_completed') },
      { key: 'failed', label: t('history.filter_failed') },
    ];

    return (
      <div className="space-y-3">
        {/* Back button + session summary */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => { setSelectedSession(null); setJobs([]); setJobPage(0); setJobFilter(null); }}
            className="flex items-center gap-1 px-2 py-1 rounded text-xs text-gray-400 hover:text-white hover:bg-gray-700/50 transition-colors"
          >
            <ChevronLeft size={14} />
            {t('history.back_to_sessions')}
          </button>
          <div className="text-sm text-white font-medium truncate">{s.name || s.source_path || `Session #${s.id}`}</div>
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${WR_STATUS_COLORS[s.status] || ''}`}>
            {s.status}
          </span>
        </div>

        {/* Summary bar */}
        <div className="flex items-center gap-4 text-xs text-gray-400 bg-gray-800/50 rounded px-3 py-2">
          <span>{t('history.total_files')}: <span className="text-white">{s.total_files}</span></span>
          <span>{t('history.completed')}: <span className="text-green-400">{s.completed_count}</span></span>
          <span>{t('history.failed')}: <span className="text-red-400">{s.failed_count}</span></span>
          <span>{t('history.duration')}: <span className="text-white">{formatDuration(s.started_at, s.completed_at)}</span></span>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-1">
          {jobFilters.map(({ key, label }) => (
            <button
              key={key ?? 'all'}
              onClick={() => setJobFilter(key)}
              className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                jobFilter === key
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600/50'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Jobs table */}
        <div className="max-h-[400px] overflow-y-auto">
          {jobs.length === 0 && !loading ? (
            <div className="text-center py-6 text-xs text-gray-500">{t('queue.no_jobs')}</div>
          ) : (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-gray-800">
                <tr className="text-gray-500 text-[10px] uppercase">
                  <th className="text-left py-1 px-2 font-medium">{t('history.col_file')}</th>
                  <th className="text-center py-1 px-2 font-medium w-20">{t('history.col_status')}</th>
                  <th className="text-center py-1 px-2 font-medium w-16">{t('history.col_phase')}</th>
                  <th className="text-left py-1 px-2 font-medium w-20">{t('history.col_worker')}</th>
                  <th className="text-right py-1 px-2 font-medium w-24">{t('history.col_started')}</th>
                  <th className="text-right py-1 px-2 font-medium w-24">{t('history.col_finished')}</th>
                  <th className="text-center py-1 px-2 font-medium w-12">{t('history.col_retries')}</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => {
                  const fname = job.file_path ? job.file_path.replace(/\\/g, '/').split('/').pop() : '';
                  const p = job.phase_completed;
                  return (
                    <tr key={job.job_id} className="border-t border-gray-700/30 hover:bg-gray-700/20">
                      <td className="py-1 px-2 truncate max-w-[200px]" title={job.file_path}>
                        <span className="text-gray-300">{fname}</span>
                        {job.error_message && (
                          <div className="flex items-center gap-1 mt-0.5">
                            <AlertCircle size={9} className="text-red-400 flex-shrink-0" />
                            <span className="text-[10px] text-red-400/80 truncate">{job.error_message}</span>
                          </div>
                        )}
                      </td>
                      <td className="py-1 px-2 text-center">
                        <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium border ${HISTORY_STATUS_COLORS[job.status] || ''}`}>
                          {job.status}
                        </span>
                      </td>
                      <td className="py-1 px-2 text-center">
                        <div className="flex items-center justify-center gap-0.5">
                          <span className={`w-2 h-2 rounded-full ${p?.parse ? 'bg-blue-400' : 'bg-gray-600'}`} title="Parse" />
                          <span className={`w-2 h-2 rounded-full ${p?.vision ? 'bg-purple-400' : 'bg-gray-600'}`} title="Vision" />
                          <span className={`w-2 h-2 rounded-full ${p?.embed ? 'bg-green-400' : 'bg-gray-600'}`} title="Embed" />
                        </div>
                      </td>
                      <td className="py-1 px-2 text-gray-400 truncate max-w-[80px]" title={job.worker_name}>
                        {job.worker_name || '-'}
                      </td>
                      <td className="py-1 px-2 text-right text-gray-500 text-[10px]">
                        {formatTime(job.started_at)}
                      </td>
                      <td className="py-1 px-2 text-right text-gray-500 text-[10px]">
                        {formatTime(job.completed_at)}
                      </td>
                      <td className="py-1 px-2 text-center text-gray-500">
                        {job.retry_count > 0 ? <span className="text-orange-400">{job.retry_count}</span> : '-'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination */}
        {jobTotal > HISTORY_PAGE_SIZE && (
          <div className="flex items-center justify-between pt-1">
            <span className="text-[10px] text-gray-500">
              {jobPage * HISTORY_PAGE_SIZE + 1}-{Math.min((jobPage + 1) * HISTORY_PAGE_SIZE, jobTotal)} / {jobTotal}
            </span>
            <div className="flex items-center gap-1">
              <button onClick={() => setJobPage(p => Math.max(0, p - 1))} disabled={jobPage === 0}
                className="p-0.5 rounded hover:bg-gray-700 disabled:opacity-30 text-gray-400">
                <ChevronLeft size={14} />
              </button>
              <span className="text-[10px] text-gray-500 px-1">{jobPage + 1}/{jobPages}</span>
              <button onClick={() => setJobPage(p => Math.min(jobPages - 1, p + 1))} disabled={jobPage >= jobPages - 1}
                className="p-0.5 rounded hover:bg-gray-700 disabled:opacity-30 text-gray-400">
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ── Session list view ──
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-medium text-white">{t('history.title')}</h3>

      {sessions.length === 0 && !loading ? (
        <div className="text-center py-8 text-xs text-gray-500">{t('history.no_sessions')}</div>
      ) : (
        <div className="max-h-[400px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-gray-800">
              <tr className="text-gray-500 text-[10px] uppercase">
                <th className="text-left py-1 px-2 font-medium">{t('history.session_name')}</th>
                <th className="text-center py-1 px-2 font-medium w-20">{t('history.status')}</th>
                <th className="text-right py-1 px-2 font-medium w-14">{t('history.total_files')}</th>
                <th className="text-right py-1 px-2 font-medium w-12">{t('history.completed')}</th>
                <th className="text-right py-1 px-2 font-medium w-12">{t('history.failed')}</th>
                <th className="text-right py-1 px-2 font-medium w-24">{t('history.started_at')}</th>
                <th className="text-right py-1 px-2 font-medium w-20">{t('history.duration')}</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr
                  key={s.id}
                  onClick={() => setSelectedSession(s)}
                  className="border-t border-gray-700/30 hover:bg-gray-700/20 cursor-pointer"
                >
                  <td className="py-1.5 px-2 truncate max-w-[200px]" title={s.source_path}>
                    <span className="text-gray-300">{s.name || s.source_path || `#${s.id}`}</span>
                    {s.source_path && s.name && (
                      <div className="text-[10px] text-gray-500 truncate">{s.source_path}</div>
                    )}
                  </td>
                  <td className="py-1.5 px-2 text-center">
                    <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${WR_STATUS_COLORS[s.status] || ''}`}>
                      {s.status}
                    </span>
                  </td>
                  <td className="py-1.5 px-2 text-right text-white">{s.total_files}</td>
                  <td className="py-1.5 px-2 text-right text-green-400">{s.completed_count}</td>
                  <td className="py-1.5 px-2 text-right">
                    {s.failed_count > 0 ? <span className="text-red-400">{s.failed_count}</span> : <span className="text-gray-600">0</span>}
                  </td>
                  <td className="py-1.5 px-2 text-right text-gray-500 text-[10px]">{formatTime(s.created_at)}</td>
                  <td className="py-1.5 px-2 text-right text-gray-400 text-[10px]">{formatDuration(s.started_at, s.completed_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {sessionTotal > HISTORY_PAGE_SIZE && (
        <div className="flex items-center justify-between pt-1">
          <span className="text-[10px] text-gray-500">
            {sessionPage * HISTORY_PAGE_SIZE + 1}-{Math.min((sessionPage + 1) * HISTORY_PAGE_SIZE, sessionTotal)} / {sessionTotal}
          </span>
          <div className="flex items-center gap-1">
            <button onClick={() => setSessionPage(p => Math.max(0, p - 1))} disabled={sessionPage === 0}
              className="p-0.5 rounded hover:bg-gray-700 disabled:opacity-30 text-gray-400">
              <ChevronLeft size={14} />
            </button>
            <span className="text-[10px] text-gray-500 px-1">{sessionPage + 1}/{sessionPages}</span>
            <button onClick={() => setSessionPage(p => Math.min(sessionPages - 1, p + 1))} disabled={sessionPage >= sessionPages - 1}
              className="p-0.5 rounded hover:bg-gray-700 disabled:opacity-30 text-gray-400">
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
