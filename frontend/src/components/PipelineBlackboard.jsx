/**
 * PipelineBlackboard — Tollgate Architecture 6-Stage Pipeline.
 *
 * STAGE 1: Work Requests (발주서)
 * STAGE 2: Job Pool (Download Lane + Direct Lane)
 * STAGE 3: Parser (Server CPU, 1 thread sequential)
 * STAGE 4: Buffer (parsed, worker 할당 대기)
 * STAGE 5: Workers (GPU Terminals: MC → VV → MV)
 * STAGE 6: Output (Storage + Failed)
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { AlertTriangle, Zap, Eye, Scan, Brain, Check, Download, FolderOpen, Pause, Play, X, ChevronRight, ChevronDown } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron, apiClient } from '../api/client';
import { getJobStats } from '../api/worker';
import { getWorkRequests, getWorkRequestDetail, listWorkerSessions, pauseWorkRequest, resumeWorkRequest, cancelWorkRequest } from '../api/admin';

// Phase config (workers do MC → VV → MV only, no Parse)
const PHASE_CFG = {
  vision:   { label: 'MC', text: 'text-purple-400', border: 'border-purple-500/60', glow: 'shadow-[0_0_10px_rgba(168,85,247,0.3)]', bg: 'bg-purple-500', icon: Eye,   anim: 'animate-pulse' },
  embed_vv: { label: 'VV', text: 'text-cyan-400',   border: 'border-cyan-500/60',   glow: 'shadow-[0_0_10px_rgba(6,182,212,0.3)]',   bg: 'bg-cyan-500',   icon: Scan,  anim: 'animate-spin-slow' },
  embed_mv: { label: 'MV', text: 'text-emerald-400',border: 'border-emerald-500/60', glow: 'shadow-[0_0_10px_rgba(16,185,129,0.3)]', bg: 'bg-emerald-500', icon: Brain, anim: 'animate-machine-pulse' },
};
const PHASES = ['vision', 'embed_vv', 'embed_mv'];

export default function PipelineBlackboard({ reloadSignal, appMode }) {
  const { t } = useLocale();
  const [stats, setStats] = useState(null);
  const [workRequests, setWorkRequests] = useState([]);
  const [workers, setWorkers] = useState([]);

  const useIPC = isElectron && window.electron?.queue;

  const load = useCallback(async () => {
    try {
      const [sData, wrData, wData] = await Promise.all([
        useIPC ? window.electron.queue.getStats() : getJobStats(),
        useIPC
          ? window.electron.queue.listWorkRequests(false).then(r => r?.work_requests || [])
          : getWorkRequests(false).catch(() => []),
        // Only fetch worker sessions in web client mode (not Electron)
        // Server auto-processing status comes via stats.embedded_worker
        !useIPC
          ? listWorkerSessions().then(d => d?.workers || []).catch(() => [])
          : Promise.resolve([]),
      ]);
      if (sData && sData.success !== false) setStats(sData);
      setWorkRequests(Array.isArray(wrData) ? wrData : []);
      setWorkers(Array.isArray(wData) ? wData : []);
    } catch { /* ignore */ }
  }, [useIPC]);

  // Adaptive polling: 3s when active, 5s when idle (setTimeout chain)
  const isActiveRef = useRef(false);
  useEffect(() => {
    let timer;
    const poll = () => {
      load();
      timer = setTimeout(poll, isActiveRef.current ? 3000 : 5000);
    };
    poll();
    return () => clearTimeout(timer);
  }, [load]);

  // Reload immediately on external triggers
  useEffect(() => { if (reloadSignal > 0) load(); }, [reloadSignal, load]);

  // ── Computed stats ──
  const s = stats || {};
  const throughput = s.throughput ?? 0;
  const dlWaiting = s.download_waiting ?? 0;
  const parsePending = s.ready_pending ?? 0;
  const parsing = s.parse_ahead_parsing ?? 0;
  const buffer = s.parse_ahead_parsed ?? 0;
  const serverMode = s.server_mode || 'parse_only';
  const ew = s.embedded_worker || {};
  const ewRunning = ew.running || false;
  const ewCompleted = ew.jobs_completed || 0;
  const ewPhase = ew.current_phase || null;
  const ewFile = ew.current_file || null;

  // Which phases the server handles
  const serverParse = true; // server always does parse + thumbnail
  const processing = (s.assigned ?? 0) + (s.processing ?? 0);
  const remaining = (s.pending ?? 0) + (s.assigned ?? 0) + (s.processing ?? 0);
  const etaMin = throughput > 0 ? Math.ceil(remaining / throughput) : null;

  // Active WRs completed/failed — from work_requests table (survives job deletion)
  const activeWRsAll = workRequests.filter(wr => wr.status === 'queued' || wr.status === 'processing' || wr.status === 'paused');

  // Group WRs by path ancestry — if WR A's source_path is a prefix of WR B's,
  // they belong to the same tree. Display only top-level roots.
  const wrGroups = useMemo(() => {
    if (activeWRsAll.length <= 1) return null;

    // Normalize paths: strip trailing slash, sort shortest first
    const items = activeWRsAll.map(wr => ({
      wr,
      path: (wr.source_path || '').replace(/\/$/, ''),
    })).sort((a, b) => a.path.length - b.path.length);

    const roots = [];      // top-level groups or singletons
    const assigned = new Set();

    // Pass 1: find WRs whose path is a prefix of others → they become roots
    for (let i = 0; i < items.length; i++) {
      if (assigned.has(items[i].wr.id)) continue;
      const root = items[i];
      const children = [];
      for (let j = i + 1; j < items.length; j++) {
        if (assigned.has(items[j].wr.id)) continue;
        if (items[j].path.startsWith(root.path + '/')) {
          children.push(items[j].wr);
          assigned.add(items[j].wr.id);
        }
      }
      if (children.length > 0) {
        assigned.add(root.wr.id);
        const parts = root.path.replace(/^webdav:\/?\/?[^/]*/, '').split('/').filter(Boolean);
        roots.push({
          type: 'group',
          displayName: parts.slice(-2).join('/') || root.wr.name,
          wrs: [root.wr, ...children],
        });
      }
    }

    // Pass 2: remaining unassigned WRs — group by common parent
    const remaining = items.filter(it => !assigned.has(it.wr.id));
    const parentGroups = {};
    for (const it of remaining) {
      const lastSlash = it.path.lastIndexOf('/');
      const parent = lastSlash > 0 ? it.path.substring(0, lastSlash) : it.path;
      if (!parentGroups[parent]) parentGroups[parent] = [];
      parentGroups[parent].push(it.wr);
    }
    for (const [parent, wrs] of Object.entries(parentGroups)) {
      if (wrs.length > 1) {
        const parts = parent.replace(/^webdav:\/?\/?[^/]*/, '').split('/').filter(Boolean);
        roots.push({ type: 'group', displayName: parts.slice(-2).join('/') || parent, wrs });
      } else {
        roots.push({ type: 'single', wr: wrs[0] });
      }
    }

    return roots.length > 0 ? roots : null;
  }, [activeWRsAll]);

  const handleWRAction = useCallback(async (wrId, action) => {
    try {
      if (action === 'pause') {
        useIPC ? await window.electron.queue.pauseWorkRequest(wrId) : await pauseWorkRequest(wrId);
      } else if (action === 'resume') {
        useIPC ? await window.electron.queue.resumeWorkRequest(wrId) : await resumeWorkRequest(wrId);
      } else if (action === 'cancel') {
        useIPC ? await window.electron.queue.cancelWorkRequest(wrId) : await cancelWorkRequest(wrId);
      }
      load();
    } catch { /* ignore */ }
  }, [useIPC, load]);
  const completed = activeWRsAll.reduce((sum, wr) => sum + (wr.completed_count || 0), 0);
  const failed = activeWRsAll.reduce((sum, wr) => sum + (wr.failed_count || 0), 0);
  const total = activeWRsAll.reduce((sum, wr) => sum + (wr.total_files || 0), 0);
  const pct = total > 0 ? ((completed / total) * 100).toFixed(1) : '0.0';

  // ── Workers (external only, embedded worker shown separately in Stage 5) ──
  const allWorkers = workers
    .filter(w => w.status === 'online')
    .filter(w => w.worker_name !== '__builtin__')
    .map(w => ({
      name: w.worker_name || w.username,
      phase: w.current_phase,
      currentFile: w.current_file,
      throughput: w.throughput,
      batchSize: w.batch_capacity || 0,
      state: w.current_phase ? 'active' : 'idle',
      mode: w.processing_mode_override || 'full',
    }));
  const activeWRs = activeWRsAll;
  const isRunning = throughput > 0 || processing > 0 || ewRunning;
  isActiveRef.current = isRunning;

  return (
    <div className="h-full flex flex-col bg-gray-950 select-none overflow-hidden">

      {/* Title Bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800/50">
        <div className="flex items-center gap-3">
          <span className="text-[11px] uppercase tracking-[0.2em] text-gray-500 font-mono font-bold">
            {t('bb.title')}
          </span>
          {isRunning && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-green-900/30 border border-green-800/40">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              <span className="text-[10px] font-mono text-green-400">{t('bb.live')}</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {throughput > 0 && (
            <span className="text-xs font-mono text-gray-400 tabular-nums">
              <Zap size={11} className="inline mr-1 text-yellow-500" />
              {throughput.toFixed(1)} {t('bb.per_min')}
            </span>
          )}
        </div>
      </div>

      {/* Factory Floor */}
      <div className="flex-1 overflow-auto p-4">

        <div className="flex flex-col gap-3 max-w-5xl mx-auto">

          {/* ── STAGE 1: Work Requests ── */}
          <Stage label="STAGE 1" title={`WORK REQUESTS${activeWRs.length > 0 ? ` (${activeWRs.length})` : ''}`} color="gray">
            <div className="flex flex-col gap-2 mt-1 max-h-[300px] overflow-y-auto">
              {activeWRs.length > 0 ? (
                wrGroups ? wrGroups.map((item, idx) =>
                  item.type === 'group' ? (
                    <WRGroupCard key={idx} group={item} onAction={handleWRAction} />
                  ) : (
                    <WRCard key={item.wr.id} wr={item.wr} onAction={handleWRAction} />
                  )
                ) : activeWRs.map(wr => (
                  <WRCard key={wr.id} wr={wr} onAction={handleWRAction} />
                ))
              ) : (
                <div className="text-[10px] text-gray-700 font-mono py-1">{t('bb.no_wr')}</div>
              )}
            </div>
          </Stage>

          <Belt active={isRunning} color="blue" label={t('bb.station_queue')} />

          {/* ── STAGE 2: Job Pool ── */}
          <Stage label="STAGE 2" title="JOB POOL" color="yellow">
            <div className="grid grid-cols-2 gap-3 mt-1">
              {/* Download Lane */}
              <div className="rounded-lg border border-blue-800/30 bg-blue-900/5 p-2">
                <div className="flex items-center gap-1.5 mb-2">
                  <Download size={14} className="text-blue-400" />
                  <span className="text-[9px] font-mono text-blue-400 font-bold uppercase">Download Lane</span>
                  {appMode === 'server' && dlWaiting > 0 && <span className="text-[6px] font-mono px-1 rounded bg-amber-800/60 text-amber-400 border border-amber-700/40">SERVER</span>}
                  <span className="text-[8px] font-mono text-gray-600 ml-auto">WebDAV → Local</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[8px] font-mono text-gray-600">{t('bb.station_queue')}</span>
                  <span className="text-lg font-mono font-bold text-blue-400 tabular-nums">{dlWaiting}</span>
                </div>
                <div className="text-[7px] text-gray-600 font-mono mt-1">DownloadPool: 3 threads</div>
              </div>
              {/* Direct Lane */}
              <div className="rounded-lg border border-teal-800/30 bg-teal-900/5 p-2">
                <div className="flex items-center gap-1.5 mb-2">
                  <FolderOpen size={14} className="text-teal-400" />
                  <span className="text-[9px] font-mono text-teal-400 font-bold uppercase">Direct Lane</span>
                  <span className="text-[8px] font-mono text-gray-600 ml-auto">Local → Ready</span>
                </div>
                <div className="text-[8px] text-gray-500 font-mono">{t('bb.station_queue')} file_ready=1</div>
                <div className="text-[7px] text-gray-600 font-mono mt-1">{t('bb.phase_idle')}</div>
              </div>
            </div>
            {/* Pool summary */}
            <div className="flex items-center gap-4 mt-2 pt-2 border-t border-gray-800/30 text-[8px] font-mono text-gray-500">
              <span>{t('bb.total')}: <span className="text-yellow-400 font-bold">{total}</span></span>
              <span>Ready: <span className="text-green-400 font-bold">{total - dlWaiting}</span></span>
              {dlWaiting > 0 && <span>DL wait: <span className="text-blue-400">{dlWaiting}</span></span>}
              <span className="text-gray-600">ORDER BY priority, created_at</span>
            </div>
          </Stage>

          <Belt active={isRunning} color="teal" label="file_ready=1" />

          {/* ── STAGE 3: Parse + Thumbnail ── */}
          <Stage label="STAGE 3" title="PARSE + THUMBNAIL" color="teal">
            <div className="mt-1 space-y-2">
              {/* Parse status */}
              <div className="flex items-center gap-1.5">
                <PhasePill label="Parse" active={serverParse} current={parsing > 0} done={parsePending === 0 && parsing === 0 && buffer > 0} />
                <span className="text-gray-700 text-[8px]">+</span>
                <PhasePill label="Thumbnail" active={serverParse} current={parsing > 0} done={parsePending === 0 && parsing === 0 && buffer > 0} />
                <span className="text-[7px] text-gray-600 font-mono ml-auto">No AI</span>
              </div>
              {/* Stats row */}
              <div className="flex items-center gap-4 text-[9px] font-mono">
                <span className="text-gray-500">parsing: <span className="text-teal-400 font-bold">{parsing}</span></span>
                <span className="text-gray-500">await: <span className="text-gray-400">{parsePending}</span></span>
              </div>
            </div>
          </Stage>

          <Belt active={isRunning && buffer > 0} color="orange" label="parsed → buffer" />

          {/* ── STAGE 4: Buffer ── */}
          <Stage label="STAGE 4" title="BUFFER" color="orange">
            <div className="flex items-center gap-4 mt-1">
              <div className={`rounded-lg border-2 ${buffer > 0 ? 'border-orange-600/40 shadow-[0_0_12px_rgba(251,146,60,0.15)]' : 'border-gray-700/30'} bg-gradient-to-b from-orange-900/15 to-gray-900 w-16 h-16 flex flex-col items-center justify-center flex-shrink-0`}>
                <span className="text-2xl">&#128230;</span>
              </div>
              <div>
                <div className="flex items-center gap-4 mb-1.5">
                  <div>
                    <div className="text-[8px] font-mono text-gray-600 mb-0.5">ready for worker</div>
                    <span className="text-2xl font-mono font-bold text-orange-400 tabular-nums">{buffer}</span>
                  </div>
                </div>
                <div className="text-[7px] text-gray-600 font-mono">
                  {allWorkers.length > 0
                    ? `target: ${allWorkers.length} workers × 2 = ${allWorkers.length * 2}`
                    : 'no workers → buffer paused'
                  }
                </div>
              </div>
            </div>
          </Stage>

          <Belt active={allWorkers.some(w => w.state === 'active')} color="purple" label={t('bb.workers_title')} />

          {/* ── STAGE 5: Workers ── */}
          <Stage label="STAGE 5" title="WORKERS (AI Processing)" color="purple">
            <div className="space-y-3 mt-1">
              {/* Embedded Worker */}
              <div className={`rounded-lg border ${ewRunning ? 'border-purple-500/40 bg-purple-900/10' : 'border-gray-700/30 bg-gray-900/20'} p-2`}>
                <div className="flex items-center gap-2 text-[9px] font-mono">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${ewRunning ? 'bg-purple-500 animate-pulse' : 'bg-gray-600'}`} />
                  <span className={`font-bold ${ewRunning ? 'text-purple-400' : 'text-gray-500'}`}>Embedded Worker</span>
                  {ewPhase && (
                    <span className="px-1.5 py-0.5 rounded text-[7px] font-mono bg-purple-900/40 text-purple-300 border border-purple-500/30">
                      {ewPhase === 'vision' ? 'MC' : ewPhase === 'embed_vv' ? 'VV' : ewPhase === 'embed_mv' ? 'MV' : ewPhase}
                    </span>
                  )}
                  {ewCompleted > 0 && <span className="text-gray-500 ml-auto">{ewCompleted} done</span>}
                  {throughput > 0 && <span className="text-yellow-500 ml-1">⚡{throughput.toFixed(1)}/m</span>}
                </div>
                {ewFile && (
                  <div className="text-[8px] text-gray-600 font-mono truncate mt-1 ml-4">
                    {ewFile}
                  </div>
                )}
                {!ewRunning && (
                  <div className="text-[8px] text-gray-600 font-mono mt-1 ml-4">
                    {buffer > 0 ? 'Standby — waiting for start' : 'Idle — no pending jobs'}
                  </div>
                )}
              </div>

              {/* External Workers */}
              {allWorkers.map((w, i) => <WorkerLine key={i} worker={w} t={t} />)}

              {allWorkers.length === 0 && !ewRunning && (
                <div className="text-[7px] text-gray-600 font-mono text-center py-1">
                  No external workers connected
                </div>
              )}
            </div>
            <div className="text-[7px] text-gray-600 font-mono mt-2 pt-2 border-t border-gray-800/30">
              MC(Vision) → VV(Visual Vector) → MV(Meaning Vector) · Phase-batch · GPU model swap
            </div>
          </Stage>

          <Belt active={completed > 0} color="green" label={t('bb.completed_label')} />

          {/* ── STAGE 6: Output ── */}
          <Stage label="STAGE 6" title="OUTPUT" color="green">
            <div className="flex items-center gap-6 mt-1">
              {/* Storage */}
              <div className="rounded-lg border-2 border-green-600/40 bg-gradient-to-b from-green-900/15 to-gray-900 px-4 py-3 text-center relative overflow-hidden min-w-[120px]">
                <div className="absolute bottom-0 left-0 right-0 bg-green-500/10 transition-all duration-500" style={{ height: `${pct}%` }} />
                <div className="relative z-10">
                  <div className="text-[7px] uppercase tracking-widest text-green-600 font-mono font-bold mb-1">{t('bb.storage')}</div>
                  <div className="text-2xl font-mono font-bold text-green-400 tabular-nums leading-none">{completed.toLocaleString()}</div>
                  <div className="text-[9px] text-green-600 font-mono mt-0.5">{pct}%</div>
                </div>
              </div>
              {/* Failed */}
              {failed > 0 && (
                <div className="rounded-lg border-2 border-red-800/40 bg-gradient-to-b from-red-900/15 to-gray-900 px-4 py-3 text-center">
                  <div className="text-[7px] uppercase tracking-widest text-red-600 font-mono font-bold mb-1">{t('bb.failed')}</div>
                  <div className="text-xl font-mono font-bold text-red-400 tabular-nums">
                    <AlertTriangle size={14} className="inline mr-1 -mt-0.5" />{failed}
                  </div>
                </div>
              )}
              {/* Stats */}
              <div className="flex-1 text-[8px] font-mono text-gray-600">
                <div>{t('bb.total')}: <span className="text-gray-400">{total.toLocaleString()}</span></div>
                <div>{t('factory.summary_processing')}: <span className="text-purple-400">{processing}</span></div>
                {etaMin != null && <div>{t('bb.eta')}: <span className="text-gray-400">{etaMin >= 60 ? `${Math.floor(etaMin/60)}h ${etaMin%60}m` : `${etaMin}m`}</span></div>}
              </div>
            </div>
          </Stage>

        </div>
      </div>

      {/* Status Ribbon */}
      <div className="px-4 py-2 border-t border-gray-800/50 bg-gray-900/50">
        <div className="flex items-center justify-center gap-6 text-[11px] font-mono text-gray-500">
          {throughput > 0 && <span><span className="text-blue-400 font-bold">{throughput.toFixed(1)}</span> {t('bb.per_min')}</span>}
          {etaMin != null && <span>{t('bb.eta')}: <span className="text-gray-300">{etaMin >= 60 ? `${Math.floor(etaMin/60)}h ${etaMin%60}m` : `${etaMin}m`}</span></span>}
          {total > 0 && <span><span className="text-green-400">{completed.toLocaleString()}</span>/{total.toLocaleString()} (<span className="text-gray-300">{pct}%</span>)</span>}
          {dlWaiting > 0 && <span>DL <span className="text-blue-400">{dlWaiting}</span></span>}
          {buffer > 0 && <span>{t('bb.buffer')} <span className="text-orange-400">{buffer}</span></span>}
          {processing > 0 && <span>{t('factory.summary_processing')} <span className="text-purple-400">{processing}</span></span>}
          {failed > 0 && <span className="text-red-400"><AlertTriangle size={10} className="inline mr-0.5 -mt-0.5" />{failed}</span>}
          {!isRunning && total === 0 && <span className="text-gray-600">{t('bb.idle')}</span>}
        </div>
      </div>
    </div>
  );
}

// ─── Stage wrapper ───────────────────────────────────────

const STAGE_COLORS = {
  gray:   'border-gray-700/30 bg-gray-900/30',
  yellow: 'border-yellow-700/20 bg-yellow-900/5',
  teal:   'border-teal-700/20 bg-teal-900/5',
  orange: 'border-orange-700/20 bg-orange-900/5',
  purple: 'border-purple-700/20 bg-purple-900/5',
  green:  'border-green-700/20 bg-green-900/5',
};

const LABEL_COLORS = {
  gray:   'bg-gray-800 text-gray-400 border-gray-700/40',
  yellow: 'bg-yellow-900/80 text-yellow-500 border-yellow-700/30',
  teal:   'bg-teal-900/80 text-teal-400 border-teal-700/30',
  orange: 'bg-orange-900/80 text-orange-400 border-orange-700/30',
  purple: 'bg-purple-900/80 text-purple-400 border-purple-700/30',
  green:  'bg-green-900/80 text-green-400 border-green-700/30',
};

function Stage({ label, title, color, children, extra }) {
  return (
    <div className={`border-2 rounded-xl p-3 relative ${STAGE_COLORS[color]}`}>
      <div className="flex items-center">
        <div className={`absolute -top-3 left-3 px-2 py-0 text-[9px] uppercase tracking-widest font-mono font-bold rounded-sm border ${LABEL_COLORS[color]}`}>
          {label} · {title}
        </div>
        {extra && <div className="absolute -top-3 right-3">{extra}</div>}
      </div>
      {children}
    </div>
  );
}

// ─── Belt between stages ─────────────────────────────────

const BELT_COLORS = {
  blue: 'bg-blue-500/20', teal: 'bg-teal-500/20', orange: 'bg-orange-500/20',
  purple: 'bg-purple-500/20', green: 'bg-green-500/20',
};

function Belt({ active, color, label }) {
  return (
    <div className="flex items-center gap-2 px-6">
      <span className={`text-[8px] ${active ? 'text-gray-500' : 'text-gray-700'}`}>&#9660;</span>
      <div className="flex-1 h-1.5 rounded-full overflow-hidden bg-gray-800/40 relative">
        {active && <div className={`absolute inset-0 ${BELT_COLORS[color]} animate-pulse`} />}
      </div>
      <span className="text-[7px] text-gray-600 font-mono w-24 text-right">{label}</span>
    </div>
  );
}

// ─── Worker Line ─────────────────────────────────────────

// ─── Server Mode Control (ON/OFF + level selector) ──────

// ServerModeControl removed — Stage 3 is always parse+thumbnail only

// ─── Phase Pill (server processing indicator) ────────────

const PILL_COLORS = {
  teal:    { active: 'border-teal-500/60 bg-teal-900/30 text-teal-400', current: 'border-teal-400 bg-teal-800/40 text-teal-300 shadow-[0_0_8px_rgba(45,212,191,0.3)]' },
  purple:  { active: 'border-purple-500/60 bg-purple-900/30 text-purple-400', current: 'border-purple-400 bg-purple-800/40 text-purple-300 shadow-[0_0_8px_rgba(168,85,247,0.3)]' },
  cyan:    { active: 'border-cyan-500/60 bg-cyan-900/30 text-cyan-400', current: 'border-cyan-400 bg-cyan-800/40 text-cyan-300 shadow-[0_0_8px_rgba(6,182,212,0.3)]' },
  emerald: { active: 'border-emerald-500/60 bg-emerald-900/30 text-emerald-400', current: 'border-emerald-400 bg-emerald-800/40 text-emerald-300 shadow-[0_0_8px_rgba(16,185,129,0.3)]' },
};

function PhasePill({ label, active, current, done, color = 'teal' }) {
  const c = PILL_COLORS[color] || PILL_COLORS.teal;
  if (!active) {
    return <span className="px-2 py-0.5 rounded border border-gray-800/30 text-[8px] font-mono text-gray-700 bg-gray-900/20">
      {label} <span className="text-[6px]">—</span>
    </span>;
  }
  return (
    <span className={`px-2 py-0.5 rounded border text-[8px] font-mono font-bold transition-all ${current ? c.current + ' animate-pulse' : c.active}`}>
      {label} {current ? '●' : done ? '✓' : '○'}
    </span>
  );
}

// Which phases each mode can execute
const MODE_PHASES = {
  full:       ['vision', 'embed_vv', 'embed_mv'],
  mc_only:    ['vision'],
  embed_only: ['embed_vv', 'embed_mv'],
};

const MODE_LABELS = {
  full: 'FULL',
  mc_only: 'MC ONLY',
  embed_only: 'EMBED',
};

const MODE_COLORS = {
  full: 'text-gray-500',
  mc_only: 'text-amber-500',
  embed_only: 'text-orange-500',
};

function WorkerLine({ worker, t }) {
  const w = worker;
  const isActive = w.state === 'active' || !!w.phase;
  const phaseIdx = w.phase ? PHASES.indexOf(w.phase) : -1;
  const mode = w.mode || 'full';
  const enabledPhases = MODE_PHASES[mode] || MODE_PHASES.full;

  return (
    <div className={`flex items-center gap-3 rounded-lg border px-3 py-2 ${isActive ? 'border-gray-600/30 bg-gray-800/20' : 'border-gray-800/20 bg-gray-900/10'}`}>
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-600'}`} />
      <div className="flex flex-col w-24 flex-shrink-0">
        <span className="text-[11px] font-mono text-gray-300 font-bold truncate">{w.name}</span>
        <span className={`text-[7px] font-mono ${MODE_COLORS[mode]}`}>{MODE_LABELS[mode]}</span>
      </div>

      {/* Phase machines */}
      <div className="flex items-center gap-0">
        {PHASES.map((phase, idx) => {
          const cfg = PHASE_CFG[phase];
          const enabled = enabledPhases.includes(phase);
          const isCurrent = enabled && w.phase === phase;
          const isPast = enabled && phaseIdx > idx;
          const Icon = cfg.icon;

          let progress = 0;
          if (isCurrent && w.phaseCount > 0) progress = ((w.phaseIndex || 0) / w.phaseCount) * 100;
          else if (isPast) progress = 100;

          return (
            <div key={phase} className="flex items-center">
              <div className={`
                relative w-11 h-11 rounded-lg border-2 flex items-center justify-center flex-shrink-0 transition-all duration-300
                ${!enabled
                  ? 'border-gray-800/20 bg-gray-900/30'
                  : isCurrent
                    ? `${cfg.border} ${cfg.glow} bg-gradient-to-b from-gray-800/80 to-gray-900`
                    : isPast
                      ? 'border-gray-600/30 bg-gray-800/40'
                      : 'border-gray-700/20 bg-gray-800/60'
                }
              `}>
                <div className={`absolute -top-1.5 left-1/2 -translate-x-1/2 px-1 rounded-sm text-[5px] uppercase tracking-widest font-mono font-bold bg-gray-900/90 border border-gray-700/30 ${isCurrent ? cfg.text : !enabled ? 'text-gray-800' : ''}`}>
                  {cfg.label}
                </div>
                {enabled ? (
                  <Icon size={16} className={`transition-all ${isCurrent ? `${cfg.text} ${cfg.anim}` : isPast ? 'text-gray-500 opacity-30' : 'text-gray-700 opacity-20'}`} />
                ) : (
                  <span className="text-[8px] text-gray-800 font-mono">&#8212;</span>
                )}
                {isPast && <div className="absolute -top-0.5 -right-0.5"><Check size={7} className="text-green-500/70" /></div>}
                {/* Gauge */}
                {enabled && (isCurrent || isPast) && (
                  <div className="absolute bottom-0.5 left-1 right-1">
                    <div className="h-0.5 rounded-full bg-gray-700/50 overflow-hidden">
                      <div className={`h-full rounded-full transition-all duration-500 ${isCurrent ? cfg.bg : 'bg-gray-600/40'}`} style={{ width: `${progress}%` }} />
                    </div>
                  </div>
                )}
              </div>
              {idx < PHASES.length - 1 && <div className={`w-2 h-px flex-shrink-0 ${!enabled ? 'bg-gray-900/20' : isPast || isCurrent ? 'bg-gray-600' : 'bg-gray-800/30'}`} />}
            </div>
          );
        })}
      </div>

      {/* Worker info */}
      <div className="flex-1 min-w-0 ml-2">
        <div className="flex items-center gap-2">
          {w.batchSize > 0 && <span className="text-[8px] font-mono text-yellow-600 tabular-nums">B:{w.batchSize}</span>}
          {w.throughput > 0 && <span className="text-[8px] font-mono text-gray-500 tabular-nums">{w.throughput.toFixed(1)}/m</span>}
          {!isActive && <span className="text-[8px] font-mono text-gray-600 italic">{w.state === 'resting' ? t('bb.resting') : t('bb.phase_idle')}</span>}
        </div>
        {w.currentFile && <div className="text-[7px] text-gray-600 font-mono truncate">{w.currentFile}</div>}
      </div>
    </div>
  );
}

// ─── WR Card ─────────────────────────────────────────────

function WRCard({ wr, onAction }) {
  const { t } = useLocale();
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState(null);

  const total = wr.total_files || 0;
  const done = wr.completed_count || 0;
  const failed = wr.failed_count || 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const isActive = wr.status === 'processing';
  const isPaused = wr.status === 'paused';
  const isWebDAV = wr.source_path?.startsWith('webdav://');

  const dotColor = isActive ? 'bg-blue-500 animate-pulse' : isPaused ? 'bg-yellow-500' : 'bg-gray-600';
  const borderColor = isActive ? 'border-blue-700/50 bg-blue-900/15' : isPaused ? 'border-yellow-700/50 bg-yellow-900/10' : 'border-gray-700/30 bg-gray-800/30';
  const barColor = isActive ? 'bg-blue-500' : isPaused ? 'bg-yellow-500/60' : 'bg-gray-600';

  const handleToggle = async () => {
    if (!expanded && !detail) {
      try {
        const useIPC = isElectron && window.electron?.queue;
        const res = useIPC
          ? await window.electron.queue.getWorkRequestDetail(wr.id)
          : await getWorkRequestDetail(wr.id);
        setDetail(res);
      } catch { /* ignore */ }
    }
    setExpanded(!expanded);
  };

  return (
    <div className={`rounded-lg border ${borderColor} ${expanded ? 'w-full' : 'w-[180px]'} transition-all`}>
      {/* Header — clickable to expand */}
      <div className="px-3 py-2 cursor-pointer" onClick={handleToggle}>
        <div className="flex items-center gap-1.5 mb-1">
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dotColor}`} />
          <span className="text-[10px] text-gray-300 font-medium truncate flex-1">{wr.name}</span>
          {isWebDAV && <span className="text-[7px] font-mono text-blue-500">WebDAV</span>}
          {/* Control buttons */}
          {onAction && (
            <div className="flex items-center gap-0.5 flex-shrink-0">
              {isActive && (
                <button onClick={(e) => { e.stopPropagation(); onAction(wr.id, 'pause'); }}
                  className="p-0.5 rounded hover:bg-gray-700 text-gray-500 hover:text-yellow-400 transition-colors" title="Pause">
                  <Pause size={9} />
                </button>
              )}
              {isPaused && (
                <button onClick={(e) => { e.stopPropagation(); onAction(wr.id, 'resume'); }}
                  className="p-0.5 rounded hover:bg-gray-700 text-gray-500 hover:text-green-400 transition-colors" title="Resume">
                  <Play size={9} />
                </button>
              )}
              <button onClick={(e) => { e.stopPropagation(); onAction(wr.id, 'cancel'); }}
                className="p-0.5 rounded hover:bg-gray-700 text-gray-500 hover:text-red-400 transition-colors" title="Cancel">
                <X size={9} />
              </button>
            </div>
          )}
        </div>
        {isPaused && <div className="text-[7px] text-yellow-500 font-mono mb-1">PAUSED</div>}
        <div className="h-1 bg-gray-700/50 rounded-full overflow-hidden">
          <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
        <div className="flex justify-between mt-0.5">
          <span className="text-[8px] font-mono text-gray-600 tabular-nums">{done}/{total}</span>
          <span className="text-[8px] font-mono text-gray-500 tabular-nums">{pct.toFixed(0)}%</span>
        </div>
        {failed > 0 && <span className="text-[7px] text-red-400 font-mono"><AlertTriangle size={7} className="inline mr-0.5" />{failed}</span>}
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-3 pb-2 border-t border-gray-700/30 pt-2 space-y-1.5">
          {/* Source path */}
          <div className="text-[8px] font-mono text-gray-600 truncate" title={wr.source_path}>{wr.source_path}</div>
          {/* Subtasks */}
          {detail?.subtasks?.length > 0 ? (
            <div className="space-y-1 max-h-[120px] overflow-y-auto">
              {detail.subtasks.map((st, i) => {
                const stPct = st.total_files > 0 ? (st.completed_count / st.total_files * 100) : 0;
                return (
                  <div key={i} className="flex items-center gap-2 text-[8px] font-mono">
                    <span className="text-gray-500 truncate flex-1" title={st.folder_path}>{st.folder_path || '/'}</span>
                    <span className="text-gray-600 tabular-nums flex-shrink-0">{st.completed_count}/{st.total_files}</span>
                    <div className="w-12 h-0.5 bg-gray-700/50 rounded-full overflow-hidden flex-shrink-0">
                      <div className="h-full bg-blue-500/60 rounded-full" style={{ width: `${stPct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : detail ? (
            <div className="text-[8px] text-gray-600 font-mono">{t('bb.no_wr')}</div>
          ) : (
            <div className="text-[8px] text-gray-600 font-mono">{t('status.loading') || 'Loading...'}</div>
          )}
          {/* Timestamps */}
          {wr.created_at && <div className="text-[7px] text-gray-700 font-mono">Created: {new Date(wr.created_at).toLocaleString()}</div>}
        </div>
      )}
    </div>
  );
}

// ─── WR Group Card (collapsed parent with expandable children) ───

function WRGroupCard({ group, onAction }) {
  const [expanded, setExpanded] = useState(false);
  const totalFiles = group.wrs.reduce((s, wr) => s + (wr.total_files || 0), 0);
  const totalDone = group.wrs.reduce((s, wr) => s + (wr.completed_count || 0), 0);
  const totalFailed = group.wrs.reduce((s, wr) => s + (wr.failed_count || 0), 0);
  const pct = totalFiles > 0 ? (totalDone / totalFiles) * 100 : 0;
  const hasActive = group.wrs.some(wr => wr.status === 'processing');
  const hasPaused = group.wrs.some(wr => wr.status === 'paused');

  const handleCancelAll = (e) => {
    e.stopPropagation();
    if (confirm(`Cancel all ${group.wrs.length} work requests?`)) {
      group.wrs.forEach(wr => onAction(wr.id, 'cancel'));
    }
  };

  return (
    <div className={`rounded-lg border ${hasActive ? 'border-blue-700/40 bg-blue-900/10' : hasPaused ? 'border-yellow-700/40 bg-yellow-900/10' : 'border-gray-700/30 bg-gray-800/30'}`}>
      {/* Group header — single row, looks like one work request */}
      <div className="px-3 py-2.5 cursor-pointer flex items-center gap-2" onClick={() => setExpanded(!expanded)}>
        {expanded
          ? <ChevronDown size={14} className="text-gray-400 flex-shrink-0" />
          : <ChevronRight size={14} className="text-gray-400 flex-shrink-0" />}
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${hasActive ? 'bg-blue-500 animate-pulse' : hasPaused ? 'bg-yellow-500' : 'bg-gray-600'}`} />
        <span className="text-[11px] text-gray-200 font-medium truncate flex-1">{group.displayName}</span>
        <span className="text-[8px] font-mono text-gray-500 flex-shrink-0">{group.wrs.length}</span>
        <span className="text-[9px] font-mono text-gray-500 tabular-nums flex-shrink-0">{totalDone}/{totalFiles}</span>
        <span className="text-[8px] font-mono text-gray-600 tabular-nums flex-shrink-0 w-8 text-right">{pct.toFixed(0)}%</span>
        {/* Progress bar inline */}
        <div className="w-20 h-1.5 bg-gray-700/50 rounded-full overflow-hidden flex-shrink-0">
          <div className={`h-full rounded-full transition-all ${hasActive ? 'bg-blue-500' : hasPaused ? 'bg-yellow-500/60' : 'bg-gray-600'}`} style={{ width: `${pct}%` }} />
        </div>
        {totalFailed > 0 && <span className="text-[8px] text-red-400 font-mono flex-shrink-0"><AlertTriangle size={8} className="inline mr-0.5" />{totalFailed}</span>}
        {onAction && (
          <button onClick={handleCancelAll}
            className="p-0.5 rounded hover:bg-gray-700 text-gray-600 hover:text-red-400 transition-colors flex-shrink-0" title="Cancel all">
            <X size={10} />
          </button>
        )}
      </div>
      {/* Expanded: child WR list */}
      {expanded && (
        <div className="px-3 pb-2 border-t border-gray-700/20 pt-2 space-y-1 max-h-[300px] overflow-y-auto">
          {group.wrs.map(wr => (
            <WRChildRow key={wr.id} wr={wr} onAction={onAction} />
          ))}
        </div>
      )}
    </div>
  );
}

function WRChildRow({ wr, onAction }) {
  const total = wr.total_files || 0;
  const done = wr.completed_count || 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const isActive = wr.status === 'processing';
  const isPaused = wr.status === 'paused';

  return (
    <div className="flex items-center gap-2 text-[9px] font-mono group py-0.5">
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-blue-500 animate-pulse' : isPaused ? 'bg-yellow-500' : 'bg-gray-600'}`} />
      <span className="text-gray-300 truncate flex-1" title={wr.source_path}>{wr.name.replace(/^\[Recovery\]\s*/, '')}</span>
      <span className="text-gray-600 tabular-nums flex-shrink-0">{done}/{total}</span>
      <div className="w-16 h-1 bg-gray-700/50 rounded-full overflow-hidden flex-shrink-0">
        <div className={`h-full rounded-full ${isActive ? 'bg-blue-500' : 'bg-gray-600'}`} style={{ width: `${pct}%` }} />
      </div>
      {onAction && (
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
          {isActive && (
            <button onClick={() => onAction(wr.id, 'pause')} className="p-0.5 rounded hover:bg-gray-700 text-gray-600 hover:text-yellow-400" title="Pause">
              <Pause size={8} />
            </button>
          )}
          {isPaused && (
            <button onClick={() => onAction(wr.id, 'resume')} className="p-0.5 rounded hover:bg-gray-700 text-gray-600 hover:text-green-400" title="Resume">
              <Play size={8} />
            </button>
          )}
          <button onClick={() => onAction(wr.id, 'cancel')} className="p-0.5 rounded hover:bg-gray-700 text-gray-600 hover:text-red-400" title="Cancel">
            <X size={8} />
          </button>
        </div>
      )}
    </div>
  );
}

