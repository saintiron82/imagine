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

import { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, Zap, Eye, Scan, Brain, Check, Download, FolderOpen, LayoutList, LayoutGrid, Pause, Play, X } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { getWorkRequests, getWorkRequestDetail, listWorkerSessions, pauseWorkRequest, resumeWorkRequest, cancelWorkRequest } from '../api/admin';

// Phase config (workers do MC → VV → MV only, no Parse)
const PHASE_CFG = {
  vision:   { label: 'MC', text: 'text-purple-400', border: 'border-purple-500/60', glow: 'shadow-[0_0_10px_rgba(168,85,247,0.3)]', bg: 'bg-purple-500', icon: Eye,   anim: 'animate-pulse' },
  embed_vv: { label: 'VV', text: 'text-cyan-400',   border: 'border-cyan-500/60',   glow: 'shadow-[0_0_10px_rgba(6,182,212,0.3)]',   bg: 'bg-cyan-500',   icon: Scan,  anim: 'animate-spin-slow' },
  embed_mv: { label: 'MV', text: 'text-emerald-400',border: 'border-emerald-500/60', glow: 'shadow-[0_0_10px_rgba(16,185,129,0.3)]', bg: 'bg-emerald-500', icon: Brain, anim: 'animate-machine-pulse' },
};
const PHASES = ['vision', 'embed_vv', 'embed_mv'];

export default function PipelineBlackboard({ workerProgress, reloadSignal }) {
  const { t } = useLocale();
  const [view, setView] = useState('pipeline'); // 'pipeline' | 'board'
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
        !useIPC ? listWorkerSessions().then(d => d?.workers || []).catch(() => []) : Promise.resolve([]),
      ]);
      if (sData && sData.success !== false) setStats(sData);
      setWorkRequests(Array.isArray(wrData) ? wrData : []);
      setWorkers(Array.isArray(wData) ? wData : []);
    } catch { /* ignore */ }
  }, [useIPC]);

  useEffect(() => {
    load();
    const iv = setInterval(load, 5000);
    return () => clearInterval(iv);
  }, [load]);

  // Reload immediately when queue changes (e.g. folder scan completed)
  useEffect(() => {
    if (reloadSignal > 0) load();
  }, [reloadSignal, load]);

  // ── Computed stats ──
  const s = stats || {};
  const throughput = s.throughput ?? 0;
  const dlWaiting = s.download_waiting ?? 0;
  const parsePending = s.ready_pending ?? 0;
  const parsing = s.parse_ahead_parsing ?? 0;
  const buffer = s.parse_ahead_parsed ?? 0;
  const processing = (s.assigned ?? 0) + (s.processing ?? 0);
  const remaining = (s.pending ?? 0) + (s.assigned ?? 0) + (s.processing ?? 0);
  const etaMin = throughput > 0 ? Math.ceil(remaining / throughput) : null;

  // Active WRs completed/failed — from work_requests table (survives job deletion)
  const activeWRsAll = workRequests.filter(wr => wr.status === 'queued' || wr.status === 'processing' || wr.status === 'paused');

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

  // ── Workers ──
  const wp = workerProgress;
  const localWorker = (isElectron && wp && (wp.currentPhase || wp.completed > 0)) ? {
    name: t('bb.local_worker'),
    phase: wp.currentPhase,
    phaseIndex: wp.phaseIndex,
    phaseCount: wp.phaseCount,
    currentFile: wp.currentFile,
    throughput: wp.throughput,
    state: wp.workerState || (wp.currentPhase ? 'active' : 'idle'),
    mode: wp.processingMode || 'full',
  } : null;

  const remoteWorkers = workers.filter(w => w.status === 'online').map(w => ({
    name: w.worker_name || w.username,
    phase: w.current_phase,
    currentFile: w.current_file,
    throughput: w.throughput,
    state: w.current_phase ? 'active' : 'idle',
    mode: w.processing_mode_override || 'full',
    isBuiltin: w.worker_name === '__builtin__',
  }));

  const allWorkers = localWorker ? [localWorker, ...remoteWorkers] : remoteWorkers;
  const activeWRs = activeWRsAll;
  const isRunning = throughput > 0 || processing > 0 || (wp?.currentPhase != null);

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
          {/* View toggle */}
          <div className="flex items-center border border-gray-700/40 rounded overflow-hidden">
            <button
              onClick={() => setView('pipeline')}
              className={`px-2 py-0.5 text-[9px] font-mono flex items-center gap-1 transition-colors ${view === 'pipeline' ? 'bg-gray-700/50 text-gray-300' : 'text-gray-600 hover:text-gray-400'}`}
            >
              <LayoutList size={10} />Pipeline
            </button>
            <button
              onClick={() => setView('board')}
              className={`px-2 py-0.5 text-[9px] font-mono flex items-center gap-1 transition-colors ${view === 'board' ? 'bg-gray-700/50 text-gray-300' : 'text-gray-600 hover:text-gray-400'}`}
            >
              <LayoutGrid size={10} />Board
            </button>
          </div>
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

       {view === 'board' ? (
        /* ══ BOARD VIEW — flat summary cards ══ */
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-w-5xl mx-auto">
          {/* WR cards */}
          {activeWRs.map(wr => <WRCard key={wr.id} wr={wr} onAction={handleWRAction} />)}

          {/* DL */}
          <BoardCard label="DOWNLOAD" color="blue" icon={<Download size={18} className="text-blue-400" />}
            value={dlWaiting} sub="waiting" active={dlWaiting > 0} />

          {/* Parser */}
          <BoardCard label="PARSER" color="teal"
            icon={<span className={`text-xl ${parsing > 0 || parsePending > 0 ? 'animate-spin-slow' : 'opacity-30'}`}>&#8862;</span>}
            value={parsing} sub={`${parsePending} queued`} active={parsing > 0 || parsePending > 0} />

          {/* Buffer */}
          <BoardCard label="BUFFER" color="orange" icon={<span className="text-xl">&#128230;</span>}
            value={buffer} sub="ready" active={buffer > 0} />

          {/* Workers */}
          {allWorkers.map((w, i) => {
            const mode = w.mode || 'full';
            const modeLabel = MODE_LABELS[mode];
            const modeColor = MODE_COLORS[mode];
            const isActive = w.state === 'active' || !!w.phase;
            const phaseCfg = w.phase ? PHASE_CFG[w.phase] : null;
            const PhaseIcon = phaseCfg?.icon;
            return (
              <div key={i} className={`rounded-xl border-2 p-3 ${isActive ? 'border-gray-600/40 bg-gray-800/25' : 'border-gray-800/20 bg-gray-900/15'}`}>
                <div className="flex items-center gap-1.5 mb-2">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-600'}`} />
                  <span className="text-[11px] font-mono text-gray-300 font-bold">{w.name}</span>
                  <span className={`text-[7px] font-mono ${modeColor} ml-auto`}>{modeLabel}</span>
                </div>
                <div className="flex items-center justify-center py-2">
                  {PhaseIcon ? (
                    <PhaseIcon size={24} className={`${phaseCfg.text} ${phaseCfg.anim}`} />
                  ) : (
                    <span className="text-gray-700 text-xl">&#9881;</span>
                  )}
                </div>
                <div className="text-center">
                  {w.throughput > 0 && <div className="text-[9px] font-mono text-gray-500 tabular-nums">{w.throughput.toFixed(1)}/m</div>}
                  {w.currentFile && <div className="text-[7px] text-gray-600 font-mono truncate">{w.currentFile}</div>}
                  {!isActive && <div className="text-[8px] text-gray-600 font-mono italic">{w.state === 'resting' ? t('bb.resting') : t('bb.phase_idle')}</div>}
                </div>
              </div>
            );
          })}

          {/* Storage */}
          <BoardCard label="STORAGE" color="green" icon={<span className="text-xl">&#9745;</span>}
            value={completed} sub={`${pct}%`} active={completed > 0} />

          {/* Failed */}
          {failed > 0 && (
            <BoardCard label="FAILED" color="red" icon={<AlertTriangle size={18} className="text-red-400" />}
              value={failed} sub="errors" active />
          )}
        </div>

       ) : (
        /* ══ PIPELINE VIEW — 6 stages ══ */
        <div className="flex flex-col gap-3 max-w-5xl mx-auto">

          {/* ── STAGE 1: Work Requests ── */}
          <Stage label="STAGE 1" title="WORK REQUESTS" color="gray">
            <div className="flex flex-wrap gap-2 mt-1">
              {activeWRs.length > 0 ? activeWRs.slice(0, 8).map(wr => (
                <WRCard key={wr.id} wr={wr} onAction={handleWRAction} />
              )) : (
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

          {/* ── STAGE 3: Parser ── */}
          <Stage label="STAGE 3" title="PARSER (Server CPU)" color="teal">
            <div className="flex items-center gap-4 mt-1">
              <div className={`rounded-lg border-2 ${parsing > 0 || parsePending > 0 ? 'border-teal-600/40 shadow-[0_0_12px_rgba(45,212,191,0.15)]' : 'border-gray-700/30'} bg-gradient-to-b from-teal-900/15 to-gray-900 w-16 h-16 flex flex-col items-center justify-center flex-shrink-0`}>
                <span className={`text-2xl ${parsing > 0 || parsePending > 0 ? 'animate-spin-slow' : 'opacity-30'}`}>&#8862;</span>
                <span className="text-[6px] text-teal-600 font-mono mt-0.5">1 thread</span>
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-4 mb-1.5">
                  <div>
                    <div className="text-[8px] font-mono text-gray-600 mb-0.5">parsing</div>
                    <span className="text-lg font-mono font-bold text-teal-400 tabular-nums">{parsing}</span>
                  </div>
                  <div>
                    <div className="text-[8px] font-mono text-gray-600 mb-0.5">await parse</div>
                    <span className="text-lg font-mono font-bold text-gray-400 tabular-nums">{parsePending}</span>
                  </div>
                </div>
                <div className="text-[7px] text-gray-600 font-mono">PSD/PNG → thumbnail + metadata extraction</div>
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
          <Stage label="STAGE 5" title="WORKERS (GPU Terminals)" color="purple">
            {(() => {
              const builtinWorkers = allWorkers.filter(w => w.isBuiltin);
              const externalWorkers = allWorkers.filter(w => !w.isBuiltin);
              return (
                <div className="space-y-3 mt-1">
                  {/* Built-in worker (server internal) */}
                  {builtinWorkers.length > 0 && (
                    <div>
                      <div className="text-[8px] font-mono text-gray-500 mb-1 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-sm bg-amber-500/60 flex-shrink-0" />
                        SERVER EMBEDDED
                      </div>
                      {builtinWorkers.map((w, i) => <WorkerLine key={`b-${i}`} worker={w} t={t} />)}
                    </div>
                  )}

                  {/* External workers */}
                  {externalWorkers.length > 0 && (
                    <div>
                      {builtinWorkers.length > 0 && (
                        <div className="text-[8px] font-mono text-gray-500 mb-1 flex items-center gap-1">
                          <span className="w-1.5 h-1.5 rounded-sm bg-purple-500/60 flex-shrink-0" />
                          EXTERNAL WORKERS
                        </div>
                      )}
                      {externalWorkers.map((w, i) => <WorkerLine key={`e-${i}`} worker={w} t={t} />)}
                    </div>
                  )}

                  {/* No workers */}
                  {allWorkers.length === 0 && (
                    <div className="text-center py-4 border border-dashed border-gray-700/30 rounded-lg">
                      <div className="text-gray-700 text-2xl mb-1">&#9881;</div>
                      <div className="text-[10px] text-gray-600 font-mono">{t('bb.no_workers')}</div>
                      <div className="text-[8px] text-gray-700 font-mono mt-1">Parser buffer paused (no demand)</div>
                    </div>
                  )}
                </div>
              );
            })()}
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
       )}
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

function Stage({ label, title, color, children }) {
  return (
    <div className={`border-2 rounded-xl p-3 relative ${STAGE_COLORS[color]}`}>
      <div className={`absolute -top-3 left-3 px-2 py-0 text-[9px] uppercase tracking-widest font-mono font-bold rounded-sm border ${LABEL_COLORS[color]}`}>
        {label} · {title}
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
      <div className="flex flex-col w-20 flex-shrink-0">
        <span className="text-[11px] font-mono text-gray-300 font-bold truncate">{w.isBuiltin ? 'Embedded' : w.name}</span>
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

// ─── Board Card (flat summary card) ──────────────────────

const BOARD_COLORS = {
  blue:   { border: 'border-blue-700/40',   bg: 'bg-blue-900/10',   text: 'text-blue-400',    label: 'text-blue-600' },
  teal:   { border: 'border-teal-700/40',   bg: 'bg-teal-900/10',   text: 'text-teal-400',    label: 'text-teal-600' },
  orange: { border: 'border-orange-700/40', bg: 'bg-orange-900/10', text: 'text-orange-400',  label: 'text-orange-600' },
  green:  { border: 'border-green-700/40',  bg: 'bg-green-900/10',  text: 'text-green-400',   label: 'text-green-600' },
  red:    { border: 'border-red-800/40',    bg: 'bg-red-900/10',    text: 'text-red-400',     label: 'text-red-600' },
};

function BoardCard({ label, color, icon, value, sub, active }) {
  const c = BOARD_COLORS[color] || BOARD_COLORS.blue;
  return (
    <div className={`rounded-xl border-2 ${active ? c.border : 'border-gray-800/20'} ${active ? c.bg : 'bg-gray-900/15'} p-3 text-center`}>
      <div className={`text-[7px] uppercase tracking-widest font-mono font-bold mb-2 ${c.label}`}>{label}</div>
      <div className="flex justify-center mb-1">{icon}</div>
      <div className={`text-xl font-mono font-bold tabular-nums ${active ? c.text : 'text-gray-600'}`}>{typeof value === 'number' ? value.toLocaleString() : value}</div>
      {sub && <div className="text-[8px] font-mono text-gray-600 mt-0.5">{sub}</div>}
    </div>
  );
}
