/**
 * PipelineBlackboard — Factory-floor style pipeline visualization.
 *
 * Layout: a single factory scene with conveyor belt, machine stations,
 * worker units, and storage buffers — like a factory simulation game.
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        FACTORY FLOOR                            │
 * │                                                                 │
 * │   ┌─────┐    ╔═══════╗    ╔═══════╗    ╔═══════╗    ┌─────┐   │
 * │   │INPUT│───▶║ PARSE ║───▶║  MC   ║───▶║ EMBED ║───▶│ OUT │   │
 * │   │ 412 │    ║  ⚙ 3  ║    ║  ⚙ 1  ║    ║  ⚙ 8  ║    │ 588 │   │
 * │   └─────┘    ╚═══════╝    ╚═══════╝    ╚═══════╝    └─────┘   │
 * │                  │              │              │                 │
 * │              [Worker-1]    [Worker-1]    [Worker-1]             │
 * │               MC 3/5        idle          VV 2/8               │
 * │                                                                 │
 * │   ── Status Bar: 12.5/min | ETA ~35m | 588/1000 (58.8%) ──    │
 * └─────────────────────────────────────────────────────────────────┘
 */

import { useState, useEffect, useCallback } from 'react';
import { AlertTriangle } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { getWorkRequests, listWorkerSessions } from '../api/admin';

// ─── Main Component ────────────────────────────────────────────

export default function PipelineBlackboard({ workerProgress }) {
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

  const s = stats || {};
  const throughput = s.throughput ?? 0;
  const pending = (s.pending ?? 0) + (s.download_waiting ?? 0);
  const buffer = s.parse_ahead_parsed ?? 0;
  const active = (s.assigned ?? 0) + (s.processing ?? 0);
  const embedWaiting = Math.max(0, (s.phase_vision_done ?? 0) - (s.phase_embed_done ?? 0));
  const completed = s.completed ?? 0;
  const failed = s.failed ?? 0;
  const total = (s.pending ?? 0) + (s.assigned ?? 0) + (s.processing ?? 0) + completed + failed;
  const remaining = (s.pending ?? 0) - (s.download_waiting ?? 0) + (s.assigned ?? 0) + (s.processing ?? 0);
  const etaMin = throughput > 0 ? Math.ceil(remaining / throughput) : null;
  const pct = total > 0 ? ((completed / total) * 100).toFixed(1) : '0.0';

  // Active phase from workerProgress
  const wp = workerProgress;
  const activePhase = wp?.currentPhase || null;
  const phaseMap = { parse: 'parse', vision: 'mc', embed_vv: 'embed', embed_mv: 'embed' };
  const activeName = phaseMap[activePhase] || null;

  // Worker info for machine display
  const localWorker = (isElectron && wp && (wp.currentPhase || wp.completed > 0)) ? {
    name: t('bb.local_worker'),
    phase: wp.currentPhase,
    phaseLabel: { parse: 'P', vision: 'MC', embed_vv: 'VV', embed_mv: 'MV', uploading: 'UP' }[wp.currentPhase] || '',
    phaseIndex: wp.phaseIndex,
    phaseCount: wp.phaseCount,
    currentFile: wp.currentFile,
    throughput: wp.throughput,
    state: wp.workerState,
  } : null;

  const remoteWorkers = workers.filter(w => w.status === 'online').map(w => ({
    name: w.worker_name || w.username,
    phase: w.current_phase,
    phaseLabel: w.current_phase?.toUpperCase()?.slice(0, 2) || '',
    currentFile: w.current_file,
    throughput: w.throughput,
    state: 'active',
  }));

  const allWorkers = localWorker ? [localWorker, ...remoteWorkers] : remoteWorkers;

  // Active WRs
  const activeWRs = workRequests.filter(wr => wr.status === 'queued' || wr.status === 'processing');

  const isRunning = throughput > 0 || active > 0 || (wp?.currentPhase != null);

  return (
    <div className="h-full flex flex-col bg-gray-950 select-none overflow-hidden">

      {/* ── Title Bar ── */}
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
        {throughput > 0 && (
          <span className="text-xs font-mono text-gray-400 tabular-nums">
            {throughput.toFixed(1)} {t('bb.per_min')}
          </span>
        )}
      </div>

      {/* ── Factory Floor ── */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-4 min-h-0 gap-6">

        {/* ── Assembly Line ── */}
        <div className="flex items-stretch gap-0 w-full max-w-4xl">

          {/* INPUT hopper */}
          <StorageSilo
            label={t('bb.station_queue')}
            count={pending}
            color="yellow"
            side="input"
          />

          {/* Conveyor + Machine stations */}
          <div className="flex-1 flex items-center">
            <Conveyor active={isRunning} />
            <MachineStation
              label={t('bb.station_parse')}
              count={buffer}
              color="teal"
              active={activeName === 'parse'}
              workers={allWorkers.filter(w => w.phase === 'parse')}
            />
            <Conveyor active={isRunning} />
            <MachineStation
              label={t('bb.station_vision')}
              count={active}
              color="purple"
              active={activeName === 'mc'}
              workers={allWorkers.filter(w => w.phase === 'vision')}
            />
            <Conveyor active={isRunning} />
            <MachineStation
              label={t('bb.station_embed')}
              count={embedWaiting}
              color="cyan"
              active={activeName === 'embed'}
              workers={allWorkers.filter(w => w.phase === 'embed_vv' || w.phase === 'embed_mv')}
            />
            <Conveyor active={isRunning} />
          </div>

          {/* OUTPUT storage */}
          <StorageSilo
            label={t('bb.station_done')}
            count={completed}
            color="green"
            side="output"
            failed={failed}
          />
        </div>

        {/* ── Worker Bay (below the line) ── */}
        {allWorkers.length > 0 && (
          <div className="w-full max-w-4xl">
            <div className="text-[9px] uppercase tracking-wider text-gray-600 font-mono mb-1.5 px-1">
              {t('bb.workers_title')}
            </div>
            <div className="flex gap-2 flex-wrap">
              {allWorkers.map((w, i) => (
                <WorkerUnit key={i} worker={w} />
              ))}
            </div>
          </div>
        )}

        {/* ── Work Request Queue (left side, like order tickets) ── */}
        {activeWRs.length > 0 && (
          <div className="w-full max-w-4xl">
            <div className="text-[9px] uppercase tracking-wider text-gray-600 font-mono mb-1.5 px-1">
              {t('bb.wr_title')}
            </div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {activeWRs.slice(0, 6).map(wr => (
                <WRTicket key={wr.id} wr={wr} />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Status Ribbon (bottom) ── */}
      <div className="px-4 py-2 border-t border-gray-800/50 bg-gray-900/50">
        <div className="flex items-center justify-center gap-6 text-[11px] font-mono text-gray-500">
          {throughput > 0 && (
            <span>
              <span className="text-blue-400 font-bold">{throughput.toFixed(1)}</span> {t('bb.per_min')}
            </span>
          )}
          {etaMin != null && (
            <span>
              {t('bb.eta')}: <span className="text-gray-300">
                {etaMin >= 60 ? `${Math.floor(etaMin / 60)}h ${etaMin % 60}m` : `${etaMin}m`}
              </span>
            </span>
          )}
          {total > 0 && (
            <span>
              <span className="text-green-400">{completed.toLocaleString()}</span>
              /{total.toLocaleString()} (<span className="text-gray-300">{pct}%</span>)
            </span>
          )}
          {failed > 0 && (
            <span className="text-red-400">
              <AlertTriangle size={10} className="inline mr-0.5 -mt-0.5" />
              {failed}
            </span>
          )}
          {!isRunning && total === 0 && (
            <span className="text-gray-600">{t('bb.idle')}</span>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Storage Silo (Input / Output) ─────────────────────────────

const siloColors = {
  yellow: { border: 'border-yellow-600/40', bg: 'bg-yellow-900/20', text: 'text-yellow-400', fill: 'bg-yellow-500/30' },
  green: { border: 'border-green-600/40', bg: 'bg-green-900/20', text: 'text-green-400', fill: 'bg-green-500/30' },
};

function StorageSilo({ label, count, color, side, failed }) {
  const c = siloColors[color] || siloColors.yellow;
  const maxDisplay = 100;
  const fillPct = Math.min((count / maxDisplay) * 100, 100);

  return (
    <div className={`
      flex flex-col items-center justify-end w-20 min-h-[120px]
      rounded-lg border-2 ${c.border} ${c.bg}
      relative overflow-hidden
    `}>
      {/* Fill level */}
      <div
        className={`absolute bottom-0 left-0 right-0 ${c.fill} transition-all duration-700`}
        style={{ height: `${fillPct}%` }}
      />
      {/* Content */}
      <div className="relative z-10 flex flex-col items-center py-3 gap-1">
        <span className={`text-2xl font-mono font-bold tabular-nums ${c.text}`}>
          {count.toLocaleString()}
        </span>
        <span className="text-[9px] uppercase tracking-wider text-gray-500">
          {label}
        </span>
        {failed > 0 && (
          <span className="text-[9px] text-red-400 font-mono">
            <AlertTriangle size={9} className="inline mr-0.5" />{failed}
          </span>
        )}
      </div>
      {/* Pipe connector */}
      <div className={`absolute top-1/2 -translate-y-1/2 ${side === 'input' ? '-right-1' : '-left-1'} w-2 h-4 bg-gray-700 rounded-sm`} />
    </div>
  );
}

// ─── Machine Station ────────────────────────────────────────────

const machineColors = {
  teal: { border: 'border-teal-500/40', bg: 'bg-teal-900/15', text: 'text-teal-400', glow: 'shadow-[0_0_20px_rgba(20,184,166,0.12)]', gear: 'text-teal-500' },
  purple: { border: 'border-purple-500/40', bg: 'bg-purple-900/15', text: 'text-purple-400', glow: 'shadow-[0_0_20px_rgba(168,85,247,0.12)]', gear: 'text-purple-500' },
  cyan: { border: 'border-cyan-500/40', bg: 'bg-cyan-900/15', text: 'text-cyan-400', glow: 'shadow-[0_0_20px_rgba(6,182,212,0.12)]', gear: 'text-cyan-500' },
};

function MachineStation({ label, count, color, active, workers }) {
  const c = machineColors[color] || machineColors.teal;
  const hasWorker = workers.length > 0;

  return (
    <div className="flex flex-col items-center gap-1">
      {/* Machine body */}
      <div className={`
        flex flex-col items-center justify-center
        w-24 min-h-[100px] rounded-lg border-2
        transition-all duration-300
        ${active ? `${c.border} ${c.bg} ${c.glow}` : 'border-gray-700/30 bg-gray-800/30'}
      `}>
        {/* Gear icon (spinning when active) */}
        <span className={`text-xl mb-1 ${active ? c.gear : 'text-gray-600'} ${active ? 'animate-spin-slow' : ''}`}>
          ⚙
        </span>
        <span className={`text-lg font-mono font-bold tabular-nums ${active ? c.text : 'text-gray-500'}`}>
          {count.toLocaleString()}
        </span>
        <span className="text-[9px] uppercase tracking-wider text-gray-500 mt-0.5">
          {label}
        </span>
      </div>
      {/* Worker attachment indicator */}
      {hasWorker && (
        <div className="flex gap-0.5">
          {workers.map((_, i) => (
            <span key={i} className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Conveyor Belt ──────────────────────────────────────────────

function Conveyor({ active }) {
  return (
    <div className="flex-1 min-w-[24px] flex items-center justify-center h-[100px]">
      <div className="relative w-full h-2 flex items-center">
        {/* Belt track */}
        <div className={`absolute inset-0 rounded-full ${active ? 'bg-gray-700' : 'bg-gray-800'}`} />
        {/* Belt pattern (dashes) */}
        <div className={`
          absolute inset-0 rounded-full overflow-hidden
        `}>
          <div className={`
            h-full w-[200%] flex items-center
            ${active ? 'animate-conveyor' : ''}
          `}>
            {Array.from({ length: 12 }).map((_, i) => (
              <div key={i} className="flex items-center">
                <div className={`w-2 h-1 rounded-sm ${active ? 'bg-gray-500/60' : 'bg-gray-700/40'}`} />
                <div className="w-2" />
              </div>
            ))}
          </div>
        </div>
        {/* Arrow */}
        <div className="absolute right-0 translate-x-1">
          <div className={`
            w-0 h-0
            border-t-[5px] border-t-transparent
            border-b-[5px] border-b-transparent
            border-l-[7px] ${active ? 'border-l-gray-500' : 'border-l-gray-700'}
          `} />
        </div>
      </div>
    </div>
  );
}

// ─── Worker Unit Card ───────────────────────────────────────────

function WorkerUnit({ worker }) {
  const w = worker;
  const isActive = w.state === 'active' || w.phase;
  const dotColor = isActive ? 'bg-green-500' : w.state === 'resting' ? 'bg-yellow-500' : 'bg-gray-500';

  return (
    <div className={`
      rounded border px-2.5 py-1.5 min-w-[140px]
      ${isActive ? 'border-gray-600/50 bg-gray-800/50' : 'border-gray-800/40 bg-gray-900/30'}
    `}>
      <div className="flex items-center gap-1.5 mb-1">
        <span className={`w-2 h-2 rounded-full ${dotColor} ${isActive ? 'animate-pulse' : ''}`} />
        <span className="text-[11px] font-mono text-gray-300 truncate">{w.name}</span>
        {w.throughput > 0 && (
          <span className="text-[9px] font-mono text-gray-600 ml-auto tabular-nums">{w.throughput.toFixed(1)}/m</span>
        )}
      </div>
      {w.phase && (
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-mono text-gray-400 font-bold w-5">{w.phaseLabel}</span>
          {w.phaseCount > 0 && (
            <>
              <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${((w.phaseIndex || 0) / w.phaseCount) * 100}%` }}
                />
              </div>
              <span className="text-[9px] font-mono text-gray-600 tabular-nums">
                {w.phaseIndex}/{w.phaseCount}
              </span>
            </>
          )}
        </div>
      )}
      {w.currentFile && (
        <div className="text-[9px] text-gray-600 truncate font-mono mt-0.5">{w.currentFile}</div>
      )}
    </div>
  );
}

// ─── Work Request Ticket ────────────────────────────────────────

function WRTicket({ wr }) {
  const total = wr.total_files || 0;
  const done = wr.completed_count || 0;
  const failed = wr.failed_count || 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const isActive = wr.status === 'processing';

  return (
    <div className={`
      flex-shrink-0 w-44 rounded border px-2.5 py-2
      ${isActive ? 'border-blue-700/40 bg-blue-900/10' : 'border-gray-700/30 bg-gray-800/20'}
    `}>
      <div className="flex items-center gap-1.5 mb-1">
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-blue-500 animate-pulse' : 'bg-gray-600'}`} />
        <span className="text-[11px] text-gray-300 truncate">{wr.name}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${isActive ? 'bg-blue-500' : 'bg-gray-600'}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-[9px] font-mono text-gray-500 tabular-nums">{pct.toFixed(0)}%</span>
      </div>
      <div className="flex items-center justify-between mt-0.5">
        <span className="text-[9px] font-mono text-gray-600 tabular-nums">{done}/{total}</span>
        {failed > 0 && (
          <span className="text-[9px] text-red-400 font-mono">
            <AlertTriangle size={8} className="inline mr-0.5" />{failed}
          </span>
        )}
      </div>
    </div>
  );
}
