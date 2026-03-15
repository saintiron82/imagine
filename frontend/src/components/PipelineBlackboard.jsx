/**
 * PipelineBlackboard v2 — Factorio-style factory floor simulation.
 *
 * 3-zone layout: Input (WR tickets + buffer) | Worker Bay | Output (storage)
 * Each worker gets an independent production lane with 4 phase mini-bars.
 *
 * ┌──────────┬────────────────────────────────────┬────────────────┐
 * │ TICKETS  │          WORKER BAY                 │    STORAGE     │
 * │ ┌──────┐ │  ┌─────────────────────────────┐   │   ┌────────┐  │
 * │ │실내소품│─┤  │ W-1: ▶[P]==[MC██]==[VV]=[MV]▶│──▶│   │ 588  │  │
 * │ │58%   │ │  └─────────────────────────────┘   │   │ ██████ │  │
 * │ └──────┘ │  ┌─────────────────────────────┐   │   └────────┘  │
 * │ ┌──────┐ │  │ W-2: ▶[P██]==[MC]==[VV]=[MV]▶│──▶│              │
 * │ │BUFFER│ │  └─────────────────────────────┘   │   ⚠ 3 failed  │
 * │ │ 24   │ │                                     │              │
 * │ └──────┘ │                                     │              │
 * └──────────┴────────────────────────────────────┴────────────────┘
 */

import { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, Zap } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { getWorkRequests, listWorkerSessions } from '../api/admin';

// Phase color palette
const PHASE_COLORS = {
  parse:    { bg: 'bg-blue-500',    glow: 'shadow-[0_0_12px_rgba(59,130,246,0.5)]',  dim: 'bg-blue-900/30',    text: 'text-blue-400',    border: 'border-blue-500/60' },
  vision:   { bg: 'bg-purple-500',  glow: 'shadow-[0_0_12px_rgba(168,85,247,0.5)]',  dim: 'bg-purple-900/30',  text: 'text-purple-400',  border: 'border-purple-500/60' },
  embed_vv: { bg: 'bg-cyan-500',    glow: 'shadow-[0_0_12px_rgba(6,182,212,0.5)]',   dim: 'bg-cyan-900/30',    text: 'text-cyan-400',    border: 'border-cyan-500/60' },
  embed_mv: { bg: 'bg-emerald-500', glow: 'shadow-[0_0_12px_rgba(16,185,129,0.5)]',  dim: 'bg-emerald-900/30', text: 'text-emerald-400', border: 'border-emerald-500/60' },
};

const PHASE_LABELS = { parse: 'P', vision: 'MC', embed_vv: 'VV', embed_mv: 'MV' };
const PHASE_ORDER = ['parse', 'vision', 'embed_vv', 'embed_mv'];

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
  const completed = s.completed ?? 0;
  const failed = s.failed ?? 0;
  const total = (s.pending ?? 0) + (s.assigned ?? 0) + (s.processing ?? 0) + completed + failed;
  const remaining = (s.pending ?? 0) - (s.download_waiting ?? 0) + (s.assigned ?? 0) + (s.processing ?? 0);
  const etaMin = throughput > 0 ? Math.ceil(remaining / throughput) : null;
  const pct = total > 0 ? ((completed / total) * 100).toFixed(1) : '0.0';

  // Build unified worker list
  const wp = workerProgress;
  const localWorker = (isElectron && wp && (wp.currentPhase || wp.completed > 0)) ? {
    name: t('bb.local_worker'),
    phase: wp.currentPhase,
    phaseIndex: wp.phaseIndex,
    phaseCount: wp.phaseCount,
    currentFile: wp.currentFile,
    throughput: wp.throughput,
    state: wp.workerState || (wp.currentPhase ? 'active' : 'idle'),
  } : null;

  const remoteWorkers = workers.filter(w => w.status === 'online').map(w => ({
    name: w.worker_name || w.username,
    phase: w.current_phase,
    currentFile: w.current_file,
    throughput: w.throughput,
    state: w.current_phase ? 'active' : 'idle',
  }));

  const allWorkers = localWorker ? [localWorker, ...remoteWorkers] : remoteWorkers;
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
            <Zap size={11} className="inline mr-1 text-yellow-500" />
            {throughput.toFixed(1)} {t('bb.per_min')}
          </span>
        )}
      </div>

      {/* ── Factory Floor — 3 columns ── */}
      <div className="flex-1 flex min-h-0 relative">

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
            backgroundSize: '24px 24px',
          }}
        />

        {/* LEFT: Input Zone — WR Tickets + Buffer */}
        <InputZone
          workRequests={activeWRs}
          buffer={buffer}
          pending={pending}
          isRunning={isRunning}
          t={t}
        />

        {/* Pipe connector left→center */}
        <div className="flex flex-col justify-center">
          <PipeConnector active={isRunning} />
        </div>

        {/* CENTER: Worker Bay */}
        <WorkerBay
          workers={allWorkers}
          isRunning={isRunning}
          t={t}
        />

        {/* Pipe connector center→right */}
        <div className="flex flex-col justify-center">
          <PipeConnector active={isRunning} />
        </div>

        {/* RIGHT: Output Zone — Storage + Failed */}
        <OutputZone
          completed={completed}
          failed={failed}
          total={total}
          t={t}
        />
      </div>

      {/* ── Status Ribbon ── */}
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
          {pending > 0 && (
            <span>
              {t('bb.station_queue')} <span className="text-yellow-400">{pending.toLocaleString()}</span>
            </span>
          )}
          {active > 0 && (
            <span>
              {t('factory.summary_processing')} <span className="text-blue-400">{active}</span>
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

// ─── Input Zone (Left sidebar) ───────────────────────────────

function InputZone({ workRequests, buffer, pending, isRunning, t }) {
  return (
    <div className="w-[140px] flex-shrink-0 flex flex-col border-r border-gray-800/30 bg-gray-900/20 p-3 gap-3 overflow-y-auto">
      {/* Section label */}
      <div className="text-[9px] uppercase tracking-wider text-gray-600 font-mono font-bold">
        {t('bb.tickets')}
      </div>

      {/* WR ticket cards */}
      {workRequests.length > 0 ? (
        workRequests.slice(0, 8).map(wr => (
          <WRTicket key={wr.id} wr={wr} />
        ))
      ) : (
        <div className="text-[10px] text-gray-700 font-mono">{t('bb.no_wr')}</div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Buffer box */}
      <div className="rounded border border-yellow-700/30 bg-yellow-900/10 px-2 py-2">
        <div className="text-[9px] uppercase tracking-wider text-yellow-600/70 font-mono mb-1">
          {t('bb.buffer')}
        </div>
        <div className="text-xl font-mono font-bold text-yellow-400 tabular-nums">
          {buffer.toLocaleString()}
        </div>
      </div>

      {/* Pending total */}
      <div className="rounded border border-gray-700/30 bg-gray-800/20 px-2 py-2">
        <div className="text-[9px] uppercase tracking-wider text-gray-600 font-mono mb-1">
          {t('bb.station_queue')}
        </div>
        <div className="text-xl font-mono font-bold text-gray-400 tabular-nums">
          {pending.toLocaleString()}
        </div>
      </div>
    </div>
  );
}

// ─── Worker Bay (Center, flex-1) ─────────────────────────────

function WorkerBay({ workers, isRunning, t }) {
  return (
    <div className="flex-1 flex flex-col min-h-0 min-w-0 p-3">
      {/* Section label */}
      <div className="text-[9px] uppercase tracking-wider text-gray-600 font-mono font-bold mb-2">
        {t('bb.worker_bay')}
      </div>

      {/* Worker lanes */}
      <div className="flex-1 flex flex-col gap-2 overflow-y-auto pr-1">
        {workers.length > 0 ? (
          workers.map((w, i) => (
            <WorkerLane key={i} worker={w} isRunning={isRunning} t={t} />
          ))
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="text-gray-700 text-2xl mb-2">&#9881;</div>
              <div className="text-[11px] text-gray-600 font-mono">{t('bb.no_workers')}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Worker Lane (individual production line) ────────────────

function WorkerLane({ worker, isRunning, t }) {
  const w = worker;
  const isActive = w.state === 'active' || !!w.phase;
  const isIdle = !w.phase && w.state !== 'active';

  return (
    <div className={`
      rounded-lg border px-3 py-2 transition-all duration-300
      ${isActive
        ? 'border-gray-600/50 bg-gray-800/40'
        : 'border-gray-800/30 bg-gray-900/20'
      }
    `}>
      {/* Header: name + status + throughput */}
      <div className="flex items-center gap-2 mb-2">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
          isActive ? 'bg-green-500 animate-pulse' :
          w.state === 'resting' ? 'bg-yellow-500' : 'bg-gray-600'
        }`} />
        <span className="text-[11px] font-mono text-gray-300 font-bold truncate">
          {w.name}
        </span>
        {w.throughput > 0 && (
          <span className="text-[9px] font-mono text-gray-500 ml-auto tabular-nums flex-shrink-0">
            {w.throughput.toFixed(1)}/m
          </span>
        )}
        {isIdle && (
          <span className="text-[9px] font-mono text-gray-600 ml-auto italic">
            {w.state === 'resting' ? t('bb.resting') : t('bb.phase_idle')}
          </span>
        )}
      </div>

      {/* Phase pipeline — 4 mini-bars in a row */}
      <div className="flex items-center gap-1">
        {/* Input arrow */}
        <span className={`text-[8px] ${isActive ? 'text-gray-500' : 'text-gray-700'}`}>&#9654;</span>

        {PHASE_ORDER.map((phase, idx) => {
          const isCurrent = w.phase === phase;
          const isPast = w.phase && PHASE_ORDER.indexOf(w.phase) > idx;
          const colors = PHASE_COLORS[phase];
          const label = PHASE_LABELS[phase];

          // Progress for current phase (local worker only)
          let progress = 0;
          if (isCurrent && w.phaseCount > 0) {
            progress = ((w.phaseIndex || 0) / w.phaseCount) * 100;
          } else if (isPast) {
            progress = 100;
          }

          return (
            <div key={phase} className="flex items-center gap-1 flex-1 min-w-0">
              {/* Phase bar */}
              <div className={`
                relative flex-1 h-5 rounded overflow-hidden border transition-all duration-300
                ${isCurrent
                  ? `${colors.border} ${colors.glow}`
                  : isPast
                    ? `border-gray-600/30`
                    : 'border-gray-800/30'
                }
              `}>
                {/* Background */}
                <div className={`absolute inset-0 ${
                  isCurrent ? colors.dim :
                  isPast ? 'bg-gray-700/20' : 'bg-gray-800/20'
                }`} />
                {/* Fill bar */}
                <div
                  className={`absolute left-0 top-0 bottom-0 transition-all duration-500 ${
                    isCurrent ? colors.bg :
                    isPast ? 'bg-gray-600/40' : ''
                  } ${isCurrent ? 'opacity-80' : 'opacity-60'}`}
                  style={{ width: `${progress}%` }}
                />
                {/* Label */}
                <div className="relative z-10 flex items-center justify-center h-full">
                  <span className={`text-[9px] font-mono font-bold ${
                    isCurrent ? colors.text :
                    isPast ? 'text-gray-500' : 'text-gray-700'
                  }`}>
                    {label}
                    {isCurrent && w.phaseCount > 0 && (
                      <span className="ml-1 font-normal text-[8px] opacity-70">
                        {w.phaseIndex}/{w.phaseCount}
                      </span>
                    )}
                  </span>
                </div>
              </div>

              {/* Connector between phases */}
              {idx < PHASE_ORDER.length - 1 && (
                <div className={`w-2 h-0.5 flex-shrink-0 ${
                  isPast || isCurrent ? 'bg-gray-600' : 'bg-gray-800/40'
                }`} />
              )}
            </div>
          );
        })}

        {/* Output arrow */}
        <span className={`text-[8px] ${isActive ? 'text-gray-500' : 'text-gray-700'}`}>&#9654;</span>
      </div>

      {/* Current file */}
      {w.currentFile && (
        <div className="text-[9px] text-gray-600 truncate font-mono mt-1.5 pl-3">
          {w.currentFile}
        </div>
      )}
    </div>
  );
}

// ─── Output Zone (Right sidebar) ─────────────────────────────

function OutputZone({ completed, failed, total, t }) {
  const fillPct = total > 0 ? Math.min((completed / total) * 100, 100) : 0;

  return (
    <div className="w-[140px] flex-shrink-0 flex flex-col border-l border-gray-800/30 bg-gray-900/20 p-3 gap-3">
      {/* Section label */}
      <div className="text-[9px] uppercase tracking-wider text-gray-600 font-mono font-bold">
        {t('bb.storage')}
      </div>

      {/* Storage warehouse — fill level visualization */}
      <div className="flex-1 flex flex-col items-center justify-center">
        <div className="relative w-full aspect-square max-w-[110px] rounded-lg border-2 border-green-700/30 bg-green-900/5 overflow-hidden">
          {/* Fill level */}
          <div
            className="absolute bottom-0 left-0 right-0 bg-green-500/20 transition-all duration-700 ease-out"
            style={{ height: `${fillPct}%` }}
          >
            {/* Fill pattern — stacked blocks */}
            <div className="absolute inset-0 flex flex-wrap content-end p-1 gap-0.5 overflow-hidden">
              {Array.from({ length: Math.min(Math.ceil(fillPct / 4), 25) }).map((_, i) => (
                <div key={i} className="w-2 h-2 rounded-[2px] bg-green-500/40" />
              ))}
            </div>
          </div>

          {/* Center number */}
          <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
            <span className="text-2xl font-mono font-bold text-green-400 tabular-nums leading-none">
              {completed.toLocaleString()}
            </span>
            <span className="text-[9px] text-gray-500 font-mono mt-1">
              {t('bb.completed_label')}
            </span>
            {total > 0 && (
              <span className="text-[10px] font-mono text-green-600 mt-0.5 tabular-nums">
                {fillPct.toFixed(1)}%
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Failed count */}
      {failed > 0 && (
        <div className="rounded border border-red-800/30 bg-red-900/10 px-2 py-2 text-center">
          <AlertTriangle size={12} className="inline mr-1 text-red-500" />
          <span className="text-sm font-mono font-bold text-red-400 tabular-nums">{failed}</span>
          <div className="text-[9px] text-red-600 font-mono mt-0.5">{t('bb.failed')}</div>
        </div>
      )}

      {/* Total */}
      {total > 0 && (
        <div className="rounded border border-gray-700/30 bg-gray-800/20 px-2 py-2">
          <div className="text-[9px] uppercase tracking-wider text-gray-600 font-mono mb-1">
            {t('bb.total')}
          </div>
          <div className="text-lg font-mono font-bold text-gray-400 tabular-nums">
            {total.toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Pipe Connector ──────────────────────────────────────────

function PipeConnector({ active }) {
  return (
    <div className="w-4 flex items-center justify-center">
      <div className="flex flex-col items-center gap-0.5">
        <div className={`w-1 h-6 rounded-full ${active ? 'bg-gray-600' : 'bg-gray-800'}`} />
        <div className={`w-2 h-2 rounded-full ${active ? 'bg-gray-500 animate-pulse' : 'bg-gray-800'}`} />
        <div className={`w-1 h-6 rounded-full ${active ? 'bg-gray-600' : 'bg-gray-800'}`} />
      </div>
    </div>
  );
}

// ─── WR Ticket Card ──────────────────────────────────────────

function WRTicket({ wr }) {
  const total = wr.total_files || 0;
  const done = wr.completed_count || 0;
  const failed = wr.failed_count || 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const isActive = wr.status === 'processing';

  return (
    <div className={`
      rounded border px-2 py-1.5
      ${isActive ? 'border-blue-700/40 bg-blue-900/10' : 'border-gray-700/30 bg-gray-800/20'}
    `}>
      <div className="flex items-center gap-1.5 mb-1">
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-blue-500 animate-pulse' : 'bg-gray-600'}`} />
        <span className="text-[10px] text-gray-300 truncate leading-tight">{wr.name}</span>
      </div>
      {/* Progress bar */}
      <div className="h-1 bg-gray-700/50 rounded-full overflow-hidden mb-1">
        <div
          className={`h-full rounded-full transition-all duration-500 ${isActive ? 'bg-blue-500' : 'bg-gray-600'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-mono text-gray-600 tabular-nums">{done}/{total}</span>
        <span className="text-[9px] font-mono text-gray-500 tabular-nums">{pct.toFixed(0)}%</span>
      </div>
      {failed > 0 && (
        <span className="text-[8px] text-red-400 font-mono">
          <AlertTriangle size={7} className="inline mr-0.5" />{failed}
        </span>
      )}
    </div>
  );
}
