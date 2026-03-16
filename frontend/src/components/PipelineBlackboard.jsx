/**
 * PipelineBlackboard v5 — Factorio-style horizontal assembly lines.
 *
 * Flat factory floor, no sidebars. Left-to-right flow:
 * - Top: WR tickets + buffer + pending (flat objects)
 * - Center: Each worker = horizontal assembly line with 4 machine stations
 * - Bottom: Storage warehouse + failed count
 *
 * Each worker line: ▶═[P]═[MC]═[VV]═[MV]═▶  W-1 4.2/m hero.psd
 */

import { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, Zap, Layers, Eye, Scan, Brain, ChevronRight, Check, Package, Inbox } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { getWorkRequests, listWorkerSessions } from '../api/admin';

// ─── Phase config ────────────────────────────────────────────

const PHASE_COLORS = {
  parse:    { bg: 'bg-blue-500',    glow: 'shadow-[0_0_16px_rgba(59,130,246,0.4)]',   dim: 'bg-blue-900/20',    text: 'text-blue-400',    border: 'border-blue-500/60',    dot: 'bg-blue-400' },
  vision:   { bg: 'bg-purple-500',  glow: 'shadow-[0_0_16px_rgba(168,85,247,0.4)]',   dim: 'bg-purple-900/20',  text: 'text-purple-400',  border: 'border-purple-500/60',  dot: 'bg-purple-400' },
  embed_vv: { bg: 'bg-cyan-500',    glow: 'shadow-[0_0_16px_rgba(6,182,212,0.4)]',    dim: 'bg-cyan-900/20',    text: 'text-cyan-400',    border: 'border-cyan-500/60',    dot: 'bg-cyan-400' },
  embed_mv: { bg: 'bg-emerald-500', glow: 'shadow-[0_0_16px_rgba(16,185,129,0.4)]',   dim: 'bg-emerald-900/20', text: 'text-emerald-400', border: 'border-emerald-500/60', dot: 'bg-emerald-400' },
};

const PHASE_LABELS = { parse: 'P', vision: 'MC', embed_vv: 'VV', embed_mv: 'MV' };
const PHASE_ICONS = { parse: Layers, vision: Eye, embed_vv: Scan, embed_mv: Brain };
const PHASE_ANIMS = { parse: 'animate-bounce', vision: 'animate-pulse', embed_vv: 'animate-spin-slow', embed_mv: 'animate-machine-pulse' };
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

      {/* ── Factory Floor ── */}
      <div className="flex-1 overflow-auto relative p-4">

        {/* Grid pattern */}
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
            backgroundSize: '24px 24px',
          }}
        />

        <div className="relative z-10 flex flex-col gap-4">

          {/* ── Input objects row ── */}
          <div className="flex flex-wrap items-center gap-2">
            {activeWRs.slice(0, 6).map(wr => (
              <WRTicket key={wr.id} wr={wr} />
            ))}
            {buffer > 0 && (
              <FloorBlock icon={<Package size={14} className="text-yellow-500" />} label={t('bb.buffer')} value={buffer} color="yellow" />
            )}
            {pending > 0 && (
              <FloorBlock icon={<Inbox size={14} className="text-gray-400" />} label={t('bb.station_queue')} value={pending} color="gray" />
            )}
            {activeWRs.length === 0 && pending === 0 && buffer === 0 && !isRunning && (
              <div className="text-[10px] text-gray-700 font-mono">{t('bb.no_wr')}</div>
            )}
          </div>

          {/* ── Worker assembly lines ── */}
          <div className="flex flex-col gap-3">
            {allWorkers.length > 0 ? (
              allWorkers.map((w, i) => (
                <WorkerLine key={i} worker={w} t={t} />
              ))
            ) : (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="text-gray-700 text-3xl mb-2">&#9881;</div>
                  <div className="text-[11px] text-gray-600 font-mono">{t('bb.no_workers')}</div>
                </div>
              </div>
            )}
          </div>

          {/* ── Output objects row ── */}
          {(completed > 0 || failed > 0 || total > 0) && (
            <div className="flex flex-wrap items-center gap-3">
              <StorageBlock completed={completed} total={total} t={t} />
              {failed > 0 && (
                <div className="rounded-lg border border-red-800/40 bg-red-900/10 px-3 py-2 flex items-center gap-2">
                  <AlertTriangle size={14} className="text-red-500" />
                  <div>
                    <div className="text-sm font-mono font-bold text-red-400 tabular-nums">{failed}</div>
                    <div className="text-[8px] text-red-600 font-mono uppercase">{t('bb.failed')}</div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Status Ribbon ── */}
      <div className="px-4 py-2 border-t border-gray-800/50 bg-gray-900/50">
        <div className="flex items-center justify-center gap-6 text-[11px] font-mono text-gray-500">
          {throughput > 0 && (
            <span><span className="text-blue-400 font-bold">{throughput.toFixed(1)}</span> {t('bb.per_min')}</span>
          )}
          {etaMin != null && (
            <span>{t('bb.eta')}: <span className="text-gray-300">{etaMin >= 60 ? `${Math.floor(etaMin / 60)}h ${etaMin % 60}m` : `${etaMin}m`}</span></span>
          )}
          {total > 0 && (
            <span><span className="text-green-400">{completed.toLocaleString()}</span>/{total.toLocaleString()} (<span className="text-gray-300">{pct}%</span>)</span>
          )}
          {active > 0 && (
            <span>{t('factory.summary_processing')} <span className="text-blue-400">{active}</span></span>
          )}
          {failed > 0 && (
            <span className="text-red-400"><AlertTriangle size={10} className="inline mr-0.5 -mt-0.5" />{failed}</span>
          )}
          {!isRunning && total === 0 && (
            <span className="text-gray-600">{t('bb.idle')}</span>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Worker Line (horizontal assembly line) ──────────────────

function WorkerLine({ worker, t }) {
  const w = worker;
  const isActive = w.state === 'active' || !!w.phase;
  const isIdle = !w.phase && w.state !== 'active';
  const currentPhaseIdx = w.phase ? PHASE_ORDER.indexOf(w.phase) : -1;

  return (
    <div className={`
      rounded-lg border px-3 py-2.5 transition-all duration-300
      ${isActive ? 'border-gray-600/40 bg-gray-800/20' : 'border-gray-800/20 bg-gray-900/10'}
    `}>
      <div className="flex items-center gap-2">

        {/* Input chevron */}
        <ChevronRight size={12} className={`flex-shrink-0 ${isActive ? 'text-gray-500' : 'text-gray-700'}`} />

        {/* Machine stations + conveyors */}
        <div className="flex items-center flex-1 min-w-0">
          {PHASE_ORDER.map((phase, idx) => {
            const isCurrent = w.phase === phase;
            const isPast = currentPhaseIdx > idx;

            let progress = 0;
            if (isCurrent && w.phaseCount > 0) {
              progress = ((w.phaseIndex || 0) / w.phaseCount) * 100;
            } else if (isPast) {
              progress = 100;
            }

            const conveyorActive = isActive && (isCurrent || isPast);
            const prevPhase = idx > 0 ? PHASE_ORDER[idx - 1] : null;
            const dotColor = prevPhase ? PHASE_COLORS[prevPhase].dot : 'bg-gray-400';

            return (
              <div key={phase} className="flex items-center">
                {/* Conveyor before each machine */}
                <ConveyorSegment active={conveyorActive} dotColor={conveyorActive ? dotColor : null} />
                {/* Machine */}
                <MachineStation
                  phase={phase}
                  isCurrent={isCurrent}
                  isPast={isPast}
                  progress={progress}
                  count={isCurrent ? (w.phaseIndex || 0) : 0}
                  total={isCurrent ? (w.phaseCount || 0) : 0}
                />
              </div>
            );
          })}

          {/* Final conveyor out */}
          <ConveyorSegment
            active={isActive && currentPhaseIdx >= 0}
            dotColor={isActive && w.phase === 'embed_mv' ? PHASE_COLORS.embed_mv.dot : null}
          />
        </div>

        {/* Output chevron */}
        <ChevronRight size={12} className={`flex-shrink-0 ${isActive ? 'text-gray-500' : 'text-gray-700'}`} />

        {/* Worker info (right side) */}
        <div className="flex items-center gap-2 ml-2 flex-shrink-0 min-w-[140px]">
          <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
            isActive ? 'bg-green-500 animate-pulse' :
            w.state === 'resting' ? 'bg-yellow-500' : 'bg-gray-600'
          }`} />
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-mono text-gray-300 font-bold truncate">{w.name}</span>
              {w.throughput > 0 && (
                <span className="text-[9px] font-mono text-gray-500 tabular-nums flex-shrink-0">{w.throughput.toFixed(1)}/m</span>
              )}
              {isIdle && (
                <span className="text-[9px] font-mono text-gray-600 italic flex-shrink-0">
                  {w.state === 'resting' ? t('bb.resting') : t('bb.phase_idle')}
                </span>
              )}
            </div>
            {w.currentFile && (
              <div className="text-[9px] text-gray-600 truncate font-mono">{w.currentFile}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Machine Station (56×56 machine block) ───────────────────

function MachineStation({ phase, isCurrent, isPast, progress, count, total }) {
  const colors = PHASE_COLORS[phase];
  const label = PHASE_LABELS[phase];
  const Icon = PHASE_ICONS[phase];
  const anim = PHASE_ANIMS[phase];

  return (
    <div className={`
      relative w-14 h-14 rounded-lg border-2 flex flex-col items-center justify-center
      transition-all duration-300 flex-shrink-0
      ${isCurrent
        ? `${colors.border} ${colors.glow} bg-gradient-to-b from-gray-800/80 to-gray-900`
        : isPast
          ? 'border-gray-600/30 bg-gray-800/40'
          : 'border-gray-700/20 bg-gray-800/60'
      }
    `}>
      {/* Name plate */}
      <div className={`
        absolute -top-2 left-1/2 -translate-x-1/2 px-1.5 py-0
        rounded-sm text-[7px] uppercase tracking-widest font-mono font-bold z-10
        ${isCurrent
          ? `${colors.text} bg-gray-900/90 border ${colors.border}`
          : 'text-gray-600 bg-gray-800/90 border border-gray-700/40'
        }
      `}>
        {label}
      </div>

      {/* Center icon */}
      <Icon
        size={20}
        className={`
          transition-all duration-300
          ${isCurrent
            ? `${colors.text} ${anim}`
            : isPast
              ? 'text-gray-500 opacity-30'
              : 'text-gray-700 opacity-20'
          }
        `}
      />

      {/* Completed checkmark */}
      {isPast && (
        <div className="absolute top-1 right-1">
          <Check size={8} className="text-green-500/60" />
        </div>
      )}

      {/* Bottom gauge */}
      <div className="absolute bottom-1 left-1.5 right-1.5 flex items-center gap-0.5">
        <div className="flex-1 h-1 rounded-full bg-gray-700/50 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isCurrent ? colors.bg : isPast ? 'bg-gray-600/40' : ''
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
        {isCurrent && total > 0 && (
          <span className={`text-[6px] font-mono tabular-nums ${colors.text} opacity-70`}>
            {count}/{total}
          </span>
        )}
      </div>
    </div>
  );
}

// ─── Conveyor Segment ────────────────────────────────────────

function ConveyorSegment({ active, dotColor }) {
  return (
    <div className="relative w-8 h-14 flex items-center justify-center flex-shrink-0">
      {/* Belt track */}
      <div className={`w-full h-2 rounded-sm relative overflow-hidden ${active ? 'bg-gray-700/60' : 'bg-gray-800/40'}`}>
        <div
          className={`absolute inset-0 ${active ? 'animate-conveyor' : ''}`}
          style={{
            width: '200%',
            backgroundImage: 'repeating-linear-gradient(90deg, transparent, transparent 4px, rgba(255,255,255,0.06) 4px, rgba(255,255,255,0.06) 6px)',
          }}
        />
      </div>
      {/* Flowing dots */}
      {active && dotColor && (
        <>
          <div className={`absolute w-1.5 h-1.5 rounded-sm ${dotColor} animate-item-flow opacity-80`}
            style={{ top: '50%', marginTop: '-3px', animationDelay: '0s' }} />
          <div className={`absolute w-1.5 h-1.5 rounded-sm ${dotColor} animate-item-flow opacity-80`}
            style={{ top: '50%', marginTop: '-3px', animationDelay: '0.5s' }} />
          <div className={`absolute w-1.5 h-1.5 rounded-sm ${dotColor} animate-item-flow opacity-80`}
            style={{ top: '50%', marginTop: '-3px', animationDelay: '1s' }} />
        </>
      )}
    </div>
  );
}

// ─── Floor Block ─────────────────────────────────────────────

function FloorBlock({ icon, label, value, color }) {
  const styles = {
    yellow: { border: 'border-yellow-700/30', bg: 'bg-yellow-900/10', text: 'text-yellow-400' },
    gray:   { border: 'border-gray-700/30',   bg: 'bg-gray-800/20',   text: 'text-gray-400' },
  };
  const s = styles[color] || styles.gray;

  return (
    <div className={`rounded-lg border ${s.border} ${s.bg} px-3 py-2 flex items-center gap-2`}>
      {icon}
      <div>
        <div className={`text-sm font-mono font-bold tabular-nums ${s.text}`}>{value.toLocaleString()}</div>
        <div className="text-[8px] uppercase tracking-wider text-gray-600 font-mono">{label}</div>
      </div>
    </div>
  );
}

// ─── Storage Block ───────────────────────────────────────────

function StorageBlock({ completed, total, t }) {
  const fillPct = total > 0 ? Math.min((completed / total) * 100, 100) : 0;

  return (
    <div className="rounded-lg border border-green-700/30 bg-green-900/5 overflow-hidden">
      <div className="relative px-4 py-2.5 flex items-center gap-3">
        <div
          className="absolute bottom-0 left-0 right-0 bg-green-500/10 transition-all duration-700"
          style={{ height: `${fillPct}%` }}
        />
        <div className="relative z-10 flex items-center gap-3">
          {/* Mini fill blocks */}
          <div className="flex flex-wrap gap-0.5 w-8">
            {Array.from({ length: Math.min(Math.ceil(fillPct / 8), 12) }).map((_, i) => (
              <div key={i} className="w-1.5 h-1.5 rounded-[1px] bg-green-500/40" />
            ))}
          </div>
          <div>
            <div className="text-lg font-mono font-bold text-green-400 tabular-nums leading-none">
              {completed.toLocaleString()}
            </div>
            <div className="text-[8px] text-gray-500 font-mono uppercase">
              {t('bb.completed_label')} {total > 0 && <span className="text-green-600">{fillPct.toFixed(1)}%</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── WR Ticket ───────────────────────────────────────────────

function WRTicket({ wr }) {
  const total = wr.total_files || 0;
  const done = wr.completed_count || 0;
  const failed = wr.failed_count || 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const isActive = wr.status === 'processing';

  return (
    <div className={`rounded-lg border px-3 py-2 min-w-[130px] ${isActive ? 'border-blue-700/40 bg-blue-900/10' : 'border-gray-700/30 bg-gray-800/20'}`}>
      <div className="flex items-center gap-1.5 mb-1.5">
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-blue-500 animate-pulse' : 'bg-gray-600'}`} />
        <span className="text-[10px] text-gray-300 truncate leading-tight font-medium">{wr.name}</span>
      </div>
      <div className="h-1 bg-gray-700/50 rounded-full overflow-hidden mb-1">
        <div className={`h-full rounded-full transition-all duration-500 ${isActive ? 'bg-blue-500' : 'bg-gray-600'}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-mono text-gray-600 tabular-nums">{done}/{total}</span>
        <span className="text-[9px] font-mono text-gray-500 tabular-nums">{pct.toFixed(0)}%</span>
      </div>
      {failed > 0 && (
        <span className="text-[8px] text-red-400 font-mono"><AlertTriangle size={7} className="inline mr-0.5" />{failed}</span>
      )}
    </div>
  );
}
