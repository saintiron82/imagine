/**
 * PipelineBlackboard v4 — Factorio-style flat factory floor.
 *
 * No sidebars. Everything placed on one 2D factory plane:
 * - Top row: WR tickets + buffer/pending as flat objects
 * - Center: Worker terminals (each is a single machine with internal P→MC→VV→MV pipeline)
 * - Bottom row: Storage warehouse + failed count
 *
 * Each worker = one machine terminal block with:
 * - Active phase icon (large, animated)
 * - Compact internal phase pipeline strip (P→MC→VV→MV)
 * - Current file, throughput
 */

import { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, Zap, Layers, Eye, Scan, Brain, Check, Package, Inbox } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { getWorkRequests, listWorkerSessions } from '../api/admin';

// ─── Phase config ────────────────────────────────────────────

const PHASE_COLORS = {
  parse:    { bg: 'bg-blue-500',    glow: 'shadow-[0_0_20px_rgba(59,130,246,0.5)]',   dim: 'bg-blue-900/30',    text: 'text-blue-400',    border: 'border-blue-500/60',    ring: 'ring-blue-500/40' },
  vision:   { bg: 'bg-purple-500',  glow: 'shadow-[0_0_20px_rgba(168,85,247,0.5)]',   dim: 'bg-purple-900/30',  text: 'text-purple-400',  border: 'border-purple-500/60',  ring: 'ring-purple-500/40' },
  embed_vv: { bg: 'bg-cyan-500',    glow: 'shadow-[0_0_20px_rgba(6,182,212,0.5)]',    dim: 'bg-cyan-900/30',    text: 'text-cyan-400',    border: 'border-cyan-500/60',    ring: 'ring-cyan-500/40' },
  embed_mv: { bg: 'bg-emerald-500', glow: 'shadow-[0_0_20px_rgba(16,185,129,0.5)]',   dim: 'bg-emerald-900/30', text: 'text-emerald-400', border: 'border-emerald-500/60', ring: 'ring-emerald-500/40' },
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

      {/* ── Factory Floor — single flat plane ── */}
      <div className="flex-1 overflow-auto relative p-4">

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
            backgroundSize: '24px 24px',
          }}
        />

        <div className="relative z-10 flex flex-col gap-4 max-w-5xl mx-auto">

          {/* ── Top row: Input objects (WR tickets + buffer + pending) ── */}
          <div className="flex flex-wrap items-start gap-2">
            {/* WR tickets as flat objects */}
            {activeWRs.slice(0, 6).map(wr => (
              <WRTicket key={wr.id} wr={wr} />
            ))}

            {/* Buffer block */}
            {buffer > 0 && (
              <FloorBlock
                icon={<Package size={14} className="text-yellow-500" />}
                label={t('bb.buffer')}
                value={buffer}
                color="yellow"
              />
            )}

            {/* Pending block */}
            {pending > 0 && (
              <FloorBlock
                icon={<Inbox size={14} className="text-gray-400" />}
                label={t('bb.station_queue')}
                value={pending}
                color="gray"
              />
            )}

            {activeWRs.length === 0 && pending === 0 && buffer === 0 && (
              <div className="text-[10px] text-gray-700 font-mono py-2">{t('bb.no_wr')}</div>
            )}
          </div>

          {/* ── Conveyor line: input → workers ── */}
          {allWorkers.length > 0 && (
            <div className="flex items-center gap-1 px-2">
              <div className={`flex-1 h-0.5 rounded ${isRunning ? 'bg-gray-600' : 'bg-gray-800'}`} />
              <span className={`text-[8px] ${isRunning ? 'text-gray-500' : 'text-gray-700'}`}>&#9660;</span>
              <div className={`flex-1 h-0.5 rounded ${isRunning ? 'bg-gray-600' : 'bg-gray-800'}`} />
            </div>
          )}

          {/* ── Center: Worker terminals ── */}
          <div className="flex flex-wrap gap-3 justify-center">
            {allWorkers.length > 0 ? (
              allWorkers.map((w, i) => (
                <WorkerTerminal key={i} worker={w} t={t} />
              ))
            ) : (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="text-gray-700 text-3xl mb-2">&#9881;</div>
                  <div className="text-[11px] text-gray-600 font-mono">{t('bb.no_workers')}</div>
                </div>
              </div>
            )}
          </div>

          {/* ── Conveyor line: workers → output ── */}
          {(completed > 0 || failed > 0) && (
            <div className="flex items-center gap-1 px-2">
              <div className={`flex-1 h-0.5 rounded ${isRunning ? 'bg-gray-600' : 'bg-gray-800'}`} />
              <span className={`text-[8px] ${isRunning ? 'text-gray-500' : 'text-gray-700'}`}>&#9660;</span>
              <div className={`flex-1 h-0.5 rounded ${isRunning ? 'bg-gray-600' : 'bg-gray-800'}`} />
            </div>
          )}

          {/* ── Bottom row: Output objects (storage + failed) ── */}
          {(completed > 0 || failed > 0 || total > 0) && (
            <div className="flex flex-wrap items-start gap-3 justify-center">
              {/* Storage warehouse */}
              <StorageBlock completed={completed} total={total} t={t} />

              {/* Failed block */}
              {failed > 0 && (
                <div className="rounded-lg border-2 border-red-800/40 bg-red-900/10 px-4 py-3 text-center">
                  <AlertTriangle size={16} className="mx-auto text-red-500 mb-1" />
                  <div className="text-lg font-mono font-bold text-red-400 tabular-nums">{failed}</div>
                  <div className="text-[9px] text-red-600 font-mono">{t('bb.failed')}</div>
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

// ─── Worker Terminal (single machine block with internal pipeline) ─

function WorkerTerminal({ worker, t }) {
  const w = worker;
  const isActive = w.state === 'active' || !!w.phase;
  const isIdle = !w.phase && w.state !== 'active';
  const currentPhaseIdx = w.phase ? PHASE_ORDER.indexOf(w.phase) : -1;

  // Active phase icon
  const ActiveIcon = w.phase ? PHASE_ICONS[w.phase] : null;
  const activeColors = w.phase ? PHASE_COLORS[w.phase] : null;
  const activeAnim = w.phase ? PHASE_ANIMS[w.phase] : '';

  return (
    <div className={`
      rounded-lg border-2 w-[180px] transition-all duration-300
      ${isActive
        ? `${activeColors?.border || 'border-gray-600/50'} ${activeColors?.glow || ''} bg-gradient-to-b from-gray-800/60 to-gray-900/80`
        : 'border-gray-700/30 bg-gray-800/50'
      }
    `}>
      {/* Header: name + status */}
      <div className="flex items-center gap-1.5 px-3 pt-2.5 pb-1">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${
          isActive ? 'bg-green-500 animate-pulse' :
          w.state === 'resting' ? 'bg-yellow-500' : 'bg-gray-600'
        }`} />
        <span className="text-[11px] font-mono text-gray-300 font-bold truncate flex-1">
          {w.name}
        </span>
        {w.throughput > 0 && (
          <span className="text-[9px] font-mono text-gray-500 tabular-nums flex-shrink-0">
            {w.throughput.toFixed(1)}/m
          </span>
        )}
      </div>

      {/* Active phase icon — large center display */}
      <div className="flex items-center justify-center py-3">
        {ActiveIcon ? (
          <div className="flex flex-col items-center gap-1">
            <ActiveIcon
              size={28}
              className={`${activeColors.text} ${activeAnim}`}
            />
            {w.phaseCount > 0 && (
              <span className={`text-[10px] font-mono font-bold tabular-nums ${activeColors.text}`}>
                {w.phaseIndex || 0}/{w.phaseCount}
              </span>
            )}
          </div>
        ) : (
          <div className="text-gray-700 text-xl">
            {w.state === 'resting' ? '&#9788;' : '&#9881;'}
          </div>
        )}
      </div>

      {/* Internal phase pipeline strip — compact P→MC→VV→MV */}
      <div className="px-2 pb-1.5">
        <div className="flex items-center gap-0.5">
          {PHASE_ORDER.map((phase, idx) => {
            const isCurrent = w.phase === phase;
            const isPast = currentPhaseIdx > idx;
            const colors = PHASE_COLORS[phase];
            const label = PHASE_LABELS[phase];

            let progress = 0;
            if (isCurrent && w.phaseCount > 0) {
              progress = ((w.phaseIndex || 0) / w.phaseCount) * 100;
            } else if (isPast) {
              progress = 100;
            }

            return (
              <div key={phase} className="flex items-center flex-1 min-w-0">
                {/* Phase cell */}
                <div className={`
                  relative flex-1 h-4 rounded-sm overflow-hidden border transition-all duration-300
                  ${isCurrent
                    ? `${colors.border}`
                    : isPast
                      ? 'border-gray-600/20'
                      : 'border-gray-800/30'
                  }
                `}>
                  {/* Background */}
                  <div className={`absolute inset-0 ${
                    isCurrent ? colors.dim : isPast ? 'bg-gray-700/15' : 'bg-gray-800/30'
                  }`} />
                  {/* Fill */}
                  <div
                    className={`absolute left-0 top-0 bottom-0 transition-all duration-500 ${
                      isCurrent ? `${colors.bg} opacity-70` : isPast ? 'bg-gray-600/30' : ''
                    }`}
                    style={{ width: `${progress}%` }}
                  />
                  {/* Label */}
                  <div className="relative z-10 flex items-center justify-center h-full">
                    <span className={`text-[7px] font-mono font-bold ${
                      isCurrent ? colors.text : isPast ? 'text-gray-500' : 'text-gray-700'
                    }`}>
                      {label}
                    </span>
                  </div>
                  {/* Completed check */}
                  {isPast && (
                    <div className="absolute top-0 right-0.5 z-10">
                      <Check size={6} className="text-green-500/50" />
                    </div>
                  )}
                </div>

                {/* Arrow between phases */}
                {idx < PHASE_ORDER.length - 1 && (
                  <span className={`text-[5px] mx-px flex-shrink-0 ${
                    isPast || isCurrent ? 'text-gray-500' : 'text-gray-800'
                  }`}>&#9654;</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Current file / idle label */}
      <div className="px-3 pb-2.5 min-h-[18px]">
        {w.currentFile ? (
          <div className="text-[9px] text-gray-600 truncate font-mono">{w.currentFile}</div>
        ) : isIdle ? (
          <div className="text-[9px] text-gray-700 font-mono italic">
            {w.state === 'resting' ? t('bb.resting') : t('bb.phase_idle')}
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ─── Floor Block (generic small object on the floor) ─────────

function FloorBlock({ icon, label, value, color }) {
  const borderMap = { yellow: 'border-yellow-700/30', gray: 'border-gray-700/30' };
  const bgMap = { yellow: 'bg-yellow-900/10', gray: 'bg-gray-800/20' };
  const textMap = { yellow: 'text-yellow-400', gray: 'text-gray-400' };

  return (
    <div className={`rounded-lg border ${borderMap[color] || borderMap.gray} ${bgMap[color] || bgMap.gray} px-3 py-2 flex items-center gap-2`}>
      {icon}
      <div>
        <div className={`text-sm font-mono font-bold tabular-nums ${textMap[color] || textMap.gray}`}>
          {value.toLocaleString()}
        </div>
        <div className="text-[8px] uppercase tracking-wider text-gray-600 font-mono">{label}</div>
      </div>
    </div>
  );
}

// ─── Storage Block (output warehouse with fill level) ────────

function StorageBlock({ completed, total, t }) {
  const fillPct = total > 0 ? Math.min((completed / total) * 100, 100) : 0;

  return (
    <div className="rounded-lg border-2 border-green-700/30 bg-green-900/5 overflow-hidden w-[160px]">
      {/* Fill level background */}
      <div className="relative px-4 py-3">
        <div
          className="absolute bottom-0 left-0 right-0 bg-green-500/15 transition-all duration-700"
          style={{ height: `${fillPct}%` }}
        />
        <div className="relative z-10 text-center">
          <div className="text-2xl font-mono font-bold text-green-400 tabular-nums leading-none">
            {completed.toLocaleString()}
          </div>
          <div className="text-[9px] text-gray-500 font-mono mt-1">
            {t('bb.completed_label')}
          </div>
          {total > 0 && (
            <div className="text-[10px] font-mono text-green-600 mt-0.5 tabular-nums">
              {fillPct.toFixed(1)}%
            </div>
          )}
          {/* Mini fill blocks */}
          <div className="flex flex-wrap justify-center gap-0.5 mt-2">
            {Array.from({ length: Math.min(Math.ceil(fillPct / 5), 20) }).map((_, i) => (
              <div key={i} className="w-1.5 h-1.5 rounded-[1px] bg-green-500/40" />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── WR Ticket (flat floor object) ───────────────────────────

function WRTicket({ wr }) {
  const total = wr.total_files || 0;
  const done = wr.completed_count || 0;
  const failed = wr.failed_count || 0;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const isActive = wr.status === 'processing';

  return (
    <div className={`
      rounded-lg border px-3 py-2 min-w-[130px]
      ${isActive ? 'border-blue-700/40 bg-blue-900/10' : 'border-gray-700/30 bg-gray-800/20'}
    `}>
      <div className="flex items-center gap-1.5 mb-1.5">
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-blue-500 animate-pulse' : 'bg-gray-600'}`} />
        <span className="text-[10px] text-gray-300 truncate leading-tight font-medium">{wr.name}</span>
      </div>
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
