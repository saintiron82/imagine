/**
 * PipelineBlackboard — Game HUD-style pipeline visualization.
 * 4-panel layout: Pipeline Flow, Worker Units, Work Requests, System Gauges.
 * Data: queue stats (5s poll) + work requests + workerProgress (props).
 */

import { useState, useEffect, useCallback } from 'react';
import { Cpu, Zap, AlertTriangle } from 'lucide-react';
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

  const throughput = stats?.throughput ?? 0;

  return (
    <div className="h-full flex flex-col bg-gray-950 p-3 gap-3">
      {/* Title bar */}
      <div className="flex items-center justify-between px-2">
        <h2 className="text-[11px] uppercase tracking-widest text-gray-500 font-mono font-bold">
          {t('bb.title')}
        </h2>
        <div className="flex items-center gap-2">
          {throughput > 0 && (
            <span className="text-[11px] font-mono text-gray-400">
              {throughput.toFixed(1)}{t('bb.per_min')}
            </span>
          )}
          <span className="flex items-center gap-1 text-[10px] font-mono text-green-500">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            {t('bb.live')}
          </span>
        </div>
      </div>

      {/* 4-panel grid */}
      <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-3 min-h-0">
        <Panel title={t('bb.flow_title')}>
          <PipelineFlowPanel stats={stats} workerProgress={workerProgress} />
        </Panel>
        <Panel title={t('bb.workers_title')}>
          <WorkerUnitsPanel workerProgress={workerProgress} workers={workers} />
        </Panel>
        <Panel title={t('bb.wr_title')}>
          <WorkRequestsPanel workRequests={workRequests} />
        </Panel>
        <Panel title={t('bb.gauges_title')}>
          <SystemGaugesPanel stats={stats} />
        </Panel>
      </div>
    </div>
  );
}

// ─── Panel Wrapper ──────────────────────────────────────────────

function Panel({ title, children }) {
  return (
    <div className="rounded-lg border border-gray-700/50 bg-gray-900/60 shadow-[0_0_15px_rgba(59,130,246,0.04)] flex flex-col overflow-hidden">
      <div className="px-3 py-1.5 border-b border-gray-800">
        <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono">
          {title}
        </span>
      </div>
      <div className="flex-1 p-3 min-h-0 overflow-auto">
        {children}
      </div>
    </div>
  );
}

// ─── Panel 1: Pipeline Flow ─────────────────────────────────────

function PipelineFlowPanel({ stats, workerProgress }) {
  const { t } = useLocale();
  if (!stats) return <Idle />;

  const pending = (stats.pending ?? 0) + (stats.download_waiting ?? 0);
  const buffer = stats.parse_ahead_parsed ?? 0;
  const active = (stats.assigned ?? 0) + (stats.processing ?? 0);
  const embedWaiting = Math.max(0, (stats.phase_vision_done ?? 0) - (stats.phase_embed_done ?? 0));
  const completed = stats.completed ?? 0;

  // Determine active phase from workerProgress
  const activePhase = workerProgress?.currentPhase || null;
  const phaseMap = { parse: 'parse', vision: 'vision', embed_vv: 'embed', embed_mv: 'embed' };
  const activeName = phaseMap[activePhase] || null;

  const stations = [
    { id: 'queue', label: t('bb.station_queue'), count: pending, color: 'yellow', active: pending > 0 },
    { id: 'parse', label: t('bb.station_parse'), count: buffer, color: 'teal', active: activeName === 'parse' },
    { id: 'vision', label: t('bb.station_vision'), count: active, color: 'purple', active: activeName === 'vision' },
    { id: 'embed', label: t('bb.station_embed'), count: embedWaiting, color: 'cyan', active: activeName === 'embed' },
    { id: 'done', label: t('bb.station_done'), count: completed, color: 'green', active: false },
  ];

  return (
    <div className="flex items-center justify-center gap-1 h-full">
      {stations.map((st, i) => (
        <div key={st.id} className="flex items-center">
          <StationNode {...st} />
          {i < stations.length - 1 && <FlowConnector active={st.count > 0} />}
        </div>
      ))}
    </div>
  );
}

const colorMap = {
  yellow: { border: 'border-yellow-500/50', bg: 'bg-yellow-500/10', text: 'text-yellow-400', glow: 'shadow-[0_0_12px_rgba(234,179,8,0.15)]' },
  teal: { border: 'border-teal-500/50', bg: 'bg-teal-500/10', text: 'text-teal-400', glow: 'shadow-[0_0_12px_rgba(20,184,166,0.15)]' },
  purple: { border: 'border-purple-500/50', bg: 'bg-purple-500/10', text: 'text-purple-400', glow: 'shadow-[0_0_12px_rgba(168,85,247,0.15)]' },
  cyan: { border: 'border-cyan-500/50', bg: 'bg-cyan-500/10', text: 'text-cyan-400', glow: 'shadow-[0_0_12px_rgba(6,182,212,0.15)]' },
  green: { border: 'border-green-500/50', bg: 'bg-green-500/10', text: 'text-green-400', glow: 'shadow-[0_0_12px_rgba(34,197,94,0.15)]' },
};

function StationNode({ label, count, color, active }) {
  const c = colorMap[color] || colorMap.green;
  return (
    <div className={`
      flex flex-col items-center justify-center rounded-lg border px-3 py-2 min-w-[64px]
      transition-all duration-300
      ${active ? `${c.border} ${c.bg} ${c.glow}` : 'border-gray-700/40 bg-gray-800/40'}
    `}>
      <span className={`text-lg font-mono font-bold tabular-nums ${active ? c.text : 'text-gray-400'}`}>
        {count.toLocaleString()}
      </span>
      <span className="text-[9px] uppercase tracking-wider text-gray-500 mt-0.5">
        {label}
      </span>
    </div>
  );
}

function FlowConnector({ active }) {
  return (
    <div className="flex items-center mx-0.5">
      <div className={`w-4 h-px ${active ? 'bg-blue-500/60' : 'bg-gray-700'} transition-colors`} />
      <div className={`
        w-0 h-0
        border-t-[3px] border-t-transparent
        border-b-[3px] border-b-transparent
        border-l-[5px] ${active ? 'border-l-blue-500/60' : 'border-l-gray-700'}
        transition-colors
      `} />
    </div>
  );
}

// ─── Panel 2: Worker Units ──────────────────────────────────────

function WorkerUnitsPanel({ workerProgress, workers }) {
  const { t } = useLocale();
  const wp = workerProgress;

  // Electron local mode: show single local worker from workerProgress
  if (isElectron && wp && (wp.currentPhase || wp.completed > 0)) {
    return (
      <div className="space-y-2">
        <WorkerCard
          name={t('bb.local_worker')}
          status={wp.workerState === 'active' ? 'active' : wp.workerState === 'resting' ? 'resting' : 'idle'}
          phase={wp.currentPhase}
          phaseIndex={wp.phaseIndex}
          phaseCount={wp.phaseCount}
          currentFile={wp.currentFile}
          throughput={wp.throughput}
        />
      </div>
    );
  }

  // Server mode: show worker sessions
  if (workers.length > 0) {
    return (
      <div className="space-y-2">
        {workers.filter(w => w.status === 'online').map(w => (
          <WorkerCard
            key={w.id}
            name={w.worker_name || w.username}
            status="active"
            phase={w.current_phase}
            currentFile={w.current_file}
            throughput={w.throughput}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center h-full">
      <span className="text-xs text-gray-600 font-mono">{t('bb.no_workers')}</span>
    </div>
  );
}

const phaseColors = {
  parse: 'bg-blue-500', vision: 'bg-purple-500',
  embed_vv: 'bg-cyan-500', embed_mv: 'bg-emerald-500',
};
const phaseLabels = { parse: 'P', vision: 'MC', embed_vv: 'VV', embed_mv: 'MV', uploading: 'UP' };

function WorkerCard({ name, status, phase, phaseIndex, phaseCount, currentFile, throughput }) {
  const dotColor = status === 'active' ? 'bg-green-500' : status === 'resting' ? 'bg-yellow-500' : 'bg-gray-500';
  const phasePct = phaseCount > 0 ? ((phaseIndex || 0) / phaseCount) * 100 : 0;

  return (
    <div className="rounded border border-gray-700/50 bg-gray-800/40 px-2.5 py-2 space-y-1">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${dotColor} ${status === 'active' ? 'animate-pulse' : ''}`} />
          <span className="text-xs font-mono text-gray-300">{name}</span>
        </div>
        {throughput > 0 && (
          <span className="text-[10px] font-mono text-gray-500">{throughput.toFixed(1)}/m</span>
        )}
      </div>
      {phase && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-gray-400 w-6">
            {phaseLabels[phase] || phase}
          </span>
          <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${phaseColors[phase] || 'bg-gray-500'}`}
              style={{ width: `${phasePct}%` }}
            />
          </div>
          {phaseCount > 0 && (
            <span className="text-[10px] font-mono text-gray-500 tabular-nums">
              {phaseIndex}/{phaseCount}
            </span>
          )}
        </div>
      )}
      {currentFile && (
        <div className="text-[10px] text-gray-600 truncate font-mono">{currentFile}</div>
      )}
    </div>
  );
}

// ─── Panel 3: Work Requests ─────────────────────────────────────

function WorkRequestsPanel({ workRequests }) {
  const { t } = useLocale();
  const activeWRs = workRequests.filter(wr => wr.status === 'queued' || wr.status === 'processing');

  if (activeWRs.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <span className="text-xs text-gray-600 font-mono">{t('bb.no_wr')}</span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {activeWRs.slice(0, 8).map(wr => {
        const total = wr.total_files || 0;
        const done = wr.completed_count || 0;
        const failed = wr.failed_count || 0;
        const pct = total > 0 ? (done / total) * 100 : 0;
        const isActive = wr.status === 'processing';

        return (
          <div key={wr.id} className="rounded border border-gray-700/40 bg-gray-800/30 px-2.5 py-1.5">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-blue-500 animate-pulse' : 'bg-gray-600'}`} />
                <span className="text-xs text-gray-300 truncate">{wr.name}</span>
              </div>
              <span className="text-[10px] font-mono text-gray-500 tabular-nums ml-2 flex-shrink-0">
                {done}/{total}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${isActive ? 'bg-blue-500' : 'bg-gray-600'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-[10px] font-mono text-gray-500 tabular-nums w-10 text-right">
                {pct.toFixed(0)}%
              </span>
            </div>
            {failed > 0 && (
              <div className="flex items-center gap-1 mt-0.5">
                <AlertTriangle size={10} className="text-red-500" />
                <span className="text-[10px] text-red-500 font-mono">{failed}</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── Panel 4: System Gauges ─────────────────────────────────────

function SystemGaugesPanel({ stats }) {
  const { t } = useLocale();
  if (!stats) return <Idle />;

  const throughput = stats.throughput ?? 0;
  const total = stats.total_files ?? ((stats.pending ?? 0) + (stats.assigned ?? 0) + (stats.processing ?? 0) + (stats.completed ?? 0) + (stats.failed ?? 0));
  const done = stats.completed ?? 0;
  const failed = stats.failed ?? 0;
  const buffer = stats.parse_ahead_parsed ?? 0;
  const rate = total > 0 ? ((done / total) * 100).toFixed(1) : '0.0';

  const remaining = (stats.pending ?? 0) - (stats.download_waiting ?? 0)
    + (stats.assigned ?? 0) + (stats.processing ?? 0);
  const etaMin = throughput > 0 ? Math.ceil(remaining / throughput) : null;
  const etaStr = etaMin != null
    ? (etaMin >= 60 ? `~${Math.floor(etaMin / 60)}h ${etaMin % 60}m` : `~${etaMin}m`)
    : '--';

  return (
    <div className="flex flex-col items-center justify-center h-full gap-3">
      {/* Semi-circle gauge */}
      <ThroughputGauge value={throughput} max={50} />

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-x-4 gap-y-2 w-full max-w-[280px]">
        <GaugeStat label={t('bb.total')} value={total.toLocaleString()} />
        <GaugeStat label={t('bb.done')} value={done.toLocaleString()} color="text-green-400" />
        <GaugeStat label={t('bb.eta')} value={etaStr} />
        <GaugeStat label={t('bb.failed')} value={failed.toString()} color={failed > 0 ? 'text-red-400' : undefined} />
        <GaugeStat label={t('bb.rate')} value={`${rate}%`} />
        <GaugeStat label={t('bb.buffer')} value={buffer.toString()} />
      </div>
    </div>
  );
}

function ThroughputGauge({ value, max }) {
  const { t } = useLocale();
  const pct = Math.min(value / max, 1);
  const r = 52;
  const circumference = Math.PI * r; // semi-circle
  const offset = circumference * (1 - pct);

  // Color based on value
  let strokeColor = '#ef4444'; // red
  if (value >= 15) strokeColor = '#22c55e'; // green
  else if (value >= 5) strokeColor = '#eab308'; // yellow

  return (
    <div className="relative flex flex-col items-center">
      <svg width="130" height="72" viewBox="0 0 130 72" className="overflow-visible">
        {/* Track */}
        <path
          d="M 10 65 A 52 52 0 0 1 120 65"
          fill="none"
          stroke="#374151"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Value */}
        <path
          d="M 10 65 A 52 52 0 0 1 120 65"
          fill="none"
          stroke={strokeColor}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700"
        />
      </svg>
      <div className="absolute top-6 left-1/2 -translate-x-1/2 text-center">
        <div className="text-2xl font-mono font-bold tabular-nums text-white">
          {value.toFixed(1)}
        </div>
        <div className="text-[9px] uppercase text-gray-500 tracking-wider">
          {t('bb.per_min')}
        </div>
      </div>
    </div>
  );
}

function GaugeStat({ label, value, color }) {
  return (
    <div className="text-center">
      <div className={`text-sm font-mono font-bold tabular-nums ${color || 'text-gray-200'}`}>
        {value}
      </div>
      <div className="text-[9px] uppercase tracking-wider text-gray-600">
        {label}
      </div>
    </div>
  );
}

// ─── Shared ─────────────────────────────────────────────────────

function Idle() {
  const { t } = useLocale();
  return (
    <div className="flex items-center justify-center h-full">
      <span className="text-xs text-gray-600 font-mono">{t('bb.idle')}</span>
    </div>
  );
}
