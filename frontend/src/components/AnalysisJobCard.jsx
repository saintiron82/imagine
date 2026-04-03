/**
 * AnalysisJobCard — displays a single analysis job's progress.
 *
 * Shows: name, total files, phase breakdown, progress bar, controls.
 * Phase sum = total (guaranteed by backend).
 */

import { useState } from 'react';
import { Pause, Play, X, RotateCcw, ChevronDown, ChevronRight } from 'lucide-react';

const PHASE_COLORS = {
  download: 'text-yellow-400',
  parse: 'text-sky-400',
  ai: 'text-purple-400',
  mv: 'text-green-400',
  done: 'text-emerald-400',
};

export default function AnalysisJobCard({ job, onAction }) {
  const [expanded, setExpanded] = useState(false);
  const p = job.progress || {};
  const total = p.total || 0;
  const complete = p.complete || 0;
  const pct = p.pct || 0;
  const phases = p.phases || {};
  const failed = p.failed || {};
  const totalFailed = Object.values(failed).reduce((s, v) => s + (v || 0), 0);

  const isActive = job.status === 'active';
  const isPaused = job.status === 'paused';
  const networkBad = job.network_status && job.network_status !== 'ok';

  const borderColor = isActive ? 'border-blue-700/50 bg-blue-900/10'
    : isPaused ? 'border-yellow-700/50 bg-yellow-900/10'
    : 'border-gray-700/30 bg-gray-800/30';

  return (
    <div className={`rounded-lg border ${borderColor} transition-all`}>
      {/* Header */}
      <div className="px-4 py-3 cursor-pointer" onClick={() => setExpanded(!expanded)}>
        <div className="flex items-center gap-2 mb-2">
          {expanded
            ? <ChevronDown size={14} className="text-gray-400" />
            : <ChevronRight size={14} className="text-gray-400" />}
          <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-blue-500 animate-pulse' : isPaused ? 'bg-yellow-500' : 'bg-gray-600'}`} />
          <span className="text-sm text-gray-200 font-medium flex-1">{job.name}</span>
          {networkBad && <span className="text-[8px] text-orange-400 font-mono">⚠ {job.network_status}</span>}
          <span className="text-xs font-mono text-gray-500">{complete}/{total}</span>
          <span className="text-xs font-mono text-blue-400 font-bold">{pct}%</span>

          {/* Controls */}
          {onAction && (
            <div className="flex items-center gap-1 ml-2" onClick={e => e.stopPropagation()}>
              {isActive && (
                <button onClick={() => onAction(job.id, 'pause')}
                  className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-yellow-400" title="일시정지">
                  <Pause size={12} />
                </button>
              )}
              {isPaused && (
                <button onClick={() => onAction(job.id, 'resume')}
                  className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-green-400" title="재개">
                  <Play size={12} />
                </button>
              )}
              {totalFailed > 0 && (
                <button onClick={() => onAction(job.id, 'retry')}
                  className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-blue-400" title="실패 재시도">
                  <RotateCcw size={12} />
                </button>
              )}
              <button onClick={() => onAction(job.id, 'cancel')}
                className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-red-400" title="취소">
                <X size={12} />
              </button>
            </div>
          )}
        </div>

        {/* Progress bar */}
        <div className="h-2 bg-gray-700/50 rounded-full overflow-hidden">
          <div className="h-full bg-emerald-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
        </div>

        {/* Phase breakdown — always visible */}
        <div className="flex items-center gap-3 mt-2 text-[10px] font-mono">
          {[
            ['DL', phases.download, 'download'],
            ['Parse', phases.parse, 'parse'],
            ['AI', phases.ai, 'ai'],
            ['MV', phases.mv, 'mv'],
            ['Done', phases.done, 'done'],
          ].map(([label, count, key]) => (
            <span key={key} className={`${PHASE_COLORS[key]} ${count > 0 ? 'opacity-100' : 'opacity-30'}`}>
              {label}:{count || 0}
            </span>
          ))}
          {totalFailed > 0 && (
            <span className="text-red-400">Fail:{totalFailed}</span>
          )}
        </div>
      </div>

      {/* Expanded: detailed metrics */}
      {expanded && (
        <div className="px-4 pb-3 border-t border-gray-700/30 pt-2 space-y-2">
          <div className="text-[9px] font-mono text-gray-600 truncate" title={job.source_path}>
            {job.source_path}
          </div>

          {/* Phase completion counts */}
          <div className="grid grid-cols-5 gap-2 text-[10px] font-mono">
            {[
              ['MC', p.mc_done, total],
              ['VV', p.vv_done, total],
              ['MV', p.mv_done, total],
            ].map(([label, done, t]) => (
              <div key={label} className="text-center">
                <div className="text-gray-400">{label}</div>
                <div className="text-gray-300">{done || 0}<span className="text-gray-600">/{t}</span></div>
              </div>
            ))}
          </div>

          {/* Failure details */}
          {totalFailed > 0 && (
            <div className="text-[9px] font-mono text-red-400/70">
              실패: {Object.entries(failed).filter(([,v]) => v > 0).map(([k,v]) => `${k}:${v}`).join(' · ')}
            </div>
          )}

          {/* Timestamps */}
          {job.created_at && (
            <div className="text-[8px] text-gray-700 font-mono">
              생성: {new Date(job.created_at).toLocaleString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
