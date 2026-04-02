/**
 * WRCards — Work Request card components.
 * Extracted from PipelineBlackboard for reuse in WorkersPanel.
 */

import { useState } from 'react';
import { AlertTriangle, Pause, Play, X, ChevronRight, ChevronDown } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getWorkRequestDetail } from '../api/admin';


export function WRCard({ wr, onAction }) {
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
    <div className={`rounded-lg border ${borderColor} transition-all`}>
      {/* Header — clickable to expand */}
      <div className="px-3 py-2 cursor-pointer" onClick={handleToggle}>
        <div className="flex items-center gap-1.5 mb-1">
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${dotColor}`} />
          <span className="text-[11px] text-gray-300 font-medium truncate flex-1">{wr.name}</span>
          {isWebDAV && <span className="text-[7px] font-mono text-blue-500">WebDAV</span>}
          {/* Control buttons */}
          {onAction && (
            <div className="flex items-center gap-0.5 flex-shrink-0">
              {isActive && (
                <button onClick={(e) => { e.stopPropagation(); onAction(wr.id, 'pause'); }}
                  className="p-0.5 rounded hover:bg-gray-700 text-gray-500 hover:text-yellow-400 transition-colors" title="Pause">
                  <Pause size={10} />
                </button>
              )}
              {isPaused && (
                <button onClick={(e) => { e.stopPropagation(); onAction(wr.id, 'resume'); }}
                  className="p-0.5 rounded hover:bg-gray-700 text-gray-500 hover:text-green-400 transition-colors" title="Resume">
                  <Play size={10} />
                </button>
              )}
              <button onClick={(e) => { e.stopPropagation(); onAction(wr.id, 'cancel'); }}
                className="p-0.5 rounded hover:bg-gray-700 text-gray-500 hover:text-red-400 transition-colors" title="Cancel">
                <X size={10} />
              </button>
            </div>
          )}
        </div>
        {isPaused && <div className="text-[7px] text-yellow-500 font-mono mb-1">PAUSED</div>}
        <div className="h-1.5 bg-gray-700/50 rounded-full overflow-hidden">
          <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-[9px] font-mono text-gray-500 tabular-nums">{done}/{total}</span>
          <span className="text-[9px] font-mono text-gray-400 tabular-nums">{pct.toFixed(0)}%</span>
        </div>
        {failed > 0 && <span className="text-[8px] text-red-400 font-mono"><AlertTriangle size={8} className="inline mr-0.5" />{failed} failed</span>}
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-3 pb-2 border-t border-gray-700/30 pt-2 space-y-1.5">
          <div className="text-[8px] font-mono text-gray-600 truncate" title={wr.source_path}>{wr.source_path}</div>
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
          {wr.created_at && <div className="text-[7px] text-gray-700 font-mono">Created: {new Date(wr.created_at).toLocaleString()}</div>}
        </div>
      )}
    </div>
  );
}


export function WRGroupCard({ group, onAction }) {
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
      <div className="px-3 py-2.5 cursor-pointer flex items-center gap-2" onClick={() => setExpanded(!expanded)}>
        {expanded
          ? <ChevronDown size={14} className="text-gray-400 flex-shrink-0" />
          : <ChevronRight size={14} className="text-gray-400 flex-shrink-0" />}
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${hasActive ? 'bg-blue-500 animate-pulse' : hasPaused ? 'bg-yellow-500' : 'bg-gray-600'}`} />
        <span className="text-[11px] text-gray-200 font-medium truncate flex-1">{group.displayName}</span>
        <span className="text-[8px] font-mono text-gray-500 flex-shrink-0">{group.wrs.length}</span>
        <span className="text-[9px] font-mono text-gray-500 tabular-nums flex-shrink-0">{totalDone}/{totalFiles}</span>
        <span className="text-[8px] font-mono text-gray-600 tabular-nums flex-shrink-0 w-8 text-right">{pct.toFixed(0)}%</span>
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
