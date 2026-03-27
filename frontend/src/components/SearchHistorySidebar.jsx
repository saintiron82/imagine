import { Clock, X, Trash2, Search } from 'lucide-react';
import { useLocale } from '../i18n';

/**
 * SearchHistorySidebar — session-only search history with cached results.
 *
 * Displays recent search queries in a sidebar list. Clicking an item
 * restores cached results instantly (no API re-call).
 * History and cache are session-only (cleared on app restart).
 */
export default function SearchHistorySidebar({ history, activeQuery, onSelect, onDelete, onClearAll }) {
  const { t } = useLocale();

  if (!history || history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-600 px-3 py-8">
        <Search size={20} className="mb-2 opacity-30" />
        <span className="text-[10px] font-mono">{t('search.no_history')}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700/50">
        <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">
          {t('search.recent')}
        </span>
        <button
          onClick={onClearAll}
          className="text-[9px] text-gray-600 hover:text-red-400 font-mono transition-colors"
          title={t('action.clear')}
        >
          <Trash2 size={11} />
        </button>
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto">
        {history.map((item, i) => {
          const isActive = item.query === activeQuery;
          const timeStr = new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

          return (
            <button
              key={`${item.query}-${item.timestamp}`}
              onClick={() => onSelect(item.query)}
              className={`w-full text-left px-3 py-2 border-b border-gray-800/30 transition-colors group
                ${isActive
                  ? 'bg-blue-900/20 border-l-2 border-l-blue-500'
                  : 'hover:bg-gray-800/40 border-l-2 border-l-transparent'
                }`}
            >
              <div className="flex items-center gap-2">
                <Clock size={10} className={`flex-shrink-0 ${isActive ? 'text-blue-400' : 'text-gray-600'}`} />
                <span className={`text-[11px] font-mono truncate flex-1 ${isActive ? 'text-blue-300' : 'text-gray-400'}`}>
                  {item.query}
                </span>
                <span
                  onClick={(e) => { e.stopPropagation(); onDelete(item.query); }}
                  className="text-gray-700 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                >
                  <X size={10} />
                </span>
              </div>
              <div className="flex items-center gap-2 mt-0.5 ml-[18px]">
                <span className="text-[9px] text-gray-600 font-mono">{item.resultCount} results</span>
                <span className="text-[9px] text-gray-700 font-mono">{timeStr}</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
