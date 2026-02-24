/**
 * DomainSelectModal — shown on first launch when no classification domain is configured.
 * Presents available domain presets as cards and lets the user pick one (or skip).
 */

import { useState, useEffect } from 'react';
import { Tag, Check } from 'lucide-react';
import { useLocale } from '../i18n';
import { listDomains, setActiveDomain } from '../services/bridge';

export default function DomainSelectModal({ onClose }) {
  const { t } = useLocale();
  const [domains, setDomains] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selecting, setSelecting] = useState(null); // domainId being saved
  const [selected, setSelected] = useState(null);   // domainId just saved

  useEffect(() => {
    (async () => {
      try {
        const result = await listDomains();
        setDomains(Array.isArray(result) ? result : result?.data || []);
      } catch {
        // Domains unavailable — close modal silently
        onClose();
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleSelect = async (domainId) => {
    if (selecting || selected) return;
    setSelecting(domainId);
    try {
      await setActiveDomain(domainId);
      setSelected(domainId);
      setTimeout(() => onClose(), 600);
    } catch {
      setSelecting(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl max-w-2xl w-full mx-4 p-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
            <Tag className="w-5 h-5 text-blue-400" />
          </div>
          <h2 className="text-lg font-semibold text-white">
            {t('domain_select.title')}
          </h2>
        </div>
        <p className="text-sm text-neutral-400 mb-6 ml-[52px]">
          {t('domain_select.desc')}
        </p>

        {/* Domain Cards */}
        {loading ? (
          <div className="flex justify-center py-12">
            <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : domains.length === 0 ? (
          <p className="text-sm text-neutral-500 text-center py-8">
            No domains available
          </p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
            {domains.map((d) => {
              const id = d.id;
              const isSelecting = selecting === id;
              const isSelected = selected === id;

              return (
                <div
                  key={id}
                  className={`
                    relative border rounded-lg p-4 transition-all cursor-pointer
                    ${isSelected
                      ? 'border-green-500 bg-green-500/10'
                      : isSelecting
                        ? 'border-blue-500 bg-blue-500/10 opacity-70'
                        : 'border-neutral-700 bg-neutral-800 hover:border-blue-500/50 hover:bg-neutral-750'
                    }
                  `}
                  onClick={() => handleSelect(id)}
                >
                  <h3 className="font-medium text-white text-sm mb-0.5">
                    {d.name}
                  </h3>
                  <p className="text-xs text-neutral-400 mb-2">
                    {d.name_ko}
                  </p>
                  {d.description && (
                    <p className="text-xs text-neutral-500 mb-3 line-clamp-2">
                      {d.description}
                    </p>
                  )}
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-neutral-500">
                      {t('domain_select.types_count', {
                        count: d.image_types_count ?? d.image_types?.length ?? 0,
                      })}
                    </span>
                    {isSelected ? (
                      <span className="flex items-center gap-1 text-xs text-green-400 font-medium">
                        <Check className="w-3.5 h-3.5" />
                        {t('domain_select.selected')}
                      </span>
                    ) : isSelecting ? (
                      <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <span className="text-xs text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity">
                        {t('domain_select.select')}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Skip link */}
        <div className="text-center">
          <button
            onClick={onClose}
            className="text-sm text-neutral-500 hover:text-neutral-300 transition-colors"
            disabled={!!selecting}
          >
            {t('domain_select.skip')}
          </button>
        </div>
      </div>
    </div>
  );
}
