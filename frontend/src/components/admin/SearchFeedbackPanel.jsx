/**
 * SearchFeedbackPanel — Sprint 2 γ3 of perceived search quality.
 *
 * Shows recent user 'irrelevant' labels: total 30-day count, top flagged
 * files, top flagged queries. Refreshing the page reloads.
 */

import { useCallback, useEffect, useState } from 'react';
import { useLocale } from '../../i18n';
import { getSearchFeedbackSummary } from '../../api/admin';

export default function SearchFeedbackPanel() {
  const { t } = useLocale();
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const resp = await getSearchFeedbackSummary();
      setSummary(resp.data ?? resp);
      setError(null);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'load failed');
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">{t('admin.search_feedback_title')}</h2>
      {error && <p className="text-xs text-red-400">{error}</p>}
      {summary && (
        <>
          <section className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <p className="text-xs text-gray-400">{t('admin.search_feedback_total_30d')}</p>
            <p className="text-2xl text-white font-mono">{summary.total_30d}</p>
          </section>

          <section className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-white mb-2">
              {t('admin.search_feedback_top_files')}
            </h3>
            {summary.top_files.length === 0 ? (
              <p className="text-xs text-gray-500">{t('admin.search_feedback_empty')}</p>
            ) : (
              <table className="w-full text-xs text-gray-300">
                <thead>
                  <tr className="text-gray-500">
                    <th className="text-left py-1">file_id</th>
                    <th className="text-right py-1">{t('admin.search_feedback_count')}</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.top_files.map(row => (
                    <tr key={row.file_id} className="border-t border-gray-700/40">
                      <td className="py-1 font-mono">{row.file_id}</td>
                      <td className="py-1 text-right font-mono">{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-white mb-2">
              {t('admin.search_feedback_top_queries')}
            </h3>
            {summary.top_queries.length === 0 ? (
              <p className="text-xs text-gray-500">{t('admin.search_feedback_empty')}</p>
            ) : (
              <ul className="text-xs text-gray-300 space-y-1">
                {summary.top_queries.map(q => (
                  <li key={q.query} className="flex justify-between border-b border-gray-700/40 py-1">
                    <span className="truncate">{q.query}</span>
                    <span className="font-mono text-gray-400 ml-2">{q.count}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </div>
  );
}
