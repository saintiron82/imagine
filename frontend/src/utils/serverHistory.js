/**
 * Server connection history — stores recent server URLs in localStorage.
 */

const HISTORY_KEY = 'imagine-server-history';
const MAX_ENTRIES = 10;

/**
 * Get server history list.
 * @returns {Array<{url: string, name: string, version: string, lastConnected: number, lastUsername: string}>}
 */
export function getServerHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch {
    return [];
  }
}

/**
 * Add or update a server entry in history (most recent first).
 */
export function addServerToHistory({ url, name, version, lastUsername }) {
  const history = getServerHistory();
  const filtered = history.filter((h) => h.url !== url);
  filtered.unshift({
    url,
    name: name || '',
    version: version || '',
    lastConnected: Date.now(),
    lastUsername: lastUsername || '',
  });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(filtered.slice(0, MAX_ENTRIES)));
}

/**
 * Remove a server entry from history.
 */
export function removeServerFromHistory(url) {
  const history = getServerHistory().filter((h) => h.url !== url);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

/**
 * Format relative time from timestamp.
 */
export function formatRelativeTime(timestamp, t) {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return t('auth.time_now');
  if (minutes < 60) return t('auth.time_min', { count: minutes });
  if (hours < 24) return t('auth.time_hour', { count: hours });
  return t('auth.time_day', { count: days });
}
