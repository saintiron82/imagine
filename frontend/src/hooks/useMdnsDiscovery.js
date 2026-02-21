/**
 * React hook for mDNS server discovery.
 *
 * Only works in Electron mode. In Web mode, returns empty array.
 */

import { useState, useEffect } from 'react';
import { isElectron } from '../api/client';

export function useMdnsDiscovery(enabled = true) {
  const [servers, setServers] = useState([]);
  const [browsing, setBrowsing] = useState(false);

  useEffect(() => {
    if (!isElectron || !enabled || !window.electron?.mdns) return;

    setBrowsing(true);
    window.electron.mdns.startBrowse();

    const handleEvent = (data) => {
      if (data.type === 'found') {
        setServers((prev) => {
          const filtered = prev.filter((s) => s.name !== data.name);
          return [...filtered, data];
        });
      } else if (data.type === 'lost') {
        setServers((prev) => prev.filter((s) => s.name !== data.name));
      }
    };

    window.electron.mdns.onServerEvent(handleEvent);

    // Also fetch initial list
    window.electron.mdns.getServers().then((list) => {
      if (list && list.length > 0) {
        setServers(list);
      }
    });

    return () => {
      window.electron.mdns.stopBrowse();
      window.electron.mdns.offServerEvent();
      setBrowsing(false);
      setServers([]);
    };
  }, [enabled]);

  return { servers, browsing };
}
