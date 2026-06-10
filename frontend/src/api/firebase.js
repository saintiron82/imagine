/**
 * Firebase Firestore REST API client — lightweight, no SDK dependency.
 *
 * Used for group name → server URL resolution (discovery).
 * Firebase project: imagine-b1e9c
 * Collection: groups/{key} — key = normalized group name
 */

const PROJECT_ID = 'imagine-b1e9c';
const FS_BASE = `https://firestore.googleapis.com/v1/projects/${PROJECT_ID}/databases/(default)/documents/groups`;

/** Normalize group name → Firestore document key */
function toKey(groupName) {
    return groupName.trim().toLowerCase().replace(/\s+/g, '_');
}

/** Extract string value from Firestore field */
function str(fields, key) {
    return fields?.[key]?.stringValue || '';
}

/** Extract integer value from Firestore field */
function num(fields, key) {
    return parseInt(fields?.[key]?.integerValue || '0', 10);
}

/** Extract boolean value from Firestore field */
function bool(fields, key) {
    return fields?.[key]?.booleanValue === true;
}

/**
 * Look up a group by name and return its server info.
 *
 * URL priority (Phase 1 of secure external worker access):
 *   external_url → tunnel_url → public_ip:port → lan_url → lan_ip:port → localhost:port
 *
 * @param {string} groupName
 * @returns {Promise<{url: string, urls: string[], groupName: string, lan_ip?: string, public_ip?: string, port: number, connect_mode: string, lan_url: string, external_url: string, tunnel_url: string, relay_endpoint: string, relay_online: boolean, reachable_status: string, updated_at: string} | null>}
 */
export async function lookupGroup(groupName) {
    const key = toKey(groupName);
    try {
        const resp = await fetch(`${FS_BASE}/${key}`, {
            method: 'GET',
            signal: AbortSignal.timeout(8000),
        });
        if (!resp.ok) return null;

        const doc = await resp.json();
        const f = doc.fields;
        if (!f) return null;

        const port = num(f, 'port');
        const lan_ip = str(f, 'lan_ip');
        const public_ip = str(f, 'public_ip');
        const connect_mode = str(f, 'connect_mode') || 'direct_lan';
        const lan_url = str(f, 'lan_url');
        const external_url = str(f, 'external_url');
        const tunnel_url = str(f, 'tunnel_url');
        const relay_endpoint = str(f, 'relay_endpoint');
        const relay_online = bool(f, 'relay_online');
        const reachable_status = str(f, 'reachable_status') || 'unknown';

        if (!port && !external_url && !tunnel_url) return null;

        // Candidate URL list — caller probes in order.
        const urls = [];
        if (external_url) urls.push(external_url);
        if (tunnel_url) urls.push(tunnel_url);
        if (public_ip && port) urls.push(`http://${public_ip}:${port}`);
        if (lan_url) urls.push(lan_url);
        else if (lan_ip && port) urls.push(`http://${lan_ip}:${port}`);
        if (port) urls.push(`http://localhost:${port}`);

        return {
            url: urls[0] || '',
            urls,
            groupName: str(f, 'group_name') || groupName,
            lan_ip,
            public_ip,
            port,
            connect_mode,
            lan_url,
            external_url,
            tunnel_url,
            relay_endpoint,
            relay_online,
            reachable_status,
            updated_at: str(f, 'updated_at'),
        };
    } catch (e) {
        console.error('Firebase group lookup failed:', e);
        return null;
    }
}

/**
 * Check if a group name is already taken.
 * @param {string} groupName
 * @returns {Promise<boolean>} true if name already exists
 */
export async function isGroupNameTaken(groupName) {
    const existing = await lookupGroup(groupName);
    return existing !== null;
}

/**
 * Register a server group in Firestore.
 *
 * Phase 1 of secure external worker access: the document carries connection
 * mode metadata so clients/workers can pick a candidate URL without guessing.
 *
 * @param {string} groupName
 * @param {{lan_ip: string, public_ip?: string, port: number, connect_mode?: string, external_url?: string, tunnel_url?: string, relay_endpoint?: string, relay_online?: boolean, reachable_status?: string}} info
 */
export async function registerGroup(groupName, info) {
    const key = toKey(groupName);

    // Reject duplicate group names
    const taken = await isGroupNameTaken(groupName);
    if (taken) {
        throw new Error('GROUP_NAME_TAKEN');
    }

    const port = info.port;
    const lan_ip = info.lan_ip || '';
    const lan_url = lan_ip && port ? `http://${lan_ip}:${port}` : '';
    const external_url = info.external_url || '';
    const tunnel_url = info.tunnel_url || '';
    const relay_endpoint = info.relay_endpoint || '';
    const relay_online = info.relay_online === true;
    const reachable_status = info.reachable_status || 'unknown';

    let connect_mode = info.connect_mode;
    if (!connect_mode) {
        if (relay_online) connect_mode = 'relay_session';
        else if (external_url) connect_mode = 'manual_external';
        else if (lan_ip && lan_ip !== '127.0.0.1') connect_mode = 'direct_lan';
        else connect_mode = 'direct_local';
    }

    const fields = {
        group_name: { stringValue: groupName },
        lan_ip: { stringValue: lan_ip },
        public_ip: { stringValue: info.public_ip || '' },
        port: { integerValue: String(port) },
        connect_mode: { stringValue: connect_mode },
        lan_url: { stringValue: lan_url },
        external_url: { stringValue: external_url },
        tunnel_url: { stringValue: tunnel_url },
        relay_endpoint: { stringValue: relay_endpoint },
        relay_online: { booleanValue: relay_online },
        reachable_status: { stringValue: reachable_status },
        updated_at: { stringValue: new Date().toISOString() },
    };

    try {
        // PATCH with document mask to create or overwrite
        const resp = await fetch(`${FS_BASE}/${key}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fields }),
            signal: AbortSignal.timeout(8000),
        });
        return resp.ok;
    } catch (e) {
        console.error('Firebase group registration failed:', e);
        return false;
    }
}

