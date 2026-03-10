/**
 * Firebase Realtime DB REST API client — lightweight, no SDK dependency.
 *
 * Used for group name → server URL resolution.
 * Firebase project: imagine-b1e9c
 */

const FIREBASE_BASE = 'https://imagine-b1e9c-default-rtdb.firebaseio.com/groups';

/**
 * Look up a group by name and return its server info.
 * @param {string} groupName
 * @returns {Promise<{url: string, groupName: string, lan_ip?: string, public_ip?: string, port: number, updated_at: string} | null>}
 */
export async function lookupGroup(groupName) {
    const key = encodeURIComponent(groupName.trim().toLowerCase().replace(/\s+/g, '_'));
    try {
        const resp = await fetch(`${FIREBASE_BASE}/${key}.json`, {
            method: 'GET',
            signal: AbortSignal.timeout(8000),
        });
        if (!resp.ok) return null;

        const data = await resp.json();
        if (!data || !data.port) return null;

        // Build URL — prefer lan_ip for local network, fall back to public_ip
        const ip = data.lan_ip || data.public_ip || 'localhost';
        const url = `http://${ip}:${data.port}`;

        return {
            url,
            groupName: data.group_name || groupName,
            lan_ip: data.lan_ip,
            public_ip: data.public_ip,
            port: data.port,
            updated_at: data.updated_at,
        };
    } catch (e) {
        console.error('Firebase group lookup failed:', e);
        return null;
    }
}

/**
 * Register a server group in Firebase.
 * @param {string} groupName
 * @param {{lan_ip: string, public_ip?: string, port: number}} info
 */
/**
 * Check if a group name is already taken in Firebase RTDB.
 * @param {string} groupName
 * @returns {Promise<boolean>} true if name already exists
 */
export async function isGroupNameTaken(groupName) {
    const existing = await lookupGroup(groupName);
    return existing !== null;
}

export async function registerGroup(groupName, info) {
    const key = encodeURIComponent(groupName.trim().toLowerCase().replace(/\s+/g, '_'));

    // Reject duplicate group names
    const taken = await isGroupNameTaken(groupName);
    if (taken) {
        throw new Error('GROUP_NAME_TAKEN');
    }

    const payload = {
        group_name: groupName,
        lan_ip: info.lan_ip || '',
        public_ip: info.public_ip || '',
        port: info.port,
        updated_at: new Date().toISOString(),
    };

    try {
        const resp = await fetch(`${FIREBASE_BASE}/${key}.json`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(8000),
        });
        return resp.ok;
    } catch (e) {
        console.error('Firebase group registration failed:', e);
        return false;
    }
}
