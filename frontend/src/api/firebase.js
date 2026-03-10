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

/**
 * Look up a group by name and return its server info.
 * @param {string} groupName
 * @returns {Promise<{url: string, groupName: string, lan_ip?: string, public_ip?: string, port: number, updated_at: string} | null>}
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
        if (!port) return null;

        // Build URL — prefer lan_ip for local network, fall back to public_ip
        const ip = str(f, 'lan_ip') || str(f, 'public_ip') || 'localhost';
        const url = `http://${ip}:${port}`;

        return {
            url,
            groupName: str(f, 'group_name') || groupName,
            lan_ip: str(f, 'lan_ip'),
            public_ip: str(f, 'public_ip'),
            port,
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
 * @param {string} groupName
 * @param {{lan_ip: string, public_ip?: string, port: number}} info
 */
export async function registerGroup(groupName, info) {
    const key = toKey(groupName);

    // Reject duplicate group names
    const taken = await isGroupNameTaken(groupName);
    if (taken) {
        throw new Error('GROUP_NAME_TAKEN');
    }

    const fields = {
        group_name: { stringValue: groupName },
        lan_ip: { stringValue: info.lan_ip || '' },
        public_ip: { stringValue: info.public_ip || '' },
        port: { integerValue: String(info.port) },
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

/**
 * Unregister (delete) a group from Firestore.
 * @param {string} groupName
 */
export async function unregisterGroup(groupName) {
    const key = toKey(groupName);
    try {
        await fetch(`${FS_BASE}/${key}`, {
            method: 'DELETE',
            signal: AbortSignal.timeout(8000),
        });
    } catch (e) {
        console.error('Firebase group unregister failed:', e);
    }
}
