/**
 * mDNS browser for discovering Imagine servers on LAN.
 *
 * Uses bonjour-service (pure JS, no native modules).
 * Only runs in Electron main process.
 */

let Bonjour;
try {
    Bonjour = require('bonjour-service').Bonjour;
} catch {
    // bonjour-service not installed — mDNS disabled
    Bonjour = null;
}

let bonjourInstance = null;
let browser = null;
const discoveredServers = new Map();

/**
 * Start browsing for _imagine._tcp services on LAN.
 * @param {(type: 'found'|'lost', server: object) => void} callback
 */
function startBrowsing(callback) {
    if (!Bonjour) return;

    stopBrowsing(); // Clean up previous session

    bonjourInstance = new Bonjour();
    browser = bonjourInstance.find({ type: 'imagine' }, (service) => {
        const addresses = (service.addresses || []).filter(
            (a) => a && !a.includes(':') // IPv4 only
        );
        if (addresses.length === 0) return;

        const server = {
            name: service.name,
            host: service.host,
            port: service.port,
            addresses,
            version: service.txt?.version || '',
            serverName: service.txt?.name || service.name,
        };

        discoveredServers.set(service.name, server);
        if (callback) callback('found', server);
    });

    // Service removal
    if (browser) {
        browser.on('down', (service) => {
            discoveredServers.delete(service.name);
            if (callback) callback('lost', { name: service.name });
        });
    }
}

/**
 * Stop browsing.
 */
function stopBrowsing() {
    if (browser) {
        try { browser.stop(); } catch { /* ignore */ }
        browser = null;
    }
    if (bonjourInstance) {
        try { bonjourInstance.destroy(); } catch { /* ignore */ }
        bonjourInstance = null;
    }
    discoveredServers.clear();
}

/**
 * Get currently discovered servers.
 * @returns {Array<object>}
 */
function getDiscoveredServers() {
    return Array.from(discoveredServers.values());
}

/**
 * Check if bonjour-service is available.
 */
function isAvailable() {
    return Bonjour !== null;
}

module.exports = { startBrowsing, stopBrowsing, getDiscoveredServers, isAvailable };
