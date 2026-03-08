/**
 * License API client — license info, activation, refresh.
 */

import { apiClient } from './client';

/**
 * Get current license info.
 * @returns {Promise<{license: object, auth_mode: string}>}
 */
export async function getLicenseInfo() {
  return apiClient.get('/api/v1/license/info');
}

/**
 * Activate an offline license key (admin only).
 * @param {string} licenseKey
 */
export async function activateLicense(licenseKey) {
  return apiClient.post('/api/v1/license/activate', { license_key: licenseKey });
}

/**
 * Force re-verify license from Firebase (admin only).
 */
export async function refreshLicense() {
  return apiClient.post('/api/v1/license/refresh');
}
