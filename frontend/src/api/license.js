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
