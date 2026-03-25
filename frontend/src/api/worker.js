/**
 * Worker API — job queue interactions for distributed workers.
 */

import { apiClient } from './client';

export async function getJobStats() {
  return apiClient.get('/api/v1/jobs/stats');
}
