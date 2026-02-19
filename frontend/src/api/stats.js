/**
 * Stats API — DB statistics.
 */

import { apiClient } from './client';

export async function getDbStats() {
  return apiClient.get('/api/v1/stats/db');
}

export async function getIncompleteStats() {
  return apiClient.get('/api/v1/stats/incomplete');
}

export async function getJobStats() {
  return apiClient.get('/api/v1/jobs/stats');
}
