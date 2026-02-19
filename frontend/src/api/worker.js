/**
 * Worker API — job queue interactions for distributed workers.
 */

import { apiClient } from './client';

export async function claimJobs(count = 10) {
  return apiClient.post('/api/v1/jobs/claim', { count });
}

export async function listMyJobs() {
  return apiClient.get('/api/v1/jobs');
}

export async function reportProgress(jobId, phase) {
  return apiClient.patch(`/api/v1/jobs/${jobId}/progress`, { phase });
}

export async function completeJob(jobId, metadata, vectors) {
  return apiClient.patch(`/api/v1/jobs/${jobId}/complete`, { metadata, vectors });
}

export async function failJob(jobId, errorMessage) {
  return apiClient.patch(`/api/v1/jobs/${jobId}/fail`, { error_message: errorMessage });
}

export async function getJobStats() {
  return apiClient.get('/api/v1/jobs/stats');
}
