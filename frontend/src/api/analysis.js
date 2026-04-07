/**
 * Analysis Job API client.
 */

import { apiClient } from './client';

export async function listAnalysisJobs(includeCompleted = false) {
  return apiClient.get(`/api/v1/analysis-jobs?include_completed=${includeCompleted}`);
}

export async function createAnalysisJob(name, sourcePath, filePaths) {
  return apiClient.post('/api/v1/analysis-jobs', {
    name, source_path: sourcePath, file_paths: filePaths,
  });
}

export async function getAnalysisJob(jobId) {
  return apiClient.get(`/api/v1/analysis-jobs/${jobId}`);
}

export async function getAnalysisProgress(jobId) {
  return apiClient.get(`/api/v1/analysis-jobs/${jobId}/progress`);
}

export async function getAnalysisMetrics(jobId) {
  return apiClient.get(`/api/v1/analysis-jobs/${jobId}/metrics`);
}

export async function pauseAnalysisJob(jobId) {
  return apiClient.post(`/api/v1/analysis-jobs/${jobId}/pause`);
}

export async function resumeAnalysisJob(jobId) {
  return apiClient.post(`/api/v1/analysis-jobs/${jobId}/resume`);
}

export async function cancelAnalysisJob(jobId) {
  return apiClient.post(`/api/v1/analysis-jobs/${jobId}/cancel`);
}

export async function retryFailedTasks(jobId, phase = null) {
  return apiClient.post(`/api/v1/analysis-jobs/${jobId}/retry`, { phase });
}

export async function archiveAnalysisJob(jobId) {
  return apiClient.post(`/api/v1/analysis-jobs/${jobId}/archive`);
}

export async function getJobErrors(jobId) {
  return apiClient.get(`/api/v1/analysis-jobs/${jobId}/errors`);
}
