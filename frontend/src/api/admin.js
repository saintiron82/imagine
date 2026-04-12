/**
 * Admin API — user management, invite codes.
 */

import { apiClient } from './client';

// ── Job Queue ────────────────────────────────────────────

export async function cleanupStaleJobs() {
  return apiClient.post('/api/v1/admin/jobs/cleanup');
}

export async function listJobs(status = null, limit = 20, offset = 0) {
  const params = new URLSearchParams();
  if (status) params.set('status', status);
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  return apiClient.get(`/api/v1/jobs/list?${params.toString()}`);
}

export async function cancelJob(jobId) {
  return apiClient.patch(`/api/v1/jobs/${jobId}/cancel`);
}

export async function retryFailedJobs() {
  return apiClient.post('/api/v1/admin/jobs/retry-failed');
}

export async function forceRetryFailedJobs() {
  return apiClient.post('/api/v1/admin/jobs/force-retry-failed');
}

export async function clearCompletedJobs() {
  return apiClient.delete('/api/v1/admin/jobs/clear-completed');
}

export async function archiveJobs(jobIds) {
  return apiClient.post('/api/v1/admin/jobs/archive', { job_ids: jobIds });
}

export async function archiveCompletedJobs() {
  return apiClient.post('/api/v1/admin/jobs/archive', { status: 'completed' });
}

export async function auditIntegrity() {
  // Legacy audit removed — Analysis Job System handles integrity via file_tasks
  return { success: true, total_files: 0, complete_files: 0, incomplete_files: 0, repaired_files: 0 };
}

export async function dismissPermanentlyFailedJobs() {
  return apiClient.delete('/api/v1/admin/jobs/permanently-failed');
}

export async function cleanupQueue() {
  return apiClient.post('/api/v1/admin/jobs/cleanup');
}

// ── Discover (server filesystem) ────────────────────────

export async function browseFolders(path = '/') {
  return apiClient.get(`/api/v1/discover/browse?path=${encodeURIComponent(path)}`);
}

export async function scanFolder(folderPath, priority = 0) {
  return apiClient.post('/api/v1/discover/scan', { folder_path: folderPath, priority });
}

export async function registerPaths(filePaths, priority = 0) {
  return apiClient.post('/api/v1/upload/register-paths', { file_paths: filePaths, priority });
}

// ── Worker Sessions ──────────────────────────────────────

export async function listWorkerSessions() {
  return apiClient.get('/api/v1/admin/workers');
}

export async function stopWorkerSession(sessionId) {
  return apiClient.post(`/api/v1/admin/workers/${sessionId}/stop`);
}

export async function blockWorkerSession(sessionId) {
  return apiClient.post(`/api/v1/admin/workers/${sessionId}/block`);
}

export async function listMyWorkers() {
  return apiClient.get('/api/v1/workers/my');
}

export async function stopMyWorker(sessionId) {
  return apiClient.post(`/api/v1/workers/${sessionId}/stop`);
}

// ── Worker Config (Admin) ────────────────────────────────

export async function updateWorkerConfig(sessionId, config) {
  return apiClient.patch(`/api/v1/admin/workers/${sessionId}/config`, config);
}

export async function getAutoProcessing() {
  return apiClient.get('/api/v1/admin/workers/auto-processing');
}

export async function updateAutoProcessing(config) {
  return apiClient.patch('/api/v1/admin/workers/auto-processing', config);
}

// ── Embedded Worker ─────────────────────────────────────

export async function getEmbeddedWorker() {
  return apiClient.get('/api/v1/admin/workers/embedded-worker');
}

export async function updateEmbeddedWorker(config) {
  return apiClient.patch('/api/v1/admin/workers/embedded-worker', config);
}

// ── Phase Pause Control ─────────────────────────────────

export async function getPausedPhases() {
  return apiClient.get('/api/v1/server/paused-phases');
}

export async function setPausedPhases(phases) {
  return apiClient.post('/api/v1/server/paused-phases', phases);
}

// ── Benchmark ───────────────────────────────────────────

export async function getBenchmark() {
  return apiClient.get('/api/v1/admin/workers/benchmark');
}

export async function runBenchmark() {
  return apiClient.post('/api/v1/admin/workers/benchmark');
}

// ── Members (Firebase Auth) ─────────────────────────────

export async function listMembers() {
  return apiClient.get('/api/v1/admin/members');
}

export async function updateMemberRole(memberId, role) {
  return apiClient.patch(`/api/v1/admin/members/${memberId}/role`, { role });
}

export async function removeMember(memberId) {
  return apiClient.delete(`/api/v1/admin/members/${memberId}`);
}

export async function deactivateMember(memberId) {
  return apiClient.patch(`/api/v1/admin/members/${memberId}/deactivate`);
}

export async function activateMember(memberId) {
  return apiClient.patch(`/api/v1/admin/members/${memberId}/activate`);
}

// ── Database Management ─────────────────────────────────

export async function resetDatabase(password) {
  return apiClient.post('/api/v1/admin/database/reset', { password });
}

// ── Thumbnail Stats ──────────────────────────────────────

export async function getThumbnailStats() {
  return apiClient.get('/api/v1/stats/thumbnails');
}

// ── Work Requests ───────────────────────────────────────

export async function getWorkRequests(includeCompleted = false) {
  return apiClient.get(`/api/v1/admin/work-requests?include_completed=${includeCompleted}`);
}

export async function getWorkRequestDetail(id) {
  return apiClient.get(`/api/v1/admin/work-requests/${id}`);
}

export async function pauseWorkRequest(id) {
  return apiClient.post(`/api/v1/admin/work-requests/${id}/pause`);
}

export async function resumeWorkRequest(id) {
  return apiClient.post(`/api/v1/admin/work-requests/${id}/resume`);
}

export async function cancelWorkRequest(id) {
  return apiClient.post(`/api/v1/admin/work-requests/${id}/cancel`);
}

export async function runRecoveryScan() {
  return apiClient.post('/api/v1/admin/recovery/scan');
}

// ── History ─────────────────────────────────────────────

export async function listHistorySessions(limit = 50, offset = 0) {
  return apiClient.get(`/api/v1/admin/history/sessions?limit=${limit}&offset=${offset}`);
}

export async function listHistoryJobs(wrId, status = null, limit = 50, offset = 0) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) params.set('status', status);
  return apiClient.get(`/api/v1/admin/history/sessions/${wrId}/jobs?${params.toString()}`);
}

// ── Tools ──────────────────────────────────────────────────

export async function startRepairParse() {
  return apiClient.post('/api/v1/admin/tools/repair-parse');
}

export async function getRepairParseStatus() {
  return apiClient.get('/api/v1/admin/tools/repair-parse/status');
}

