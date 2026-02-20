/**
 * Admin API — user management, invite codes.
 */

import { apiClient } from './client';

// ── Invite Codes ─────────────────────────────────────────

export async function createInviteCode({ max_uses = 1, expires_days = 7, note = '' } = {}) {
  return apiClient.post('/api/v1/admin/invite-codes', { max_uses, expires_days, note });
}

export async function listInviteCodes() {
  return apiClient.get('/api/v1/admin/invite-codes');
}

// ── Users ────────────────────────────────────────────────

export async function listUsers() {
  return apiClient.get('/api/v1/admin/users');
}

export async function updateUser(userId, updates) {
  return apiClient.patch(`/api/v1/admin/users/${userId}`, updates);
}

export async function deleteUser(userId) {
  return apiClient.delete(`/api/v1/admin/users/${userId}`);
}

// ── Job Queue ────────────────────────────────────────────

export async function cleanupStaleJobs() {
  return apiClient.post('/api/v1/admin/jobs/cleanup');
}

export async function getJobStats() {
  return apiClient.get('/api/v1/jobs/stats');
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

export async function clearCompletedJobs() {
  return apiClient.delete('/api/v1/admin/jobs/clear-completed');
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

// ── Worker Tokens ────────────────────────────────────────

export async function createWorkerToken({ name, expires_in_days = 30 } = {}) {
  return apiClient.post('/api/v1/admin/worker-tokens', { name, expires_in_days });
}

export async function listWorkerTokens() {
  return apiClient.get('/api/v1/admin/worker-tokens');
}

export async function revokeWorkerToken(tokenId) {
  return apiClient.delete(`/api/v1/admin/worker-tokens/${tokenId}`);
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

// ── Worker Self-service ──────────────────────────────────

export async function registerWorker() {
  return apiClient.post('/api/v1/worker/register');
}
