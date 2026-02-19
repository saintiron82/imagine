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

// ── Discover (server filesystem) ────────────────────────

export async function browseFolders(path = '/') {
  return apiClient.get(`/api/v1/discover/browse?path=${encodeURIComponent(path)}`);
}

export async function scanFolder(folderPath, priority = 0) {
  return apiClient.post('/api/v1/discover/scan', { folder_path: folderPath, priority });
}
