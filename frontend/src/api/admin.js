/**
 * Admin API — user management, invite codes.
 */

import { apiClient } from './client';

// ── Analysis Job Maintenance ─────────────────────────────

function countFailedPhases(job) {
  const failed = job?.progress?.failed;
  if (!failed || typeof failed !== 'object') return 0;
  return Object.values(failed).reduce((sum, value) => sum + (value || 0), 0);
}

async function listAnalysisJobsForMaintenance() {
  const data = await apiClient.get('/api/v1/analysis-jobs?include_completed=true');
  return (data?.jobs || []).filter(job => !['archived', 'cancelled'].includes(job.status));
}

export async function retryFailedJobs() {
  const jobs = await listAnalysisJobsForMaintenance();
  let retried = 0;

  for (const job of jobs) {
    if (countFailedPhases(job) === 0) continue;
    try {
      const result = await apiClient.post(`/api/v1/analysis-jobs/${job.id}/retry`, { phase: null });
      retried += result?.retried || 0;
    } catch {
      // A job can become non-retryable between list and retry; continue with the rest.
    }
  }

  return { success: true, retried };
}

export async function forceRetryFailedJobs() {
  return retryFailedJobs();
}

export async function auditIntegrity() {
  // Deep legacy audit was removed; Analysis Job System handles task state via file_tasks.
  return { success: true, total_files: 0, complete_files: 0, incomplete_files: 0, repaired_files: 0 };
}

export async function dismissPermanentlyFailedJobs() {
  const jobs = await listAnalysisJobsForMaintenance();
  let dismissedJobs = 0;

  for (const job of jobs) {
    if (countFailedPhases(job) === 0) continue;
    try {
      const result = await apiClient.post(`/api/v1/analysis-jobs/${job.id}/dismiss`);
      dismissedJobs += result?.dismissed || 0;
    } catch {
      // Dismiss only applies to permanent failures; skip jobs that do not qualify.
    }
  }

  return { success: true, dismissed_jobs: dismissedJobs };
}

// ── Discover (server filesystem) ────────────────────────

export async function browseFolders(path = '/') {
  return apiClient.get(`/api/v1/discover/browse?path=${encodeURIComponent(path)}`);
}

export async function scanFolder(folderPath, priority = 0, options = {}) {
  const payload = { folder_path: folderPath, priority };
  if (options?.name) payload.name = options.name;
  if (options?.analysisProfile) payload.analysis_profile = options.analysisProfile;
  return apiClient.post('/api/v1/discover/scan', payload);
}

export async function registerPaths(filePaths, analysisProfile = null) {
  const paths = Array.isArray(filePaths) ? filePaths.filter(Boolean) : [];
  if (paths.length === 0) return { success: false, jobs_created: 0, total_files: 0 };

  const firstPath = String(paths[0]).replace(/\\/g, '/');
  const slash = firstPath.lastIndexOf('/');
  const sourcePath = slash > 0 ? firstPath.slice(0, slash) : 'selected-files';
  const fileName = slash >= 0 ? firstPath.slice(slash + 1) : firstPath;
  const name = paths.length === 1 ? fileName : `Selected files (${paths.length})`;

  const result = await apiClient.post('/api/v1/analysis-jobs', {
    name,
    source_path: sourcePath,
    file_paths: paths,
    ...(analysisProfile ? { analysis_profile: analysisProfile } : {}),
  });
  return {
    success: result?.success !== false,
    jobs_created: (result?.total_files || 0) > 0 ? 1 : 0,
    ...result,
  };
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

// ── Connection Info / Headless Worker Install ──────────────

export async function getConnectionInfo() {
  return apiClient.get('/api/v1/server/connection-info');
}

export async function createHeadlessWorkerCommand({
  worker_name,
  launcher = 'cloud',
  expires_minutes = 1440,
  server_url = null,
  connect_mode = 'direct_lan',
} = {}) {
  const payload = {
    worker_name,
    launcher,
    expires_minutes,
    connect_mode,
  };
  if (server_url) payload.server_url = server_url;
  return apiClient.post('/api/v1/admin/workers/headless-command', payload);
}
