/**
 * Service Bridge — 서버 HTTP API 호출 계층.
 * v2 는 항상 원격 서버(HTTP) 모드로 동작한다.
 */

import { apiClient, getServerUrl, getAccessToken } from '../api/client';


/**
 * Search images (Triaxis: VV + MV + FTS).
 * Accepts the same options format used by Electron IPC searchVector.
 */
export async function searchImages(options) {
  // Translate IPC-style options → server SearchRequest body
  const body = {
    query: options.query || '',
    limit: options.limit || 20,
    threshold: options.threshold ?? 0,
    filters: options.filters || null,
  };

  if (options.queryImage) body.query_image = options.queryImage;
  if (options.queryImages) body.query_images = options.queryImages;
  if (options.imageSearchMode) body.image_search_mode = options.imageSearchMode;
  if (options.queryFileId != null) body.query_file_id = options.queryFileId;
  if (options.use_codex != null) body.use_codex = options.use_codex;
  if (options.effort) body.effort = options.effort;
  if (options.file_ids) body.file_ids = options.file_ids;

  // Map mode to endpoint
  const modeMap = {
    triaxis: '/api/v1/search/triaxis',
    vector: '/api/v1/search/visual',
    text_vector: '/api/v1/search/semantic',
    fts: '/api/v1/search/keyword',
    structure: '/api/v1/search/structure',
  };
  const endpoint = modeMap[options.mode] || '/api/v1/search/triaxis';

  return apiClient.post(endpoint, body);
}


/**
 * Submit user feedback that a search result is irrelevant for a query.
 * Backend accumulates these to soft-demote that file in subsequent searches.
 */
export async function postSearchFeedback(query, fileId) {
  return apiClient.post('/api/v1/search/feedback', {
    query,
    file_id: fileId,
    label: 'irrelevant',
  });
}


/**
 * Get file detail by ID.
 */
export async function getFileDetail(fileId) {
  return apiClient.get(`/api/v1/files/${fileId}`);
}


/**
 * Get DB stats (total files, processed counts, format distribution).
 */
export async function getDbStats() {
  return apiClient.get('/api/v1/stats/db');
}


/**
 * Get thumbnail URL for a file (server API URL with JWT token).
 */
export function getThumbnailUrl(thumbnailPath, fileId) {
  if (!fileId) return null;
  const base = getServerUrl();
  const token = getAccessToken();
  const url = `${base}/api/v1/files/${fileId}/thumbnail`;
  return token ? `${url}?token=${token}` : url;
}


/**
 * Update user metadata (notes, tags, category, rating).
 */
export async function updateUserMeta(fileId, updates) {
  return apiClient.patch(`/api/v1/files/${fileId}/user-meta`, updates);
}


// ── Classification Domains ───────────────────────────────────

/**
 * List available classification domain presets.
 */
export async function listDomains() {
  return apiClient.get('/api/v1/admin/classification/domains');
}

/**
 * Get full details of a classification domain.
 */
export async function getDomainDetail(domainId) {
  return apiClient.get(`/api/v1/admin/classification/domains/${domainId}`);
}

/**
 * Get current active domain config.
 * Returns { active_domain: string|null, domain: DomainDetail|null }
 */
export async function getActiveDomainConfig() {
  return apiClient.get('/api/v1/admin/classification/active');
}

/**
 * Set the active classification domain.
 */
export async function setActiveDomain(domainId) {
  return apiClient.put('/api/v1/admin/classification/active', { domain_id: domainId });
}

export async function saveDomainYaml(domainId, yamlContent) {
  return apiClient.post('/api/v1/admin/classification/domains', {
    domain_id: domainId,
    yaml_content: yamlContent,
  });
}
