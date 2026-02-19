/**
 * Service Bridge — abstracts Electron IPC vs Server HTTP API.
 *
 * Electron mode: delegates to window.electron.pipeline.* via IPC
 * Web mode:      calls server API via apiClient
 */

import { apiClient, isElectron, getServerUrl, getAccessToken } from '../api/client';


/**
 * Search images (Triaxis: VV + MV + FTS).
 * Accepts the same options format used by Electron IPC searchVector.
 */
export async function searchImages(options) {
  if (isElectron) {
    return window.electron.pipeline.searchVector(options);
  }

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

  // Map mode to endpoint
  const modeMap = {
    triaxis: '/api/v1/search/triaxis',
    vector: '/api/v1/search/visual',
    text_vector: '/api/v1/search/semantic',
    fts: '/api/v1/search/keyword',
    structure: '/api/v1/search/triaxis',
  };
  const endpoint = modeMap[options.mode] || '/api/v1/search/triaxis';

  return apiClient.post(endpoint, body);
}


/**
 * Get file detail by ID.
 */
export async function getFileDetail(fileId) {
  if (isElectron) {
    return window.electron.pipeline.readMetadata(fileId);
  }
  return apiClient.get(`/api/v1/files/${fileId}`);
}


/**
 * Get DB stats (total files, processed counts, format distribution).
 */
export async function getDbStats() {
  if (isElectron) {
    return window.electron.pipeline.getDbStats();
  }
  return apiClient.get('/api/v1/stats/db');
}


/**
 * Get thumbnail URL for a file.
 * Electron: local file path. Web: server API endpoint.
 */
export function getThumbnailUrl(thumbnailPath, fileId) {
  if (isElectron) {
    return thumbnailPath || null;
  }
  if (!fileId) return null;
  const base = getServerUrl();
  const token = getAccessToken();
  const url = `${base}/api/v1/files/${fileId}/thumbnail`;
  return token ? `${url}?token=${token}` : url;
}


/**
 * Update user metadata (notes, tags, category, rating).
 */
export async function updateUserMeta(filePathOrId, updates) {
  if (isElectron) {
    return window.electron.metadata.updateUserData(filePathOrId, updates);
  }
  // Web mode: filePathOrId is file_id (number)
  return apiClient.patch(`/api/v1/files/${filePathOrId}/user-meta`, updates);
}
