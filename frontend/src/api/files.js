/**
 * Files API — file listing, metadata, thumbnails.
 */

import { apiClient, getServerUrl, getAccessToken } from './client';

/**
 * List files with pagination and filtering.
 */
export async function listFiles({
  limit = 50,
  offset = 0,
  folder_path,
  format,
  has_mc,
  has_vv,
  has_mv,
  sort_by = 'created_at',
  sort_order = 'desc',
} = {}) {
  return apiClient.get('/api/v1/files', {
    limit,
    offset,
    folder_path,
    format,
    has_mc,
    has_vv,
    has_mv,
    sort_by,
    sort_order,
  });
}

/**
 * Get single file details.
 */
export async function getFile(fileId) {
  return apiClient.get(`/api/v1/files/${fileId}`);
}

/**
 * Update user metadata (notes, custom_tags, category, rating).
 */
export async function updateUserMeta(fileId, updates) {
  return apiClient.patch(`/api/v1/files/${fileId}/user-meta`, updates);
}

/**
 * Delete a file (admin or owner only).
 */
export async function deleteFile(fileId) {
  return apiClient.delete(`/api/v1/files/${fileId}`);
}

/**
 * Get thumbnail URL for a file (authenticated).
 */
export function getThumbnailUrl(fileId) {
  const base = getServerUrl();
  const token = getAccessToken();
  return `${base}/api/v1/files/${fileId}/thumbnail?token=${encodeURIComponent(token || '')}`;
}
