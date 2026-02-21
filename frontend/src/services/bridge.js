/**
 * Service Bridge — abstracts Electron IPC vs Server HTTP API.
 *
 * Local mode (Electron server/admin): delegates to window.electron.pipeline.* via IPC
 * Remote mode (Electron client/worker, Web): calls server API via apiClient
 */

import { apiClient, isElectron, getServerUrl, getAccessToken } from '../api/client';


// ── Mode management ──────────────────────────────────────────
// Electron server mode → IPC (local Python backend)
// Electron client mode / Web → HTTP API (remote server)
let _useLocalBackend = isElectron; // safe default: existing behavior

/**
 * Set whether bridge functions should use local IPC backend.
 * Called by App.jsx once appMode is determined from config.yaml.
 *
 * true  = Electron server mode (IPC → local Python)
 * false = Electron client mode or Web (HTTP API → remote server)
 */
export function setUseLocalBackend(value) {
  _useLocalBackend = value;
}

/** Check if currently using local IPC backend. */
export function isLocalMode() {
  return _useLocalBackend;
}


/**
 * Search images (Triaxis: VV + MV + FTS).
 * Accepts the same options format used by Electron IPC searchVector.
 */
export async function searchImages(options) {
  if (_useLocalBackend) {
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
  if (_useLocalBackend) {
    return window.electron.pipeline.readMetadata(fileId);
  }
  return apiClient.get(`/api/v1/files/${fileId}`);
}


/**
 * Get DB stats (total files, processed counts, format distribution).
 */
export async function getDbStats() {
  if (_useLocalBackend) {
    return window.electron.pipeline.getDbStats();
  }
  return apiClient.get('/api/v1/stats/db');
}


/**
 * Get thumbnail URL for a file.
 * Local mode: returns raw path (caller converts to file:// URL).
 * Remote mode: returns server API URL with JWT token.
 */
export function getThumbnailUrl(thumbnailPath, fileId) {
  if (_useLocalBackend) {
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
  if (_useLocalBackend) {
    return window.electron.metadata.updateUserData(filePathOrId, updates);
  }
  // Remote mode: filePathOrId is file_id (number)
  return apiClient.patch(`/api/v1/files/${filePathOrId}/user-meta`, updates);
}


// ── Folder Sync ──────────────────────────────────────────────

/**
 * Scan a folder and compare disk state with DB records.
 * Returns: { matched, moved, missing, new_files, moved_list, missing_list }
 */
export async function syncFolder(folderPath) {
  if (_useLocalBackend) {
    return window.electron.sync.scanFolder(folderPath);
  }
  return apiClient.post('/api/v1/sync/scan', { folder_path: folderPath });
}

/**
 * Apply path updates for moved files.
 */
export async function syncApplyMoves(moves) {
  if (_useLocalBackend) {
    return window.electron.sync.applyMoves(moves);
  }
  return apiClient.post('/api/v1/sync/apply-moves', { moves });
}

/**
 * Delete DB records for files no longer on disk.
 */
export async function syncDeleteMissing(fileIds) {
  if (_useLocalBackend) {
    return window.electron.sync.deleteMissing(fileIds);
  }
  return apiClient.post('/api/v1/sync/delete-missing', { file_ids: fileIds });
}


// ── File Download ────────────────────────────────────────────

/**
 * Get download URL for original image file.
 * Local mode: returns null (use openFile instead).
 * Remote mode: returns server API URL with JWT token.
 */
export function getOriginalDownloadUrl(fileId) {
  if (_useLocalBackend) {
    return null; // Electron uses openFile / showInFolder
  }
  if (!fileId) return null;
  const base = getServerUrl();
  const token = getAccessToken();
  const url = `${base}/api/v1/files/${fileId}/download`;
  return token ? `${url}?token=${token}` : url;
}
