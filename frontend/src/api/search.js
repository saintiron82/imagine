/**
 * Search API — Triaxis search endpoints.
 */

import { apiClient } from './client';

/**
 * Triaxis search (VV + MV + FTS combined via RRF).
 */
export async function searchTriaxis({
  query = '',
  query_image,
  query_images,
  image_search_mode = 'and',
  query_file_id,
  limit = 20,
  threshold = 0.0,
  filters,
  diagnostic = false,
} = {}) {
  return apiClient.post('/api/v1/search/triaxis', {
    query,
    query_image,
    query_images,
    image_search_mode,
    query_file_id,
    limit,
    threshold,
    filters,
    diagnostic,
  });
}

/**
 * VV-only search (SigLIP2 visual similarity).
 */
export async function searchVisual(opts) {
  return apiClient.post('/api/v1/search/visual', opts);
}

/**
 * MV-only search (Qwen3-Embedding text similarity).
 */
export async function searchSemantic(opts) {
  return apiClient.post('/api/v1/search/semantic', opts);
}

/**
 * FTS-only search (FTS5 BM25 keyword matching).
 */
export async function searchKeyword(opts) {
  return apiClient.post('/api/v1/search/keyword', opts);
}

/**
 * Find similar images to a given file.
 */
export async function searchSimilar(fileId, limit = 20) {
  return apiClient.post(`/api/v1/search/similar/${fileId}`, { limit });
}
