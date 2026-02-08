-- ImageParser PostgreSQL Schema with pgvector
-- This schema supports the 3-Axis architecture:
--   1. Structural: layer_tree (JSONB)
--   2. Latent: CLIP embeddings (pgvector)
--   3. Descriptive: AI captions/tags

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text similarity search

-- Files table: File-level metadata + CLIP vectors
CREATE TABLE IF NOT EXISTS files (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- File identification
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_size BIGINT,

    -- Image properties
    format TEXT,
    width INTEGER,
    height INTEGER,

    -- AI-generated fields (Phase 4)
    ai_caption TEXT,
    ai_tags TEXT[],  -- PostgreSQL array
    ocr_text TEXT,
    dominant_color TEXT,
    ai_style TEXT,

    -- Nested metadata (JSONB for complex structures)
    -- Contains: layer_tree, translated_layer_tree, semantic_tags, text_content, etc.
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP,
    modified_at TIMESTAMP,
    parsed_at TIMESTAMP DEFAULT NOW(),

    -- References
    thumbnail_url TEXT,

    -- CLIP vector embedding (768 dimensions for CLIP-ViT-L-14)
    embedding vector(768)
);

-- Layers table: Layer-level metadata (optional, for major layers only)
CREATE TABLE IF NOT EXISTS layers (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,

    -- Layer identification
    layer_path TEXT NOT NULL,
    layer_name TEXT,
    layer_type TEXT,

    -- Layer metadata (JSONB for flexibility)
    metadata JSONB,

    -- AI analysis (layer-level)
    ai_caption TEXT,
    ai_tags TEXT[],

    -- Layer-level CLIP embedding
    embedding vector(768),

    UNIQUE(file_id, layer_path)
);

-- Performance indexes
-- Vector similarity search (HNSW algorithm)
CREATE INDEX IF NOT EXISTS idx_files_embedding ON files
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_layers_embedding ON layers
    USING hnsw (embedding vector_cosine_ops);

-- JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_files_metadata ON files
    USING gin (metadata);

-- Full-text search on AI captions
CREATE INDEX IF NOT EXISTS idx_files_ai_caption ON files
    USING gin (ai_caption gin_trgm_ops);

-- Standard indexes for common queries
CREATE INDEX IF NOT EXISTS idx_files_format ON files (format);
CREATE INDEX IF NOT EXISTS idx_files_parsed_at ON files (parsed_at DESC);
CREATE INDEX IF NOT EXISTS idx_files_resolution ON files (width, height);
CREATE INDEX IF NOT EXISTS idx_layers_file_id ON layers (file_id);

-- Comments for documentation
COMMENT ON TABLE files IS 'File-level metadata and CLIP embeddings';
COMMENT ON TABLE layers IS 'Layer-level metadata (major layers only, ~30% of total)';
COMMENT ON COLUMN files.metadata IS 'Nested JSON: layer_tree, translated_layer_tree, semantic_tags, text_content, layer_count, used_fonts';
COMMENT ON COLUMN files.embedding IS 'CLIP ViT-L-14 embedding (768 dimensions)';
COMMENT ON INDEX idx_files_embedding IS 'HNSW index for fast vector similarity search (cosine distance)';
