-- Imagine PostgreSQL Schema v2 (Client-Server Architecture)
-- Requires: PostgreSQL 15+, pgvector extension, pg_trgm extension

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector for vector similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigram for fuzzy text matching

-- ═══════════════════════════════════════════════════════════════
-- Users & Authentication
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    -- Quotas
    quota_files_per_day INTEGER DEFAULT 1000,
    quota_search_per_min INTEGER DEFAULT 60
);

CREATE TABLE IF NOT EXISTS invite_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(32) UNIQUE NOT NULL,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    max_uses INTEGER DEFAULT 1,
    use_count INTEGER DEFAULT 0,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Track which invite code each user registered with
CREATE TABLE IF NOT EXISTS invite_uses (
    id SERIAL PRIMARY KEY,
    invite_id INTEGER NOT NULL REFERENCES invite_codes(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    used_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(invite_id, user_id)
);

-- Refresh tokens for JWT renewal
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA256 of refresh token
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    revoked BOOLEAN DEFAULT FALSE
);

-- ═══════════════════════════════════════════════════════════════
-- Files & Metadata (migrated from SQLite files table)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,

    -- File identification
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_size INTEGER,
    content_hash VARCHAR(64),    -- SHA256(file_size + first_8KB + last_8KB)

    -- Image properties
    format VARCHAR(10),
    width INTEGER,
    height INTEGER,

    -- AI-generated fields (Phase V: Vision)
    mc_caption TEXT,             -- Meta-Context Caption
    ai_tags JSONB DEFAULT '[]'::jsonb,
    ocr_text TEXT,
    dominant_color VARCHAR(20),
    ai_style VARCHAR(100),

    -- v3 2-Stage Vision classification
    image_type VARCHAR(50),
    art_style VARCHAR(50),
    color_palette VARCHAR(50),
    scene_type VARCHAR(100),
    time_of_day VARCHAR(50),
    weather VARCHAR(50),
    character_type VARCHAR(50),
    item_type VARCHAR(50),
    ui_type VARCHAR(50),
    structured_meta JSONB,

    -- Path abstraction
    storage_root TEXT,
    relative_path TEXT,

    -- Nested metadata (JSON)
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Folder discovery metadata
    folder_path TEXT,
    folder_depth INTEGER DEFAULT 0,
    folder_tags JSONB DEFAULT '[]'::jsonb,

    -- Embedding version tracking
    embedding_model VARCHAR(100),
    embedding_version INTEGER DEFAULT 1,

    -- 3-Tier AI Mode metadata
    mode_tier VARCHAR(20),
    caption_model VARCHAR(100),
    text_embed_model VARCHAR(100),
    runtime_version VARCHAR(50),
    preprocess_params JSONB,

    -- Perceptual hash for de-duplication
    perceptual_hash BIGINT,
    dup_group_id INTEGER,

    -- User metadata
    user_note TEXT DEFAULT '',
    user_tags JSONB DEFAULT '[]'::jsonb,
    user_category VARCHAR(100) DEFAULT '',
    user_rating INTEGER DEFAULT 0 CHECK (user_rating BETWEEN 0 AND 5),

    -- Ownership (server mode)
    uploaded_by INTEGER REFERENCES users(id) ON DELETE SET NULL,

    -- Thumbnail
    thumbnail_url TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ,
    modified_at TIMESTAMPTZ,
    parsed_at TIMESTAMPTZ DEFAULT NOW(),

    -- Full-text search vector (tsvector)
    fts_vector tsvector
);

-- ═══════════════════════════════════════════════════════════════
-- Layers (PSD layer metadata)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS layers (
    id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    layer_path TEXT NOT NULL,
    layer_name TEXT,
    layer_type VARCHAR(50),
    metadata JSONB,
    ai_caption TEXT,
    ai_tags JSONB,
    UNIQUE(file_id, layer_path)
);

-- ═══════════════════════════════════════════════════════════════
-- Vector tables (pgvector)
-- ═══════════════════════════════════════════════════════════════

-- VV: Visual Vector (SigLIP2)
-- Max dimension 1664 (ultra tier). Actual dimension tracked in 'dimension' column.
CREATE TABLE IF NOT EXISTS vec_files (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    embedding vector(1664),
    dimension INTEGER NOT NULL    -- Actual dimension used (768/1152/1664)
);

-- MV: Meaning Vector (Qwen3-Embedding)
-- Max dimension 4096 (ultra tier).
CREATE TABLE IF NOT EXISTS vec_text (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    embedding vector(4096),
    dimension INTEGER NOT NULL    -- Actual dimension used (256/1024/4096)
);

-- Structure Vector (DINOv2)
-- Fixed 768 dimensions.
CREATE TABLE IF NOT EXISTS vec_structure (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    embedding vector(768)
);

-- ═══════════════════════════════════════════════════════════════
-- Job Queue (distributed processing)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS job_queue (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'assigned', 'processing', 'completed', 'failed', 'cancelled')),
    assigned_to INTEGER REFERENCES users(id) ON DELETE SET NULL,
    assigned_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Phase-level tracking
    phase_completed JSONB DEFAULT '{"parse":false,"vision":false,"embed":false}'::jsonb,

    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Priority (higher = first)
    priority INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════
-- System metadata
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS system_meta (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════
-- Indexes
-- ═══════════════════════════════════════════════════════════════

-- Files: standard lookups
CREATE INDEX IF NOT EXISTS idx_files_format ON files(format);
CREATE INDEX IF NOT EXISTS idx_files_parsed_at ON files(parsed_at DESC);
CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_relative_path ON files(relative_path);
CREATE INDEX IF NOT EXISTS idx_files_storage_root ON files(storage_root);
CREATE INDEX IF NOT EXISTS idx_files_image_type ON files(image_type);
CREATE INDEX IF NOT EXISTS idx_files_art_style ON files(art_style);
CREATE INDEX IF NOT EXISTS idx_files_uploaded_by ON files(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_files_perceptual_hash ON files(perceptual_hash);
CREATE INDEX IF NOT EXISTS idx_files_dup_group_id ON files(dup_group_id);

-- Files: GIN indexes (JSONB full containment search)
CREATE INDEX IF NOT EXISTS idx_files_ai_tags_gin ON files USING GIN (ai_tags);
CREATE INDEX IF NOT EXISTS idx_files_folder_tags_gin ON files USING GIN (folder_tags);
CREATE INDEX IF NOT EXISTS idx_files_metadata_gin ON files USING GIN (metadata);

-- Files: Full-text search (tsvector GIN)
CREATE INDEX IF NOT EXISTS idx_files_fts ON files USING GIN (fts_vector);

-- Layers
CREATE INDEX IF NOT EXISTS idx_layers_file_id ON layers(file_id);

-- pgvector HNSW indexes (approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_vec_files_hnsw ON vec_files
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_vec_text_hnsw ON vec_text
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_vec_structure_hnsw ON vec_structure
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Job queue
CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
CREATE INDEX IF NOT EXISTS idx_job_queue_assigned ON job_queue(assigned_to, status);
CREATE INDEX IF NOT EXISTS idx_job_queue_priority ON job_queue(priority DESC, created_at ASC);

-- Refresh tokens
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires ON refresh_tokens(expires_at);

-- ═══════════════════════════════════════════════════════════════
-- FTS trigger: auto-update tsvector on INSERT/UPDATE
-- ═══════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION files_fts_update_trigger() RETURNS trigger AS $$
BEGIN
    NEW.fts_vector :=
        setweight(to_tsvector('simple', COALESCE(NEW.file_name, '')), 'A') ||
        setweight(to_tsvector('simple', COALESCE(
            array_to_string(
                ARRAY(SELECT jsonb_array_elements_text(COALESCE(NEW.ai_tags, '[]'::jsonb))),
                ' '
            ), ''
        )), 'A') ||
        setweight(to_tsvector('simple', COALESCE(NEW.mc_caption, '')), 'B') ||
        setweight(to_tsvector('simple', COALESCE(NEW.folder_path, '')), 'C') ||
        setweight(to_tsvector('simple', COALESCE(NEW.image_type, '')), 'C') ||
        setweight(to_tsvector('simple', COALESCE(NEW.art_style, '')), 'C') ||
        setweight(to_tsvector('simple', COALESCE(NEW.user_note, '')), 'D') ||
        setweight(to_tsvector('simple', COALESCE(
            array_to_string(
                ARRAY(SELECT jsonb_array_elements_text(COALESCE(NEW.user_tags, '[]'::jsonb))),
                ' '
            ), ''
        )), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_files_fts_update ON files;
CREATE TRIGGER trg_files_fts_update
    BEFORE INSERT OR UPDATE ON files
    FOR EACH ROW EXECUTE FUNCTION files_fts_update_trigger();

-- ═══════════════════════════════════════════════════════════════
-- Job queue: auto-update updated_at timestamp
-- ═══════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION job_queue_updated_at_trigger() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_job_queue_updated_at ON job_queue;
CREATE TRIGGER trg_job_queue_updated_at
    BEFORE UPDATE ON job_queue
    FOR EACH ROW EXECUTE FUNCTION job_queue_updated_at_trigger();
