-- ImageParser SQLite Schema with sqlite-vec
-- Single-file database: imageparser.db

-- Files table: File-level metadata + CLIP vectors
CREATE TABLE IF NOT EXISTS files (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- File identification
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_size INTEGER,

    -- Image properties
    format TEXT,
    width INTEGER,
    height INTEGER,

    -- AI-generated fields (Phase 4 → v3.1)
    mc_caption TEXT,  -- Meta-Context Caption (renamed from ai_caption in v3.1)
    ai_tags TEXT,  -- JSON array as TEXT: ["tag1", "tag2"]
    ocr_text TEXT,
    dominant_color TEXT,
    ai_style TEXT,

    -- v3.1: Perceptual hash for de-duplication
    perceptual_hash INTEGER,
    dup_group_id INTEGER,

    -- Nested metadata (JSON for complex structures)
    -- Contains: layer_tree, semantic_tags, text_content, layer_count, used_fonts
    metadata TEXT,  -- JSON stored as TEXT

    -- Timestamps
    created_at TEXT,  -- ISO 8601 format
    modified_at TEXT,
    parsed_at TEXT DEFAULT (datetime('now')),

    -- References
    thumbnail_url TEXT,

    -- Folder discovery metadata
    folder_path TEXT,
    folder_depth INTEGER DEFAULT 0,
    folder_tags TEXT,  -- JSON array as TEXT: ["Characters", "Hero"]

    -- User metadata
    user_note TEXT DEFAULT '',
    user_tags TEXT DEFAULT '[]',  -- JSON array as TEXT
    user_category TEXT DEFAULT '',
    user_rating INTEGER DEFAULT 0,  -- 0-5 stars

    -- v3 P0: 2-Stage Vision classification
    image_type TEXT,              -- character, background, ui_element, item, icon, texture, effect, logo, photo, illustration, other
    art_style TEXT,               -- realistic, anime, pixel, painterly, cartoon, 3d_render, etc.
    color_palette TEXT,           -- warm, cool, monochrome, vibrant, pastel, dark, neutral
    scene_type TEXT,              -- background only: alley, forest, dungeon, castle, etc.
    time_of_day TEXT,             -- background only: dawn, morning, noon, sunset, night, etc.
    weather TEXT,                 -- background only: clear, rain, snow, fog, storm, etc.
    character_type TEXT,          -- character only: human, monster, animal, etc.
    item_type TEXT,               -- item only: weapon, armor, potion, etc.
    ui_type TEXT,                 -- ui_element only: button, panel, hud, etc.
    structured_meta TEXT,         -- Full Stage 2 JSON output

    -- v3 P0: Path abstraction
    storage_root TEXT,            -- POSIX-normalized storage root
    relative_path TEXT,           -- POSIX-normalized relative path

    -- v3 P0: Embedding version tracking
    embedding_model TEXT DEFAULT '{VISUAL_MODEL}',
    embedding_version INTEGER DEFAULT 1,

    -- v3.1: 3-Tier AI Mode metadata
    mode_tier TEXT DEFAULT '{DEFAULT_TIER}',           -- standard | pro | ultra
    caption_model TEXT,                              -- VLM model (e.g., Qwen/Qwen3-VL-4B-Instruct)
    text_embed_model TEXT,                           -- MV model (e.g., qwen3-embedding:0.6b)
    runtime_version TEXT,                            -- Ollama/runtime version (e.g., ollama-0.15.2)
    preprocess_params TEXT                           -- JSON: {max_edge, aspect_ratio_mode, padding_color}
);

-- Layers table: Layer-level metadata (optional)
CREATE TABLE IF NOT EXISTS layers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,

    -- Layer identification
    layer_path TEXT NOT NULL,
    layer_name TEXT,
    layer_type TEXT,

    -- Layer metadata (JSON)
    metadata TEXT,

    -- AI analysis (layer-level)
    ai_caption TEXT,
    ai_tags TEXT,  -- JSON array as TEXT

    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    UNIQUE(file_id, layer_path)
);

-- Virtual table for file embeddings (sqlite-vec)
-- Dimension is set dynamically from active tier config
CREATE VIRTUAL TABLE IF NOT EXISTS vec_files USING vec0(
    file_id INTEGER PRIMARY KEY,
    embedding FLOAT[{VISUAL_DIM}]
);

-- Virtual table for layer embeddings (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_layers USING vec0(
    layer_id INTEGER PRIMARY KEY,
    embedding FLOAT[{VISUAL_DIM}]
);

-- Virtual table for MV (Meaning Vector) (sqlite-vec)
-- Generated from MC (mc_caption + ai_tags) via MV model
CREATE VIRTUAL TABLE IF NOT EXISTS vec_text USING vec0(
    file_id INTEGER PRIMARY KEY,
    embedding FLOAT[{TEXT_DIM}]
);

-- Standard indexes for common queries
CREATE INDEX IF NOT EXISTS idx_files_format ON files(format);
CREATE INDEX IF NOT EXISTS idx_files_parsed_at ON files(parsed_at DESC);
CREATE INDEX IF NOT EXISTS idx_files_resolution ON files(width, height);
CREATE INDEX IF NOT EXISTS idx_layers_file_id ON layers(file_id);
CREATE INDEX IF NOT EXISTS idx_files_folder_path ON files(folder_path);
CREATE INDEX IF NOT EXISTS idx_image_type ON files(image_type);
CREATE INDEX IF NOT EXISTS idx_art_style ON files(art_style);
CREATE INDEX IF NOT EXISTS idx_scene_type ON files(scene_type);
CREATE INDEX IF NOT EXISTS idx_relative_path ON files(relative_path);
CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON files(perceptual_hash);
CREATE INDEX IF NOT EXISTS idx_dup_group_id ON files(dup_group_id);

-- Full-text search (FTS5) — 2-column BM25-weighted architecture
--
-- meta_strong (BM25 weight: 3.0) — Direct identification facts:
--   file_name, layer_names, used_fonts, user_tags, ocr_text
--
-- meta_weak (BM25 weight: 1.5) — Contextual information:
--   file_path, text_content, user_note, folder_tags, image_type, scene_type, art_style
--
-- NOTE: caption column (mc_caption, ai_tags) will be added when AI caption feature is implemented
--
-- Design: Candidate First
--   1. FTS MATCH → candidate doc_id list (top 2000 if >10k results)
--   2. NumPy scoring only on candidates (VV/MV vectors)
--   3. RRF merge → de-dup → return
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    meta_strong,      -- BM25 3.0: file_name, layer_names, used_fonts, user_tags, ocr_text
    meta_weak         -- BM25 1.5: file_path, text_content, user_note, folder_tags, image_type, scene_type, art_style
);

-- Triggers: 2-column FTS
-- meta_strong, meta_weak are built by Python (complex JSON walking)
-- SQL triggers insert empty strings; Python updates immediately after
CREATE TRIGGER IF NOT EXISTS files_fts_insert AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, meta_strong, meta_weak)
    VALUES (new.id, '', '');
END;

CREATE TRIGGER IF NOT EXISTS files_fts_update AFTER UPDATE ON files BEGIN
    DELETE FROM files_fts WHERE rowid = old.id;
    INSERT INTO files_fts(rowid, meta_strong, meta_weak)
    VALUES (new.id, '', '');
END;

CREATE TRIGGER IF NOT EXISTS files_fts_delete AFTER DELETE ON files BEGIN
    DELETE FROM files_fts WHERE rowid = old.id;
END;
