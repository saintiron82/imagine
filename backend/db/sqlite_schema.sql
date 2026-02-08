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

    -- AI-generated fields (Phase 4)
    ai_caption TEXT,
    ai_tags TEXT,  -- JSON array as TEXT: ["tag1", "tag2"]
    ocr_text TEXT,
    dominant_color TEXT,
    ai_style TEXT,

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
    embedding_model TEXT DEFAULT 'clip-ViT-L-14',
    embedding_version INTEGER DEFAULT 1
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
CREATE VIRTUAL TABLE IF NOT EXISTS vec_files USING vec0(
    file_id INTEGER PRIMARY KEY,
    embedding FLOAT[1152]  -- SigLIP 2 So400m embeddings
);

-- Virtual table for layer embeddings (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_layers USING vec0(
    layer_id INTEGER PRIMARY KEY,
    embedding FLOAT[1152]
);

-- Virtual table for T-axis text embeddings (sqlite-vec)
-- Generated from ai_caption + ai_tags via Qwen3-Embedding-0.6B
CREATE VIRTUAL TABLE IF NOT EXISTS vec_text USING vec0(
    file_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024]
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

-- Full-text search (FTS5) — 16 columns, 3-axis architecture + v3.
--
-- 2축 Descriptive (AI 비저닝 — Qwen/Ollama):
--   ai_caption, ai_tags, ai_style, dominant_color, ocr_text
--   Note: ai_tags contains keywords NOT present in ai_caption (independently generated).
--
-- 3축 Structural (PSD 파싱 + 파일 + 사용자):
--   file_path, file_name, layer_names, text_content, used_fonts, user_note, user_tags, folder_tags
--
-- v3 P0 (2-Stage Vision classification):
--   image_type, scene_type, art_style
--
-- Merged (별도 컬럼 불필요):
--   semantic_tags → layer_names에 병합
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    file_path,        -- 3축: file identity + folder path
    file_name,        -- 3축: file name
    ai_caption,       -- 2축: AI description
    ai_tags,          -- 2축: AI keywords (independent from caption)
    ai_style,         -- 2축: AI style (independent via Ollama)
    layer_names,      -- 3축: PSD layers: original + KR + EN merged (Python)
    text_content,     -- 3축: PSD text: original + KR + EN merged (Python)
    ocr_text,         -- 2축: AI OCR
    dominant_color,   -- 2축: image color
    used_fonts,       -- 3축: PSD fonts (Python)
    user_note,        -- 3축: user memo
    user_tags,        -- 3축: user tags
    image_type,       -- v3: AI classification
    scene_type,       -- v3: background scene type
    art_style,        -- v3: art style
    folder_tags       -- 3축: folder name tags
);

-- Triggers: layer_names, text_content, used_fonts, folder_tags require Python,
-- so triggers set them to ''. Python patches immediately after INSERT.
CREATE TRIGGER IF NOT EXISTS files_fts_insert AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid,
        file_path, file_name,
        ai_caption, ai_tags, ai_style,
        layer_names, text_content,
        ocr_text, dominant_color,
        used_fonts,
        user_note, user_tags,
        image_type, scene_type, art_style, folder_tags)
    VALUES (new.id,
        new.file_path, new.file_name,
        new.ai_caption, new.ai_tags, new.ai_style,
        '', '',
        new.ocr_text, new.dominant_color,
        '',
        new.user_note, new.user_tags,
        new.image_type, new.scene_type, new.art_style, '');
END;

CREATE TRIGGER IF NOT EXISTS files_fts_update AFTER UPDATE ON files BEGIN
    DELETE FROM files_fts WHERE rowid = old.id;
    INSERT INTO files_fts(rowid,
        file_path, file_name,
        ai_caption, ai_tags, ai_style,
        layer_names, text_content,
        ocr_text, dominant_color,
        used_fonts,
        user_note, user_tags,
        image_type, scene_type, art_style, folder_tags)
    VALUES (new.id,
        new.file_path, new.file_name,
        new.ai_caption, new.ai_tags, new.ai_style,
        '', '',
        new.ocr_text, new.dominant_color,
        '',
        new.user_note, new.user_tags,
        new.image_type, new.scene_type, new.art_style, '');
END;

CREATE TRIGGER IF NOT EXISTS files_fts_delete AFTER DELETE ON files BEGIN
    DELETE FROM files_fts WHERE rowid = old.id;
END;
