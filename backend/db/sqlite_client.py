"""
SQLite client with sqlite-vec support.

This module replaces pg_client.py with SQLite-based storage,
maintaining API compatibility for minimal code changes.
"""

import logging
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class SQLiteDB:
    """SQLite database client with sqlite-vec support."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite connection.

        Args:
            db_path: Path to SQLite database file.
                    Default: ./imageparser.db
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "imageparser.db")

        self.db_path = db_path
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access

            # Enable sqlite-vec extension
            try:
                self.conn.enable_load_extension(True)
                self.conn.load_extension("vec0")  # Try loading directly first
                self.conn.enable_load_extension(False)
                logger.info("✅ sqlite-vec loaded via load_extension")
            except:
                # Fallback: try sqlite_vec Python package
                try:
                    import sqlite_vec
                    self.conn.enable_load_extension(True)
                    sqlite_vec.load(self.conn)
                    self.conn.enable_load_extension(False)
                    logger.info("✅ sqlite-vec loaded via Python package")
                except Exception as e:
                    logger.warning(f"⚠️ sqlite-vec not loaded: {e}")
                    logger.warning("Vector search will not work. Install: pip install sqlite-vec")

            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")

            # Performance optimizations
            self.conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
            self.conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes
            self.conn.execute("PRAGMA cache_size = -64000")  # 64MB cache

            # Auto-migrate existing DB on connect
            self._migrate_folder_columns()
            self._migrate_v3_columns()
            self._ensure_fts()

            logger.info(f"✅ Connected to SQLite database: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to SQLite: {e}")
            raise

    def init_schema(self):
        """
        Initialize database schema from sqlite_schema.sql file.
        Creates tables and indexes if they don't exist.
        """
        try:
            # Read schema file
            schema_path = Path(__file__).parent / "sqlite_schema.sql"
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")

            with open(schema_path, encoding='utf-8') as f:
                schema_sql = f.read()

            # Execute schema (split by semicolons for multiple statements)
            self.conn.executescript(schema_sql)
            self.conn.commit()

            logger.info("✅ SQLite schema initialized successfully")

            # Verify sqlite-vec is loaded
            try:
                cursor = self.conn.execute("SELECT sqlite_version()")
                sqlite_ver = cursor.fetchone()[0]
                logger.info(f"✅ SQLite version: {sqlite_ver}")

                # Try to check vec_version
                try:
                    cursor = self.conn.execute("SELECT vec_version()")
                    vec_ver = cursor.fetchone()[0]
                    logger.info(f"✅ sqlite-vec version: {vec_ver}")
                except:
                    logger.warning("⚠️ vec_version() not available - sqlite-vec may not be loaded")
            except Exception as e:
                logger.warning(f"Version check failed: {e}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Schema initialization failed: {e}")
            raise

    def _migrate_folder_columns(self):
        """Add folder columns to existing databases if missing."""
        try:
            self.conn.execute("SELECT folder_path FROM files LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Migrating: adding folder columns to files table...")
            self.conn.execute("ALTER TABLE files ADD COLUMN folder_path TEXT")
            self.conn.execute("ALTER TABLE files ADD COLUMN folder_depth INTEGER DEFAULT 0")
            self.conn.execute("ALTER TABLE files ADD COLUMN folder_tags TEXT")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_files_folder_path ON files(folder_path)")
            self.conn.commit()

            # FTS5 virtual tables cannot be ALTERed - drop and recreate
            logger.info("Migrating: rebuilding FTS5 table with folder_tags...")
            self.conn.execute("DROP TRIGGER IF EXISTS files_fts_insert")
            self.conn.execute("DROP TRIGGER IF EXISTS files_fts_update")
            self.conn.execute("DROP TRIGGER IF EXISTS files_fts_delete")
            self.conn.execute("DROP TABLE IF EXISTS files_fts")
            self.conn.commit()
            # FTS5 + triggers will be recreated by init_schema() executescript

            logger.info("✅ Folder columns migration complete")

    def _migrate_v3_columns(self):
        """Add v3 P0 columns to existing databases if missing."""
        v3_cols = [
            ("image_type", "TEXT"),
            ("art_style", "TEXT"),
            ("color_palette", "TEXT"),
            ("scene_type", "TEXT"),
            ("time_of_day", "TEXT"),
            ("weather", "TEXT"),
            ("character_type", "TEXT"),
            ("item_type", "TEXT"),
            ("ui_type", "TEXT"),
            ("structured_meta", "TEXT"),
            ("storage_root", "TEXT"),
            ("relative_path", "TEXT"),
            ("embedding_model", "TEXT DEFAULT 'clip-ViT-L-14'"),
            ("embedding_version", "INTEGER DEFAULT 1"),
        ]
        try:
            existing = {row[1] for row in self.conn.execute("PRAGMA table_info(files)").fetchall()}
            added = 0
            for col_name, col_def in v3_cols:
                if col_name not in existing:
                    self.conn.execute(f"ALTER TABLE files ADD COLUMN {col_name} {col_def}")
                    added += 1
            if added:
                # v3 indexes
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_image_type ON files(image_type)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_art_style ON files(art_style)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_scene_type ON files(scene_type)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_relative_path ON files(relative_path)")
                self.conn.commit()
                logger.info(f"✅ v3 migration: added {added} columns + indexes")
        except Exception as e:
            logger.warning(f"v3 migration check failed (non-fatal): {e}")

    # ── FTS5 columns: 16 columns across 3 axes + v3 ─────────────────
    #
    # 2축 Descriptive (AI 비저닝 — Qwen/Ollama 출력):
    #   ai_caption      AI가 생성한 이미지 설명 문장
    #   ai_tags         AI가 독립 생성한 키워드 (caption과 다른 정보 포함)
    #   ai_style        AI가 독립 생성한 스타일 설명
    #   dominant_color   AI 색상 분석
    #   ocr_text        AI OCR (이미지 안 글자)
    #
    # 3축 Structural (PSD 파싱 + 파일 정보 + 사용자 입력):
    #   file_path       파일 경로 (폴더 포함)
    #   file_name       파일명
    #   layer_names     PSD 레이어명 (원본+KR+EN 통합, Python 빌드)
    #   text_content    PSD 텍스트 (원본+KR+EN 통합, Python 빌드)
    #   used_fonts      사용 폰트 (Python 빌드)
    #   user_note       사용자 메모
    #   user_tags       사용자 태그
    #
    # Merged (별도 컬럼 불필요):
    #   semantic_tags → layer_names에 병합
    #   folder_tags → file_path에 포함
    _FTS_COLUMNS = [
        # 3축 Structural
        'file_path',        # file identity + folder path
        'file_name',        # file name search
        # 2축 Descriptive
        'ai_caption',       # AI vision description
        'ai_tags',          # AI keywords (independently generated, NOT subset of caption)
        'ai_style',         # AI style (Ollama generates independently)
        # 3축 Structural (Python-built from metadata JSON)
        'layer_names',      # PSD layer names: original + KR + EN merged
        'text_content',     # PSD text layers: original + KR + EN merged
        # 2축 Descriptive
        'ocr_text',         # AI OCR (text visible in image)
        'dominant_color',   # image color analysis
        # 3축 Structural (Python-built)
        'used_fonts',       # PSD fonts
        # 3축 Structural (user input)
        'user_note',        # user memo
        'user_tags',        # user tags
        # v3 P0: 2-Stage Vision classification (filterable + searchable)
        'image_type',       # character, background, ui_element, item, ...
        'scene_type',       # alley, forest, dungeon, ... (backgrounds)
        'art_style',        # realistic, anime, pixel, ...
        'folder_tags',      # folder name tags
    ]

    def _ensure_fts(self):
        """Ensure FTS5 table exists with correct schema and is populated."""
        needs_rebuild = False

        try:
            # Check if table exists and has the right columns
            cursor = self.conn.execute("PRAGMA table_info(files_fts)")
            existing_cols = [row[1] for row in cursor.fetchall()]
            if not existing_cols or set(existing_cols) != set(self._FTS_COLUMNS):
                logger.info(f"FTS5 schema mismatch — rebuilding")
                needs_rebuild = True
            else:
                # Check if populated
                fts_count = self.conn.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
                files_count = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                if fts_count == 0 and files_count > 0:
                    logger.info(f"FTS5 empty but {files_count} files exist — backfilling")
                    needs_rebuild = True
        except sqlite3.OperationalError:
            needs_rebuild = True

        if not needs_rebuild:
            return

        self._rebuild_fts()

    def _rebuild_fts(self):
        """Drop and recreate FTS5 table, backfilling from files table."""
        logger.info("Rebuilding FTS5 table (16 columns, 3-axis + v3)...")

        # Drop old FTS + triggers, then create fresh
        self.conn.executescript("""
            DROP TRIGGER IF EXISTS files_fts_insert;
            DROP TRIGGER IF EXISTS files_fts_update;
            DROP TRIGGER IF EXISTS files_fts_delete;
            DROP TABLE IF EXISTS files_fts;

            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                file_path, file_name,
                ai_caption, ai_tags, ai_style,
                layer_names, text_content,
                ocr_text, dominant_color,
                used_fonts,
                user_note, user_tags,
                image_type, scene_type, art_style, folder_tags
            );

            -- Trigger: layer_names, text_content, used_fonts, folder_tags need Python.
            -- Set to '' here; patched immediately after INSERT in insert_file().
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
        """)

        # Backfill: Python merges all layer/text/font data per row
        cursor = self.conn.execute(
            "SELECT id, file_path, file_name, ai_caption, ai_tags, ai_style, "
            "metadata, ocr_text, dominant_color, user_note, user_tags, "
            "image_type, scene_type, art_style, folder_tags FROM files"
        )

        rows_inserted = 0
        for row in cursor.fetchall():
            file_id = row[0]
            ai_tags_raw = row[4] or ''
            metadata_str = row[6] or '{}'
            try:
                meta = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                meta = {}

            # ai_tags: stored as JSON array string, flatten to space-separated for FTS
            ai_tags_fts = ''
            if ai_tags_raw:
                try:
                    tags_list = json.loads(ai_tags_raw)
                    if isinstance(tags_list, list):
                        ai_tags_fts = ' '.join(str(t) for t in tags_list)
                    else:
                        ai_tags_fts = str(tags_list)
                except (json.JSONDecodeError, TypeError):
                    ai_tags_fts = str(ai_tags_raw)

            layer_names = self._build_fts_layer_names(meta)
            text_content = self._build_fts_text_content(meta)
            used_fonts = meta.get('used_fonts', [])
            if isinstance(used_fonts, list):
                used_fonts = ' '.join(used_fonts)
            else:
                used_fonts = str(used_fonts) if used_fonts else ''

            # v3: folder_tags JSON → space-separated for FTS
            folder_tags_raw = row[14] or ''
            folder_tags_fts = ''
            if folder_tags_raw:
                try:
                    ft_list = json.loads(folder_tags_raw)
                    if isinstance(ft_list, list):
                        folder_tags_fts = ' '.join(str(t) for t in ft_list)
                except (json.JSONDecodeError, TypeError):
                    folder_tags_fts = str(folder_tags_raw)

            self.conn.execute(
                "INSERT INTO files_fts(rowid, "
                "file_path, file_name, "
                "ai_caption, ai_tags, ai_style, "
                "layer_names, text_content, "
                "ocr_text, dominant_color, "
                "used_fonts, "
                "user_note, user_tags, "
                "image_type, scene_type, art_style, folder_tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (file_id, row[1], row[2],
                 row[3], ai_tags_fts, row[5],
                 layer_names, text_content,
                 row[7], row[8],
                 used_fonts,
                 row[9], row[10],
                 row[11] or '', row[12] or '', row[13] or '', folder_tags_fts)
            )
            rows_inserted += 1

        self.conn.commit()
        logger.info(f"✅ FTS5 rebuilt and backfilled ({rows_inserted} rows)")

    @staticmethod
    def _extract_tree_names(tree) -> str:
        """Extract names from a single layer_tree dict as space-separated text."""
        if not tree or not isinstance(tree, dict):
            return ''
        names = []

        def _walk(node):
            name = node.get('name', '')
            if name and name != 'Root':
                names.append(name)
            cleaned = node.get('cleaned_name', '')
            if cleaned and cleaned != name and cleaned != 'Root':
                names.append(cleaned)
            for child in node.get('children', []):
                _walk(child)

        _walk(tree)
        seen = set()
        unique = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return ' '.join(unique)

    @classmethod
    def _build_fts_layer_names(cls, meta: dict) -> str:
        """Merge layer-name-related text into one FTS column.

        Combines: layer_tree names (original) + semantic_tags
        Cross-language search is handled at query time by QueryDecomposer.
        """
        parts = []
        parts.append(cls._extract_tree_names(meta.get('layer_tree')))
        val = meta.get('semantic_tags')
        if val and isinstance(val, str):
            parts.append(val)
        return ' '.join(p for p in parts if p)

    @staticmethod
    def _build_fts_text_content(meta: dict) -> str:
        """Build FTS column from PSD text content (original only).

        Cross-language search is handled at query time by QueryDecomposer.
        """
        val = meta.get('text_content')
        if val and isinstance(val, list):
            return ' '.join(str(t) for t in val if t)
        return ''

    def get_file_modified_at(self, file_path: str) -> Optional[str]:
        """
        Get stored modified_at timestamp for a file.

        Args:
            file_path: Absolute file path

        Returns:
            ISO 8601 modified_at string, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT modified_at FROM files WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        return row[0] if row else None

    def insert_file(
        self,
        file_path: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray
    ) -> int:
        """
        Insert or update file metadata + CLIP vector.

        Args:
            file_path: Absolute file path (unique identifier)
            metadata: Full metadata dict from AssetMeta.model_dump()
            embedding: CLIP embedding vector (768 dimensions)

        Returns:
            Database ID of inserted/updated record
        """
        cursor = self.conn.cursor()

        try:
            # Extract nested data for JSON storage
            metadata_json = {
                "layer_tree": metadata.get("layer_tree"),
                "semantic_tags": metadata.get("semantic_tags"),
                "text_content": metadata.get("text_content"),
                "layer_count": metadata.get("layer_count"),
                "used_fonts": metadata.get("used_fonts"),
            }

            # Extract resolution tuple
            resolution = metadata.get("resolution", (None, None))
            width = resolution[0] if isinstance(resolution, (list, tuple)) else None
            height = resolution[1] if isinstance(resolution, (list, tuple)) else None

            # Convert ai_tags array to JSON string
            ai_tags = metadata.get("ai_tags", [])
            ai_tags_json = json.dumps(ai_tags) if ai_tags else None

            # Extract folder discovery metadata
            folder_path = metadata.get("folder_path")
            folder_depth = metadata.get("folder_depth", 0)
            folder_tags = metadata.get("folder_tags", [])
            folder_tags_json = json.dumps(folder_tags) if folder_tags else None

            # v3 P0: structured vision fields
            image_type = metadata.get("image_type")
            art_style_val = metadata.get("art_style")
            color_palette = metadata.get("color_palette")
            scene_type = metadata.get("scene_type")
            time_of_day = metadata.get("time_of_day")
            weather = metadata.get("weather")
            character_type = metadata.get("character_type")
            item_type = metadata.get("item_type")
            ui_type = metadata.get("ui_type")
            structured_meta = metadata.get("structured_meta")
            storage_root = metadata.get("storage_root")
            relative_path = metadata.get("relative_path")
            embedding_model = metadata.get("embedding_model", "google/siglip2-so400m-patch14-384")
            embedding_version = metadata.get("embedding_version", 1)

            # Insert/update file record
            cursor.execute("""
                INSERT INTO files (
                    file_path, file_name, file_size, format, width, height,
                    ai_caption, ai_tags, ocr_text, dominant_color, ai_style,
                    metadata, thumbnail_url,
                    created_at, modified_at, parsed_at,
                    folder_path, folder_depth, folder_tags,
                    image_type, art_style, color_palette,
                    scene_type, time_of_day, weather,
                    character_type, item_type, ui_type,
                    structured_meta,
                    storage_root, relative_path,
                    embedding_model, embedding_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                -- CRITICAL: user_note, user_tags, user_category, user_rating는
                -- 여기에 절대 추가하지 말 것. 재분석 시 사용자 입력이 덮어쓰기됨.
                -- 사용자 메타데이터는 update_user_metadata()로만 업데이트.
                ON CONFLICT(file_path) DO UPDATE SET
                    ai_caption = excluded.ai_caption,
                    ai_tags = excluded.ai_tags,
                    ocr_text = excluded.ocr_text,
                    dominant_color = excluded.dominant_color,
                    ai_style = excluded.ai_style,
                    metadata = excluded.metadata,
                    parsed_at = datetime('now'),
                    folder_path = excluded.folder_path,
                    folder_depth = excluded.folder_depth,
                    folder_tags = excluded.folder_tags,
                    image_type = excluded.image_type,
                    art_style = excluded.art_style,
                    color_palette = excluded.color_palette,
                    scene_type = excluded.scene_type,
                    time_of_day = excluded.time_of_day,
                    weather = excluded.weather,
                    character_type = excluded.character_type,
                    item_type = excluded.item_type,
                    ui_type = excluded.ui_type,
                    structured_meta = excluded.structured_meta,
                    storage_root = excluded.storage_root,
                    relative_path = excluded.relative_path,
                    embedding_model = excluded.embedding_model,
                    embedding_version = excluded.embedding_version
            """, (
                file_path,
                metadata.get("file_name"),
                metadata.get("file_size"),
                metadata.get("format"),
                width,
                height,
                metadata.get("ai_caption"),
                ai_tags_json,
                metadata.get("ocr_text"),
                metadata.get("dominant_color"),
                metadata.get("ai_style"),
                json.dumps(metadata_json),
                metadata.get("thumbnail_url"),
                metadata.get("created_at"),
                metadata.get("modified_at"),
                folder_path,
                folder_depth,
                folder_tags_json,
                image_type, art_style_val, color_palette,
                scene_type, time_of_day, weather,
                character_type, item_type, ui_type,
                structured_meta,
                storage_root, relative_path,
                embedding_model, embedding_version,
            ))

            file_id = cursor.lastrowid
            # ON CONFLICT UPDATE sets lastrowid=0; fetch actual id
            if not file_id:
                row = cursor.execute(
                    "SELECT id FROM files WHERE file_path = ?", (file_path,)
                ).fetchone()
                file_id = row[0] if row else 0

            # Insert/update vector embedding (if sqlite-vec is available)
            if embedding is not None:
                try:
                    # Serialize numpy array to JSON list for sqlite-vec
                    embedding_list = embedding.astype(np.float32).tolist()

                    # Virtual tables don't support ON CONFLICT, so delete + insert
                    cursor.execute("DELETE FROM vec_files WHERE file_id = ?", (file_id,))
                    cursor.execute("""
                        INSERT INTO vec_files (file_id, embedding)
                        VALUES (?, ?)
                    """, (file_id, json.dumps(embedding_list)))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to insert embedding (sqlite-vec may not be loaded): {e}")

            # Post-trigger FTS fix: layer_names, text_content, used_fonts, folder_tags
            # need Python (SQL triggers set these to '' because JSON walking is not possible)
            layer_names = self._build_fts_layer_names(metadata_json)
            text_content = self._build_fts_text_content(metadata_json)
            used_fonts_raw = metadata_json.get('used_fonts', [])
            used_fonts = ' '.join(used_fonts_raw) if isinstance(used_fonts_raw, list) else ''
            # folder_tags: flatten JSON array to space-separated for FTS
            folder_tags_list = metadata.get("folder_tags", [])
            folder_tags_fts = ' '.join(str(t) for t in folder_tags_list) if isinstance(folder_tags_list, list) else ''
            if layer_names or text_content or used_fonts or folder_tags_fts:
                try:
                    cursor.execute(
                        "UPDATE files_fts SET layer_names = ?, text_content = ?, "
                        "used_fonts = ?, folder_tags = ? WHERE rowid = ?",
                        (layer_names, text_content, used_fonts, folder_tags_fts, file_id)
                    )
                except Exception as e:
                    logger.warning(f"⚠️ FTS post-trigger update failed: {e}")

            self.conn.commit()

            logger.debug(f"✅ Indexed file to SQLite: {file_path} (ID: {file_id})")
            return file_id

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Failed to insert file {file_path}: {e}")
            raise

    def insert_layer(
        self,
        file_id: int,
        layer_path: str,
        layer_metadata: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> int:
        """
        Insert or update layer metadata + embedding.

        Args:
            file_id: Parent file database ID
            layer_path: Layer path (e.g., "Root/Group 1/Layer 2")
            layer_metadata: Layer properties dict
            embedding: Optional CLIP embedding for this layer

        Returns:
            Database ID of inserted/updated layer
        """
        cursor = self.conn.cursor()

        try:
            ai_tags = layer_metadata.get("ai_tags", [])
            ai_tags_json = json.dumps(ai_tags) if ai_tags else None

            cursor.execute("""
                INSERT INTO layers (
                    file_id, layer_path, layer_name, layer_type,
                    metadata, ai_caption, ai_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id, layer_path) DO UPDATE SET
                    layer_name = excluded.layer_name,
                    layer_type = excluded.layer_type,
                    metadata = excluded.metadata,
                    ai_caption = excluded.ai_caption,
                    ai_tags = excluded.ai_tags
            """, (
                file_id,
                layer_path,
                layer_metadata.get("name"),
                layer_metadata.get("kind"),
                json.dumps(layer_metadata),
                layer_metadata.get("ai_caption"),
                ai_tags_json,
            ))

            layer_id = cursor.lastrowid

            # Insert vector if provided (if sqlite-vec is available)
            if embedding is not None:
                try:
                    embedding_list = embedding.astype(np.float32).tolist()
                    # Virtual tables don't support ON CONFLICT, so delete + insert
                    cursor.execute("DELETE FROM vec_layers WHERE layer_id = ?", (layer_id,))
                    cursor.execute("""
                        INSERT INTO vec_layers (layer_id, embedding)
                        VALUES (?, ?)
                    """, (layer_id, json.dumps(embedding_list)))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to insert layer embedding: {e}")

            self.conn.commit()

            logger.debug(f"✅ Indexed layer: {layer_path} (ID: {layer_id})")
            return layer_id

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Failed to insert layer {layer_path}: {e}")
            raise

    def get_file_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve file metadata by path.

        Args:
            file_path: Absolute file path

        Returns:
            File record as dict, or None if not found
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                f.id, f.file_path, f.file_name, f.file_size, f.format,
                f.width, f.height, f.ai_caption, f.ai_tags, f.ocr_text,
                f.dominant_color, f.ai_style, f.metadata, f.thumbnail_url,
                f.folder_path, f.folder_depth, f.folder_tags,
                f.created_at, f.modified_at, f.parsed_at
            FROM files f
            WHERE f.file_path = ?
        """, (file_path,))

        row = cursor.fetchone()
        if row:
            result = dict(row)
            # Parse JSON fields
            if result['ai_tags']:
                try:
                    result['ai_tags'] = json.loads(result['ai_tags'])
                except:
                    result['ai_tags'] = []
            if result['metadata']:
                try:
                    result['metadata'] = json.loads(result['metadata'])
                except:
                    result['metadata'] = {}
            if result.get('folder_tags'):
                try:
                    result['folder_tags'] = json.loads(result['folder_tags'])
                except:
                    result['folder_tags'] = []
            return result
        return None

    def count_files(self) -> int:
        """Count total files in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM files")
        return cursor.fetchone()[0]

    def count_layers(self) -> int:
        """Count total layers in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM layers")
        return cursor.fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()
        stats = {}

        # File count
        cursor.execute("SELECT COUNT(*) FROM files")
        stats['total_files'] = cursor.fetchone()[0]

        # Layer count
        cursor.execute("SELECT COUNT(*) FROM layers")
        stats['total_layers'] = cursor.fetchone()[0]

        # Files with AI captions
        cursor.execute("SELECT COUNT(*) FROM files WHERE ai_caption IS NOT NULL")
        stats['files_with_ai_caption'] = cursor.fetchone()[0]

        # Average layers per file
        cursor.execute("""
            SELECT AVG(CAST(json_extract(metadata, '$.layer_count') AS INTEGER))
            FROM files
            WHERE json_extract(metadata, '$.layer_count') IS NOT NULL
        """)
        result = cursor.fetchone()[0]
        stats['avg_layers_per_file'] = int(result) if result else 0

        # Format distribution
        cursor.execute("""
            SELECT format, COUNT(*) as count
            FROM files
            GROUP BY format
            ORDER BY count DESC
        """)
        stats['format_distribution'] = dict(cursor.fetchall())

        return stats

    def update_user_metadata(
        self,
        file_path: str,
        user_note: Optional[str] = None,
        user_tags: Optional[List[str]] = None,
        user_category: Optional[str] = None,
        user_rating: Optional[int] = None
    ) -> bool:
        """
        Update user metadata for a file.

        Args:
            file_path: Absolute path to the file
            user_note: User's personal notes (optional)
            user_tags: List of custom user tags (optional)
            user_category: User-defined category/folder (optional)
            user_rating: Rating from 0-5 stars (optional)

        Returns:
            True if update succeeded, False otherwise
        """
        cursor = self.conn.cursor()

        try:
            # Build UPDATE query dynamically (only update provided fields)
            updates = {}
            if user_note is not None:
                updates['user_note'] = user_note
            if user_tags is not None:
                updates['user_tags'] = json.dumps(user_tags)
            if user_category is not None:
                updates['user_category'] = user_category
            if user_rating is not None:
                if not 0 <= user_rating <= 5:
                    logger.error(f"Invalid rating: {user_rating} (must be 0-5)")
                    return False
                updates['user_rating'] = user_rating

            if not updates:
                logger.warning("No user metadata updates provided")
                return False

            # Build SQL
            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [file_path]

            cursor.execute(
                f"UPDATE files SET {set_clause} WHERE file_path = ?",
                values
            )
            self.conn.commit()

            if cursor.rowcount == 0:
                logger.warning(f"File not found in database: {file_path}")
                return False

            logger.debug(f"✅ Updated user metadata for: {file_path}")
            return True

        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Failed to update user metadata for {file_path}: {e}")
            return False

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("SQLite connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
