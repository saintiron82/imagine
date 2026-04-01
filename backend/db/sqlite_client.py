"""
SQLite client with sqlite-vec support.

This module replaces pg_client.py with SQLite-based storage,
maintaining API compatibility for minimal code changes.
"""

import logging
import sqlite3
import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

# Max retries for DB-locked write operations (each wait = busy_timeout)
_LOCKED_MAX_RETRIES = 3


def _retry_on_locked(func):
    """Decorator: retry a method up to _LOCKED_MAX_RETRIES on 'database is locked'."""
    import functools
    import time as _time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_err = None
        for attempt in range(_LOCKED_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" not in str(e):
                    raise
                last_err = e
                if attempt < _LOCKED_MAX_RETRIES - 1:
                    wait = 0.5 * (attempt + 1)
                    logger.warning(
                        f"{func.__name__}: DB locked, retry {attempt + 1}/{_LOCKED_MAX_RETRIES} "
                        f"(wait {wait:.1f}s)"
                    )
                    _time.sleep(wait)
        raise last_err
    return wrapper


class SQLiteDB:
    """SQLite database client with sqlite-vec support."""
    CURRENT_DATA_BUILD_LEVEL = 2
    CURRENT_FTS_INDEX_VERSION = 2

    _META_KEY_DATA_BUILD_LEVEL = "data_build_level"
    _META_KEY_FTS_INDEX_VERSION = "fts_index_version"
    _META_KEY_LAST_REBUILD_AT = "last_rebuild_at"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite connection with thread-local connection pool.

        Each thread gets its own sqlite3.Connection via threading.local().
        Migrations run once on a dedicated setup connection.

        Args:
            db_path: Path to SQLite database file.
                    Default: ./imageparser.db
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "imageparser.db")

        self.db_path = db_path
        self._local = threading.local()
        self._vec_extension_loaded = False
        self._setup_conn = None
        self._connect_setup()

    def _load_vec_extension(self, conn: sqlite3.Connection):
        """Load sqlite-vec extension into a connection."""
        try:
            conn.enable_load_extension(True)
            conn.load_extension("vec0")
            conn.enable_load_extension(False)
            self._vec_extension_loaded = True
        except:
            try:
                import sqlite_vec
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self._vec_extension_loaded = True
            except Exception as e:
                self._vec_extension_loaded = False
                logger.warning(f"sqlite-vec not loaded: {e}")

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with standard settings."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")
        conn.execute("PRAGMA busy_timeout = 10000")
        self._load_vec_extension(conn)
        return conn

    def _connect_setup(self):
        """Create setup connection and run migrations (once at init)."""
        from backend.db.sqlite_migrations import run_migrations

        try:
            self._setup_conn = self._create_connection()
            # Store setup conn in thread-local so migrations use it via self.conn
            self._local.conn = self._setup_conn
            if self._vec_extension_loaded:
                logger.info("sqlite-vec loaded")

            # Auto-migrate existing DB on connect (only if files table exists)
            if self._table_exists('files'):
                run_migrations(self, existing_db=True)
            else:
                logger.info("Empty database detected — auto-initializing schema")
                self.init_schema()
                run_migrations(self, existing_db=False)

            logger.info(f"Connected to SQLite database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    @property
    def conn(self) -> sqlite3.Connection:
        """Return current thread's connection (create if needed)."""
        c = getattr(self._local, 'conn', None)
        if c is None:
            c = self._create_connection()
            self._local.conn = c
        return c

    def _get_default_embedding_model(self) -> str:
        """Get the VV model name from active tier config."""
        try:
            from backend.utils.tier_config import get_active_tier
            _, tier_config = get_active_tier()
            return tier_config.get("visual", {}).get("model", "unknown")
        except Exception:
            return "unknown"

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone()[0] > 0

    def _ensure_system_meta(self):
        """Create system metadata table used for build/version tracking."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()


    def _get_system_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch a value from system_meta."""
        try:
            row = self.conn.execute(
                "SELECT value FROM system_meta WHERE key = ?",
                (key,)
            ).fetchone()
            return row[0] if row else default
        except Exception:
            return default

    def _set_system_meta(self, key: str, value: Any, commit: bool = True):
        """Upsert a key/value into system_meta."""
        self.conn.execute("""
            INSERT INTO system_meta (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = datetime('now')
        """, (key, str(value)))
        if commit:
            self.conn.commit()

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

            # Replace dimension placeholders with active tier config
            try:
                from backend.utils.tier_config import get_active_tier
                tier_name, tier_config = get_active_tier()
                visual_dim = tier_config.get("visual", {}).get("dimensions", 768)
                text_dim = tier_config.get("text_embed", {}).get("dimensions", 1024)
                visual_model = tier_config.get("visual", {}).get("model", "unknown")
            except Exception:
                visual_dim = 768
                text_dim = 1024
                visual_model = "unknown"
                tier_name = "standard"

            schema_sql = schema_sql.replace("{VISUAL_DIM}", str(visual_dim))
            schema_sql = schema_sql.replace("{TEXT_DIM}", str(text_dim))
            schema_sql = schema_sql.replace("{VISUAL_MODEL}", visual_model)
            schema_sql = schema_sql.replace("{DEFAULT_TIER}", tier_name)

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


    # ── FTS5 columns: 2-column BM25-weighted architecture ──
    #
    # meta_strong (BM25 3.0): Direct identification facts
    #   file_name, layer_names, used_fonts, user_tags, ocr_text
    #
    # meta_weak (BM25 1.5): Contextual information
    #   file_path, text_content, user_note, folder_tags,
    #   image_type, scene_type, art_style
    #
    # NOTE: caption column will be added when AI caption feature is implemented
    _FTS_COLUMNS = ['meta_strong', 'meta_weak']

    def _ensure_fts(self):
        """Ensure FTS5 table exists with correct schema and is populated."""
        needs_rebuild = False
        version_mismatch = False

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

                # Build-level check: index policy version drift
                db_fts_ver_raw = self._get_system_meta(self._META_KEY_FTS_INDEX_VERSION, "0")
                try:
                    db_fts_ver = int(db_fts_ver_raw or 0)
                except Exception:
                    db_fts_ver = 0
                if files_count > 0 and db_fts_ver < self.CURRENT_FTS_INDEX_VERSION:
                    version_mismatch = True
                    logger.warning(
                        f"FTS index version outdated (db={db_fts_ver}, expected={self.CURRENT_FTS_INDEX_VERSION})"
                    )
        except sqlite3.OperationalError:
            needs_rebuild = True

        if version_mismatch and not needs_rebuild:
            # Optional auto rebuild for version mismatch (defaults to True)
            auto_rebuild = True
            try:
                from backend.utils.config import get_config
                auto_rebuild = bool(get_config().get("search.fts.auto_rebuild_on_version_mismatch", True))
            except Exception:
                logger.warning("Config unavailable; using default auto rebuild for FTS mismatch")
            if auto_rebuild:
                logger.info("Auto rebuild enabled for FTS version mismatch")
                needs_rebuild = True

        if not needs_rebuild:
            return

        self._rebuild_fts()

    def _rebuild_fts(self):
        """Drop and recreate FTS5 table, backfilling from files table."""
        logger.info("Rebuilding FTS5 table (Triaxis: 2 columns, metadata-only)...")

        # Drop old FTS + triggers, then create fresh
        self.conn.executescript("""
            DROP TRIGGER IF EXISTS files_fts_insert;
            DROP TRIGGER IF EXISTS files_fts_update;
            DROP TRIGGER IF EXISTS files_fts_delete;
            DROP TABLE IF EXISTS files_fts;

            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                meta_strong,
                meta_weak
            );

            -- Triggers: meta_strong, meta_weak need Python (complex JSON walking)
            -- Set to '' here; patched immediately after INSERT in insert_file()
            CREATE TRIGGER IF NOT EXISTS files_fts_insert AFTER INSERT ON files BEGIN
                INSERT INTO files_fts(rowid, meta_strong, meta_weak)
                VALUES (new.id, '', '');
            END;

            -- UPDATE trigger removed: FTS is managed by Python (_refresh_fts_row)
            -- Previous trigger reset FTS to '' on every UPDATE, losing data
            -- when UPDATE didn't call _refresh_fts_row afterwards.

            CREATE TRIGGER IF NOT EXISTS files_fts_delete AFTER DELETE ON files BEGIN
                DELETE FROM files_fts WHERE rowid = old.id;
            END;
        """)

        # Backfill: Triaxis 2-column FTS (metadata-only)
        cursor = self.conn.execute(
            "SELECT id, file_path, file_name, mc_caption, ai_tags, "
            "metadata, ocr_text, user_note, user_tags, "
            "folder_path, relative_path, "
            "image_type, scene_type, art_style, folder_tags FROM files"
        )

        rows_inserted = 0
        for row in cursor.fetchall():
            file_id = row[0]
            metadata_str = row[5] or '{}'
            try:
                meta = json.loads(metadata_str)
            except (json.JSONDecodeError, TypeError):
                meta = {}

            # Build 2-column FTS (metadata-only, no caption)
            meta_strong = self._build_fts_meta_strong(row, meta)
            meta_weak = self._build_fts_meta_weak(row, meta)

            self.conn.execute(
                "INSERT INTO files_fts(rowid, meta_strong, meta_weak) VALUES (?, ?, ?)",
                (file_id, meta_strong, meta_weak)
            )
            rows_inserted += 1

        self.conn.commit()
        self._set_system_meta(self._META_KEY_FTS_INDEX_VERSION, self.CURRENT_FTS_INDEX_VERSION)
        self._set_system_meta(self._META_KEY_DATA_BUILD_LEVEL, self.CURRENT_DATA_BUILD_LEVEL)
        self._set_system_meta(self._META_KEY_LAST_REBUILD_AT, "fts")
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

    @staticmethod
    def _row_value(row, key: str, idx: int = -1):
        """Read sqlite row by key first (sqlite3.Row), fallback to tuple index."""
        try:
            if hasattr(row, "keys") and key in row.keys():
                return row[key]
        except Exception:
            pass
        if idx >= 0:
            try:
                return row[idx]
            except Exception:
                return None
        return None

    @staticmethod
    def _build_fts_path_terms(path_text: str) -> str:
        """
        Build searchable path terms from a path-like string.

        Includes:
        - Raw normalized path
        - Path segments
        - Segment-split tokens (snake/kebab/mixed)
        """
        if not path_text:
            return ""

        normalized = str(path_text).replace('\\', '/').strip().lower()
        if not normalized:
            return ""

        tokens = []
        segments = [seg for seg in normalized.split('/') if seg]
        for seg in segments:
            # Keep simple folder-like segments as whole words only.
            if re.fullmatch(r"[0-9a-zA-Z가-힣]{2,}", seg) and SQLiteDB._is_meaningful_path_token(seg):
                tokens.append(seg)
            for part in re.split(r"[^0-9a-zA-Z가-힣]+", seg):
                part = part.strip().lower()
                if SQLiteDB._is_meaningful_path_token(part):
                    tokens.append(part)

        seen = set()
        uniq = []
        for tok in tokens:
            if tok and tok not in seen:
                seen.add(tok)
                uniq.append(tok)
        return ' '.join(uniq)

    @staticmethod
    def _is_meaningful_path_token(token: str) -> bool:
        """
        Path token quality gate.

        Rejects low-information tokens such as pure numbers ("1", "2024")
        and version-only fragments ("v2", "v10").
        """
        if not token:
            return False
        t = str(token).strip().lower()
        if len(t) < 2:
            return False
        if t.isdigit():
            return False
        if re.fullmatch(r'v\d+', t):
            return False
        if t in {"psd", "png", "jpg", "jpeg"}:
            return False
        if t in {"assets", "asset", "images", "image", "img", "files", "file", "data", "resource", "resources", "output", "outputs", "tmp", "temp", "test"}:
            return False
        return True

    @classmethod
    def _build_fts_meta_strong(cls, row, meta: dict) -> str:
        """Build meta_strong: file_name, layer_names, used_fonts, user_tags, ocr_text.

        v3.1: BM25 weight 3.0 (highest priority for direct identification)
        """
        parts = []

        # file_name
        file_name = cls._row_value(row, "file_name", 2)
        parts.append(str(file_name) if file_name else '')

        # layer_names (from metadata layer_tree)
        layer_names = cls._build_fts_layer_names(meta)
        if layer_names:
            parts.append(layer_names)

        # used_fonts
        fonts = meta.get('used_fonts', [])
        if isinstance(fonts, list):
            parts.append(' '.join(fonts))

        # user_tags
        user_tags_raw = cls._row_value(row, "user_tags", 8) or ''
        if user_tags_raw:
            try:
                tags = json.loads(user_tags_raw)
                if isinstance(tags, list):
                    parts.append(' '.join(str(t) for t in tags))
            except:
                pass

        # ocr_text
        ocr_text = cls._row_value(row, "ocr_text", 6)
        parts.append(str(ocr_text) if ocr_text else '')

        return ' '.join(str(p) for p in parts if p)

    @staticmethod
    def _build_fts_meta_weak(row, meta: dict) -> str:
        """Build meta_weak: path terms, text_content, user_note, folder_tags, image_type, scene_type, art_style.

        v3.1: BM25 weight 1.5 (contextual information)
        """
        parts = []

        # path terms
        file_path = SQLiteDB._row_value(row, "file_path", 1)
        folder_path = SQLiteDB._row_value(row, "folder_path", 9)
        relative_path = SQLiteDB._row_value(row, "relative_path", 10)
        parts.append(SQLiteDB._build_fts_path_terms(file_path))
        parts.append(SQLiteDB._build_fts_path_terms(folder_path))
        parts.append(SQLiteDB._build_fts_path_terms(relative_path))

        # text_content (from metadata)
        text_content = meta.get('text_content', [])
        if isinstance(text_content, list):
            parts.append(' '.join(str(t) for t in text_content))

        # user_note
        user_note = SQLiteDB._row_value(row, "user_note", 7)
        parts.append(str(user_note) if user_note else '')

        # folder_tags
        folder_tags_raw = SQLiteDB._row_value(row, "folder_tags", 14) or ''
        if folder_tags_raw:
            try:
                ft = json.loads(folder_tags_raw)
                if isinstance(ft, list):
                    parts.append(' '.join(str(t) for t in ft))
            except:
                pass

        # image_type, scene_type, art_style
        image_type = SQLiteDB._row_value(row, "image_type", 11)
        scene_type = SQLiteDB._row_value(row, "scene_type", 12)
        art_style = SQLiteDB._row_value(row, "art_style", 13)
        parts.append(str(image_type) if image_type else '')
        parts.append(str(scene_type) if scene_type else '')
        parts.append(str(art_style) if art_style else '')

        return ' '.join(str(p) for p in parts if p)

    # Triaxis: _build_fts_caption() removed - AI content now handled by MV

    # ── Phase-specific storage methods (v3.3) ─────────────────────────

    def _refresh_fts_row(self, cursor, file_id: int):
        """Refresh FTS entry with actual data after INSERT/UPDATE trigger."""
        try:
            file_data = cursor.execute(
                "SELECT id, file_path, file_name, mc_caption, ai_tags, "
                "metadata, ocr_text, user_note, user_tags, "
                "folder_path, relative_path, "
                "image_type, scene_type, art_style, folder_tags "
                "FROM files WHERE id = ?",
                (file_id,)
            ).fetchone()

            if file_data:
                metadata_str = file_data[5] or '{}'
                try:
                    meta = json.loads(metadata_str)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

                meta_strong = self._build_fts_meta_strong(file_data, meta)
                meta_weak = self._build_fts_meta_weak(file_data, meta)

                cursor.execute(
                    "UPDATE files_fts SET meta_strong = ?, meta_weak = ? WHERE rowid = ?",
                    (meta_strong, meta_weak, file_id)
                )
        except Exception as e:
            logger.warning(f"⚠️ FTS refresh failed for file_id={file_id}: {e}")

    @_retry_on_locked
    def upsert_metadata(self, file_path: str, metadata: Dict[str, Any], commit: bool = True, preview_only: bool = False) -> int:
        """
        Phase 1 storage: INSERT basic metadata, preserve existing AI fields on conflict.

        Returns database file ID.
        """
        file_path = unicodedata.normalize('NFC', file_path)
        cursor = self.conn.cursor()

        try:
            metadata_json = {
                "layer_tree": metadata.get("layer_tree"),
                "semantic_tags": metadata.get("semantic_tags"),
                "text_content": metadata.get("text_content"),
                "layer_count": metadata.get("layer_count"),
                "used_fonts": metadata.get("used_fonts"),
            }

            resolution = metadata.get("resolution", (None, None))
            width = resolution[0] if isinstance(resolution, (list, tuple)) else None
            height = resolution[1] if isinstance(resolution, (list, tuple)) else None

            folder_path = metadata.get("folder_path")
            folder_depth = metadata.get("folder_depth", 0)
            folder_tags = metadata.get("folder_tags", [])
            folder_tags_json = json.dumps(folder_tags) if folder_tags else None

            storage_root = metadata.get("storage_root")
            relative_path = metadata.get("relative_path")
            embedding_model = metadata.get("embedding_model", self._get_default_embedding_model())
            embedding_version = metadata.get("embedding_version", 1)

            mode_tier = metadata.get("mode_tier")
            caption_model = metadata.get("caption_model")
            text_embed_model = metadata.get("text_embed_model")
            runtime_version = metadata.get("runtime_version")
            preprocess_params_json = json.dumps(metadata.get("preprocess_params", {})) if metadata.get("preprocess_params") else None
            content_hash = metadata.get("content_hash")

            cursor.execute("""
                INSERT INTO files (
                    file_path, file_name, file_size, format, width, height,
                    metadata, thumbnail_url,
                    created_at, modified_at, parsed_at,
                    folder_path, folder_depth, folder_tags,
                    storage_root, relative_path,
                    embedding_model, embedding_version,
                    mode_tier, caption_model, text_embed_model,
                    runtime_version, preprocess_params,
                    content_hash, preview_only
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_name = excluded.file_name,
                    file_size = excluded.file_size,
                    format = excluded.format,
                    width = excluded.width,
                    height = excluded.height,
                    metadata = excluded.metadata,
                    thumbnail_url = excluded.thumbnail_url,
                    modified_at = excluded.modified_at,
                    parsed_at = datetime('now'),
                    folder_path = excluded.folder_path,
                    folder_depth = excluded.folder_depth,
                    folder_tags = excluded.folder_tags,
                    storage_root = excluded.storage_root,
                    relative_path = excluded.relative_path,
                    embedding_model = excluded.embedding_model,
                    embedding_version = excluded.embedding_version,
                    mode_tier = excluded.mode_tier,
                    caption_model = excluded.caption_model,
                    text_embed_model = excluded.text_embed_model,
                    runtime_version = excluded.runtime_version,
                    preprocess_params = excluded.preprocess_params,
                    content_hash = excluded.content_hash,
                    preview_only = MIN(files.preview_only, excluded.preview_only)
            """, (
                file_path,
                metadata.get("file_name"),
                metadata.get("file_size"),
                metadata.get("format"),
                width, height,
                json.dumps(metadata_json),
                metadata.get("thumbnail_url"),
                metadata.get("created_at"),
                metadata.get("modified_at"),
                folder_path, folder_depth, folder_tags_json,
                storage_root, relative_path,
                embedding_model, embedding_version,
                mode_tier, caption_model, text_embed_model,
                runtime_version, preprocess_params_json,
                content_hash,
                1 if preview_only else 0,
            ))

            # cursor.lastrowid is unreliable for INSERT ON CONFLICT DO UPDATE:
            # when the UPDATE path triggers, it may return the rowid from a
            # PREVIOUS INSERT instead of the current row's actual ID.
            # Always use explicit SELECT to get the correct file_id.
            row = cursor.execute(
                "SELECT id FROM files WHERE file_path = ?", (file_path,)
            ).fetchone()
            if row:
                file_id = row[0]
            else:
                raise RuntimeError(f"UPSERT succeeded but file not found: {file_path}")

            self._refresh_fts_row(cursor, file_id)
            if commit:
                self.conn.commit()

            logger.debug(f"✅ Phase 1 metadata stored: {file_path} (ID: {file_id})")
            return file_id

        except Exception as e:
            if commit:
                self.conn.rollback()
            logger.error(f"❌ upsert_metadata failed for {file_path}: {e}")
            raise

    @_retry_on_locked
    def update_vision_fields(self, file_path: str, fields: Dict[str, Any], commit: bool = True) -> bool:
        """
        Phase 2 storage: UPDATE only VLM-generated fields.

        fields dict may contain: mc_caption, ai_tags, ocr_text, dominant_color,
        ai_style, image_type, art_style, color_palette, scene_type, time_of_day,
        weather, character_type, item_type, ui_type, structured_meta,
        perceptual_hash, dup_group_id, caption_model
        """
        file_path = unicodedata.normalize('NFC', file_path)
        cursor = self.conn.cursor()

        try:
            # Build dynamic UPDATE
            allowed_cols = {
                'mc_caption', 'ai_tags', 'ocr_text', 'dominant_color', 'ai_style',
                'image_type', 'art_style', 'color_palette', 'scene_type',
                'time_of_day', 'weather', 'character_type', 'item_type', 'ui_type',
                'structured_meta', 'perceptual_hash', 'dup_group_id', 'caption_model',
            }

            updates = {}
            for col in allowed_cols:
                if col in fields:
                    val = fields[col]
                    # SQLite cannot bind list/dict — serialize to JSON string
                    if isinstance(val, (list, dict)):
                        val = json.dumps(val)
                    updates[col] = val

            if not updates:
                return False

            set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [file_path]

            cursor.execute(
                f"UPDATE files SET {set_clause} WHERE file_path = ?",
                values
            )

            if cursor.rowcount == 0:
                logger.warning(f"update_vision_fields: file not found: {file_path}")
                return False

            # Refresh FTS with new MC data
            row = cursor.execute(
                "SELECT id FROM files WHERE file_path = ?", (file_path,)
            ).fetchone()
            if row:
                self._refresh_fts_row(cursor, row[0])

            if commit:
                self.conn.commit()
            logger.debug(f"✅ Phase 2 vision fields updated: {file_path}")
            return True

        except Exception as e:
            if commit:
                self.conn.rollback()
            logger.error(f"❌ update_vision_fields failed for {file_path}: {e}")
            raise

    @_retry_on_locked
    def upsert_vectors(self, file_id: int, vv_vec=None, mv_vec=None, structure_vec=None, commit: bool = True) -> bool:
        """
        Phase 3 storage: INSERT/REPLACE VV, MV, and Structure vectors.

        Args:
            file_id: Database file ID (from upsert_metadata)
            vv_vec: numpy array for VV (Visual Vector), or None to skip
            mv_vec: numpy array for MV (Meaning Vector), or None to skip
            structure_vec: numpy array for Structure Vector (DINOv2), or None to skip
        """
        cursor = self.conn.cursor()

        try:
            if vv_vec is not None:
                embedding_list = vv_vec.astype(np.float32).tolist()
                cursor.execute("DELETE FROM vec_files WHERE file_id = ?", (file_id,))
                cursor.execute(
                    "INSERT INTO vec_files (file_id, embedding) VALUES (?, ?)",
                    (file_id, json.dumps(embedding_list))
                )

            if mv_vec is not None:
                mv_list = mv_vec.astype(np.float32).tolist()
                cursor.execute("DELETE FROM vec_text WHERE file_id = ?", (file_id,))
                cursor.execute(
                    "INSERT INTO vec_text (file_id, embedding) VALUES (?, ?)",
                    (file_id, json.dumps(mv_list))
                )

            if structure_vec is not None:
                struct_list = structure_vec.astype(np.float32).tolist()
                cursor.execute("DELETE FROM vec_structure WHERE file_id = ?", (file_id,))
                cursor.execute(
                    "INSERT INTO vec_structure (file_id, embedding) VALUES (?, ?)",
                    (file_id, json.dumps(struct_list))
                )

            if commit:
                self.conn.commit()
            logger.debug(f"✅ Phase 3 vectors stored for file_id={file_id}")
            return True

        except Exception as e:
            if commit:
                self.conn.rollback()
            logger.error(f"❌ upsert_vectors failed for file_id={file_id}: {e}")
            raise

    def get_file_modified_at(self, file_path: str) -> Optional[str]:
        """
        Get stored modified_at timestamp for a file.

        Args:
            file_path: Absolute file path

        Returns:
            ISO 8601 modified_at string, or None if not found
        """
        file_path = unicodedata.normalize('NFC', file_path)
        cursor = self.conn.cursor()
        cursor.execute("SELECT modified_at FROM files WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_file_mode_tier(self, file_path: str) -> Optional[str]:
        """
        Get stored mode_tier for a file.

        Args:
            file_path: Absolute file path

        Returns:
            Tier name string (e.g. 'standard', 'pro', 'ultra'), or None if not found
        """
        file_path = unicodedata.normalize('NFC', file_path)
        cursor = self.conn.cursor()
        cursor.execute("SELECT mode_tier FROM files WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_file_phase_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get per-phase completion info for smart skip.

        Returns dict with:
            - caption_model, embedding_model, text_embed_model, mode_tier
            - has_mc: bool (mc_caption is non-empty)
            - has_vv: bool (vec_files entry exists)
            - has_mv: bool (vec_text entry exists)
            - modified_at: stored mtime
            - file_size, content_hash
        """
        file_path = unicodedata.normalize('NFC', file_path)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT f.id, f.caption_model, f.embedding_model, f.text_embed_model,
                   f.mode_tier, f.modified_at, f.file_size, f.content_hash,
                   f.mc_caption, f.ai_tags,
                   f.image_type, f.scene_type, f.art_style,
                   f.relative_path
            FROM files f
            WHERE f.file_path = ?
        """, (file_path,))
        row = cursor.fetchone()
        if not row:
            return None

        file_id = row[0]

        # Check vec_files existence
        cursor.execute("SELECT COUNT(*) FROM vec_files WHERE file_id = ?", (file_id,))
        has_vv = cursor.fetchone()[0] > 0

        # Check vec_text existence
        cursor.execute("SELECT COUNT(*) FROM vec_text WHERE file_id = ?", (file_id,))
        has_mv = cursor.fetchone()[0] > 0
        
        # Check vec_structure existence
        # Note: vec_structure uses file_id as PK in our schema (though named 'file_id' explicitly)
        cursor.execute("SELECT COUNT(*) FROM vec_structure WHERE file_id = ?", (file_id,))
        has_structure = cursor.fetchone()[0] > 0

        mc_caption = row[8] or ""
        return {
            "file_id": file_id,
            "caption_model": row[1],
            "embedding_model": row[2],
            "text_embed_model": row[3],
            "mode_tier": row[4],
            "modified_at": row[5],
            "file_size": row[6],
            "content_hash": row[7],
            "has_mc": len(mc_caption.strip()) > 0,
            "mc_caption": mc_caption,
            "ai_tags": row[9],
            "image_type": row[10],
            "scene_type": row[11],
            "art_style": row[12],
            "relative_path": row[13],
            "has_relative_path": bool((row[13] or "").strip()),
            "has_vv": has_vv,
            "has_mv": has_mv,
            "has_structure": has_structure,
        }

    def verify_data_integrity(self, file_id: int,
                              expect_mc: bool = False,
                              expect_vv: bool = False,
                              expect_mv: bool = False) -> dict:
        """Verify actual DB data existence for a file, independent of flags.

        Uses PK-based O(1) queries to check whether each data type truly
        exists in the database. Returns a dict describing what was found
        and what is missing vs. expectations.

        Args:
            file_id: The files.id to check
            expect_mc: Whether MC (mc_caption) should exist
            expect_vv: Whether VV (vec_files row) should exist
            expect_mv: Whether MV (vec_text row) should exist

        Returns:
            {
                "valid": bool,         # All expectations met
                "has_mc": bool,        # mc_caption IS NOT NULL AND != ''
                "has_vv": bool,        # vec_files row exists
                "has_mv": bool,        # vec_text row exists
                "missing": [str],      # List of missing items: "mc", "vv", "mv"
                "actual_phases": {     # Data-driven phase status
                    "parse": bool,
                    "vision": bool,
                    "embed": bool,
                },
            }
        """
        cursor = self.conn.cursor()

        # Check mc_caption
        cursor.execute(
            "SELECT 1 FROM files WHERE id = ? AND mc_caption IS NOT NULL AND mc_caption != ''",
            (file_id,)
        )
        has_mc = cursor.fetchone() is not None

        # Check vec_files
        cursor.execute("SELECT 1 FROM vec_files WHERE file_id = ?", (file_id,))
        has_vv = cursor.fetchone() is not None

        # Check vec_text
        cursor.execute("SELECT 1 FROM vec_text WHERE file_id = ?", (file_id,))
        has_mv = cursor.fetchone() is not None

        # Determine missing items
        missing = []
        if expect_mc and not has_mc:
            missing.append("mc")
        if expect_vv and not has_vv:
            missing.append("vv")
        if expect_mv and not has_mv:
            missing.append("mv")

        return {
            "valid": len(missing) == 0,
            "has_mc": has_mc,
            "has_vv": has_vv,
            "has_mv": has_mv,
            "missing": missing,
            "actual_phases": {
                "parse": True,  # If file_id exists, parse is done
                "vision": has_mc,
                "embed": has_vv and has_mv,
            },
        }

    def find_by_content_hash(self, content_hash: str) -> List[Dict[str, Any]]:
        """
        Find files by content_hash (may return multiple results for copies).

        Returns list of dicts with id, file_path, content_hash, has_mc, has_vv, has_mv.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT f.id, f.file_path, f.content_hash, f.mc_caption
            FROM files f
            WHERE f.content_hash = ?
        """, (content_hash,))

        results = []
        for row in cursor.fetchall():
            file_id = row[0]
            mc = row[3] or ""
            has_vv = cursor.execute(
                "SELECT COUNT(*) FROM vec_files WHERE file_id = ?", (file_id,)
            ).fetchone()[0] > 0
            has_mv = cursor.execute(
                "SELECT COUNT(*) FROM vec_text WHERE file_id = ?", (file_id,)
            ).fetchone()[0] > 0
            results.append({
                "id": file_id,
                "file_path": row[1],
                "content_hash": row[2],
                "has_mc": len(mc.strip()) > 0,
                "has_vv": has_vv,
                "has_mv": has_mv,
            })
        return results

    def relink_file(self, content_hash: str, new_file_path: str) -> bool:
        """
        Update file_path for a file matched by content_hash (DB migration/relink).

        Returns True if a row was updated.
        """
        new_file_path = unicodedata.normalize('NFC', new_file_path)
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE files SET file_path = ?, parsed_at = datetime('now')
            WHERE content_hash = ? AND file_path != ?
            LIMIT 1
        """, (new_file_path, content_hash, new_file_path))
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_file(self, file_id: int) -> bool:
        """
        Delete a file row by ID.

        Vec cascade triggers automatically clean up vec_files/vec_text.
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def reset_file_data(self) -> dict:
        """
        Delete all file data while preserving auth tables and thumbnails.

        Clears: files, layers, vec_files, vec_text, vec_structure, files_fts, job_queue
        Preserves: users, invite_codes, worker_tokens, worker_sessions, system_meta (reset values)
        """
        cursor = self.conn.cursor()
        try:
            # Count before delete (for reporting)
            file_count = cursor.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            vec_count = (
                cursor.execute("SELECT COUNT(*) FROM vec_files").fetchone()[0]
                + cursor.execute("SELECT COUNT(*) FROM vec_text").fetchone()[0]
            )
            job_count = 0
            if self._table_exists('job_queue'):
                job_count = cursor.execute("SELECT COUNT(*) FROM job_queue").fetchone()[0]

            # Delete order: FTS → vectors → layers → files → jobs
            cursor.execute("DELETE FROM files_fts")
            cursor.execute("DELETE FROM vec_files")
            cursor.execute("DELETE FROM vec_text")
            if self._table_exists('vec_structure'):
                cursor.execute("DELETE FROM vec_structure")
            cursor.execute("DELETE FROM layers")
            cursor.execute("DELETE FROM files")
            if self._table_exists('job_queue'):
                cursor.execute("DELETE FROM job_queue")
            if self._table_exists('job_completions'):
                cursor.execute("DELETE FROM job_completions")
            if self._table_exists('work_subtasks'):
                cursor.execute("DELETE FROM work_subtasks")
            if self._table_exists('work_requests'):
                cursor.execute("DELETE FROM work_requests")

            # Reset system meta
            self._set_system_meta(self._META_KEY_DATA_BUILD_LEVEL, "0", commit=False)
            self._set_system_meta(self._META_KEY_FTS_INDEX_VERSION, "0", commit=False)
            self._set_system_meta(self._META_KEY_LAST_REBUILD_AT, "", commit=False)

            self.conn.commit()
            logger.info(f"Database reset: {file_count} files, {vec_count} vectors, {job_count} jobs cleared")
            return {"success": True, "files": file_count, "vectors": vec_count, "jobs": job_count}
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Database reset failed: {e}")
            return {"success": False, "error": str(e)}

    def insert_file(
        self,
        file_path: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray,
        structure_embedding: Optional[np.ndarray] = None
    ) -> int:
        """
        Insert or update file metadata + VV (SigLIP2) + Structure (DINOv2).

        Args:
            file_path: Absolute file path (unique identifier)
            metadata: Full metadata dict from AssetMeta.model_dump()
            embedding: VV (Visual Vector) (dimension from active tier)
            structure_embedding: DINOv2 (Structure Vector) (768-dim)
        Returns:
            Database ID of inserted/updated record
        """
        file_path = unicodedata.normalize('NFC', file_path)
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
            embedding_model = metadata.get("embedding_model", self._get_default_embedding_model())
            embedding_version = metadata.get("embedding_version", 1)

            # v3.1: Extract perceptual_hash and dup_group_id
            perceptual_hash = metadata.get("perceptual_hash")
            dup_group_id = metadata.get("dup_group_id")

            # v3.1: Tier tracking metadata
            mode_tier = metadata.get("mode_tier")
            caption_model = metadata.get("caption_model")
            text_embed_model = metadata.get("text_embed_model")
            runtime_version = metadata.get("runtime_version")
            preprocess_params_json = json.dumps(metadata.get("preprocess_params", {})) if metadata.get("preprocess_params") else None

            # Insert/update file record
            cursor.execute("""
                INSERT INTO files (
                    file_path, file_name, file_size, format, width, height,
                    mc_caption, ai_tags, ocr_text, dominant_color, ai_style,
                    metadata, thumbnail_url,
                    created_at, modified_at, parsed_at,
                    folder_path, folder_depth, folder_tags,
                    image_type, art_style, color_palette,
                    scene_type, time_of_day, weather,
                    character_type, item_type, ui_type,
                    structured_meta,
                    storage_root, relative_path,
                    embedding_model, embedding_version,
                    perceptual_hash, dup_group_id,
                    mode_tier, caption_model, text_embed_model,
                    runtime_version, preprocess_params
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?)
                -- CRITICAL: user_note, user_tags, user_category, user_rating는
                -- 여기에 절대 추가하지 말 것. 재분석 시 사용자 입력이 덮어쓰기됨.
                -- 사용자 메타데이터는 update_user_metadata()로만 업데이트.
                ON CONFLICT(file_path) DO UPDATE SET
                    mc_caption = excluded.mc_caption,
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
                    embedding_version = excluded.embedding_version,
                    perceptual_hash = excluded.perceptual_hash,
                    dup_group_id = excluded.dup_group_id,
                    mode_tier = excluded.mode_tier,
                    caption_model = excluded.caption_model,
                    text_embed_model = excluded.text_embed_model,
                    runtime_version = excluded.runtime_version,
                    preprocess_params = excluded.preprocess_params
            """, (
                file_path,
                metadata.get("file_name"),
                metadata.get("file_size"),
                metadata.get("format"),
                width,
                height,
                metadata.get("mc_caption"),
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
                perceptual_hash, dup_group_id,
                mode_tier, caption_model, text_embed_model,
                runtime_version, preprocess_params_json,
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

            # Triaxis: Post-trigger FTS fix (2 columns: meta_strong, meta_weak)
            # SQL triggers set these to ''; Python updates with actual data
            # Insert/update Structure vector (DINOv2)
            if structure_embedding is not None:
                try:
                    struct_list = structure_embedding.astype(np.float32).tolist()
                    cursor.execute("DELETE FROM vec_structure WHERE file_id = ?", (file_id,))
                    cursor.execute(
                        "INSERT INTO vec_structure (file_id, embedding) VALUES (?, ?)",
                        (file_id, json.dumps(struct_list))
                    )
                except Exception as e:
                     logger.warning(f"⚠️ Failed to insert structure embedding: {e}")

            # Triaxis: Post-trigger FTS fix (2 columns: meta_strong, meta_weak)
            # SQL triggers set these to ''; Python updates with actual data
            try:
                # Get file data for FTS building
                file_data = cursor.execute(
                    "SELECT id, file_path, file_name, mc_caption, ai_tags, "
                    "metadata, ocr_text, user_note, user_tags, "
                    "folder_path, relative_path, "
                    "image_type, scene_type, art_style, folder_tags "
                    "FROM files WHERE id = ?",
                    (file_id,)
                ).fetchone()

                if file_data:
                    meta_strong = self._build_fts_meta_strong(file_data, metadata_json)
                    meta_weak = self._build_fts_meta_weak(file_data, metadata_json)

                    cursor.execute(
                        "UPDATE files_fts SET meta_strong = ?, meta_weak = ? WHERE rowid = ?",
                        (meta_strong, meta_weak, file_id)
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
            embedding: Optional VV for this layer

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
        file_path = unicodedata.normalize('NFC', file_path)
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                f.id, f.file_path, f.file_name, f.file_size, f.format,
                f.width, f.height, f.mc_caption, f.ai_tags, f.ocr_text,
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

    def get_build_status(self) -> Dict[str, Any]:
        """
        Get data-build compatibility status for user-visible rebuild guidance.
        """
        cursor = self.conn.cursor()

        total_files = cursor.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        db_fts_ver_raw = self._get_system_meta(self._META_KEY_FTS_INDEX_VERSION, "0")
        db_level_raw = self._get_system_meta(self._META_KEY_DATA_BUILD_LEVEL, "0")

        try:
            db_fts_ver = int(db_fts_ver_raw or 0)
        except Exception:
            db_fts_ver = 0
        try:
            db_level = int(db_level_raw or 0)
        except Exception:
            db_level = 0

        # Legacy quality gaps that require rebuild/reprocess
        # vec_structure (DINOv2) is experimental and excluded from standard checks.
        vector_extension_available = True
        missing_structure = 0

        missing_relative = cursor.execute("""
            SELECT COUNT(*) FROM files
            WHERE relative_path IS NULL OR TRIM(relative_path) = ''
        """).fetchone()[0]

        reasons = []
        if total_files > 0 and db_fts_ver < self.CURRENT_FTS_INDEX_VERSION:
            reasons.append(
                f"FTS index version is outdated (db={db_fts_ver}, expected={self.CURRENT_FTS_INDEX_VERSION})"
            )
        if missing_structure > 0:
            reasons.append(f"{missing_structure} files are missing Structure vectors (DINOv2)")
        if missing_relative > 0:
            reasons.append(f"{missing_relative} files are missing relative_path metadata")

        needs_rebuild = len(reasons) > 0

        # If explicit db-level metadata is absent, infer from quality gaps.
        if db_level <= 0:
            inferred_level = self.CURRENT_DATA_BUILD_LEVEL if not needs_rebuild else max(1, self.CURRENT_DATA_BUILD_LEVEL - 1)
        else:
            inferred_level = min(db_level, self.CURRENT_DATA_BUILD_LEVEL if not needs_rebuild else db_level)

        return {
            "needs_rebuild": needs_rebuild,
            "db_data_build_level": inferred_level,
            "current_data_build_level": self.CURRENT_DATA_BUILD_LEVEL,
            "db_fts_index_version": db_fts_ver,
            "current_fts_index_version": self.CURRENT_FTS_INDEX_VERSION,
            "missing_structure_count": missing_structure,
            "missing_relative_path_count": missing_relative,
            "vector_extension_available": vector_extension_available,
            "reasons": reasons,
        }

    def fix_missing_relative_paths(self) -> int:
        """Auto-fill missing relative_path from file_path and storage_root.

        For files where relative_path is NULL/empty but storage_root exists,
        compute relative_path = file_path - storage_root prefix.
        For files where storage_root is also empty, derive from file_path directory.

        Returns: number of rows fixed.
        """
        cursor = self.conn.cursor()
        rows = cursor.execute("""
            SELECT id, file_path, file_name, storage_root
            FROM files
            WHERE relative_path IS NULL OR TRIM(relative_path) = ''
        """).fetchall()

        if not rows:
            return 0

        fixed = 0
        for row in rows:
            file_id, file_path, file_name, storage_root = row
            if not file_path:
                continue

            # Compute relative_path from file_path
            import re as _re
            webdav_match = _re.match(r'(webdav://[^/]+/)', file_path)
            if webdav_match:
                # WebDAV: prefix is webdav://SOURCE_ID/
                storage_root = webdav_match.group(1)
                rel = file_path[len(storage_root):]
            elif storage_root and storage_root != file_path and file_path.startswith(storage_root):
                # Local with valid storage_root (not equal to file_path)
                rel = file_path[len(storage_root):].lstrip("/\\")
            else:
                # Derive from file_path directory structure
                parts = file_path.replace("\\", "/").rsplit("/", 1)
                storage_root = parts[0] + "/" if len(parts) > 1 else ""
                rel = parts[1] if len(parts) > 1 else file_path

            cursor.execute("""
                UPDATE files SET relative_path = ?, storage_root = COALESCE(NULLIF(TRIM(storage_root), ''), ?)
                WHERE id = ?
            """, (rel, storage_root, file_id))
            fixed += 1

        if fixed > 0:
            self.conn.commit()
            logger.info(f"Fixed {fixed} files with missing relative_path")

        return fixed

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

        # Files with MC captions (v3.1: renamed from ai_caption)
        cursor.execute("SELECT COUNT(*) FROM files WHERE mc_caption IS NOT NULL")
        stats['files_with_mc_caption'] = cursor.fetchone()[0]

        # Average layers per file
        cursor.execute("""
            SELECT AVG(CAST(json_extract(metadata, '$.layer_count') AS INTEGER))
            FROM files
            WHERE json_extract(metadata, '$.layer_count') IS NOT NULL
        """)
        result = cursor.fetchone()[0]
        stats['avg_layers_per_file'] = int(result) if result else 0

        # Fully archived (MC + VV + MV all done)
        # vec_structure is excluded — it's experimental and not part of
        # the standard Triaxis pipeline (VV + MV + FTS).
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM files f
                WHERE (mc_caption IS NOT NULL AND mc_caption != '')
                  AND EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
                  AND EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id)
            """)
            stats['fully_archived'] = cursor.fetchone()[0]
        except Exception:
            stats['fully_archived'] = 0

        # Preview-only vs searchable breakdown
        try:
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN preview_only = 0 OR preview_only IS NULL THEN 1 ELSE 0 END) as searchable,
                    SUM(CASE WHEN preview_only = 1 THEN 1 ELSE 0 END) as preview_only
                FROM files
            """)
            row = cursor.fetchone()
            stats['total'] = row[0]
            stats['searchable'] = row[1]
            stats['preview_only'] = row[2]
        except Exception:
            stats['total'] = stats['total_files']
            stats['searchable'] = stats['total_files']
            stats['preview_only'] = 0

        # Format distribution
        cursor.execute("""
            SELECT format, COUNT(*) as count
            FROM files
            GROUP BY format
            ORDER BY count DESC
        """)
        stats['format_distribution'] = dict(cursor.fetchall())
        stats['build_status'] = self.get_build_status()

        return stats

    def get_incomplete_stats(self) -> Dict[str, Any]:
        """Get incomplete file stats grouped by storage_root.

        Returns dict with total_files, total_incomplete, and per-folder breakdown.
        Only folders with incomplete files are included.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                f.storage_root,
                COUNT(*) as total,
                COUNT(CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 END) as mc,
                COUNT(CASE
                    WHEN EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
                    THEN 1
                END) as vv,
                COUNT(CASE WHEN EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) THEN 1 END) as mv
            FROM files f
            GROUP BY f.storage_root
        """)
        folders = []
        total_files = 0
        total_incomplete = 0
        for row in cursor.fetchall():
            sr, total, mc, vv, mv = row
            done = min(mc, vv, mv)
            incomplete = total - done
            total_files += total
            total_incomplete += incomplete
            if incomplete > 0:
                folders.append({
                    "storage_root": sr or "",
                    "total": total, "done": done, "incomplete": incomplete,
                    "mc": mc, "vv": vv, "mv": mv,
                })
        return {
            "total_files": total_files,
            "total_incomplete": total_incomplete,
            "folders": folders,
        }

    def get_folder_phase_stats(self, root_path: str) -> List[Dict[str, Any]]:
        """Get per-storage_root phase completion stats under root_path prefix.

        Uses file_path LIKE prefix match and EXISTS subqueries on vec0 tables.
        Returns one row per storage_root with MC/VV/MV counts.
        """
        root_path = unicodedata.normalize('NFC', root_path)
        cursor = self.conn.cursor()
        prefix = root_path.rstrip('/') + '/'
        # Global index-version drift affects all folders in this DB.
        db_fts_ver_raw = self._get_system_meta(self._META_KEY_FTS_INDEX_VERSION, "0")
        try:
            db_fts_ver = int(db_fts_ver_raw or 0)
        except Exception:
            db_fts_ver = 0
        fts_version_mismatch = db_fts_ver < self.CURRENT_FTS_INDEX_VERSION

        # Use COALESCE to derive folder from file_path when storage_root is NULL.
        # Without this, files with NULL storage_root are invisible to folder stats.
        effective_root_expr = "COALESCE(NULLIF(TRIM(f.storage_root), ''), REPLACE(f.file_path, '/' || f.file_name, ''))"

        rows = []
        vector_extension_available = True
        try:
            cursor.execute(f"""
                SELECT
                    {effective_root_expr} as effective_root,
                    COUNT(*) as total,
                    COUNT(CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 END) as mc,
                    COUNT(CASE
                        WHEN EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
                        THEN 1
                    END) as vv,
                    COUNT(CASE WHEN EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) THEN 1 END) as mv,
                    COUNT(CASE WHEN f.relative_path IS NULL OR TRIM(f.relative_path) = '' THEN 1 END) as missing_relative,
                    0 as missing_structure
                FROM files f
                WHERE f.file_path LIKE ? || '%'
                GROUP BY effective_root
            """, (prefix,))
            rows = cursor.fetchall()
        except Exception:
            # vec0 unavailable: fallback query without vec tables
            vector_extension_available = False
            cursor.execute(f"""
                SELECT
                    {effective_root_expr} as effective_root,
                    COUNT(*) as total,
                    COUNT(CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 END) as mc,
                    0 as vv,
                    0 as mv,
                    COUNT(CASE WHEN f.relative_path IS NULL OR TRIM(f.relative_path) = '' THEN 1 END) as missing_relative,
                    0 as missing_structure
                FROM files f
                WHERE f.file_path LIKE ? || '%'
                GROUP BY effective_root
            """, (prefix,))
            rows = cursor.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            total = row[1] or 0
            missing_relative = row[5] or 0
            missing_structure = row[6] or 0
            reasons = []

            if missing_relative > 0:
                reasons.append("missing_relative_path")
            if missing_structure > 0:
                reasons.append("missing_structure_vector")

            results.append({
                "storage_root": row[0] or "",
                "total": total,
                "mc": row[2] or 0,
                "vv": row[3] or 0,
                "mv": row[4] or 0,
                "missing_relative_path_count": missing_relative,
                "missing_structure_count": missing_structure,
                "fts_version_mismatch": bool(total > 0 and fts_version_mismatch),
                "vector_extension_available": vector_extension_available,
                "rebuild_needed": len(reasons) > 0,
                "rebuild_reasons": reasons,
            })

        return results

    def get_files_phase_status(self, file_paths: List[str]) -> Dict[str, Dict[str, bool]]:
        """Return per-file MC/VV/MV presence status.

        Args:
            file_paths: List of absolute file paths to check.

        Returns:
            { file_path: { "mc": bool, "vv": bool, "mv": bool } }
        """
        if not file_paths:
            return {}

        cursor = self.conn.cursor()
        result: Dict[str, Dict[str, bool]] = {}

        # Process in batches of 100 to avoid SQLite variable limit
        batch_size = 100
        use_vec = True

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))

            if use_vec:
                try:
                    rows = cursor.execute(f"""
                        SELECT
                            f.file_path,
                            CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 ELSE 0 END as has_mc,
                            CASE WHEN EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id) THEN 1 ELSE 0 END as has_vv,
                            CASE WHEN EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) THEN 1 ELSE 0 END as has_mv
                        FROM files f
                        WHERE f.file_path IN ({placeholders})
                    """, batch).fetchall()
                except Exception:
                    use_vec = False

            if not use_vec:
                # Fallback without vec tables
                rows = cursor.execute(f"""
                    SELECT
                        f.file_path,
                        CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 ELSE 0 END as has_mc,
                        0 as has_vv,
                        0 as has_mv
                    FROM files f
                    WHERE f.file_path IN ({placeholders})
                """, batch).fetchall()

            for row in rows:
                result[row[0]] = {
                    "mc": bool(row[1]),
                    "vv": bool(row[2]),
                    "mv": bool(row[3]),
                }

        # Files not found in DB → all False
        for fp in file_paths:
            if fp not in result:
                result[fp] = {"mc": False, "vv": False, "mv": False}

        return result

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
        file_path = unicodedata.normalize('NFC', file_path)
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

    def get_db_tier(self) -> Optional[str]:
        """
        Get the tier of existing data in the database.

        Returns:
            Tier name ('standard', 'pro', 'ultra') or None if DB is empty
        """
        try:
            cursor = self.conn.execute(
                "SELECT mode_tier FROM files WHERE mode_tier IS NOT NULL LIMIT 1"
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to get DB tier: {e}")
            return None

    def get_db_embedding_dimension(self) -> Optional[int]:
        """
        Get the embedding dimension used in the database.

        Returns:
            Dimension size or None if cannot determine
        """
        try:
            # Query vec_files table info
            cursor = self.conn.execute("SELECT sql FROM sqlite_master WHERE name='vec_files'")
            row = cursor.fetchone()
            if row:
                # Parse CREATE VIRTUAL TABLE statement
                # Example: "CREATE VIRTUAL TABLE vec_files USING vec0(..., embedding FLOAT[1152])"
                import re
                match = re.search(r'embedding FLOAT\[(\d+)\]', row[0])
                if match:
                    return int(match.group(1))
            return None
        except Exception as e:
            logger.error(f"Failed to get DB embedding dimension: {e}")
            return None

    def check_tier_compatibility(self, current_tier: str, current_dimension: int) -> Dict[str, Any]:
        """
        Check if current tier is compatible with existing DB data.
        Uses tier compatibility matrix for intelligent decision-making.

        Args:
            current_tier: Current tier being used ('standard', 'pro', 'ultra')
            current_dimension: Expected embedding dimension for current tier

        Returns:
            Dict with detailed compatibility information:
                - compatible: bool
                - action: str (TierAction: 'none', 'reprocess_optional', 'reprocess_required', 'block')
                - reason: str (CompatibilityReason)
                - message: str (short description)
                - user_prompt: str or None (detailed message for user)
                - auto_allow: bool (can proceed automatically)
                - db_tier: str or None
                - current_tier: str
                - db_dimension: int or None
                - current_dimension: int
        """
        from backend.utils.tier_compatibility import check_tier_transition

        db_tier = self.get_db_tier()
        db_dimension = self.get_db_embedding_dimension()

        return check_tier_transition(
            db_tier=db_tier,
            current_tier=current_tier,
            db_dimension=db_dimension,
            current_dimension=current_dimension
        )

    def migrate_tier(self, new_tier: str, new_dimension: int) -> bool:
        """
        Migrate database to a new tier by recreating vec_files table.
        WARNING: This deletes all existing embeddings!

        Args:
            new_tier: Target tier name
            new_dimension: Target embedding dimension

        Returns:
            True if successful
        """
        try:
            logger.warning(f"[TIER MIGRATION] Migrating to {new_tier} (dimension: {new_dimension})")
            logger.warning("[TIER MIGRATION] This will delete all existing embeddings!")

            # Drop vec_files table
            self.conn.execute("DROP TABLE IF EXISTS vec_files")
            logger.info("[TIER MIGRATION] Dropped vec_files table")

            # Recreate with new dimension
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE vec_files USING vec0(
                    file_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{new_dimension}]
                )
            """)
            logger.info(f"[TIER MIGRATION] Created vec_files with dimension={new_dimension}")

            # Clear mode_tier from all files (will be repopulated on reprocessing)
            self.conn.execute("UPDATE files SET mode_tier = NULL, embedding_model = NULL")
            self.conn.commit()
            logger.info("[TIER MIGRATION] Cleared tier metadata from files table")

            logger.info(f"[TIER MIGRATION] Migration complete. Reprocess all files to populate embeddings.")
            return True

        except Exception as e:
            self.conn.rollback()
            logger.error(f"[TIER MIGRATION] Failed: {e}")
            return False

    def checkpoint(self):
        """Force WAL checkpoint to flush pending writes to main DB."""
        if self.conn:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception as e:
                logger.warning(f"WAL checkpoint failed: {e}")

    # ── Archive Browse ─────────────────────────────────────────

    def get_archive_folders(self) -> List[Dict[str, Any]]:
        """Get distinct folder_path values with file counts and phase stats.

        Returns list of dicts: { folder_path, total, mc, vv, mv, image_types }
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT
                    COALESCE(NULLIF(TRIM(f.folder_path), ''), '[root]') as fp,
                    COUNT(*) as total,
                    COUNT(CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 END) as mc,
                    COUNT(CASE
                        WHEN EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
                        THEN 1
                    END) as vv,
                    COUNT(CASE WHEN EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) THEN 1 END) as mv
                FROM files f
                WHERE f.preview_only = 0 OR f.preview_only IS NULL
                GROUP BY fp
                ORDER BY fp
            """)
        except Exception:
            # vec0 unavailable fallback
            cursor.execute("""
                SELECT
                    COALESCE(NULLIF(TRIM(f.folder_path), ''), '[root]') as fp,
                    COUNT(*) as total,
                    COUNT(CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 END) as mc,
                    0 as vv,
                    0 as mv
                FROM files f
                WHERE f.preview_only = 0 OR f.preview_only IS NULL
                GROUP BY fp
                ORDER BY fp
            """)

        results = []
        for row in cursor.fetchall():
            results.append({
                "folder_path": row[0],
                "total": row[1],
                "mc": row[2],
                "vv": row[3],
                "mv": row[4],
            })

        # Get image_type distribution per folder
        cursor.execute("""
            SELECT
                COALESCE(NULLIF(TRIM(folder_path), ''), '[root]') as fp,
                image_type,
                COUNT(*) as cnt
            FROM files
            WHERE (preview_only = 0 OR preview_only IS NULL)
              AND image_type IS NOT NULL AND image_type != ''
            GROUP BY fp, image_type
        """)
        type_map = {}
        for row in cursor.fetchall():
            fp = row[0]
            if fp not in type_map:
                type_map[fp] = {}
            type_map[fp][row[1]] = row[2]

        for r in results:
            r["image_types"] = type_map.get(r["folder_path"], {})

        return results

    def get_archive_files(
        self,
        folder_path: Optional[str] = None,
        image_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get files for archive browsing with phase status.

        Returns { total, files: [...] }
        """
        cursor = self.conn.cursor()

        conditions = ["(f.preview_only = 0 OR f.preview_only IS NULL)"]
        params: list = []

        if folder_path and folder_path != '[all]':
            if folder_path == '[root]':
                conditions.append("(f.folder_path IS NULL OR TRIM(f.folder_path) = '')")
            else:
                conditions.append("f.folder_path = ?")
                params.append(folder_path)

        if image_type:
            conditions.append("f.image_type = ?")
            params.append(image_type)

        where = f"WHERE {' AND '.join(conditions)}"

        # Total count
        cursor.execute(f"SELECT COUNT(*) FROM files f {where}", params)
        total = cursor.fetchone()[0]

        # Fetch files with phase status
        use_vec = True
        try:
            cursor.execute(f"""
                SELECT
                    f.id, f.file_path, f.file_name, f.format,
                    f.width, f.height, f.thumbnail_url,
                    f.mc_caption, f.ai_tags, f.image_type, f.art_style,
                    f.folder_path, f.storage_root,
                    f.user_note, f.user_tags, f.user_rating,
                    f.mode_tier, f.parsed_at,
                    CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 ELSE 0 END as has_mc,
                    CASE WHEN EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id) THEN 1 ELSE 0 END as has_vv,
                    CASE WHEN EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) THEN 1 ELSE 0 END as has_mv
                FROM files f
                {where}
                ORDER BY f.file_name ASC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
        except Exception:
            use_vec = False
            cursor.execute(f"""
                SELECT
                    f.id, f.file_path, f.file_name, f.format,
                    f.width, f.height, f.thumbnail_url,
                    f.mc_caption, f.ai_tags, f.image_type, f.art_style,
                    f.folder_path, f.storage_root,
                    f.user_note, f.user_tags, f.user_rating,
                    f.mode_tier, f.parsed_at,
                    CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != '' THEN 1 ELSE 0 END as has_mc,
                    0 as has_vv,
                    0 as has_mv
                FROM files f
                {where}
                ORDER BY f.file_name ASC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])

        files = []
        for row in cursor.fetchall():
            ai_tags = row[8]
            user_tags = row[14]
            if isinstance(ai_tags, str):
                try:
                    ai_tags = json.loads(ai_tags)
                except Exception:
                    ai_tags = []
            if isinstance(user_tags, str):
                try:
                    user_tags = json.loads(user_tags)
                except Exception:
                    user_tags = []

            files.append({
                "id": row[0],
                "file_path": row[1],
                "file_name": row[2],
                "format": row[3],
                "width": row[4],
                "height": row[5],
                "thumbnail_url": row[6],
                "mc_caption": row[7],
                "ai_tags": ai_tags,
                "image_type": row[9],
                "art_style": row[10],
                "folder_path": row[11],
                "storage_root": row[12],
                "user_note": row[13],
                "user_tags": user_tags,
                "user_rating": row[15],
                "mode_tier": row[16],
                "parsed_at": row[17],
                "phase": {
                    "mc": bool(row[18]),
                    "vv": bool(row[19]),
                    "mv": bool(row[20]),
                },
            })

        return {"total": total, "files": files}

    def get_image_type_stats(self) -> List[Dict[str, Any]]:
        """Get image_type distribution across all non-preview files.

        Returns list of { image_type, count }.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                COALESCE(image_type, 'unknown') as it,
                COUNT(*) as cnt
            FROM files
            WHERE (preview_only = 0 OR preview_only IS NULL)
            GROUP BY it
            ORDER BY cnt DESC
        """)
        return [{"image_type": row[0], "count": row[1]} for row in cursor.fetchall()]

    def close(self):
        """Close database connections."""
        # Close current thread's connection
        c = getattr(self._local, 'conn', None)
        if c is not None:
            try:
                c.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
            c.close()
            self._local.conn = None
        # Close setup connection (if different)
        if self._setup_conn is not None and self._setup_conn is not c:
            try:
                self._setup_conn.close()
            except Exception:
                pass
            self._setup_conn = None
        logger.info("SQLite connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
