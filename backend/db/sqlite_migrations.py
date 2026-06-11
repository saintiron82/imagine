"""
SQLite migration functions extracted from sqlite_client.py.

Each migration is a standalone function that receives a SQLiteDB instance (db).
Access the connection via db.conn and use db._table_exists() for table checks.
run_migrations(db) calls all migrations in the correct order.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Individual migration functions
# ──────────────────────────────────────────────────────────────

def migrate_folder_columns(db):
    """Add folder columns to existing databases if missing."""
    try:
        db.conn.execute("SELECT folder_path FROM files LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating: adding folder columns to files table...")
        db.conn.execute("ALTER TABLE files ADD COLUMN folder_path TEXT")
        db.conn.execute("ALTER TABLE files ADD COLUMN folder_depth INTEGER DEFAULT 0")
        db.conn.execute("ALTER TABLE files ADD COLUMN folder_tags TEXT")
        db.conn.execute("CREATE INDEX IF NOT EXISTS idx_files_folder_path ON files(folder_path)")
        db.conn.commit()

        # FTS5 virtual tables cannot be ALTERed - drop and recreate
        logger.info("Migrating: rebuilding FTS5 table with folder_tags...")
        db.conn.execute("DROP TRIGGER IF EXISTS files_fts_insert")
        db.conn.execute("DROP TRIGGER IF EXISTS files_fts_update")
        db.conn.execute("DROP TRIGGER IF EXISTS files_fts_delete")
        db.conn.execute("DROP TABLE IF EXISTS files_fts")
        db.conn.commit()
        # FTS5 + triggers will be recreated by init_schema() executescript

        logger.info("Folder columns migration complete")


def migrate_v3_columns(db):
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
        # NOTE: Legacy default from pre-v3.1. Runtime uses SigLIP2 from tier config.
        ("embedding_model", "TEXT DEFAULT 'clip-ViT-L-14'"),
        ("embedding_version", "INTEGER DEFAULT 1"),
    ]
    try:
        existing = {row[1] for row in db.conn.execute("PRAGMA table_info(files)").fetchall()}
        added = 0
        for col_name, col_def in v3_cols:
            if col_name not in existing:
                db.conn.execute(f"ALTER TABLE files ADD COLUMN {col_name} {col_def}")
                added += 1
        if added:
            # v3 indexes
            db.conn.execute("CREATE INDEX IF NOT EXISTS idx_image_type ON files(image_type)")
            db.conn.execute("CREATE INDEX IF NOT EXISTS idx_art_style ON files(art_style)")
            db.conn.execute("CREATE INDEX IF NOT EXISTS idx_scene_type ON files(scene_type)")
            db.conn.execute("CREATE INDEX IF NOT EXISTS idx_relative_path ON files(relative_path)")
            db.conn.commit()
            logger.info(f"v3 migration: added {added} columns + indexes")
    except Exception as e:
        logger.warning(f"v3 migration check failed (non-fatal): {e}")


def migrate_content_hash(db):
    """Add content_hash column and vec cascade delete triggers."""
    try:
        db.conn.execute("SELECT content_hash FROM files LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating: adding content_hash column to files table...")
        db.conn.execute("ALTER TABLE files ADD COLUMN content_hash TEXT")
        db.conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON files(content_hash)")
        db.conn.commit()
        logger.info("content_hash migration complete")

    # Vec cascade delete triggers (best-effort; non-fatal when vec module is unavailable)
    try:
        db.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS vec_files_cascade_delete
            AFTER DELETE ON files BEGIN
                DELETE FROM vec_files WHERE file_id = old.id;
            END
        """)
        db.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS vec_text_cascade_delete
            AFTER DELETE ON files BEGIN
                DELETE FROM vec_text WHERE file_id = old.id;
            END
        """)
        db.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS vec_structure_cascade_delete
            AFTER DELETE ON files BEGIN
                DELETE FROM vec_structure WHERE file_id = old.id;
            END
        """)
        db.conn.commit()
    except sqlite3.OperationalError as e:
        logger.warning(f"Skipping vec cascade trigger ensure: {e}")


def migrate_structure_table(db):
    """Ensure vec_structure table exists (for DINOv2)."""
    if not db._vec_extension_loaded:
        logger.warning("Skipping vec_structure migration: sqlite-vec extension not loaded")
        return
    try:
        db.conn.execute("SELECT count(*) FROM vec_structure")
    except sqlite3.OperationalError:
        logger.info("Migrating: creating vec_structure table...")
        try:
            db.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_structure USING vec0(file_id INTEGER PRIMARY KEY, embedding FLOAT[768])")
            db.conn.commit()
        except sqlite3.OperationalError as e:
            logger.warning(f"Skipping vec_structure creation: {e}")


def migrate_uploaded_by(db):
    """Add uploaded_by column for server-mode file ownership tracking."""
    try:
        db.conn.execute("SELECT uploaded_by FROM files LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating: adding uploaded_by column to files table...")
        db.conn.execute("ALTER TABLE files ADD COLUMN uploaded_by INTEGER REFERENCES users(id) ON DELETE SET NULL")
        db.conn.execute("CREATE INDEX IF NOT EXISTS idx_files_uploaded_by ON files(uploaded_by)")
        db.conn.commit()
        logger.info("uploaded_by migration complete")


def migrate_preview_only(db):
    """Add preview_only column for browse-time pre-parsing."""
    try:
        db.conn.execute("SELECT preview_only FROM files LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating: adding preview_only column to files table...")
        db.conn.execute("ALTER TABLE files ADD COLUMN preview_only INTEGER DEFAULT 0")
        db.conn.commit()
        logger.info("preview_only migration complete")


def migrate_backfill_storage_root(db):
    """Backfill storage_root for files where it's NULL.

    Derives parent directory from file_path.
    Without storage_root, folder-level phase stats (Sidebar green/orange dots) break.
    """
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM files
        WHERE storage_root IS NULL OR TRIM(storage_root) = ''
    """)
    null_count = cursor.fetchone()[0]
    if null_count == 0:
        return

    logger.info(f"Migrating: backfilling storage_root for {null_count} files...")
    # Derive parent directory: remove '/' + file_name from file_path
    cursor.execute("""
        UPDATE files
        SET storage_root = REPLACE(file_path, '/' || file_name, '')
        WHERE (storage_root IS NULL OR TRIM(storage_root) = '')
          AND file_name IS NOT NULL AND file_name != ''
          AND file_path IS NOT NULL AND file_path != ''
    """)
    updated = cursor.rowcount
    db.conn.commit()
    logger.info(f"storage_root backfill complete: {updated} files updated")


def migrate_auth_tables(db):
    """Create auth and worker tables for client-server mode if missing."""
    if db._table_exists('users'):
        return  # Already migrated

    logger.info("Migrating: creating auth and worker tables...")
    auth_schema_path = Path(__file__).parent / "sqlite_schema_auth.sql"
    if auth_schema_path.exists():
        with open(auth_schema_path, encoding='utf-8') as f:
            db.conn.executescript(f.read())
        db.conn.commit()
        logger.info("Auth and worker tables created")
    else:
        logger.warning(f"Auth schema file not found: {auth_schema_path}")


def migrate_drop_worker_tokens(db):
    """Drop legacy worker_tokens table (removed when WK_ token flow was retired)."""
    if not db._table_exists('worker_tokens'):
        return
    logger.info("Migrating: dropping legacy worker_tokens table...")
    db.conn.execute("DROP INDEX IF EXISTS idx_worker_tokens_hash")
    db.conn.execute("DROP TABLE IF EXISTS worker_tokens")
    db.conn.commit()
    logger.info("worker_tokens table dropped")


def migrate_drop_legacy_queue_tables(db):
    """Drop retired queue/work tables after the analysis task system takes over."""
    legacy_tables = ("job_queue", "work_subtasks", "work_requests", "job_completions")
    if not any(db._table_exists(table) for table in legacy_tables):
        return

    logger.info("Migrating: dropping retired queue/work tables...")
    try:
        db.conn.execute("PRAGMA foreign_keys = OFF")
        for table in legacy_tables:
            db.conn.execute(f"DROP TABLE IF EXISTS {table}")
        db.conn.commit()
        logger.info("Retired queue/work tables dropped")
    finally:
        db.conn.execute("PRAGMA foreign_keys = ON")


def migrate_worker_sessions(db):
    """Create worker_sessions table if missing (added in v4.10)."""
    if db._table_exists('worker_sessions'):
        return
    logger.info("Migrating: creating worker_sessions table...")
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS worker_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            worker_name TEXT NOT NULL,
            hostname TEXT,
            origin TEXT DEFAULT 'headless'
                CHECK (origin IN ('server-local', 'client-launched', 'headless')),
            launcher TEXT DEFAULT 'cli'
                CHECK (launcher IN ('server', 'electron', 'cli', 'service', 'cloud')),
            status TEXT DEFAULT 'online'
                CHECK (status IN ('online', 'offline', 'blocked')),
            batch_capacity INTEGER DEFAULT 5,
            jobs_completed INTEGER DEFAULT 0,
            jobs_failed INTEGER DEFAULT 0,
            current_job_id INTEGER,
            current_file TEXT,
            current_phase TEXT,
            pending_command TEXT DEFAULT NULL
                CHECK (pending_command IN (NULL, 'stop', 'pause', 'block')),
            connected_at TEXT DEFAULT (datetime('now')),
            last_heartbeat TEXT DEFAULT (datetime('now')),
            disconnected_at TEXT
        )
    """)
    db.conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_sessions_user ON worker_sessions(user_id, status)")
    db.conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_sessions_status ON worker_sessions(status)")
    db.conn.commit()
    logger.info("worker_sessions table created")


def migrate_worker_session_overrides(db):
    """Add per-worker override columns to worker_sessions (v10.6 real-time control)."""
    if not db._table_exists('worker_sessions'):
        return
    try:
        db.conn.execute("SELECT processing_mode_override FROM worker_sessions LIMIT 1")
    except Exception:
        logger.info("Migrating: adding per-worker override columns to worker_sessions...")
        db.conn.execute("ALTER TABLE worker_sessions ADD COLUMN processing_mode_override TEXT DEFAULT NULL")
        db.conn.execute("ALTER TABLE worker_sessions ADD COLUMN batch_capacity_override INTEGER DEFAULT NULL")
        db.conn.commit()
        logger.info("per-worker override columns added to worker_sessions")


def migrate_worker_resources_json(db):
    """Add resources_json column to worker_sessions for resource metrics."""
    if not db._table_exists('worker_sessions'):
        return
    try:
        db.conn.execute("SELECT resources_json FROM worker_sessions LIMIT 1")
    except Exception:
        logger.info("Migrating: adding resources_json column to worker_sessions...")
        db.conn.execute("ALTER TABLE worker_sessions ADD COLUMN resources_json TEXT DEFAULT NULL")
        db.conn.commit()
        logger.info("resources_json column added to worker_sessions")


def migrate_worker_origin_launcher(db):
    """Add worker origin/launcher columns for runtime classification."""
    if not db._table_exists('worker_sessions'):
        return
    try:
        db.conn.execute("SELECT origin FROM worker_sessions LIMIT 1")
    except Exception:
        logger.info("Migrating: adding worker origin/launcher columns...")
        db.conn.execute(
            "ALTER TABLE worker_sessions ADD COLUMN origin TEXT DEFAULT 'headless'"
        )
        db.conn.execute(
            "ALTER TABLE worker_sessions ADD COLUMN launcher TEXT DEFAULT 'cli'"
        )
        db.conn.commit()
        logger.info("worker origin/launcher columns added")


def migrate_audit_log(db):
    """Phase 8: append-only audit log for security-relevant events."""
    if db._table_exists('audit_log'):
        return
    logger.info("Migrating: creating audit_log table...")
    db.conn.executescript(
        """
        CREATE TABLE audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            actor_user_id INTEGER,
            actor_username TEXT,
            target_kind TEXT,
            target_id TEXT,
            ip_address TEXT,
            detail TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_audit_log_created ON audit_log(created_at);
        CREATE INDEX idx_audit_log_event ON audit_log(event);
        CREATE INDEX idx_audit_log_actor ON audit_log(actor_user_id);
        """
    )
    db.conn.commit()
    logger.info("audit_log table created")


def migrate_search_feedback(db):
    """Phase D: store user 'irrelevant' feedback for search results."""
    if db._table_exists('search_feedback'):
        return
    logger.info("Migrating: creating search_feedback table...")
    db.conn.executescript(
        """
        CREATE TABLE search_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            label TEXT NOT NULL CHECK (label IN ('irrelevant')),
            user_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_search_feedback_query ON search_feedback(query);
        CREATE INDEX idx_search_feedback_file ON search_feedback(file_id);
        """
    )
    db.conn.commit()
    logger.info("search_feedback table created")


def migrate_worker_enrollment_tokens(db):
    """Phase 8: short-lived, single-use tokens for worker attachment.

    Separate from refresh_tokens so they can never grant admin API
    access. Hash stored, plaintext never persisted.
    """
    if db._table_exists('worker_enrollment_tokens'):
        return
    logger.info("Migrating: creating worker_enrollment_tokens table...")
    db.conn.executescript(
        """
        CREATE TABLE worker_enrollment_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash TEXT UNIQUE NOT NULL,
            created_by INTEGER,
            worker_name TEXT,
            max_uses INTEGER DEFAULT 1,
            use_count INTEGER DEFAULT 0,
            expires_at TEXT NOT NULL,
            revoked INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            last_used_at TEXT
        );
        CREATE INDEX idx_enrollment_hash ON worker_enrollment_tokens(token_hash);
        CREATE INDEX idx_enrollment_expires ON worker_enrollment_tokens(expires_at);
        """
    )
    db.conn.commit()
    logger.info("worker_enrollment_tokens table created")


def migrate_users_email_nullable(db):
    """Make users.email nullable (remove NOT NULL constraint).

    Group-based registration makes email optional.
    Uses PRAGMA writable_schema since SQLite cannot ALTER COLUMN.
    """
    if not db._table_exists('users'):
        return
    row = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()
    if not row:
        return
    schema_sql = row[0]
    # Check if email column has NOT NULL -- if not present, already nullable
    if 'email TEXT UNIQUE NOT NULL' not in schema_sql:
        return

    logger.info("Migrating: making users.email nullable...")
    db.conn.execute("PRAGMA writable_schema = ON")
    db.conn.execute("""
        UPDATE sqlite_master
        SET sql = REPLACE(sql,
            'email TEXT UNIQUE NOT NULL',
            'email TEXT UNIQUE')
        WHERE type = 'table' AND name = 'users'
    """)
    db.conn.execute("PRAGMA writable_schema = OFF")
    # Integrity check after schema modification
    db.conn.execute("PRAGMA integrity_check")
    db.conn.commit()
    logger.info("users.email is now nullable")


def migrate_users_firebase_uid(db):
    """Add firebase_uid column to users table for 2-layer auth (Firebase identity + server password)."""
    if not db._table_exists('users'):
        return
    try:
        db.conn.execute("SELECT firebase_uid FROM users LIMIT 1")
    except Exception:
        logger.info("Migrating: adding firebase_uid column to users...")
        db.conn.execute("ALTER TABLE users ADD COLUMN firebase_uid TEXT")
        db.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_firebase_uid "
            "ON users(firebase_uid)"
        )
        db.conn.commit()
        logger.info("firebase_uid column + unique index added to users")


def migrate_files_processing_status(db):
    """Add processing_status and processing_error columns to files table."""
    if not db._table_exists('files'):
        return
    try:
        db.conn.execute("SELECT processing_status FROM files LIMIT 1")
    except Exception:
        logger.info("Migrating: adding processing_status columns to files...")
        db.conn.execute(
            "ALTER TABLE files ADD COLUMN processing_status TEXT DEFAULT NULL"
        )
        db.conn.execute(
            "ALTER TABLE files ADD COLUMN processing_error TEXT DEFAULT NULL"
        )
        db.conn.commit()
        logger.info("processing_status columns added to files")


def migrate_members_table(db):
    """Create members table if missing (added for group/Firestore migration)."""
    if db._table_exists('members'):
        return
    try:
        logger.info("Migrating: creating members table...")
        db.conn.execute("""
            CREATE TABLE IF NOT EXISTS members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                firebase_uid TEXT UNIQUE,
                email TEXT,
                display_name TEXT,
                role TEXT NOT NULL DEFAULT 'user'
                    CHECK (role IN ('admin', 'user', 'viewer')),
                is_active INTEGER NOT NULL DEFAULT 1,
                joined_at TEXT DEFAULT (datetime('now')),
                last_seen_at TEXT,
                invited_by INTEGER REFERENCES members(id),
                quota_files_per_day INTEGER NOT NULL DEFAULT 100,
                quota_search_per_min INTEGER NOT NULL DEFAULT 30
            )
        """)
        db.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_members_firebase_uid "
            "ON members(firebase_uid)"
        )
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_members_email ON members(email)"
        )
        db.conn.commit()
        logger.info("members table created")
    except Exception as e:
        logger.warning(f"members migration failed (non-fatal): {e}")


def migrate_drop_fts_update_trigger(db):
    """Remove FTS UPDATE trigger that was resetting FTS to empty strings."""
    try:
        db.conn.execute("DROP TRIGGER IF EXISTS files_fts_update")
        db.conn.commit()
    except Exception:
        pass


def migrate_file_objects(db):
    """Ensure normalized spatial object evidence storage exists."""
    try:
        db._ensure_file_objects_table()
    except Exception as e:
        logger.warning(f"file_objects migration failed (non-fatal): {e}")


def migrate_vlm_raw_outputs(db):
    """Ensure raw VLM output preservation storage exists."""
    try:
        db._ensure_vlm_raw_outputs_table()
    except Exception as e:
        logger.warning(f"vlm_raw_outputs migration failed (non-fatal): {e}")


def migrate_search_logs(db):
    """Create search_logs table for search request tracking."""
    if db._table_exists('search_logs'):
        return
    try:
        logger.info("Migrating: creating search_logs table...")
        db.conn.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'triaxis',
                result_count INTEGER DEFAULT 0,
                elapsed_ms INTEGER DEFAULT 0,
                username TEXT,
                ip_address TEXT,
                filters TEXT,
                threshold REAL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_search_logs_created "
            "ON search_logs(created_at)"
        )
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_search_logs_user "
            "ON search_logs(username)"
        )
        db.conn.commit()
        logger.info("search_logs table created")
    except Exception as e:
        logger.warning(f"search_logs migration failed: {e}")


def migrate_worker_phase_tracking(db):
    """Add assigned_mode + phase_job_count to worker_sessions for dynamic mode tracking."""
    if not db._table_exists('worker_sessions'):
        return
    try:
        db.conn.execute("SELECT assigned_mode FROM worker_sessions LIMIT 1")
    except Exception:
        logger.info("Migrating: adding assigned_mode, phase_job_count to worker_sessions...")
        db.conn.execute("ALTER TABLE worker_sessions ADD COLUMN assigned_mode TEXT DEFAULT NULL")
        db.conn.execute("ALTER TABLE worker_sessions ADD COLUMN phase_job_count INTEGER DEFAULT 0")
        db.conn.commit()
        logger.info("worker_sessions phase tracking columns added")


def _ensure_analysis_tables(db):
    """Create analysis_jobs + file_tasks if they don't exist."""
    schema_path = Path(__file__).parent / "schema_analysis.sql"
    if schema_path.exists():
        db.conn.executescript(schema_path.read_text())
        logger.info("Analysis Job tables verified")


def migrate_dismissed_at(db):
    """Add dismissed_at column to file_tasks for acknowledged permanent failures."""
    try:
        db.conn.execute("SELECT dismissed_at FROM file_tasks LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating: adding dismissed_at to file_tasks")
        db.conn.execute("ALTER TABLE file_tasks ADD COLUMN dismissed_at TEXT")
        db.conn.commit()


# ──────────────────────────────────────────────────────────────
# Orchestrator: run all migrations in order
# ──────────────────────────────────────────────────────────────

def migrate_derivations(db):
    """CAS M1: content-addressed derivation cache + model registry.

    derivations: (content_hash, phase, model_version) → result. Write path
    only in M1 (shadow write); reads activate in M3.
    """
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS derivations (
            content_hash  TEXT NOT NULL,
            phase         TEXT NOT NULL CHECK (phase IN ('mc','vv','mv')),
            model_version TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'done'
                CHECK (status IN ('done','failed')),
            result_json   TEXT,
            vector_blob   BLOB,
            error_message TEXT,
            created_at    TEXT DEFAULT (datetime('now')),
            created_by    TEXT,
            PRIMARY KEY (content_hash, phase, model_version)
        )
    """)
    db.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_deriv_phase_ver
        ON derivations(phase, model_version)
    """)
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            phase          TEXT NOT NULL CHECK (phase IN ('mc','vv','mv')),
            model_version  TEXT NOT NULL,
            is_active      INTEGER NOT NULL DEFAULT 0,
            activated_at   TEXT,
            PRIMARY KEY (phase, model_version)
        )
    """)
    db.conn.commit()


def run_migrations(db, *, existing_db: bool = True):
    """Run all migrations in the correct order.

    Args:
        db: SQLiteDB instance (access db.conn, db._table_exists(), etc.)
        existing_db: True if files table already exists (upgrade path),
                     False if empty database (fresh install path).
    """
    if existing_db:
        import time as _t
        _total_start = _t.perf_counter()

        migrations = [
            ("folder_columns", lambda: migrate_folder_columns(db)),
            ("v3_columns", lambda: migrate_v3_columns(db)),
            ("content_hash", lambda: migrate_content_hash(db)),
            ("structure_table", lambda: migrate_structure_table(db)),
            ("uploaded_by", lambda: migrate_uploaded_by(db)),
            ("preview_only", lambda: migrate_preview_only(db)),
            ("backfill_storage_root", lambda: migrate_backfill_storage_root(db)),
            ("system_meta", lambda: db._ensure_system_meta()),
            ("file_objects", lambda: migrate_file_objects(db)),
            ("vlm_raw_outputs", lambda: migrate_vlm_raw_outputs(db)),
            ("fts", lambda: db._ensure_fts()),
            ("auth_tables", lambda: migrate_auth_tables(db)),
            ("drop_worker_tokens", lambda: migrate_drop_worker_tokens(db)),
            ("drop_legacy_queue_tables", lambda: migrate_drop_legacy_queue_tables(db)),
            ("worker_sessions", lambda: migrate_worker_sessions(db)),
            ("worker_session_overrides", lambda: migrate_worker_session_overrides(db)),
            ("worker_resources_json", lambda: migrate_worker_resources_json(db)),
            ("worker_origin_launcher", lambda: migrate_worker_origin_launcher(db)),
            ("audit_log", lambda: migrate_audit_log(db)),
            ("search_feedback", lambda: migrate_search_feedback(db)),
            ("worker_enrollment_tokens", lambda: migrate_worker_enrollment_tokens(db)),
            ("users_email_nullable", lambda: migrate_users_email_nullable(db)),
            ("users_firebase_uid", lambda: migrate_users_firebase_uid(db)),
            ("files_processing_status", lambda: migrate_files_processing_status(db)),
            ("members_table", lambda: migrate_members_table(db)),
            ("drop_fts_update_trigger", lambda: migrate_drop_fts_update_trigger(db)),
            ("search_logs", lambda: migrate_search_logs(db)),
            ("worker_phase_tracking", lambda: migrate_worker_phase_tracking(db)),
            ("analysis_tables", lambda: _ensure_analysis_tables(db)),
            ("dismissed_at", lambda: migrate_dismissed_at(db)),
            ("derivations", lambda: migrate_derivations(db)),
        ]

        slow = []
        for name, fn in migrations:
            t0 = _t.perf_counter()
            fn()
            elapsed = _t.perf_counter() - t0
            if elapsed > 0.1:
                slow.append((name, elapsed))

        total = _t.perf_counter() - _total_start
        if slow:
            slow_str = ", ".join(f"{n}={e:.1f}s" for n, e in slow)
            msg = f"Migrations: {total:.1f}s total. Slow: {slow_str}"
        else:
            msg = f"Migrations: {total:.1f}s total (all fast)"
        logger.info(msg)
        # stderr, not stdout: subprocess consumers (e.g. api_search.py daemon)
        # use stdout as a JSON IPC channel. A raw print() here pollutes the
        # protocol and wedges main.cjs's line parser.
        import sys as _sys
        print(f"[DB] {msg}", flush=True, file=_sys.stderr)
    else:
        # Fresh install path: empty DB, schema just initialized
        db._ensure_system_meta()
        migrate_file_objects(db)
        migrate_vlm_raw_outputs(db)
        migrate_auth_tables(db)
        migrate_worker_sessions(db)
        migrate_worker_session_overrides(db)
        migrate_worker_resources_json(db)
        migrate_worker_origin_launcher(db)
        migrate_audit_log(db)
        migrate_search_feedback(db)
        migrate_worker_enrollment_tokens(db)
        migrate_files_processing_status(db)
        migrate_members_table(db)
        migrate_search_logs(db)
        migrate_worker_phase_tracking(db)
        _ensure_analysis_tables(db)
        migrate_dismissed_at(db)
        migrate_derivations(db)
