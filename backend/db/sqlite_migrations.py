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
    """Create auth & job queue tables for client-server mode if missing."""
    if db._table_exists('users'):
        return  # Already migrated

    logger.info("Migrating: creating auth & job queue tables...")
    auth_schema_path = Path(__file__).parent / "sqlite_schema_auth.sql"
    if auth_schema_path.exists():
        with open(auth_schema_path, encoding='utf-8') as f:
            db.conn.executescript(f.read())
        db.conn.commit()
        logger.info("Auth & job queue tables created")
    else:
        logger.warning(f"Auth schema file not found: {auth_schema_path}")


def migrate_worker_tokens(db):
    """Create worker_tokens table if missing (added in v4.7)."""
    if db._table_exists('worker_tokens'):
        return
    logger.info("Migrating: creating worker_tokens table...")
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS worker_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_by INTEGER REFERENCES users(id) ON DELETE CASCADE,
            is_active INTEGER DEFAULT 1,
            expires_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            last_used_at TEXT
        )
    """)
    db.conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_tokens_hash ON worker_tokens(token_hash)")
    db.conn.commit()
    logger.info("worker_tokens table created")


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


def migrate_parse_ahead_columns(db):
    """Add parse_status / parsed_metadata / parsed_at to job_queue (v10.3 Parse-ahead Pool)."""
    if not db._table_exists('job_queue'):
        return
    try:
        db.conn.execute("SELECT parse_status FROM job_queue LIMIT 1")
    except Exception:
        logger.info("Migrating: adding parse-ahead columns to job_queue...")
        db.conn.execute("ALTER TABLE job_queue ADD COLUMN parse_status TEXT DEFAULT NULL")
        db.conn.execute("ALTER TABLE job_queue ADD COLUMN parsed_metadata TEXT DEFAULT NULL")
        db.conn.execute("ALTER TABLE job_queue ADD COLUMN parsed_at TEXT DEFAULT NULL")
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_queue_parse_status "
            "ON job_queue(parse_status, priority DESC, created_at ASC)"
        )
        db.conn.commit()
        logger.info("parse-ahead columns added to job_queue")


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


def migrate_worker_session_tracking(db):
    """Add worker_session_id to job_queue for per-worker throughput tracking."""
    if not db._table_exists('job_queue'):
        return
    try:
        db.conn.execute("SELECT worker_session_id FROM job_queue LIMIT 1")
    except Exception:
        logger.info("Migrating: adding worker_session_id to job_queue...")
        db.conn.execute("ALTER TABLE job_queue ADD COLUMN worker_session_id INTEGER")
        db.conn.commit()
        logger.info("worker_session_id column added to job_queue")


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


def migrate_mc_completed_at(db):
    """Add mc_completed_at column to job_queue for MC throughput measurement."""
    if not db._table_exists('job_queue'):
        return
    try:
        db.conn.execute("SELECT mc_completed_at FROM job_queue LIMIT 1")
    except Exception:
        logger.info("Migrating: adding mc_completed_at column to job_queue...")
        db.conn.execute("ALTER TABLE job_queue ADD COLUMN mc_completed_at TEXT")
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_queue_mc_completed "
            "ON job_queue(mc_completed_at)"
        )
        db.conn.commit()
        logger.info("mc_completed_at column + index added to job_queue")


def migrate_backfill_parse_status(db):
    """Extend parse_status CHECK constraint to allow 'backfill' value.

    SQLite CHECK constraints cannot be altered with ALTER TABLE,
    so we use PRAGMA writable_schema to modify the schema SQL directly.
    """
    if not db._table_exists('job_queue'):
        return
    row = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='job_queue'"
    ).fetchone()
    if not row or "'backfill'" in row[0]:
        return  # Already includes 'backfill' or table missing

    logger.info("Migrating: extending parse_status CHECK to include 'backfill'...")
    db.conn.execute("PRAGMA writable_schema = ON")
    db.conn.execute("""
        UPDATE sqlite_master
        SET sql = REPLACE(sql,
            "'pending', 'parsing', 'parsed', 'failed')",
            "'pending', 'parsing', 'parsed', 'failed', 'backfill')")
        WHERE type = 'table' AND name = 'job_queue'
    """)
    db.conn.execute("PRAGMA writable_schema = OFF")
    db.conn.commit()
    logger.info("parse_status CHECK extended to include 'backfill'")


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


def migrate_error_code(db):
    """Add error_code column to job_queue for structured error classification."""
    if not db._table_exists('job_queue'):
        return
    try:
        db.conn.execute("SELECT error_code FROM job_queue LIMIT 1")
    except Exception:
        logger.info("Migrating: adding error_code column to job_queue...")
        db.conn.execute("ALTER TABLE job_queue ADD COLUMN error_code TEXT DEFAULT NULL")
        db.conn.commit()
        logger.info("error_code column added to job_queue")

    # Backfill: infer error_code from legacy error_message for failed jobs
    cursor = db.conn.execute(
        "SELECT COUNT(*) FROM job_queue WHERE status='failed' AND error_code IS NULL AND error_message IS NOT NULL"
    )
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        logger.info(f"Backfilling error_code for {null_count} legacy failed jobs...")
        # FILE_NOT_FOUND pattern
        db.conn.execute("""
            UPDATE job_queue SET error_code = 'FILE_NOT_FOUND'
            WHERE status = 'failed' AND error_code IS NULL
            AND (LOWER(error_message) LIKE '%file unavailable%'
                 OR LOWER(error_message) LIKE '%file not found%'
                 OR LOWER(error_message) LIKE '%cannot access%')
        """)
        # THUMB_MISSING pattern
        db.conn.execute("""
            UPDATE job_queue SET error_code = 'THUMB_MISSING'
            WHERE status = 'failed' AND error_code IS NULL
            AND LOWER(error_message) LIKE '%thumbnail%requires%'
        """)
        # PARSE_FAILED pattern
        db.conn.execute("""
            UPDATE job_queue SET error_code = 'PARSE_FAILED'
            WHERE status = 'failed' AND error_code IS NULL
            AND LOWER(error_message) LIKE '%parse failed%'
        """)
        # UNKNOWN fallback for everything else
        db.conn.execute("""
            UPDATE job_queue SET error_code = 'UNKNOWN'
            WHERE status = 'failed' AND error_code IS NULL
        """)
        db.conn.commit()
        filled = null_count - db.conn.execute(
            "SELECT COUNT(*) FROM job_queue WHERE status='failed' AND error_code IS NULL AND error_message IS NOT NULL"
        ).fetchone()[0]
        logger.info(f"Backfilled error_code for {filled}/{null_count} jobs")


def migrate_file_ready(db):
    """Add file_ready column to job_queue (2-stage pipeline gate).

    file_ready=1: file is locally available for processing (default)
    file_ready=0: file needs preparation (WebDAV download pending)

    Also resets existing WebDAV failed jobs to pending + file_ready=0
    so DownloadAheadPool can pick them up fresh.
    """
    if not db._table_exists('job_queue'):
        return
    try:
        db.conn.execute("SELECT file_ready FROM job_queue LIMIT 1")
    except Exception:
        logger.info("Migrating: adding file_ready column to job_queue...")
        db.conn.execute(
            "ALTER TABLE job_queue ADD COLUMN file_ready INTEGER NOT NULL DEFAULT 1"
        )
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_queue_file_ready "
            "ON job_queue(file_ready, status, priority DESC, created_at ASC)"
        )
        # Backfill: mark existing WebDAV jobs as not ready
        cursor = db.conn.execute(
            """UPDATE job_queue SET file_ready = 0
               WHERE file_path LIKE 'webdav://%'
                 AND status != 'completed'"""
        )
        webdav_count = cursor.rowcount
        # Reset permanently failed WebDAV jobs to pending
        cursor2 = db.conn.execute(
            """UPDATE job_queue SET status = 'pending', retry_count = 0,
                   error_message = NULL, error_code = NULL
               WHERE file_path LIKE 'webdav://%'
                 AND status = 'failed'"""
        )
        reset_count = cursor2.rowcount
        db.conn.commit()
        logger.info(
            f"file_ready column added to job_queue "
            f"(webdav={webdav_count} marked not-ready, "
            f"{reset_count} failed jobs reset to pending)"
        )


def migrate_job_completions(db):
    """Create job_completions table for throughput tracking."""
    if db._table_exists('job_completions'):
        return
    try:
        logger.info("Migrating: creating job_completions table...")
        db.conn.execute("""
            CREATE TABLE IF NOT EXISTS job_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER,
                completed_at TEXT NOT NULL DEFAULT (datetime('now')),
                worker_session_id INTEGER
            )
        """)
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_completions_at "
            "ON job_completions(completed_at)"
        )
        db.conn.commit()
        logger.info("job_completions table created")
    except Exception as e:
        logger.warning(f"job_completions migration failed (non-fatal): {e}")


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
        # Backfill: mark files associated with permanently failed jobs
        cursor = db.conn.execute("""
            UPDATE files SET processing_status = 'failed',
                processing_error = jq.error_message
            FROM (
                SELECT file_id, error_message FROM job_queue
                WHERE status = 'failed'
                  AND (error_code IN ('FILE_NOT_FOUND', 'PARSE_FAILED')
                       OR retry_count >= 3)
            ) AS jq
            WHERE files.id = jq.file_id
        """)
        backfilled = cursor.rowcount
        db.conn.commit()
        logger.info(
            f"processing_status columns added to files "
            f"({backfilled} permanently failed files backfilled)"
        )


def migrate_work_requests(db):
    """Create work_requests and work_subtasks tables, add FK columns to job_queue."""
    # 1) work_requests table
    if not db._table_exists('work_requests'):
        try:
            logger.info("Migrating: creating work_requests table...")
            db.conn.execute("""
                CREATE TABLE IF NOT EXISTS work_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source_path TEXT,
                    status TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'processing', 'completed', 'paused', 'cancelled')),
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    total_files INTEGER NOT NULL DEFAULT 0,
                    completed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    started_at TEXT,
                    completed_at TEXT
                )
            """)
            db.conn.commit()
            logger.info("work_requests table created")
        except Exception as e:
            logger.warning(f"work_requests migration failed (non-fatal): {e}")

    # 2) work_subtasks table
    if not db._table_exists('work_subtasks'):
        try:
            logger.info("Migrating: creating work_subtasks table...")
            db.conn.execute("""
                CREATE TABLE IF NOT EXISTS work_subtasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    work_request_id INTEGER NOT NULL REFERENCES work_requests(id) ON DELETE CASCADE,
                    folder_path TEXT NOT NULL,
                    folder_name TEXT NOT NULL,
                    total_files INTEGER NOT NULL DEFAULT 0,
                    completed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(work_request_id, folder_path)
                )
            """)
            db.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_work_subtasks_wr "
                "ON work_subtasks(work_request_id)"
            )
            db.conn.commit()
            logger.info("work_subtasks table created")
        except Exception as e:
            logger.warning(f"work_subtasks migration failed (non-fatal): {e}")

    # 3) job_queue FK columns
    if db._table_exists('job_queue'):
        try:
            db.conn.execute("SELECT work_request_id FROM job_queue LIMIT 1")
        except Exception:
            try:
                logger.info("Migrating: adding work_request_id/work_subtask_id to job_queue...")
                db.conn.execute(
                    "ALTER TABLE job_queue ADD COLUMN work_request_id INTEGER "
                    "REFERENCES work_requests(id) ON DELETE SET NULL"
                )
                db.conn.execute(
                    "ALTER TABLE job_queue ADD COLUMN work_subtask_id INTEGER "
                    "REFERENCES work_subtasks(id) ON DELETE SET NULL"
                )
                db.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_job_queue_work_request "
                    "ON job_queue(work_request_id)"
                )
                db.conn.commit()
                logger.info("work_request_id/work_subtask_id columns added to job_queue")
            except Exception as e:
                logger.warning(f"job_queue work_request columns migration failed (non-fatal): {e}")

    # 4) work_requests.started_at column (added post-initial migration)
    if db._table_exists('work_requests'):
        try:
            db.conn.execute("SELECT started_at FROM work_requests LIMIT 1")
        except Exception:
            try:
                logger.info("Migrating: adding started_at to work_requests...")
                db.conn.execute(
                    "ALTER TABLE work_requests ADD COLUMN started_at TEXT"
                )
                db.conn.commit()
                logger.info("started_at column added to work_requests")
            except Exception as e:
                logger.warning(f"work_requests started_at migration failed (non-fatal): {e}")


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


def migrate_job_queue_unique_file_id(db):
    """Add partial unique index on job_queue(file_id) for active jobs."""
    try:
        db.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_job_queue_file_id_active "
            "ON job_queue(file_id) WHERE status IN ('pending', 'assigned', 'processing')"
        )
        db.conn.commit()
    except Exception:
        pass  # Index may already exist


def migrate_job_queue_archived_at(db):
    """Add archived_at column for job history soft delete."""
    if not db._table_exists('job_queue'):
        return
    try:
        db.conn.execute("SELECT archived_at FROM job_queue LIMIT 1")
    except Exception:
        logger.info("Migrating: adding archived_at to job_queue...")
        db.conn.execute(
            "ALTER TABLE job_queue ADD COLUMN archived_at TEXT DEFAULT NULL"
        )
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_queue_archived "
            "ON job_queue(archived_at)"
        )
        db.conn.commit()
        logger.info("archived_at column added to job_queue")


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


def migrate_phase_completed_vv_mv(db):
    """Split phase_completed 'embed' key into separate 'vv' and 'mv' keys.
    Also sync with actual vec_files/vec_text data."""
    if not db._table_exists('job_queue'):
        return
    import json as _json
    cursor = db.conn.cursor()

    # 1. Convert embed → vv + mv
    cursor.execute("SELECT id, phase_completed FROM job_queue WHERE phase_completed LIKE '%embed%'")
    rows = cursor.fetchall()
    if rows:
        for jid, pc_str in rows:
            pc = _json.loads(pc_str or '{}')
            if 'embed' in pc:
                val = pc.pop('embed')
                pc['vv'] = val
                pc['mv'] = val
                cursor.execute("UPDATE job_queue SET phase_completed = ? WHERE id = ?",
                             (_json.dumps(pc), jid))
        db.conn.commit()
        logger.info(f"Migrated {len(rows)} jobs: embed → vv + mv")

    # 2. Sync vv/mv with actual DB data (vec_files/vec_text)
    cursor.execute("""SELECT jq.id, jq.file_id, jq.phase_completed FROM job_queue jq
                      WHERE jq.file_id IS NOT NULL""")
    rows = cursor.fetchall()
    fixed = 0
    for jid, fid, pc_str in rows:
        pc = _json.loads(pc_str or '{}')
        cursor.execute("SELECT COUNT(*) FROM vec_files WHERE file_id = ?", (fid,))
        actual_vv = cursor.fetchone()[0] > 0
        cursor.execute("SELECT COUNT(*) FROM vec_text WHERE file_id = ?", (fid,))
        actual_mv = cursor.fetchone()[0] > 0
        if pc.get('vv') != actual_vv or pc.get('mv') != actual_mv:
            pc['vv'] = actual_vv
            pc['mv'] = actual_mv
            cursor.execute("UPDATE job_queue SET phase_completed = ? WHERE id = ?",
                         (_json.dumps(pc), jid))
            fixed += 1
    if fixed:
        db.conn.commit()
        logger.info(f"Synced {fixed} jobs vv/mv with actual DB data")


# ──────────────────────────────────────────────────────────────
# Orchestrator: run all migrations in order
# ──────────────────────────────────────────────────────────────

def run_migrations(db, *, existing_db: bool = True):
    """Run all migrations in the correct order.

    Args:
        db: SQLiteDB instance (access db.conn, db._table_exists(), etc.)
        existing_db: True if files table already exists (upgrade path),
                     False if empty database (fresh install path).
    """
    if existing_db:
        # Upgrade path: existing DB with files table
        migrate_folder_columns(db)
        migrate_v3_columns(db)
        migrate_content_hash(db)
        migrate_structure_table(db)
        migrate_uploaded_by(db)
        migrate_preview_only(db)
        migrate_backfill_storage_root(db)
        db._ensure_system_meta()
        db._ensure_fts()
        migrate_auth_tables(db)
        migrate_worker_tokens(db)
        migrate_worker_sessions(db)
        migrate_parse_ahead_columns(db)
        migrate_worker_session_tracking(db)
        migrate_worker_session_overrides(db)
        migrate_worker_resources_json(db)
        migrate_mc_completed_at(db)
        migrate_backfill_parse_status(db)
        migrate_users_email_nullable(db)
        migrate_users_firebase_uid(db)
        migrate_error_code(db)
        migrate_file_ready(db)
        migrate_job_completions(db)
        migrate_files_processing_status(db)
        migrate_work_requests(db)
        migrate_members_table(db)
        migrate_drop_fts_update_trigger(db)
        migrate_job_queue_unique_file_id(db)
        migrate_job_queue_archived_at(db)
        migrate_search_logs(db)
        migrate_worker_phase_tracking(db)
        migrate_phase_completed_vv_mv(db)
    else:
        # Fresh install path: empty DB, schema just initialized
        db._ensure_system_meta()
        migrate_auth_tables(db)
        migrate_worker_tokens(db)
        migrate_worker_sessions(db)
        migrate_parse_ahead_columns(db)
        migrate_worker_session_tracking(db)
        migrate_worker_session_overrides(db)
        migrate_worker_resources_json(db)
        migrate_mc_completed_at(db)
        migrate_backfill_parse_status(db)
        migrate_error_code(db)
        migrate_file_ready(db)
        migrate_job_completions(db)
        migrate_files_processing_status(db)
        migrate_work_requests(db)
        migrate_members_table(db)
