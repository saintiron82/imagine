-- Imagine SQLite Auth & Job Queue Schema Extension
-- Added for client-server architecture (v4.0)
-- Applied via auto-migration in sqlite_client.py

-- ═══════════════════════════════════════════════════════════════
-- Users & Authentication
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,  -- nullable (email is optional for group-based registration)
    firebase_uid TEXT,  -- Firebase Auth UID (for 2-layer auth: Firebase identity + server password)
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    is_active INTEGER DEFAULT 1,  -- boolean: 0/1
    created_at TEXT DEFAULT (datetime('now')),
    last_login_at TEXT,
    -- Quotas
    quota_files_per_day INTEGER DEFAULT 1000,
    quota_search_per_min INTEGER DEFAULT 60
);

CREATE TABLE IF NOT EXISTS invite_codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    max_uses INTEGER DEFAULT 1,
    use_count INTEGER DEFAULT 0,
    expires_at TEXT,  -- ISO 8601
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS invite_uses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invite_id INTEGER NOT NULL REFERENCES invite_codes(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    used_at TEXT DEFAULT (datetime('now')),
    UNIQUE(invite_id, user_id)
);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT UNIQUE NOT NULL,  -- SHA256 of refresh token
    expires_at TEXT NOT NULL,         -- ISO 8601
    created_at TEXT DEFAULT (datetime('now')),
    revoked INTEGER DEFAULT 0        -- boolean: 0/1
);

-- ═══════════════════════════════════════════════════════════════
-- Worker Sessions (live monitoring & control)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS worker_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    worker_name TEXT NOT NULL,
    hostname TEXT,
    -- Status
    status TEXT DEFAULT 'online'
        CHECK (status IN ('online', 'offline', 'blocked')),
    -- Metrics (reported by worker via heartbeat)
    batch_capacity INTEGER DEFAULT 5,
    jobs_completed INTEGER DEFAULT 0,
    jobs_failed INTEGER DEFAULT 0,
    current_job_id INTEGER,
    current_file TEXT,
    current_phase TEXT,
    -- Resource metrics (JSON blob from resource_monitor.collect_metrics())
    resources_json TEXT DEFAULT NULL,
    -- Command queue (server → worker, consumed on heartbeat)
    pending_command TEXT DEFAULT NULL
        CHECK (pending_command IN (NULL, 'stop', 'pause', 'block')),
    -- Dynamic mode tracking (server-assigned, reset on phase switch)
    assigned_mode TEXT DEFAULT NULL,              -- Current server-assigned mode (mc/vv/mv/parse)
    phase_job_count INTEGER DEFAULT 0,            -- Jobs completed in current assigned_mode (reset on switch)
    -- Per-worker overrides (admin-controlled, applied via heartbeat)
    processing_mode_override TEXT DEFAULT NULL,   -- NULL = global config, "mc" | "vv" | "mv"
    batch_capacity_override INTEGER DEFAULT NULL,  -- NULL = worker default
    -- Timestamps
    connected_at TEXT DEFAULT (datetime('now')),
    last_heartbeat TEXT DEFAULT (datetime('now')),
    disconnected_at TEXT
);

-- ═══════════════════════════════════════════════════════════════
-- Job Queue (distributed processing)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS job_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'assigned', 'processing', 'completed', 'failed', 'cancelled')),
    assigned_to INTEGER REFERENCES users(id) ON DELETE SET NULL,
    assigned_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    mc_completed_at TEXT,          -- MC(Vision) completion timestamp (throughput measurement)
    vv_completed_at TEXT,          -- VV(SigLIP2) completion timestamp
    mv_completed_at TEXT,          -- MV(Qwen3-Embedding) completion timestamp

    -- Phase-level tracking (JSON)
    phase_completed TEXT DEFAULT '{"parse":false,"vision":false,"embed":false}',

    -- Error handling
    error_message TEXT,
    error_code TEXT DEFAULT NULL,      -- structured error code (THUMB_MISSING, VLM_FAILED, etc.)
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Priority (higher = first)
    priority INTEGER DEFAULT 0,

    -- Per-worker tracking (for multi-worker throughput)
    worker_session_id INTEGER REFERENCES worker_sessions(id) ON DELETE SET NULL,

    -- Folder-level work tracking
    work_request_id INTEGER REFERENCES work_requests(id) ON DELETE SET NULL,
    work_subtask_id INTEGER REFERENCES work_subtasks(id) ON DELETE SET NULL,

    -- File readiness gate (2-stage pipeline: preparation → processing)
    -- 1 = file is locally available for processing (default for local files)
    -- 0 = file needs preparation (e.g., WebDAV download pending)
    file_ready INTEGER NOT NULL DEFAULT 1,

    -- Parse-ahead (server-side pre-parsing for worker optimization)
    parse_status TEXT DEFAULT NULL
        CHECK (parse_status IN (NULL, 'pending', 'parsing', 'parsed', 'failed', 'backfill')),
    parsed_metadata TEXT DEFAULT NULL,   -- Phase P result JSON (metadata + thumb_path + mc_raw)
    parsed_at TEXT DEFAULT NULL,

    -- Timestamps
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),

    -- Soft delete for history (NULL = active in queue, NOT NULL = archived)
    archived_at TEXT DEFAULT NULL
);

-- ═══════════════════════════════════════════════════════════════
-- Indexes
-- ═══════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_firebase_uid ON users(firebase_uid);
CREATE INDEX IF NOT EXISTS idx_invite_codes_code ON invite_codes(code);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_hash ON refresh_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
CREATE INDEX IF NOT EXISTS idx_job_queue_assigned ON job_queue(assigned_to, status);
CREATE INDEX IF NOT EXISTS idx_job_queue_priority ON job_queue(priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_worker_sessions_user ON worker_sessions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_worker_sessions_status ON worker_sessions(status);
CREATE INDEX IF NOT EXISTS idx_job_queue_file_ready
    ON job_queue(file_ready, status, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_job_queue_parse_status
    ON job_queue(parse_status, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_job_queue_mc_completed
    ON job_queue(mc_completed_at);
CREATE INDEX IF NOT EXISTS idx_job_queue_archived
    ON job_queue(archived_at);

-- Prevent duplicate active jobs for the same file
CREATE UNIQUE INDEX IF NOT EXISTS idx_job_queue_file_id_active
    ON job_queue(file_id) WHERE status IN ('pending', 'assigned', 'processing');

-- ═══════════════════════════════════════════════════════════════
-- Members (Firebase Auth based group membership)
-- Replaces users table for Firebase Auth integration
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firebase_uid TEXT NOT NULL,
    email TEXT NOT NULL,
    display_name TEXT,
    role TEXT CHECK (role IN ('admin', 'user')) DEFAULT 'user',
    is_active INTEGER DEFAULT 1,
    invited_by INTEGER REFERENCES members(id),
    joined_at TEXT DEFAULT (datetime('now')),
    last_seen_at TEXT,
    quota_files_per_day INTEGER DEFAULT 1000,
    quota_search_per_min INTEGER DEFAULT 60
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_members_firebase_uid ON members(firebase_uid);
CREATE INDEX IF NOT EXISTS idx_members_email ON members(email);

-- ═══════════════════════════════════════════════════════════════
-- Work Requests (folder-level work tracking)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS work_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_path TEXT,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'completed', 'paused', 'cancelled')),
    sort_order INTEGER NOT NULL DEFAULT 0,

    -- Denormalized counters
    total_files INTEGER NOT NULL DEFAULT 0,
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,

    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS work_subtasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_request_id INTEGER NOT NULL REFERENCES work_requests(id) ON DELETE CASCADE,
    folder_path TEXT NOT NULL,
    folder_name TEXT NOT NULL,

    -- Counters
    total_files INTEGER NOT NULL DEFAULT 0,
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,

    UNIQUE(work_request_id, folder_path)
);

CREATE INDEX IF NOT EXISTS idx_work_subtasks_wr ON work_subtasks(work_request_id);
CREATE INDEX IF NOT EXISTS idx_job_queue_work_request ON job_queue(work_request_id);

-- ═══════════════════════════════════════════════════════════════
-- Search Logs
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'triaxis',       -- triaxis/vector/text_vector/fts
    result_count INTEGER DEFAULT 0,
    elapsed_ms INTEGER DEFAULT 0,
    username TEXT,
    ip_address TEXT,
    filters TEXT,                                -- JSON
    threshold REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_search_logs_created ON search_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_search_logs_user ON search_logs(username);
