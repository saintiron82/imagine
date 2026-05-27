-- Imagine SQLite Auth & Worker Schema Extension
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
    origin TEXT DEFAULT 'headless'
        CHECK (origin IN ('server-local', 'client-launched', 'headless')),
    launcher TEXT DEFAULT 'cli'
        CHECK (launcher IN ('server', 'electron', 'cli', 'service', 'cloud')),
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
-- Indexes
-- ═══════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_firebase_uid ON users(firebase_uid);
CREATE INDEX IF NOT EXISTS idx_invite_codes_code ON invite_codes(code);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_hash ON refresh_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_worker_sessions_user ON worker_sessions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_worker_sessions_status ON worker_sessions(status);

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

-- ═══════════════════════════════════════════════════════════════
-- Phase 8 — Audit Log
-- Append-only ledger of security-relevant events.
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,                          -- enum below
    actor_user_id INTEGER,                        -- nullable: system events
    actor_username TEXT,
    target_kind TEXT,                             -- 'worker'/'session'/'group'/'token'/etc
    target_id TEXT,                               -- free-form id (worker name, session id, ...)
    ip_address TEXT,
    detail TEXT,                                  -- short human-readable
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_event ON audit_log(event);
CREATE INDEX IF NOT EXISTS idx_audit_log_actor ON audit_log(actor_user_id);

-- ═══════════════════════════════════════════════════════════════
-- Phase 8 — Worker Enrollment Tokens
-- Short-lived, single-use (or limit-bound), revocable tokens that
-- attach a worker. These are separate from access/refresh tokens so
-- they never grant admin API access.
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS worker_enrollment_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash TEXT UNIQUE NOT NULL,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    worker_name TEXT,
    max_uses INTEGER DEFAULT 1,
    use_count INTEGER DEFAULT 0,
    expires_at TEXT NOT NULL,
    revoked INTEGER DEFAULT 0,                    -- boolean 0/1
    created_at TEXT DEFAULT (datetime('now')),
    last_used_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_enrollment_hash ON worker_enrollment_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_enrollment_expires ON worker_enrollment_tokens(expires_at);

-- ═══════════════════════════════════════════════════════════════
-- Phase D — Search Feedback (irrelevant labels)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS search_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    file_id INTEGER NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('irrelevant')),
    user_id INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_search_feedback_query ON search_feedback(query);
CREATE INDEX IF NOT EXISTS idx_search_feedback_file ON search_feedback(file_id);
