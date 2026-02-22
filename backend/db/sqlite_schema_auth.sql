-- Imagine SQLite Auth & Job Queue Schema Extension
-- Added for client-server architecture (v4.0)
-- Applied via auto-migration in sqlite_client.py

-- ═══════════════════════════════════════════════════════════════
-- Users & Authentication
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
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

CREATE TABLE IF NOT EXISTS worker_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_hash TEXT UNIQUE NOT NULL,  -- SHA256 of token secret
    name TEXT NOT NULL,               -- Human-readable label (e.g. "김철수 GPU PC")
    created_by INTEGER REFERENCES users(id) ON DELETE CASCADE,
    is_active INTEGER DEFAULT 1,      -- boolean: 0/1
    expires_at TEXT,                   -- ISO 8601
    created_at TEXT DEFAULT (datetime('now')),
    last_used_at TEXT
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
    -- Command queue (server → worker, consumed on heartbeat)
    pending_command TEXT DEFAULT NULL
        CHECK (pending_command IN (NULL, 'stop', 'pause', 'block')),
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

    -- Phase-level tracking (JSON)
    phase_completed TEXT DEFAULT '{"parse":false,"vision":false,"embed":false}',

    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Priority (higher = first)
    priority INTEGER DEFAULT 0,

    -- Parse-ahead (server-side pre-parsing for worker optimization)
    parse_status TEXT DEFAULT NULL
        CHECK (parse_status IN (NULL, 'pending', 'parsing', 'parsed', 'failed')),
    parsed_metadata TEXT DEFAULT NULL,   -- Phase P result JSON (metadata + thumb_path + mc_raw)
    parsed_at TEXT DEFAULT NULL,

    -- Timestamps
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- ═══════════════════════════════════════════════════════════════
-- Indexes
-- ═══════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_invite_codes_code ON invite_codes(code);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_hash ON refresh_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_worker_tokens_hash ON worker_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
CREATE INDEX IF NOT EXISTS idx_job_queue_assigned ON job_queue(assigned_to, status);
CREATE INDEX IF NOT EXISTS idx_job_queue_priority ON job_queue(priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_worker_sessions_user ON worker_sessions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_worker_sessions_status ON worker_sessions(status);
CREATE INDEX IF NOT EXISTS idx_job_queue_parse_status
    ON job_queue(parse_status, priority DESC, created_at ASC);
