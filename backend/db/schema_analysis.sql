-- Analysis Job System v1 — replaces job_queue + work_requests + work_subtasks
-- See docs/specs/analysis_job_system_v1.md

-- ── Analysis Jobs (user-created analysis requests) ──────────────

CREATE TABLE IF NOT EXISTS analysis_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                      -- display name ("크랑베르무")
    source_path TEXT NOT NULL,               -- folder path ("webdav://..." or local)
    status TEXT DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'cancelled', 'archived')),
    network_status TEXT DEFAULT 'ok'
        CHECK (network_status IN ('ok', 'degraded', 'paused')),
    total_files INTEGER NOT NULL DEFAULT 0,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT (datetime('now')),
    started_at TEXT,                             -- first task claimed
    completed_at TEXT,

    -- Completion snapshot (written once when status → 'completed')
    completed_files INTEGER,
    avg_mc_speed REAL,                          -- files/min
    avg_vv_speed REAL,
    avg_mv_speed REAL,
    total_elapsed_s REAL,                       -- wall time (started → completed)
    worker_count INTEGER,                       -- distinct workers used
    worker_stats_json TEXT                      -- per-worker breakdown (JSON)
    -- Example: [{"worker_id":3,"name":"Mac-M5","mc":1500,"vv":2904,"mv":0,"mc_speed":7.8,"vv_speed":81.0},
    --           {"worker_id":5,"name":"RTX-4090","mc":1404,"vv":0,"mv":2904,"mc_speed":12.0,"mv_speed":119.0}]
);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status
    ON analysis_jobs(status);


-- ── File Tasks (per-file phase tracking) ────────────────────────

CREATE TABLE IF NOT EXISTS file_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_job_id INTEGER NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,

    -- Preparation phases
    download_status TEXT DEFAULT 'pending'
        CHECK (download_status IN ('pending', 'assigned', 'done', 'failed', 'n/a')),
    download_assigned_to INTEGER,
    download_started_at TEXT,
    download_completed_at TEXT,

    parse_status TEXT DEFAULT 'pending'
        CHECK (parse_status IN ('pending', 'assigned', 'done', 'failed')),
    parse_assigned_to INTEGER,
    parse_started_at TEXT,
    parse_completed_at TEXT,

    -- AI phases (MC and VV are independent; MV depends on MC)
    mc_status TEXT DEFAULT 'pending'
        CHECK (mc_status IN ('pending', 'assigned', 'done', 'failed')),
    mc_assigned_to INTEGER,
    mc_started_at TEXT,
    mc_completed_at TEXT,
    mc_elapsed_s REAL,              -- actual inference time (worker-measured)

    vv_status TEXT DEFAULT 'pending'
        CHECK (vv_status IN ('pending', 'assigned', 'done', 'failed')),
    vv_assigned_to INTEGER,
    vv_started_at TEXT,
    vv_completed_at TEXT,
    vv_elapsed_s REAL,              -- actual inference time (worker-measured)

    mv_status TEXT DEFAULT 'pending'
        CHECK (mv_status IN ('pending', 'assigned', 'done', 'failed')),
    mv_assigned_to INTEGER,
    mv_started_at TEXT,
    mv_completed_at TEXT,
    mv_elapsed_s REAL,              -- actual inference time (worker-measured)

    -- Error / retry
    error_message TEXT,
    error_code TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Dismiss (acknowledged permanent failure)
    dismissed_at TEXT,              -- NULL = active, non-NULL = user acknowledged

    -- Metadata
    priority INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),

    UNIQUE(analysis_job_id, file_id)
);

-- Claim indexes: one per phase for fast worker claim queries
CREATE INDEX IF NOT EXISTS idx_ft_download
    ON file_tasks(analysis_job_id, download_status)
    WHERE download_status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_ft_parse
    ON file_tasks(analysis_job_id, parse_status)
    WHERE parse_status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_ft_mc
    ON file_tasks(analysis_job_id, mc_status, parse_status)
    WHERE mc_status IN ('pending', 'failed') AND parse_status = 'done';

CREATE INDEX IF NOT EXISTS idx_ft_vv
    ON file_tasks(analysis_job_id, vv_status, parse_status)
    WHERE vv_status IN ('pending', 'failed') AND parse_status = 'done';

CREATE INDEX IF NOT EXISTS idx_ft_mv
    ON file_tasks(analysis_job_id, mv_status, mc_status)
    WHERE mv_status IN ('pending', 'failed') AND mc_status = 'done';
