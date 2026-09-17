-- database/schema.sql — Conductor v2
--
-- Changes from v1:
--   tasks   → added required_gpus, required_mem_gb
--   workers → added total_gpus, total_mem_gb, available_gpus, available_mem_gb
--
-- Run:
--   psql -U $(whoami) -d postgres -c "CREATE DATABASE conductorv2;"
--   psql -U $(whoami) -d conductorv2 -f database/schema.sql

CREATE TYPE task_status AS ENUM (
    'pending',
    'claimed',
    'running',
    'completed',
    'failed'
);

CREATE TABLE IF NOT EXISTS tasks (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    status          task_status NOT NULL DEFAULT 'pending',
    payload         JSONB NOT NULL DEFAULT '{}',
    priority        INTEGER NOT NULL DEFAULT 5,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    retry_count     INTEGER NOT NULL DEFAULT 0,

    -- Resource requirements — what this task needs to run
    required_gpus   INTEGER NOT NULL DEFAULT 1,
    required_mem_gb FLOAT   NOT NULL DEFAULT 4,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    worker_id       VARCHAR(255),
    last_heartbeat  TIMESTAMPTZ,
    claimed_at      TIMESTAMPTZ,
    result          JSONB,
    error_message   TEXT,

    CONSTRAINT retry_count_within_max CHECK (retry_count <= max_retries)
);

-- Index for the resource-aware claim query:
-- "give me the highest-priority pending task that fits my available resources"
CREATE INDEX IF NOT EXISTS idx_tasks_claim
    ON tasks (status, priority DESC, created_at ASC, required_gpus, required_mem_gb);

CREATE INDEX IF NOT EXISTS idx_tasks_heartbeat
    ON tasks (last_heartbeat)
    WHERE status IN ('claimed', 'running');


CREATE TABLE IF NOT EXISTS workers (
    worker_id           VARCHAR(255) PRIMARY KEY,
    hostname            VARCHAR(255),
    status              VARCHAR(50) NOT NULL DEFAULT 'idle',

    -- Total resources this worker has (fixed at registration)
    total_gpus          INTEGER NOT NULL DEFAULT 1,
    total_mem_gb        FLOAT   NOT NULL DEFAULT 8,

    -- Available resources right now (updated as tasks are claimed/released)
    -- Starts equal to total; decremented on claim, incremented on complete
    available_gpus      INTEGER NOT NULL DEFAULT 1,
    available_mem_gb    FLOAT   NOT NULL DEFAULT 8,

    current_task_id     INTEGER REFERENCES tasks(id),
    registered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS attempts (
    id               SERIAL PRIMARY KEY,
    task_id          INTEGER NOT NULL REFERENCES tasks(id),
    worker_id        VARCHAR(255) NOT NULL,
    attempt_number   INTEGER NOT NULL,
    status           VARCHAR(50) NOT NULL,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at      TIMESTAMPTZ,
    epochs_completed INTEGER DEFAULT 0,
    error_message    TEXT,
    UNIQUE (task_id, attempt_number)
);

CREATE INDEX IF NOT EXISTS idx_attempts_task_id ON attempts (task_id);


-- Auto-update updated_at on tasks
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();