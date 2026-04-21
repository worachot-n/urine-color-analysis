-- =============================================================================
-- Urine Analysis System — PostgreSQL database setup
-- Usage:
--   psql -U postgres -f configs/setup_db.sql
-- =============================================================================

-- 1. Database + user
CREATE DATABASE urine_db ENCODING 'UTF8';
CREATE USER "user" WITH PASSWORD 'pass';
GRANT ALL PRIVILEGES ON DATABASE urine_db TO "user";

-- 2. Connect to the new database
\c urine_db

-- 3. Schema privileges
GRANT ALL ON SCHEMA public TO "user";

-- =============================================================================
-- Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS people (
    id           SERIAL PRIMARY KEY,
    full_name    VARCHAR NOT NULL,
    personnel_id VARCHAR UNIQUE,
    department   VARCHAR
);
CREATE INDEX IF NOT EXISTS ix_people_personnel_id ON people (personnel_id);

CREATE TABLE IF NOT EXISTS trays (
    id             SERIAL PRIMARY KEY,
    tray_name      VARCHAR,
    is_active      BOOLEAN NOT NULL DEFAULT FALSE,
    rows           INTEGER DEFAULT 13,
    cols           INTEGER DEFAULT 15,
    total_slots    INTEGER DEFAULT 195,
    dimension_info VARCHAR DEFAULT '13x15',
    created_at     TIMESTAMP DEFAULT NOW(),
    layout_json    JSONB
);

CREATE TABLE IF NOT EXISTS scan_sessions (
    id                   SERIAL PRIMARY KEY,
    tray_id              INTEGER NOT NULL REFERENCES trays(id),
    scanned_at           TIMESTAMP DEFAULT NOW(),
    image_raw_path       VARCHAR,
    image_annotated_path VARCHAR,
    color_0              INTEGER DEFAULT 0,
    color_1              INTEGER DEFAULT 0,
    color_2              INTEGER DEFAULT 0,
    color_3              INTEGER DEFAULT 0,
    color_4              INTEGER DEFAULT 0,
    error_count          INTEGER DEFAULT 0,
    is_clean             BOOLEAN DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS ix_scan_sessions_tray_id    ON scan_sessions (tray_id);
CREATE INDEX IF NOT EXISTS ix_scan_sessions_scanned_at ON scan_sessions (scanned_at);

CREATE TABLE IF NOT EXISTS test_slots (
    id             SERIAL PRIMARY KEY,
    session_id     INTEGER NOT NULL REFERENCES scan_sessions(id),
    position_index INTEGER NOT NULL,
    color_result   INTEGER,
    is_error       BOOLEAN DEFAULT FALSE,
    person_id      INTEGER REFERENCES people(id)
);
CREATE INDEX IF NOT EXISTS ix_test_slots_session_id ON test_slots (session_id);

-- Only one tray may be active at a time
CREATE UNIQUE INDEX IF NOT EXISTS uq_trays_single_active
    ON trays (is_active)
    WHERE is_active = TRUE;
