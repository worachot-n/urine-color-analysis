-- =============================================================================
-- Urine Analysis System — PostgreSQL database setup
-- Usage:
--   psql -U postgres -f configs/setup_db.sql
-- =============================================================================

-- 1. Database + user
CREATE DATABASE urine_db ENCODING 'UTF8' TEMPLATE template0;
CREATE USER "user" WITH PASSWORD 'pass';
GRANT ALL PRIVILEGES ON DATABASE urine_db TO "user";

-- 2. Connect to the new database
\c urine_db

-- 3. Schema privileges
GRANT ALL ON SCHEMA public TO "user";

-- =============================================================================
-- Tables
-- =============================================================================

-- 1. สร้างตาราง People (Master)
CREATE TABLE IF NOT EXISTS people (
    id           SERIAL PRIMARY KEY,
    full_name    VARCHAR NOT NULL,
    personnel_id VARCHAR UNIQUE,
    department   VARCHAR
);
CREATE INDEX IF NOT EXISTS ix_people_personnel_id ON people (personnel_id);

-- 2. สร้างตาราง Trays
CREATE TABLE IF NOT EXISTS trays (
    id             SERIAL PRIMARY KEY,
    tray_name      VARCHAR,
    is_active      BOOLEAN NOT NULL DEFAULT FALSE,
    rows           INTEGER DEFAULT 13,
    cols           INTEGER DEFAULT 15,
    total_slots    INTEGER DEFAULT 195,
    dimension_info VARCHAR DEFAULT '13x15',
    created_at     TIMESTAMP DEFAULT NOW(),
    layout_json    JSONB,
    grid_json      JSONB    -- {calibration_date, corners, grid_pts[14][17][2], sample_centres, ref_centres, grid_spacing}
);

-- บังคับให้ Active ได้ทีละอัน
CREATE UNIQUE INDEX IF NOT EXISTS uq_trays_single_active
    ON trays (is_active)
    WHERE is_active = TRUE;

-- ==================================================

-- 3. สร้าง Scan Sessions
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

-- 4. สร้าง Test Slots (ตารางที่เชื่อมโยงทุกอย่าง)
CREATE TABLE IF NOT EXISTS test_slots (
    id             SERIAL PRIMARY KEY,
    session_id     INTEGER NOT NULL REFERENCES scan_sessions(id),
    position_index INTEGER NOT NULL,
    color_result   INTEGER,
    is_error       BOOLEAN DEFAULT FALSE,
    person_id      INTEGER REFERENCES people(id)
);

-- =============================================================================
-- Seed Data
-- =============================================================================

-- Initial tray — grid_json left NULL, fill via /settings calibration UI
INSERT INTO trays (tray_name, is_active, rows, cols, total_slots, dimension_info, layout_json, grid_json)
VALUES (
    'Standard Tray 13x15',
    TRUE, 13, 15, 195, '13x15',
    '{
      "1":  "R01_0", "2":  "R02_0", "3":  "R03_0", "4":  "R01_1", "5":  "R02_1", "6":  "R03_1", "7":  "R01_2", "8":  "R02_2", "9":  "R03_2", "10": "R01_3", "11": "R02_3", "12": "R03_3", "13": "R01_4", "14": "R02_4", "15": "R03_4",
      "16": "A11_0", "17": "A12_0", "18": "A13_0", "19": "A11_1", "20": "A12_1", "21": "A13_1", "22": "A11_2", "23": "A12_2", "24": "A13_2", "25": "A11_3", "26": "A12_3", "27": "A13_3", "28": "A11_4", "29": "A12_4", "30": "A13_4",
      "31": "A14_0", "32": "A15_0", "33": "A16_0", "34": "A14_1", "35": "A15_1", "36": "A16_1", "37": "A14_2", "38": "A15_2", "39": "A16_2", "40": "A14_3", "41": "A15_3", "42": "A16_3", "43": "A14_4", "44": "A15_4", "45": "A16_4",
      "46": "A17_0", "47": "A18_0", "48": "A19_0", "49": "A17_1", "50": "A18_1", "51": "A19_1", "52": "A17_2", "53": "A18_2", "54": "A19_2", "55": "A17_3", "56": "A18_3", "57": "A19_3", "58": "A17_4", "59": "A18_4", "60": "A19_4",
      "61": "A21_0", "62": "A22_0", "63": "A23_0", "64": "A21_1", "65": "A22_1", "66": "A23_1", "67": "A21_2", "68": "A22_2", "69": "A23_2", "70": "A21_3", "71": "A22_3", "72": "A23_3", "73": "A21_4", "74": "A22_4", "75": "A23_4",
      "76": "A24_0", "77": "A25_0", "78": "A26_0", "79": "A24_1", "80": "A25_1", "81": "A26_1", "82": "A24_2", "83": "A25_2", "84": "A26_2", "85": "A24_3", "86": "A25_3", "87": "A26_3", "88": "A24_4", "89": "A25_4", "90": "A26_4",
      "91": "A27_0", "92": "A28_0", "93": "A29_0", "94": "A27_1", "95": "A28_1", "96": "A29_1", "97": "A27_2", "98": "A28_2", "99": "A29_2", "100": "A27_3", "101": "A28_3", "102": "A29_3", "103": "A27_4", "104": "A28_4", "105": "A29_4",
      "106": "A31_0", "107": "A32_0", "108": "A33_0", "109": "A31_1", "110": "A32_1", "111": "A33_1", "112": "A31_2", "113": "A32_2", "114": "A33_2", "115": "A31_3", "116": "A32_3", "117": "A33_3", "118": "A31_4", "119": "A32_4", "120": "A33_4",
      "121": "A34_0", "122": "A35_0", "123": "A36_0", "124": "A34_1", "125": "A35_1", "126": "A36_1", "127": "A34_2", "128": "A35_2", "129": "A36_2", "130": "A34_3", "131": "A35_3", "132": "A36_3", "133": "A34_4", "134": "A35_4", "135": "A36_4",
      "136": "A37_0", "137": "A38_0", "138": "A39_0", "139": "A37_1", "140": "A38_1", "141": "A39_1", "142": "A37_2", "143": "A38_2", "144": "A39_2", "145": "A37_3", "146": "A38_3", "147": "A39_3", "148": "A37_4", "149": "A38_4", "150": "A39_4",
      "151": "A41_0", "152": "A42_0", "153": "A43_0", "154": "A41_1", "155": "A42_1", "156": "A43_1", "157": "A41_2", "158": "A42_2", "159": "A43_2", "160": "A41_3", "161": "A42_3", "162": "A43_3", "163": "A41_4", "164": "A42_4", "165": "A43_4",
      "166": "A44_0", "167": "A45_0", "168": "A46_0", "169": "A44_1", "170": "A45_1", "171": "A46_1", "172": "A44_2", "173": "A45_2", "174": "A46_2", "175": "A44_3", "176": "A45_3", "177": "A46_3", "178": "A44_4", "179": "A45_4", "180": "A46_4",
      "181": "A47_0", "182": "A48_0", "183": "A49_0", "184": "A47_1", "185": "A48_1", "186": "A49_1", "187": "A47_2", "188": "A48_2", "189": "A49_2", "190": "A47_3", "191": "A48_3", "192": "A49_3", "193": "A47_4", "194": "A48_4", "195": "A49_4"
    }'::JSONB,
    NULL   -- grid_json: calibrate via /settings after first startup
);
