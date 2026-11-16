CREATE TABLE IF NOT EXISTS wg_users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(120) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS wg_inspections (
    id BIGSERIAL PRIMARY KEY,
    wafer_name TEXT NOT NULL,
    predicted_class VARCHAR(16) NOT NULL CHECK (predicted_class IN ('Normal', 'Abnormal')),
    predicted_class_code SMALLINT NOT NULL CHECK (predicted_class_code IN (0, 1)),
    abnormal_probability DOUBLE PRECISION NOT NULL CHECK (abnormal_probability BETWEEN 0 AND 1),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    model_source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wg_feedback (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(120),
    category VARCHAR(80) NOT NULL,
    rating SMALLINT CHECK (rating BETWEEN 1 AND 5),
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
