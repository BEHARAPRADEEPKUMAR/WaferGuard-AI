"""Optional PostgreSQL persistence for WaferGuard inspections and users."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def database_configured() -> bool:
    """Return whether a PostgreSQL connection string is configured."""
    return bool(os.getenv("DATABASE_URL", "").strip())


def _connect():
    """Open a PostgreSQL connection using psycopg 3."""
    if not database_configured():
        raise RuntimeError("DATABASE_URL is not configured.")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install psycopg[binary] to enable PostgreSQL persistence.") from exc
    return psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=8)


def init_database() -> bool:
    """Create the application tables when PostgreSQL is configured."""
    if not database_configured():
        return False
    statements = [
        """CREATE TABLE IF NOT EXISTS wg_users (
            id BIGSERIAL PRIMARY KEY,
            username VARCHAR(120) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        )""",
        """CREATE TABLE IF NOT EXISTS wg_inspections (
            id BIGSERIAL PRIMARY KEY,
            wafer_name TEXT NOT NULL,
            predicted_class VARCHAR(16) NOT NULL CHECK (predicted_class IN ('Normal','Abnormal')),
            predicted_class_code SMALLINT NOT NULL CHECK (predicted_class_code IN (0,1)),
            abnormal_probability DOUBLE PRECISION NOT NULL CHECK (abnormal_probability BETWEEN 0 AND 1),
            confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
            model_source TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS wg_feedback (
            id BIGSERIAL PRIMARY KEY,
            username VARCHAR(120),
            category VARCHAR(80) NOT NULL,
            rating SMALLINT CHECK (rating BETWEEN 1 AND 5),
            message TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""",
    ]
    with _connect() as conn:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
    return True


def save_inspection(wafer_name: str, prediction: dict[str, Any]) -> None:
    """Persist one validated model prediction."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO wg_inspections
                (wafer_name, predicted_class, predicted_class_code,
                 abnormal_probability, confidence, model_source)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    wafer_name,
                    prediction["predicted_label"],
                    int(prediction["predicted_class"]),
                    float(prediction["abnormal_probability"]),
                    float(prediction.get("confidence", prediction.get("prediction_confidence"))),
                    prediction.get("source", "Unknown"),
                ),
            )


def save_feedback(category: str, rating: int, message: str, username: str | None = None) -> None:
    """Store product feedback with a bounded rating and non-empty message."""
    if not 1 <= int(rating) <= 5:
        raise ValueError("Feedback rating must be between 1 and 5.")
    if not message.strip():
        raise ValueError("Feedback message cannot be empty.")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO wg_feedback (username, category, rating, message) VALUES (%s, %s, %s, %s)",
                (username, category.strip()[:80], int(rating), message.strip()),
            )


def _password_hash(password: str, salt: bytes | None = None) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random salt."""
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password hash without storing or comparing plaintext values."""
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        calculated = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        ).hex()
        return hmac.compare_digest(calculated, digest_hex)
    except (ValueError, TypeError):
        return False


def create_user(username: str, password: str) -> None:
    """Create a local application user with a salted password hash."""
    normalized = username.strip().lower()
    if len(normalized) < 3 or len(normalized) > 120:
        raise ValueError("Username must be between 3 and 120 characters.")
    password_hash = _password_hash(password)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO wg_users (username, password_hash) VALUES (%s, %s)",
                (normalized, password_hash),
            )


def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user against the stored salted password hash."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash, is_active FROM wg_users WHERE username = %s",
                (username.strip().lower(),),
            )
            row = cur.fetchone()
    return bool(row and row[1] and _verify_password(password, row[0]))


def fetch_recent_inspections(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent persisted inspection records for the history screen."""
    limit = max(1, min(int(limit), 1000))
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT wafer_name, predicted_class, abnormal_probability,
                confidence, model_source, created_at
                FROM wg_inspections ORDER BY created_at DESC LIMIT %s""",
                (limit,),
            )
            rows = cur.fetchall()
    return [
        {"wafer": row[0], "result": row[1], "abnormal_probability": row[2],
         "confidence": row[3], "source": row[4], "time": row[5].isoformat()}
        for row in rows
    ]
