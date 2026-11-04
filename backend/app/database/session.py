"""
Database session management.
Creates the SQLite database and tables on first use.
"""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from backend.app.config import settings
from backend.app.database.models import Base


# Ensure the data directory exists
Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# Create engine with connection args for SQLite thread safety
engine = create_engine(
    settings.DB_URL,
    connect_args={"check_same_thread": False},
    echo=settings.DEBUG,
)


def _set_wal_mode(dbapi_conn, connection_record):
    """Enable WAL mode for better concurrent read performance."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


event.listen(engine, "connect", _set_wal_mode)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables. Safe to call multiple times (CREATE IF NOT EXISTS).
    Also performs lightweight column migrations for new attributes."""
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        cursor = conn.connection.cursor()
        try:
            # Check experiments table columns
            cursor.execute("PRAGMA table_info(experiments)")
            exp_cols = {row[1] for row in cursor.fetchall()}
            if "min_entropy" not in exp_cols:
                cursor.execute("ALTER TABLE experiments ADD COLUMN min_entropy FLOAT")
            if "num_qubits" not in exp_cols:
                cursor.execute("ALTER TABLE experiments ADD COLUMN num_qubits INTEGER DEFAULT 1")

            # Check debiasing_records table columns
            cursor.execute("PRAGMA table_info(debiasing_records)")
            deb_cols = {row[1] for row in cursor.fetchall()}
            if "output_min_entropy" not in deb_cols:
                cursor.execute("ALTER TABLE debiasing_records ADD COLUMN output_min_entropy FLOAT")
            if "output_to_input_ratio" not in deb_cols:
                cursor.execute("ALTER TABLE debiasing_records ADD COLUMN output_to_input_ratio FLOAT")
            if "pair_retention_ratio" not in deb_cols:
                cursor.execute("ALTER TABLE debiasing_records ADD COLUMN pair_retention_ratio FLOAT")
            if "discarded_bits" not in deb_cols:
                cursor.execute("ALTER TABLE debiasing_records ADD COLUMN discarded_bits INTEGER")

            conn.connection.commit()
        except Exception:
            pass
        finally:
            cursor.close()


def get_db():
    """FastAPI dependency that yields a database session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
