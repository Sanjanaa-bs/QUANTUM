"""
Configuration module for QRNG backend.
Reads from environment variables and .env file.
No secrets are stored in code.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parents[3]
load_dotenv(_project_root / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    # Application
    APP_TITLE: str = "QRNG - Quantum Random Number Generator"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Quantum Random Number Generator with Debiasing and "
        "Statistical Randomness Analysis (Course: 25AM3504, Project #15)"
    )
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # API
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["http://localhost:8501", "http://127.0.0.1:8501"]

    # Limits
    MAX_BITS: int = int(os.getenv("MAX_BITS", "100000"))
    MIN_BITS: int = int(os.getenv("MIN_BITS", "100"))
    DEFAULT_BITS: int = int(os.getenv("DEFAULT_BITS", "10000"))
    MAX_EXPERIMENTS: int = int(os.getenv("MAX_EXPERIMENTS", "1000"))

    # Database
    DB_PATH: str = os.getenv(
        "DB_PATH",
        str(_project_root / "data" / "qrng.db")
    )
    DB_URL: str = f"sqlite:///{DB_PATH}"

    # Reports directory
    REPORTS_DIR: str = os.getenv(
        "REPORTS_DIR",
        str(_project_root / "reports")
    )

    # NIST tests configuration
    NIST_ALPHA: float = float(os.getenv("NIST_ALPHA", "0.01"))

    # IBM Quantum (optional - never hardcode credentials)
    IBM_QUANTUM_TOKEN: Optional[str] = os.getenv("IBM_QUANTUM_TOKEN")
    IBM_QUANTUM_INSTANCE: Optional[str] = os.getenv("IBM_QUANTUM_INSTANCE")
    IBM_QUANTUM_BACKEND: Optional[str] = os.getenv("IBM_QUANTUM_BACKEND")
    HARDWARE_JOB_TIMEOUT_SECONDS: int = int(
        os.getenv("HARDWARE_JOB_TIMEOUT_SECONDS", "300")
    )

    @property
    def hardware_configured(self) -> bool:
        """True only if IBM Quantum token is set in environment."""
        return bool(self.IBM_QUANTUM_TOKEN)


settings = Settings()
