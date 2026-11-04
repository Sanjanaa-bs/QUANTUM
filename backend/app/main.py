"""
FastAPI application entry point.
Sets up the app, CORS, database initialization, and API router.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.database.session import init_db
from backend.app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    logger.info("QRNG backend starting up…")
    init_db()
    logger.info("Database initialized at %s", settings.DB_PATH)
    logger.info("Hardware configured: %s", settings.hardware_configured)
    yield
    logger.info("QRNG backend shutting down.")


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS — restrict to local Streamlit frontend for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
)

# Mount API router
app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "QRNG API is running.",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health",
        "info": f"{settings.API_PREFIX}/info",
    }
