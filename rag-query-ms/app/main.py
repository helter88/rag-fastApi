from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.api.endpoints import query
from app.core.config import settings
from app.core.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(f"Starting up {settings.APP_NAME}...")
    yield

    logger.info("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    version="1.0.0"
)

app.include_router(query.router, prefix="/api/v1", tags=["RAG API"])

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}