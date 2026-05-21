"""
FastAPI server setup with router registration
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import ErrorHandlerMiddleware
from app.api.routes import documents, search, collections, advanced_search, query
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# App version
APP_VERSION = "1.0.2"

# Initialize FastAPI app
app = FastAPI(
    title="SPLADE Content Server",
    description="An in-memory SPLADE content server with FAISS integration",
    version=APP_VERSION,
    # Disable automatic redirection when accessing routes without trailing slash
    redirect_slashes=False
)

# Log startup info
logger.info(f"Starting SPLADE Content Server v{APP_VERSION}")
logger.info(f"Model name: {settings.MODEL_NAME}")
logger.info(f"Model directory template: {settings.MODEL_DIR}")
logger.info(f"Min score threshold: {settings.MIN_SCORE_THRESHOLD}")
logger.info(f"Default results per query: {settings.DEFAULT_TOP_K}")

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handler middleware
app.add_middleware(ErrorHandlerMiddleware)

# Register routers
app.include_router(collections.router, prefix="/collections", tags=["collections"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(advanced_search.router, prefix="/advanced-search", tags=["advanced-search"])
app.include_router(query.router, prefix="/query", tags=["query"])

@app.get("/", tags=["root"])
async def root():
    """Root endpoint returning service information"""
    import os
    from app.core.config import settings
    from app.core.splade_service import splade_service

    # Check if model directory exists
    model_dir = settings.MODEL_DIR.format(model_name=settings.MODEL_NAME)
    model_exists = os.path.exists(model_dir)

    # Check if SPLADE service is initialized
    service_initialized = splade_service.is_initialized

    return {
        "service": "SPLADE Content Server",
        "version": APP_VERSION,
        "status": "operational" if service_initialized else "degraded",
        "model": {
            "name": settings.MODEL_NAME,
            "path": model_dir,
            "exists": model_exists,
            "loaded": service_initialized
        },
        "settings": {
            "max_length": settings.MAX_LENGTH,
            "data_dir": settings.DATA_DIR,
            "default_top_k": settings.DEFAULT_TOP_K
        },
        "llm": {
            "enabled": settings.LLM_ENABLED,
            "default_model": settings.LLM_DEFAULT_MODEL
        }
    }
