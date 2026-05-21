"""
FastAPI dependencies
"""

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.splade_service import splade_service


async def get_splade_service():
    """
    Dependency that ensures the SPLADE service is initialized.
    Returns the service if initialized, or initializes it if not.
    """
    try:
        return splade_service.get_service()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"SPLADE service is not available: {str(e)}. Please check if the model directory exists: {settings.MODEL_DIR}"
        ) from e
