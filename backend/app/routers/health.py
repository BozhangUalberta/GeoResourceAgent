from fastapi import APIRouter

router = APIRouter(
    prefix="/api/v1/health",
    tags=["health"]
)

@router.get("")
async def health_check():
    """
    Simple health check endpoint to verify API is alive.
    """
    return {"status": "ok"}
