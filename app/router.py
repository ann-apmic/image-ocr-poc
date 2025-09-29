from fastapi import APIRouter

from app.health.router import router as health_router

router = APIRouter()

router.include_router(health_router, tags=["health"], prefix="/health")
