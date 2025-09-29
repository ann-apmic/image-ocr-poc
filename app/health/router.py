from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/liveliness", name="health:liveliness", include_in_schema=False)
async def liveliness():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "pass"},
        media_type="application/health+json",
        headers={"cache-control": "no-cache", "http_status": "200"},
    )


@router.get("/readiness", name="health:readiness", include_in_schema=False)
async def readiness():
    headers = {"Cache-Control": "no-cache"}
    headers["http_status"] = "200"
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "pass"},
        media_type="application/health+json",
        headers=headers,
    )
