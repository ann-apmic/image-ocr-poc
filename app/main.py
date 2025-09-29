import subprocess

import sentry_sdk
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.middleware import SentryMiddleware
from app.helper.hasura import HasuraException
from app.router import router as api_router

from .core.config import ALLOWED_HOSTS, DEBUG, PROJECT_NAME, STAGE

# NOTE: 架構參考：https://github.com/nsidnev/fastapi-realworld-example-app


def get_git_head_hash() -> str:
    command = ["git", "rev-parse", "HEAD"]
    process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)
    commit_bash = process.stdout.readlines()[0]
    return commit_bash.decode("utf8").strip()


def get_application() -> FastAPI:
    application = FastAPI(
        title=PROJECT_NAME,
        debug=DEBUG,
        version=get_git_head_hash()[:8],
    )

    if STAGE in ("staging", "prod"):
        application.add_middleware(SentryMiddleware)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_HOSTS or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix="/api")
    return application


app = get_application()


@app.exception_handler(HasuraException)
async def hasura_exception_handler(request: Request, e: HasuraException):
    with sentry_sdk.push_scope() as scope:
        user_id = "hasura"  # when available

        scope.set_context("request", request)
        scope.user = {
            "ip_address": request.client.host,
            "id": user_id,
        }
        scope.set_extra("request", e.message)
        sentry_sdk.capture_message("Hasura 401", e.message)

    return JSONResponse(
        status_code=e.status_code,
        content={"message": e.message},
    )
