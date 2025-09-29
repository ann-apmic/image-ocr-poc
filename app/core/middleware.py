import sentry_sdk
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# sentry
class SentryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            with sentry_sdk.push_scope() as scope:
                scope.set_context("request", request)
                user_id = "database_user_id"  # when available
                scope.user = {"ip_address": request.client.host, "id": user_id}
                sentry_sdk.capture_exception(e)
            raise e
