from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BearerAuthMiddleware:
    """Requires Authorization: Bearer <token> on protected API routes."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        exempt_paths: set[str] | None = None,
        exempt_get_paths: set[str] | None = None,
    ):
        self.app = app
        self.token = token
        self.exempt_paths = exempt_paths or {"/health", "/docs", "/redoc", "/openapi.json"}
        # CopilotKit React >= 1.57's inspector can request this endpoint without
        # forwarding the provider Authorization header. The compat middleware
        # answers it with an empty thread list, so this does not expose app data.
        self.exempt_get_paths = exempt_get_paths or {"/copilotkit/threads"}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        if method == "OPTIONS" or path in self.exempt_paths or (method == "GET" and path in self.exempt_get_paths):
            await self.app(scope, receive, send)
            return

        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        expected = f"Bearer {self.token}"
        if headers.get("authorization") != expected:
            response = JSONResponse(status_code=401, content={"detail": "Missing or invalid bearer token."})
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
