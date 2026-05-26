from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from backend.app.v1.utils.security import BearerAuthMiddleware
from backend.app.core.agent import build_agent, close_agent_resources
from backend.app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
settings = get_settings()


def runtime_info_payload() -> dict[str, Any]:
    """Small compatibility payload used by CopilotKit's /info handshake.

    Some CopilotKit + ag-ui-langgraph version combinations currently return 422
    for POST /copilotkit/info. Keeping this route in our app makes the Vite demo
    resilient while the actual graph run endpoint remains provided by
    add_langgraph_fastapi_endpoint below.
    """
    return {
        "actions": [],
        "agents": [
            {
                "name": settings.agent_name,
                "description": settings.agent_description,
                "type": "langgraph",
            }
        ],
        "sdkVersion": "0.1.90",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph = await build_agent()
    app.state.graph = graph

    add_langgraph_fastapi_endpoint(
        app=app,
        agent=LangGraphAGUIAgent(
            name=settings.agent_name,
            description=settings.agent_description,
            graph=graph,
        ),
        path="/copilotkit",
    )

    yield

    await close_agent_resources()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(BearerAuthMiddleware, token=settings.api_bearer_token)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def copilotkit_info_guard(request: Request, call_next):
    """Return CopilotKit runtime info before ag-ui-langgraph can validate the body.

    Some versions of ag-ui-langgraph register /copilotkit/info as a normal run
    endpoint and then validate the body as a RunAgentInput. CopilotKit's docs
    recommend testing this endpoint with `curl -d '{}'`, which sends
    application/x-www-form-urlencoded by default. This guard makes the info
    handshake body/content-type agnostic and prevents 422s or request-level
    crashes caused by version mismatches.
    """
    if request.url.path.rstrip("/") == "/copilotkit/info":
        return JSONResponse(runtime_info_payload())

    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "agent": settings.agent_name,
        "runtime": "copilotkit-ag-ui",
        "checkpointer": "postgres",
    }


# Must be registered before add_langgraph_fastapi_endpoint so this permissive
# handler wins for the runtime handshake.
@app.api_route("/copilotkit/info", methods=["GET", "POST", "OPTIONS"])
async def copilotkit_runtime_info() -> dict[str, Any]:
    return runtime_info_payload()
