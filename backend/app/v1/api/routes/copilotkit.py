from __future__ import annotations

import asyncio
from uuid import uuid4
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from copilotkit import LangGraphAGUIAgent
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse


def _copy_run_input(input_data: RunAgentInput, **updates: Any) -> RunAgentInput:
    """
    Supports both Pydantic v1 and v2 objects.
    """
    if hasattr(input_data, "model_copy"):
        return input_data.model_copy(update=updates)
    return input_data.copy(update=updates)


def _bearer_from_request(request: Request) -> str:
    authorization = request.headers.get("authorization")

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    return authorization


def _build_run_config(
    *,
    principal,
    request: Request,
    input_data: RunAgentInput,
    request_id: str,
    session_id: str,
    thread_id: str,
    auth_context: dict[str, Any],
) -> dict[str, Any]:
    """
    This replaces the config block from your old /chat route.
    Do not put raw access tokens in metadata because metadata often goes to logs/traces.
    """
    return {
        "metadata": safe_langsmith_metadata(
            principal=principal,
            request_metadata={
                "frontend_url": request.headers.get("referer"),
                "user_agent": request.headers.get("user-agent"),
                "agui_run_id": getattr(input_data, "run_id", None),
            },
            extra={
                "request_id": request_id,
                "session_id": session_id,
            },
        ),
        "configurable": {
            "thread_id": thread_id,
            "request_id": request_id,
            "session_id": session_id,
            "user_id": principal.identity,
            "tenant_id": principal.tenant or "default",
            "auth_context": auth_context,
            "langgraph_auth_user": principal.to_langgraph_user(),

            # Put backend-only tool config here too:
            "ai_search": {
                "authorized_index_names": principal.authorized_index_names
                if hasattr(principal, "authorized_index_names")
                else [],
            },
        },
        "recursion_limit": settings.max_steps,
        # "callbacks": [langfuse_handler],  # add this only if defined/imported
    }

def create_copilotkit_router(app_state: Any,) -> APIRouter:
    router = APIRouter()

    @router.post("/copilotkit")
    async def copilotkit_run(input_data: RunAgentInput, request: Request):
        graph = getattr(app_state, "graph", None)
        if graph is None:
            raise HTTPException(status_code=503, detail="Graph is not ready")

        authorization = _bearer_from_request(request)

        principal, access_token = _authenticate_parent_request(
            authorization,
            request,
        )

        request_id = str(uuid4())

        # Prefer AG-UI/CopilotKit thread_id as the session id.
        # If it is missing, generate one.
        raw_thread_id = getattr(input_data, "thread_id", None)
        session_id = _normalize_session_id(raw_thread_id or request_id)

        # Namespace by user so two users cannot collide on same frontend thread id.
        thread_id = f"{principal.identity}:{session_id}"

        auth_context = _state_safe_auth_context(principal)

        config = _build_run_config(
            principal=principal,
            request=request,
            input_data=input_data,
            request_id=request_id,
            session_id=session_id,
            thread_id=thread_id,
            auth_context=auth_context,
        )

        # Important: make the AG-UI input use your backend-scoped thread_id.
        # This is what your checkpointer will see.
        scoped_input = _copy_run_input(
            input_data,
            thread_id=thread_id,
        )

        agent = LangGraphAGUIAgent(
            name=settings.agent_name,
            description=settings.agent_description,
            graph=graph,
            langgraph_config=config,
        )

        encoder = EventEncoder(accept=request.headers.get("accept"))

        async def event_generator():
            # Important: streaming happens after this function returns the response object.
            # So request_runtime_context must be inside the generator, not outside it.
            with request_runtime_context(
                access_token=access_token,
                request_id=request_id,
            ):
                try:
                    async for event in agent.run(scoped_input):
                        yield encoder.encode(event)
                finally:
                    await asyncio.to_thread(
                        _emit_chat_audit_event,
                        request_id=request_id,
                        principal=principal,
                        route=None,
                        session_id=session_id,
                        thread_id=thread_id,
                        result={},  # You can replace this with final graph state if needed.
                    )

        return StreamingResponse(
            event_generator(),
            media_type=getattr(encoder, "get_content_type", lambda: "text/event-stream")(),
        )

    return router
