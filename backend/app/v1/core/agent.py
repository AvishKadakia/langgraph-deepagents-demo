from __future__ import annotations

import logging
import os
from typing import Any

from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver  
from app.v1.utils.checkpointer import AsyncCheckpointerBundle, create_postgres_checkpointer
from app.v1.core.config import get_settings
from app.v1.core.tools import (

    ai_search_tool,
)
from app.v1.api.models import ChatRequest
from app.v1.core.subagents import SERVICENOW_SUBAGENT, close_servicenow_resources
from app.v1.core.middlewares.safety import SafetyGateMiddleware

logger = logging.getLogger(__name__)
settings = get_settings()

_checkpointer_bundle: AsyncCheckpointerBundle | None = None

def build_azure_chat_model(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> AzureChatOpenAI:
    logger.info("Building AzureChatOpenAI model with endpoint: %s, deployment: %s, api_version: %s", settings.endpoint, settings.chat_deployment, settings.api_version)
    kwargs: dict[str, Any] = {
        "azure_endpoint": settings.endpoint,
        "api_key": settings.api_key,
        "azure_deployment": settings.chat_deployment,
        "api_version": settings.api_version,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    # Azure/OpenAI chat models support top_p, not top_k, for sampling.
    if top_p is not None:
        kwargs["top_p"] = top_p

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return AzureChatOpenAI(**kwargs)
SYSTEM_PROMPT = """
You are a helpful agent.

You coordinate user requests and delegate ServiceNow ticket work to the
servicenow-ticket-agent subagent. Do not answer ServiceNow ticket questions
from memory; delegate them.
""".strip()

async def build_agent(checkpointer=InMemorySaver(),config = None) -> Any:
    payload:ChatRequest = config.payload
    model = build_azure_chat_model(
        temperature=getattr(payload, "temperature", settings.ai_llm_default_temperature),
        top_p=getattr(payload, "top_p", settings.ai_llm_default_top_p),
        max_tokens=getattr(payload, "max_tokens", settings.ai_llm_default_tmax_token),
    )
    return create_deep_agent(
        model=model,
        tools=[
            ai_search_tool,
        ],
        subagents=[
            SERVICENOW_SUBAGENT,
        ],
        middleware=[
            SafetyGateMiddleware(),
        ],
        system_prompt=SYSTEM_PROMPT + payload.SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


async def close_agent_resources() -> None:
    if _checkpointer_bundle is not None:
        await _checkpointer_bundle.close()

    await close_servicenow_resources()