from __future__ import annotations

import logging

from copilotkit import CopilotKitMiddleware
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI

from app.v1.utils.checkpointer import AsyncCheckpointerBundle, create_postgres_checkpointer
from app.v1.core.config import get_settings
from app.v1.core.tools import (

    ai_search_tool,
)
from app.v1.core.subagents import SERVICENOW_SUBAGENT, close_servicenow_resources
from app.v1.core.middlewares.safety import SafetyGateMiddleware

logger = logging.getLogger(__name__)
settings = get_settings()

_checkpointer_bundle: AsyncCheckpointerBundle | None = None

def build_azure_chat_model() -> AzureChatOpenAI:
    logger.info("Building AzureChatOpenAI model with endpoint: %s, deployment: %s, api_version: %s", settings.endpoint, settings.chat_deployment, settings.api_version)
    return AzureChatOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        azure_deployment=settings.chat_deployment,
        api_version=settings.api_version,
    )

SYSTEM_PROMPT = """
You are a helpful agent.

You coordinate user requests and delegate ServiceNow ticket work to the
servicenow-ticket-agent subagent. Do not answer ServiceNow ticket questions
from memory; delegate them.
""".strip()


async def build_agent():
    global _checkpointer_bundle

    _checkpointer_bundle = await create_postgres_checkpointer(settings.database_url)

    return create_deep_agent(
        model=build_azure_chat_model(),
        tools=[
            ai_search_tool,
        ],
        subagents=[
            SERVICENOW_SUBAGENT,
        ],
        middleware=[
            CopilotKitMiddleware(),
            SafetyGateMiddleware(),
        ],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer_bundle.saver,
    )


async def close_agent_resources() -> None:
    if _checkpointer_bundle is not None:
        await _checkpointer_bundle.close()

    await close_servicenow_resources()