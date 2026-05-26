from __future__ import annotations

import logging

from copilotkit import CopilotKitMiddleware
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI

from app.v1.utils.checkpointer import AsyncCheckpointerBundle, create_postgres_checkpointer
from app.v1.core.config import get_settings
from app.v1.core.tools import (
    add_demo_task,
    create_launch_checklist,
    list_demo_tasks,
    summarize_demo_architecture,
    ai_search_tool,
)
from app.v1.core.middlewares.safety import SafetyGateMiddleware

logger = logging.getLogger(__name__)
settings = get_settings()

_checkpointer_bundle: AsyncCheckpointerBundle | None = None

def build_azure_chat_model() -> AzureChatOpenAI:

    return AzureChatOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        azure_deployment=settings.chat_deployment,
        api_version=settings.api_version,
        temperature=0,
    )

SYSTEM_PROMPT = """
You are a helpful agent 

""".strip()


async def build_agent():
    global _checkpointer_bundle

    _checkpointer_bundle = await create_postgres_checkpointer(settings.database_url)

    return create_deep_agent(
        model=build_azure_chat_model(),
        tools=[
            summarize_demo_architecture,
            create_launch_checklist,
            add_demo_task,
            list_demo_tasks,
            ai_search_tool
        ],
        middleware=[CopilotKitMiddleware(), SafetyGateMiddleware()],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer_bundle.saver,
    )


async def close_agent_resources() -> None:
    if _checkpointer_bundle is not None:
        await _checkpointer_bundle.close()