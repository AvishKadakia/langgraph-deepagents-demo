from __future__ import annotations

import logging

from copilotkit import CopilotKitMiddleware
from deepagents import create_deep_agent

from app.checkpointer import AsyncCheckpointerBundle, create_postgres_checkpointer
from app.config import get_settings
from app.tools import (
    add_demo_task,
    create_launch_checklist,
    list_demo_tasks,
    summarize_demo_architecture,
)

logger = logging.getLogger(__name__)
settings = get_settings()

_checkpointer_bundle: AsyncCheckpointerBundle | None = None

SYSTEM_PROMPT = """
You are a senior full-stack AI engineer embedded in a demo application.

Your job:
- Help the user understand and test a LangGraph DeepAgent + CopilotKit + AG-UI app.
- Use tools when the user asks about the demo architecture, launch checklist, or task cards.
- Keep responses practical, concise, and implementation-focused.
- Remember user-provided project facts within the same thread; persistence is handled by LangGraph checkpoints.

When asked to prove persistence, ask the user to send a fact, then later recall it in the same thread.
""".strip()


async def build_agent():
    global _checkpointer_bundle

    _checkpointer_bundle = await create_postgres_checkpointer(settings.database_url)

    return create_deep_agent(
        model=settings.openai_model,
        tools=[
            summarize_demo_architecture,
            create_launch_checklist,
            add_demo_task,
            list_demo_tasks,
        ],
        middleware=[CopilotKitMiddleware()],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer_bundle.saver,
    )


async def close_agent_resources() -> None:
    if _checkpointer_bundle is not None:
        await _checkpointer_bundle.close()