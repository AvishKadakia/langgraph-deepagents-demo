from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal


TaskStatus = Literal["todo", "doing", "done"]


# In-memory demo side data. LangGraph conversation state is persisted by PostgresSaver.
# This list is intentionally simple so you can replace it with real DB tables later.
_DEMO_TASKS: list[dict[str, str]] = []


def add_demo_task(title: str, status: TaskStatus = "todo") -> dict[str, str]:
    """Create a demo task card that the frontend team could later render in the UI."""

    task = {
        "id": f"task-{len(_DEMO_TASKS) + 1}",
        "title": title,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _DEMO_TASKS.append(task)
    return task


def list_demo_tasks() -> list[dict[str, str]]:
    """Return the current demo task cards created during this backend process."""

    return _DEMO_TASKS


def summarize_demo_architecture() -> str:
    """Explain this starter app's architecture in one concise paragraph."""

    return (
        "The React frontend uses CopilotKit v2 components to talk AG-UI to a FastAPI "
        "runtime. FastAPI exposes a LangGraph DeepAgent through the CopilotKit AG-UI "
        "bridge. The agent is compiled with LangGraph's PostgresSaver so thread-level "
        "checkpoints survive backend restarts. Docker Compose runs Postgres, backend, "
        "and frontend together for local development."
    )


def create_launch_checklist(project_name: str) -> list[str]:
    """Create a practical launch checklist for a small AI demo app."""

    return [
        f"Confirm {project_name} starts with docker compose up --build.",
        "Send one chat message and verify an AG-UI streaming response appears in React.",
        "Restart the backend container and verify the same thread can retain context.",
        "Check the Postgres checkpoint tables for new checkpoint rows.",
        "Record the demo flow and known limitations in the README.",
    ]
