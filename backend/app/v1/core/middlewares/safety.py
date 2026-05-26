from typing import List
import pluggy
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from typing import Any, Callable, Optional, Literal, List, Dict
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
import json
import re

from langchain_core.runnables import RunnableConfig
hookimpl = pluggy.HookimplMarker("jade")

# 1. ADD THE HELPER FUNCTION HERE
def _last_user_text(state: AgentState) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
            return msg[1]
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""

SECRET_PATTERNS = (
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)\b", re.I),
    re.compile(r"\bpassword\s*[:=]", re.I),
    re.compile(r"\b[A-Za-z0-9_=-]{24,}\.[A-Za-z0-9_=-]{24,}\.[A-Za-z0-9_=-]{16,}\b"),
)

DESTRUCTIVE_PATTERNS = (
    re.compile(r"\bdelete\s+(?:all|every)\b", re.I),
    re.compile(r"\bdrop\s+(?:table|database)\b", re.I),
    re.compile(r"\bdisable\s+(?:logging|audit|security)\b", re.I),
)

OUTPUT_REDACTIONS = (
    (
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)"
            r"\s*[:=]\s*\S+",
            re.I,
        ),
        "[redacted secret]",
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted identifier]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[redacted number]"),
)


def assess_input_safety(text: str) -> tuple[bool, list[str]]:
    """Return whether the request is safe enough to route plus reasons."""

    reasons: list[str] = []
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        reasons.append("request_contains_sensitive_secret_reference")
    if any(pattern.search(text) for pattern in DESTRUCTIVE_PATTERNS):
        reasons.append("request_contains_high_risk_destructive_action")
    return not reasons, reasons

class SafetyGateMiddleware(AgentMiddleware):
    def __init__(self):
        pass

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state: AgentState, runtime,config: RunnableConfig = None) -> Optional[dict[str, Any]]:
        message = _last_user_text(state)
        safe, reasons = assess_input_safety(message)
        if not safe:
            return {
            "messages": [
                AIMessage(content=f"I cannot process that request because it may expose sensitive data or perform a high-risk action. Reasons: {', '.join(reasons)}")
            ],
            "jump_to": "end",
        }
        print("Safety Gate: Didn't block request.")
        return {
            "messages": [AIMessage(content=f"Request allowed.")],
            "jump_to": "continue",
        }