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
from jade_safety_plugin.safety_prompt import prompt as safety_prompt
from jade import middleware_priority
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
class SafetyGateMiddleware(AgentMiddleware):
    def __init__(self):
        pass

    @hook_config(can_jump_to=["end"])
    @middleware_priority(10000)  
    async def abefore_agent(self, state: AgentState, runtime,config: RunnableConfig = None) -> Optional[dict[str, Any]]:
        user_text = _last_user_text(state)
        context = config.get("configurable", {}).get("jade_context")
        # Minimal safety prompt. Replace with your policy + categories.
        resp = await context.invoke_vlm(
            [
                {"role": "system", "content": safety_prompt},
                {"role": "user", "content": json.dumps({
                    "task": "classify",
                    "input": user_text
                })},
            ]
        )
        if resp.get("status") == "error":
            print(f"Safety Gate Error: VLM invocation failed - {resp.get('message')}")
            # You can decide to block or allow if the safety check itself fails
            return None
        data = resp["message"]
        if data.get("allowed") == False:
           
            print("Safety Gate: Blocking request.")
            return {
                "messages": [AIMessage(content=f"Request blocked. Reason: {str(data.get('reason'))}")],
                "jump_to": "end",
            }
        print("Safety Gate: Didn't block request.")
        return None