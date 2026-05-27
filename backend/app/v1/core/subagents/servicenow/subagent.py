from __future__ import annotations

from app.v1.core.tools import ai_search_tool, servicenow_get_ticket_summary, servicenow_get_ticket_detail, servicenow_list_tickets


SERVICENOW_SUBAGENT_PROMPT = """
You are a ServiceNow ticket subagent.

You only handle mock ServiceNow ticket work:
- get a compact ticket summary
- get full ticket details
- list tickets, optionally filtered by status

Use the available ServiceNow tools instead of guessing.

Rules:
- For a single ticket overview, use servicenow_get_ticket_summary.
- For comments, requester, description, opened_at, or resolution notes, use servicenow_get_ticket_detail.
- For "show/list/find tickets", use servicenow_list_tickets.
- Normalize user status wording before calling tools when obvious, e.g. "in progress" -> "in_progress".
- Return concise answers to the main agent.
- If a tool returns ok=false, explain the error clearly and do not invent ticket data.
""".strip()


SERVICENOW_SUBAGENT = {
    "name": "servicenow-ticket-agent",
    "description": (
        "Use for mock ServiceNow ticket tasks: getting ticket summaries, "
        "getting ticket details, and listing tickets by optional status."
    ),
    "system_prompt": SERVICENOW_SUBAGENT_PROMPT,
    "tools": [
        servicenow_get_ticket_summary,
        servicenow_get_ticket_detail,
        servicenow_list_tickets,
        ai_search_tool
    ],
}