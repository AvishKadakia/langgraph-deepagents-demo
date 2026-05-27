from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from app.v1.clients.mock_servicenow import MockServiceNowError, TicketNotFoundError

def _error_payload(exc: Exception) -> dict:
    return {
        "ok": False,
        "source": "mock_servicenow",
        "kind": "servicenow_error",
        "error": str(exc),
    }


@tool
async def servicenow_get_ticket_summary(
    ticket_number: Annotated[
        str,
        Field(description="ServiceNow ticket number, e.g. INC0001001, REQ0001003, CHG0001004."),
    ],
) -> dict:
    """Get a compact normalized summary for one ServiceNow ticket."""
    
    try:
        normalized_number = validate_ticket_number(ticket_number)
        raw_ticket = await get_ticket_summary(normalized_number)
        return normalize_ticket_summary(raw_ticket)
    except (MockServiceNowError, TicketNotFoundError) as exc:
        return _error_payload(exc)


@tool
async def servicenow_get_ticket_detail(
    ticket_number: Annotated[
        str,
        Field(description="ServiceNow ticket number, e.g. INC0001001, REQ0001003, CHG0001004."),
    ],
) -> dict:
    """Get full normalized details for one ServiceNow ticket, including description, comments, requester, and resolution notes."""
    try:
        normalized_number = validate_ticket_number(ticket_number)
        raw_ticket = await get_ticket(normalized_number)
        return normalize_ticket_detail(raw_ticket)
    except (MockServiceNowError, TicketNotFoundError) as exc:
        return _error_payload(exc)


@tool
async def servicenow_list_tickets(
    statuses: Annotated[
        str | None,
        Field(
            description=(
                "Optional comma-separated status filters. "
                "Supported: new, open, in_progress, on_hold, resolved, closed, canceled. "
                "Aliases like 'in progress' and 'on hold' are accepted."
            )
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Field(description="Maximum tickets to return. Defaults to the backend default and must not exceed the backend max."),
    ] = None,
) -> dict:
    """List normalized ServiceNow tickets, optionally filtered by status."""
    try:
        normalized_statuses = normalize_status_filters(statuses)
        normalized_limit = resolve_ticket_limit(limit=limit, count=count)
        raw_tickets = await self.client.list_tickets(
            statuses=normalized_statuses,
            limit=normalized_limit,
        )
        return normalize_ticket_list(
            raw_tickets,
            statuses=normalized_statuses,
            limit=normalized_limit,
        )
    except (MockServiceNowError, TicketNotFoundError) as exc:
        return _error_payload(exc)


SERVICENOW_TOOLS = [
    servicenow_get_ticket_summary,
    servicenow_get_ticket_detail,
    servicenow_list_tickets,
]


async def close_servicenow_resources() -> None:
    global _servicenow_agent

    if _servicenow_agent is not None:
        await _servicenow_agent.aclose()
        _servicenow_agent = None