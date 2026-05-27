"""LangChain tools for the mock ServiceNow ticket helper.

These tools expose a small, stable surface area for a ServiceNow-focused
subagent:
- get a compact ticket summary
- get full ticket details
- list tickets, optionally filtered by status

The backend remains the existing MockServiceNowClient so the demo stays local
and deterministic unless that client is later extended.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import Field

from app.v1.clients.mock_servicenow import (
    MockServiceNowClient,
    MockServiceNowError,
    TicketNotFoundError,
    normalize_status,
    normalize_status_filters,
    resolve_ticket_limit,
    validate_ticket_number,
)

SOURCE = "mock_servicenow"

_servicenow_client: MockServiceNowClient | None = None


def get_servicenow_client() -> MockServiceNowClient:
    """Return the shared mock ServiceNow client for this process."""

    global _servicenow_client

    if _servicenow_client is None:
        # Use the deterministic local fixture client. This avoids depending on
        # env-only remote configuration for the demo tools.
        _servicenow_client = MockServiceNowClient()

    return _servicenow_client


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "source": SOURCE,
        "kind": "servicenow_error",
        "error": str(exc),
    }


def _ticket_base(raw_ticket: Mapping[str, Any]) -> dict[str, Any]:
    ticket_number = validate_ticket_number(str(raw_ticket.get("number", "")))

    return {
        "ticket_number": ticket_number,
        "short_description": str(raw_ticket.get("short_description", "")),
        "status": normalize_status(str(raw_ticket.get("state", ""))),
        "priority": str(raw_ticket.get("priority", "")),
        "assignment_group": str(raw_ticket.get("assignment_group", "")),
        "updated_at": raw_ticket.get("updated_at"),
    }


def normalize_ticket_summary(raw_ticket: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a raw mock ticket payload for compact summaries."""

    return {
        "ok": True,
        "source": SOURCE,
        "kind": "ticket_summary",
        "ticket": _ticket_base(raw_ticket),
    }


def normalize_ticket_detail(raw_ticket: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a raw mock ticket payload for full ticket details."""

    ticket = _ticket_base(raw_ticket)
    ticket.update(
        {
            "description": str(raw_ticket.get("description", "")),
            "requested_by": str(raw_ticket.get("requested_by", "")),
            "opened_at": raw_ticket.get("opened_at"),
            "comments": [str(comment) for comment in raw_ticket.get("comments", [])],
            "resolution_notes": raw_ticket.get("resolution_notes"),
        }
    )

    return {
        "ok": True,
        "source": SOURCE,
        "kind": "ticket_detail",
        "ticket": ticket,
    }


def normalize_ticket_list(
    raw_tickets: Iterable[Mapping[str, Any]],
    *,
    statuses: str | Iterable[str] | None = None,
    limit: int | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    """Normalize raw mock ticket list results with validated filter metadata."""

    normalized_statuses = normalize_status_filters(statuses)
    normalized_limit = resolve_ticket_limit(limit=limit, count=count)
    tickets = [_ticket_base(ticket) for ticket in raw_tickets]

    return {
        "ok": True,
        "source": SOURCE,
        "kind": "ticket_list",
        "count": len(tickets),
        "limit": normalized_limit,
        "status_filter": list(normalized_statuses or []),
        "tickets": tickets,
    }


@tool
async def servicenow_get_ticket_summary(
    ticket_number: Annotated[
        str,
        Field(
            description=(
                "ServiceNow ticket number, e.g. INC0001001, REQ0001003, CHG0001004."
            )
        ),
    ],
) -> dict[str, Any]:
    """Get a compact normalized summary for one ServiceNow ticket."""

    try:
        normalized_number = validate_ticket_number(ticket_number)
        raw_ticket = await get_servicenow_client().get_ticket_summary(normalized_number)
        return normalize_ticket_summary(raw_ticket)
    except (MockServiceNowError, TicketNotFoundError) as exc:
        return _error_payload(exc)


@tool
async def servicenow_get_ticket_detail(
    ticket_number: Annotated[
        str,
        Field(
            description=(
                "ServiceNow ticket number, e.g. INC0001001, REQ0001003, CHG0001004."
            )
        ),
    ],
) -> dict[str, Any]:
    """Get full normalized details for one ServiceNow ticket."""

    try:
        normalized_number = validate_ticket_number(ticket_number)
        raw_ticket = await get_servicenow_client().get_ticket(normalized_number)
        return normalize_ticket_detail(raw_ticket)
    except (MockServiceNowError, TicketNotFoundError) as exc:
        return _error_payload(exc)


@tool
async def servicenow_list_tickets(
    statuses: Annotated[
        str | None,
        Field(
            description=(
                "Optional comma-separated status filters. Supported values: "
                "new, open, in_progress, on_hold, resolved, closed, canceled. "
                "Aliases like 'in progress' and 'on hold' are accepted."
            )
        ),
    ] = None,
    limit: Annotated[
        int | None,
        Field(
            description=(
                "Maximum tickets to return. Defaults to the backend default and "
                "must not exceed the backend max."
            )
        ),
    ] = None,
    count: Annotated[
        int | None,
        Field(description="Alias for limit. Do not pass both unless they match."),
    ] = None,
) -> dict[str, Any]:
    """List normalized ServiceNow tickets, optionally filtered by status."""

    try:
        normalized_statuses = normalize_status_filters(statuses)
        normalized_limit = resolve_ticket_limit(limit=limit, count=count)
        raw_tickets = await get_servicenow_client().list_tickets(
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
    global _servicenow_client

    if _servicenow_client is not None:
        await _servicenow_client.aclose()
        _servicenow_client = None


__all__ = [
    "SERVICENOW_TOOLS",
    "close_servicenow_resources",
    "get_servicenow_client",
    "normalize_ticket_detail",
    "normalize_ticket_list",
    "normalize_ticket_summary",
    "servicenow_get_ticket_detail",
    "servicenow_get_ticket_summary",
    "servicenow_list_tickets",
]