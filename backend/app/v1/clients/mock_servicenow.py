"""Async mock ServiceNow client for local MVP orchestration.

This module intentionally contains no external ServiceNow integration, customer
configuration or credentials. It simulates a tiny API surface
with deterministic fixture data so sub-agents can be wired and exercised
without external dependencies.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

import httpx
from app.v1.core.config import get_settings

settings = get_settings()
from app.v1.utils.retry import http_retry_async

DEFAULT_TICKET_LIMIT = 10
MAX_TICKET_LIMIT = 50

VALID_STATUSES = frozenset(
    {
        "new",
        "open",
        "in_progress",
        "on_hold",
        "resolved",
        "closed",
        "canceled",
    }
)

_STATUS_ALIASES = {
    "cancelled": "canceled",
    "in progress": "in_progress",
    "in-progress": "in_progress",
    "on hold": "on_hold",
    "on-hold": "on_hold",
}

_TICKET_NUMBER_RE = re.compile(r"^(?:INC|RITM|REQ|CHG|PRB|TASK|CASE)\d{7}$", re.IGNORECASE)

_MOCK_TICKETS: tuple[dict[str, Any], ...] = (
    {
        "number": "INC0001001",
        "short_description": "Unable to access internal portal",
        "description": (
            "User reports sign-in succeeds but the portal returns an access denied page."
        ),
        "state": "open",
        "priority": "2",
        "assignment_group": "Service Desk",
        "requested_by": "Sample User",
        "opened_at": "2026-05-01T13:15:00Z",
        "updated_at": "2026-05-03T18:40:00Z",
        "comments": [
            "Confirmed user account is active.",
            "Access group membership review is pending.",
        ],
    },
    {
        "number": "INC0001002",
        "short_description": "Laptop battery drains quickly",
        "description": "Battery health check shows reduced capacity after recent travel.",
        "state": "in_progress",
        "priority": "3",
        "assignment_group": "Endpoint Support",
        "requested_by": "Sample User",
        "opened_at": "2026-05-02T09:05:00Z",
        "updated_at": "2026-05-04T15:22:00Z",
        "comments": [
            "Diagnostics collected.",
            "Replacement battery approval requested.",
        ],
    },
    {
        "number": "REQ0001003",
        "short_description": "Request standard productivity software",
        "description": "Request for approved productivity software on a managed workstation.",
        "state": "resolved",
        "priority": "4",
        "assignment_group": "Software Fulfillment",
        "requested_by": "Sample User",
        "opened_at": "2026-04-28T10:00:00Z",
        "updated_at": "2026-04-29T19:10:00Z",
        "comments": [
            "License was available.",
            "Software deployment completed successfully.",
        ],
        "resolution_notes": "Fulfilled from the standard software catalog.",
    },
    {
        "number": "CHG0001004",
        "short_description": "Routine patch window",
        "description": "Scheduled patching for a non-production application host.",
        "state": "closed",
        "priority": "3",
        "assignment_group": "Platform Operations",
        "requested_by": "Change Coordinator",
        "opened_at": "2026-04-20T12:00:00Z",
        "updated_at": "2026-04-24T23:30:00Z",
        "comments": [
            "Change approved.",
            "Patch window completed with no reported impact.",
        ],
        "resolution_notes": "Closed after post-change validation.",
    },
)


class MockServiceNowError(ValueError):
    """Raised when mock ServiceNow request input is invalid."""


class TicketNotFoundError(LookupError):
    """Raised when a valid mock ticket number does not exist."""


def validate_ticket_number(ticket_number: str) -> str:
    """Return a normalized ticket number or raise for invalid input."""

    if not isinstance(ticket_number, str):
        raise MockServiceNowError("ticket_number must be a string")

    normalized = ticket_number.strip().upper()
    if not _TICKET_NUMBER_RE.fullmatch(normalized):
        raise MockServiceNowError(
            "ticket_number must look like INC0001001, RITM0001001, REQ0001001, "
            "CHG0001001, PRB0001001, TASK0001001, or CASE0001001"
        )

    return normalized


def normalize_status(status: str) -> str:
    """Return a canonical status value or raise for unsupported filters."""

    if not isinstance(status, str):
        raise MockServiceNowError("status filters must be strings")

    normalized = status.strip().lower().replace("_", " ")
    normalized = _STATUS_ALIASES.get(normalized, normalized.replace(" ", "_"))
    if normalized not in VALID_STATUSES:
        valid = ", ".join(sorted(VALID_STATUSES))
        raise MockServiceNowError(f"unsupported status filter '{status}'. Valid statuses: {valid}")

    return normalized


def normalize_status_filters(statuses: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Normalize optional status filters while preserving caller order."""

    if statuses is None:
        return None

    if isinstance(statuses, str):
        raw_statuses = [status for status in statuses.split(",") if status.strip()]
    else:
        try:
            raw_statuses = list(statuses)
        except TypeError as exc:
            raise MockServiceNowError(
                "statuses must be a string, iterable of strings, or None"
            ) from exc

    normalized: list[str] = []
    for status in raw_statuses:
        canonical = normalize_status(status)
        if canonical not in normalized:
            normalized.append(canonical)

    if not normalized:
        raise MockServiceNowError(
            "at least one status filter is required when statuses is provided"
        )

    return tuple(normalized)


def validate_ticket_limit(limit: int | None) -> int:
    """Validate a ticket list count limit."""

    if limit is None:
        return DEFAULT_TICKET_LIMIT

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise MockServiceNowError("limit must be an integer")

    if limit < 1:
        raise MockServiceNowError("limit must be at least 1")

    if limit > MAX_TICKET_LIMIT:
        raise MockServiceNowError(f"limit must be {MAX_TICKET_LIMIT} or less")

    return limit


def resolve_ticket_limit(*, limit: int | None = None, count: int | None = None) -> int:
    """Resolve supported limit/count aliases into one validated value."""

    if limit is not None and count is not None and limit != count:
        raise MockServiceNowError("limit and count cannot disagree")

    return validate_ticket_limit(count if limit is None else limit)


class MockServiceNowClient:
    """Async client for the mock ServiceNow API, with fixture fallback for tests."""

    def __init__(
        self,
        tickets: Sequence[Mapping[str, Any]] | None = None,
        *,
        base_url: str | None = None,
        client_id: str = "local-client",
        client_secret: str = "local-secret",  # noqa: S107 - mock-only default.
        access_token: str | None = None,
        latency_seconds: float = 0,
        _http_pool_owner: MockServiceNowClient | None = None,
    ) -> None:
        self._tickets = {
            validate_ticket_number(ticket["number"]): dict(ticket)
            for ticket in (tickets or _MOCK_TICKETS)
        }
        self.base_url = base_url.rstrip("/") if base_url else None
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token = access_token
        self._latency_seconds = latency_seconds
        # Lazy shared httpx client for remote ServiceNow API + token paths so
        # we reuse the connection pool instead of opening a new TCP/TLS
        # session per call. When this instance was created via
        # ``with_access_token`` we delegate to the owner's pool so per-request
        # token-scoped clones reuse the same long-lived connection pool.
        self._remote_http_client: httpx.AsyncClient | None = None
        self._http_pool_owner = _http_pool_owner

    def _get_remote_http_client(self) -> httpx.AsyncClient:
        if self._http_pool_owner is not None:
            return self._http_pool_owner._get_remote_http_client()
        if self._remote_http_client is None:
            self._remote_http_client = httpx.AsyncClient(
                base_url=self.base_url or "",
                timeout=httpx.Timeout(10.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return self._remote_http_client

    async def aclose(self) -> None:
        # Per-request clones borrow the owner's pool — they must not close it.
        if self._http_pool_owner is not None:
            return
        if self._remote_http_client is not None:
            await self._remote_http_client.aclose()
            self._remote_http_client = None

    def with_access_token(self, access_token: str | None) -> MockServiceNowClient:
        """Return a request-scoped client that reuses this client's configuration."""

        return MockServiceNowClient(
            tickets=list(self._tickets.values()),
            base_url=self.base_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
            access_token=access_token or self._access_token,
            latency_seconds=self._latency_seconds,
            _http_pool_owner=self,
        )

    @classmethod
    def from_env(cls) -> MockServiceNowClient:
        """Create a mock API client from environment variables."""

        # Strict monolith: env config cannot route the default sub-agent
        # to an HTTP sidecar.
        base_url = None
        access_token = settings["MOCK_SERVICENOW_TOKEN"] or None
        if _prod_like_env() and (base_url or access_token):
            raise RuntimeError("Mock ServiceNow remote credentials are local/dev only")
        return cls(
            base_url=base_url,
            client_id=settings["MOCK_SERVICENOW_CLIENT_ID"],
            client_secret=settings["MOCK_SERVICENOW_CLIENT_SECRET"],
            access_token=access_token,
        )

    async def get_ticket(self, ticket_number: str) -> dict[str, Any]:
        """Fetch full mock ticket details."""

        payload = await self._request("ticket.get", ticket_number=ticket_number)
        return cast(dict[str, Any], payload["result"])

    async def get_ticket_summary(self, ticket_number: str) -> dict[str, Any]:
        """Fetch a summary-shaped mock ticket payload."""

        payload = await self._request("ticket.summary", ticket_number=ticket_number)
        return cast(dict[str, Any], payload["result"])

    async def list_tickets(
        self,
        *,
        statuses: str | Iterable[str] | None = None,
        limit: int | None = None,
        count: int | None = None,
    ) -> list[dict[str, Any]]:
        """List mock tickets, optionally filtered by state."""

        payload = await self._request("ticket.list", statuses=statuses, limit=limit, count=count)
        return cast(list[dict[str, Any]], payload["result"])

    async def _request(self, operation: str, **params: Any) -> dict[str, Any]:
        """Route a mock API request and return a raw API-like envelope."""

        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)
        else:
            await asyncio.sleep(0)

        if self.base_url:
            return await self._remote_request(operation, **params)

        if operation == "ticket.get":
            ticket = self._find_ticket(params["ticket_number"])
            return {"result": deepcopy(ticket)}

        if operation == "ticket.summary":
            ticket = self._find_ticket(params["ticket_number"])
            return {
                "result": {
                    "number": ticket["number"],
                    "short_description": ticket["short_description"],
                    "state": ticket["state"],
                    "priority": ticket["priority"],
                    "assignment_group": ticket["assignment_group"],
                    "updated_at": ticket["updated_at"],
                }
            }

        if operation == "ticket.list":
            statuses = normalize_status_filters(params.get("statuses"))
            limit = resolve_ticket_limit(limit=params.get("limit"), count=params.get("count"))
            tickets = list(self._tickets.values())
            if statuses:
                allowed = set(statuses)
                tickets = [
                    ticket for ticket in tickets if normalize_status(ticket["state"]) in allowed
                ]
            return {"result": deepcopy(tickets[:limit]), "count": min(len(tickets), limit)}

        raise MockServiceNowError(f"unsupported mock ServiceNow operation '{operation}'")

    async def _remote_request(self, operation: str, **params: Any) -> dict[str, Any]:
        token = await self._token()
        headers = {"Authorization": f"Bearer {token}"}
        client = self._get_remote_http_client()
        if operation in {"ticket.get", "ticket.summary"}:
            ticket_number = validate_ticket_number(params["ticket_number"])

            @http_retry_async()
            async def _do_get_ticket() -> httpx.Response:
                resp = await client.get(
                    f"/api/now/table/incident/{ticket_number}",
                    headers=headers,
                )
                resp.raise_for_status()
                return resp

            response = await _do_get_ticket()
            return self._response_payload(response)

        if operation == "ticket.list":
            statuses = normalize_status_filters(params.get("statuses"))
            limit = resolve_ticket_limit(limit=params.get("limit"), count=params.get("count"))
            query_params: dict[str, Any] = {"sysparm_limit": limit}
            if statuses:
                query_params["state"] = _status_to_mock_api_state(statuses[0])

            @http_retry_async()
            async def _do_list_tickets() -> httpx.Response:
                resp = await client.get(
                    "/api/now/table/incident",
                    params=query_params,
                    headers=headers,
                )
                resp.raise_for_status()
                return resp

            response = await _do_list_tickets()
            return self._response_payload(response)

        raise MockServiceNowError(f"unsupported mock ServiceNow operation '{operation}'")

    async def _token(self) -> str:
        if self._access_token:
            return self._access_token

        client = self._get_remote_http_client()

        @http_retry_async()
        async def _do_token() -> httpx.Response:
            resp = await client.post(
                "/oauth_token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            resp.raise_for_status()
            return resp

        response = await _do_token()
        payload = self._response_payload(response)
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise MockServiceNowError(
                "mock ServiceNow token endpoint did not return an access token"
            )
        self._access_token = token
        return token

    def _response_payload(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MockServiceNowError("mock ServiceNow API returned non-JSON response") from exc

        if response.status_code >= 400:
            message = payload.get("detail") or payload.get("error") or payload
            raise MockServiceNowError(
                f"mock ServiceNow API returned {response.status_code}: {message}"
            )
        if not isinstance(payload, dict):
            raise MockServiceNowError("mock ServiceNow API returned an unexpected response shape")
        return payload

    def _find_ticket(self, ticket_number: str) -> dict[str, Any]:
        normalized = validate_ticket_number(ticket_number)
        try:
            return self._tickets[normalized]
        except KeyError as exc:
            raise TicketNotFoundError(
                f"ticket '{normalized}' was not found in the mock dataset"
            ) from exc


def _status_to_mock_api_state(status: str) -> str:
    return {
        "new": "New",
        "open": "New",
        "in_progress": "In Progress",
        "on_hold": "On Hold",
        "resolved": "Resolved",
        "closed": "Closed",
        "canceled": "Canceled",
    }[normalize_status(status)]


def _prod_like_env() -> bool:
    return os.getenv("APP_ENV", "local").strip().lower() in {"stage", "prod", "production"}


__all__ = [
    "DEFAULT_TICKET_LIMIT",
    "MAX_TICKET_LIMIT",
    "VALID_STATUSES",
    "MockServiceNowClient",
    "MockServiceNowError",
    "TicketNotFoundError",
    "normalize_status",
    "normalize_status_filters",
    "resolve_ticket_limit",
    "validate_ticket_limit",
    "validate_ticket_number",
]
