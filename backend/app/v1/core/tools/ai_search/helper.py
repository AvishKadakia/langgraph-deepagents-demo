import re
import httpx
from typing import Any

from collections.abc import Mapping, Sequence
from app.v1.core.tools.ai_search.interfaces import AISearchRequest, AISearchResponse
from app.v1.utils.retry import http_retry_async
from app.v1.core.tools.ai_search.interfaces import SearchResult

_INDEX_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

def _validate_index_name(index_name: str) -> str:
    candidate = (index_name or "").strip()
    if not candidate or not _INDEX_NAME_PATTERN.match(candidate):
        raise ValueError("index_name must be a structured, non-empty search index identifier")
    return candidate

def _query_with_context(query: str, context: str | None) -> str:
    query_text = (query or "").strip()
    if not context:
        return query_text
    context_text = " ".join(context.split())[:1200]
    return f"{query_text}\n\nServiceNow context: {context_text}".strip()

def _authorize_index(index_name: str, request_allowed: frozenset[str]) -> str:
        validated = _validate_index_name(index_name)
        allowed = request_allowed
        if allowed and validated not in allowed:
            raise ValueError("index_name is not authorized for this request")
        return validated



def _get_remote_http_client(request) -> httpx.AsyncClient:
        if request._remote_http_client is None:
            base_url = str(request.agent_base_url or "").rstrip("/")
            timeout = httpx.Timeout(request.agent_timeout_seconds)
            request._remote_http_client = httpx.AsyncClient(
                base_url=base_url,
                timeout=timeout,
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return request._remote_http_client

def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]

async def _remote_agent_search(
        request: AISearchRequest,
    ) -> AISearchResponse:
        headers = {}
        if request.access_token:
            headers["Authorization"] = f"Bearer {request.access_token}"
        payload = {
            "query": request.query,
            "index_name": request.index_name,
            "top_k": request.top_k,
            "filter_expression": request.filter_expression,
            "context": request.context,
        }
        client = _get_remote_http_client(request)

        @http_retry_async()
        async def _do_search() -> httpx.Response:
            resp = await client.post("/search", json=payload, headers=headers)
            resp.raise_for_status()
            return resp

        try:
            response = await _do_search()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"AI Search agent returned {exc.response.status_code}") from exc

        body = response.json()
        if not isinstance(body, Mapping):
            raise RuntimeError("AI Search agent returned an unexpected response shape")

        return AISearchResponse(
            answer=str(body.get("answer") or "No matching content found."),
            citations=_list_of_dicts(body.get("citations")),
            results=_list_of_dicts(body.get("results")),
            metadata=dict(body.get("metadata") or {}),
        )

def _compose_answer(
    results: Sequence[SearchResult],
    *,
    chars_per_result: int = 2000,
    total_char_budget: int = 16000,
) -> str:
    """Stitch retrieval results into the grounded text fed to synthesis.

    The synthesizer can only summarize what we put in front of it. Capping at
    top-3 x 500 chars (the old default) starved multi-entity questions, since
    the dominant entity's hits crowded the others out of the window. We now
    include every retrieved result up to the per-result and total-char budget
    — search has already ranked them, so trust the ranking but give the
    synthesizer enough material to actually find each entity.
    """

    if not results:
        return "No matching content found."

    per_result_cap = max(0, chars_per_result)
    total_cap = max(0, total_char_budget)
    if per_result_cap == 0 or total_cap == 0:
        return "Matching documents were found."

    snippets: list[str] = []
    remaining = total_cap
    for result in results:
        if remaining <= 0:
            break
        text = " ".join(result.content.split())
        if not text:
            continue
        snippet = text[: min(per_result_cap, remaining)]
        if not snippet:
            continue
        snippets.append(snippet)
        # +2 accounts for the "\n\n" separator we'll join with — keeps the
        # output strictly under the total budget instead of overshooting.
        remaining -= len(snippet) + 2

    return "\n\n".join(snippets) if snippets else "Matching documents were found."

def _citations(results: Sequence[SearchResult]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for position, result in enumerate(results, start=1):
        citations.append(
            {
                "rank": position,
                "title": result.title,
                "document_id": result.document_id,
                "source": result.source,
                "score": result.score,
            }
        )
    return citations


def _result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "content": result.content,
        "score": result.score,
        "title": result.title,
        "document_id": result.document_id,
        "source": result.source,
        "metadata": result.metadata,
    }


def _mode(*, semantic: bool, vector: bool, hybrid: bool) -> str:
    if hybrid and vector:
        return "hybrid_semantic" if semantic else "hybrid"
    if vector:
        return "vector_semantic" if semantic else "vector"
    return "semantic" if semantic else "keyword"