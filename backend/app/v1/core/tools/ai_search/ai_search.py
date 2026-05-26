import logging
import httpx
from collections.abc import Sequence
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

from .interfaces import AISearchRequest, AISearchResponse
from .helper import _query_with_context, _authorize_index, _remote_agent_search, _compose_answer, _citations, _result_to_dict, _mode


logger = logging.getLogger(__name__)


@tool("ai_search")
async def ai_search_tool(request: AISearchRequest,
        config: RunnableConfig=None
    ) -> AISearchResponse:
    """
    Azure AI Search Tool

    This tool performs a search query against a specified index in azure.

    Parameters:
        request (AISearchRequest): The search request containing the query, index name, and other parameters.
        AISearchRequest(
            query: str | The search query to execute.
            index_name: str | The name of the index to search against.
            authorized_index_names: frozenset[str] | A set of index names that are authorized for this request.
            top_k: int | None | The maximum number of search results to return. If None, the default value from the configuration will be used.
            filter_expression: str | None | An optional filter expression to apply to the search results.
            context: str | None | Optional contextual information to enhance the search query.
            access_token: str | None | An optional access token for authenticating the search request.
        )

    Returns:
        AISearchResponse(
            answer: str | A concise answer generated from the search results.
            citations: list[dict[str, Any]] | A list of citations for the search results.
            results: list[dict[str, Any]] | A list of detailed search results.
            metadata: dict[str, Any] | Metadata about the search results.
        ): The response containing the search results.
    """
    return await search(request)

async def search(self, request: AISearchRequest) -> AISearchResponse:
        query = _query_with_context(request.query, request.context)
        if not query:
            raise ValueError("query is required")

        index_name = _authorize_index(request.index_name, request.authorized_index_names)
        
        
        if request.agent_base_url:
            return await _remote_agent_search(request)

        embedding: list[float] | None = None
        remote_vector_available = (
            request.use_vector
            and request.openai_client.supports_remote_embeddings
            and request.search_client.supports_remote_search
        )
        if remote_vector_available:
            embedding = await request.openai_client.create_embedding(query)

        results = await request.search_client.search(
            query,
            index_name=index_name,
            embedding=embedding,
            top_k=request.top_k,
            filter_expression=request.filter_expression,
            use_semantic=request.use_semantic,
            use_vector=embedding is not None and request.use_vector,
            use_hybrid=embedding is not None and request.use_hybrid,
        )

        return AISearchResponse(
            answer=_compose_answer(
                results,
                chars_per_result= request.compose_chars_per_result,
                total_char_budget=request.compose_total_char_budget,
            ),
            citations=_citations(results),
            results=[_result_to_dict(result) for result in results],
            metadata={
                "agent": self.name,
                "index_name": index_name,
                "result_count": len(results),
                "grounded_context": bool(request.context),
                "mode": _mode(
                    semantic=request.use_semantic,
                    vector=embedding is not None and request.use_vector,
                    hybrid=embedding is not None and request.use_hybrid,
                ),
            },
        )

