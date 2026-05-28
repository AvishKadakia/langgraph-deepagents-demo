import os
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from langchain_core.tools import tool
from langchain_openai import AzureOpenAIEmbeddings

from app.v1.core.config import get_settings

settings = get_settings()


# --- Embeddings: must match the model/dimensions used to populate your index ---
embeddings = AzureOpenAIEmbeddings(
model=settings.embedding_deployment,
azure_endpoint=settings.endpoint,
api_key=settings.api_key,
openai_api_version=settings.api_version,
)


@tool("ai_search_tool")
async def ai_search_tool(query: str,index_name: str, top_k: int = settings.ai_search_default_top_k) -> str:
    """Search the company knowledge base and return cited passages."""
    # --- Azure AI Search client ---
    search_client = SearchClient(
    endpoint=settings.azure_search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(settings.azure_search_api_key),
    )
    query_vector = embeddings.embed_query(query)

    vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=top_k,
    fields=settings.azure_search_select_fields,  # Ensure this matches the fields in your index
    )

    results = search_client.search(
    search_text=query, # makes this hybrid: keyword + vector
    vector_queries=[vector_query],
    select=settings.azure_search_select_fields,
    top=top_k,
    )

    passages: List[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")
    passages.append(
    f"[{i}] {title}\nURL: {url}\nCONTENT:\n{content}"
    )

    return "\n\n---\n\n".join(passages)

