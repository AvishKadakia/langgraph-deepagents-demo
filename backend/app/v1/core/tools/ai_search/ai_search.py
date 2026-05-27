import os
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from langchain_core.tools import tool
from langchain_openai import AzureOpenAIEmbeddings
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from azure.identity import DefaultAzureCredential

from deepagents import create_deep_agent

from app.v1.core.config import get_settings

settings = get_settings()
# --- Azure AI Search client ---
search_client = SearchClient(
endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
)

# --- Embeddings: must match the model/dimensions used to populate your index ---
embeddings = AzureOpenAIEmbeddings(
model=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
api_key=os.environ["AZURE_OPENAI_API_KEY"],
openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)


@tool("ai_search_tool")
async def ai_search_tool(query: str, top_k: int = 5) -> str:
    """Search the company knowledge base and return cited passages."""

    query_vector = embeddings.embed_query(query)

    vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=top_k,
    fields=os.environ.get("AZURE_SEARCH_VECTOR_FIELD", "content_vector"),
    )

    results = search_client.search(
    search_text=query, # makes this hybrid: keyword + vector
    vector_queries=[vector_query],
    select=[
    "title",
    "content",
    "url",
    ],
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

