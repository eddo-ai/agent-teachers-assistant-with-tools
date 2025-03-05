"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import os
import logging
from typing import Any, List, Optional, Dict, cast
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


async def retrieve_instructional_materials(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> Optional[List[Dict[str, Any]]]:
    """Search for instructional materials from OpenSciEd that are relevant to the user's query.

    Args:
        query: The search query to find relevant instructional materials
            (e.g. "What materials are available for teaching about forces and motion?").
        config: Configuration for the runnable.

    Returns:
        A list of relevant documents if found, None if configuration is missing.

    Raises:
        ValueError: If required Azure Search configuration is missing.
    """
    # Get required environment variables
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    semantic_config = os.getenv(
        "AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME", "unit_lesson"
    )

    # Validate required environment variables
    if not all([endpoint, api_key, index_name]):
        raise ValueError(
            "Missing required Azure Search configuration. Please set AZURE_AI_SEARCH_ENDPOINT, "
            "AZURE_AI_SEARCH_API_KEY, and AZURE_AI_SEARCH_INDEX_NAME environment variables."
        )

    # Initialize Azure Search client
    embeddings = OpenAIEmbeddings()
    retriever = AzureSearch(
        azure_search_endpoint=cast(str, endpoint),
        azure_search_key=cast(str, api_key),
        index_name=cast(str, index_name),
        embedding_function=embeddings,
    ).as_retriever(
        search_type="semantic_hybrid",
        semantic_configuration_name=semantic_config,
    )

    # Convert Document objects to dictionaries
    documents = retriever.get_relevant_documents(query)
    return [
        {"content": doc.page_content, "metadata": doc.metadata} for doc in documents
    ]
