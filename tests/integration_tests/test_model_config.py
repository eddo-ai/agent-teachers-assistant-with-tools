"""End-to-end tests for model configuration and tool calling capabilities."""

import os
from typing import Dict, List, cast

import pytest
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langsmith import Client

from agent_arcade_tools.backend.configuration import AgentConfigurable

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def langsmith_client() -> Client:
    """Initialize LangSmith client for tracing."""
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        pytest.skip("LANGCHAIN_TRACING_V2 not enabled")
    return Client()


@pytest.fixture
def test_config() -> RunnableConfig:
    """Test configuration fixture."""
    return {
        "configurable": {
            "model": "azure_openai/gpt-4o-mini",
            "user_id": "test-user",
            "max_search_results": 5,
            "thread_id": "test-thread",
            "debug_mode": True,
            "log_level": "DEBUG",
        },
        "tags": ["e2e-test"],  # Add tags for LangSmith tracing
        "metadata": {
            "test_name": "model_config_test",
            "environment": "test",
        },
    }


@pytest.fixture
def required_env_vars() -> List[str]:
    """List of required environment variables."""
    return [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "ARCADE_API_KEY",
        "LANGCHAIN_API_KEY",  # LangChain requirements
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_ENDPOINT",
    ]


def test_env_vars(required_env_vars: List[str]) -> None:
    """Test that all required environment variables are set."""
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        pytest.skip(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def test_model_config_from_runnable(test_config: Dict) -> None:
    """Test that model configuration can be created from runnable config."""
    config = AgentConfigurable.from_runnable_config(cast(RunnableConfig, test_config))
    assert config.model == "azure_openai/gpt-4o-mini"
    assert config.user_id == "test-user"
    assert config.max_search_results == 5
    assert config.debug_mode is True
    assert config.log_level == "DEBUG"
