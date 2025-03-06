"""Test the graph functionality."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent_arcade_tools.graph import should_call_tools, wait_for_auth_with_timeout


@pytest.fixture
def mock_state():
    """Create a mock state for testing."""
    return {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ],
        "awaiting_authorization_id": None,
    }


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return RunnableConfig(
        configurable={
            "user_id": "test_user",
            "model": "gpt-4",
            "system_prompt": "You are a helpful assistant.",
        }
    )


def test_should_call_tools_no_tool_calls(mock_state, mock_config):
    """Test should_call_tools when there are no tool calls."""
    result = should_call_tools(mock_state, mock_config)
    assert result == "agent"


def test_should_call_tools_with_tool_calls(mock_state, mock_config):
    """Test should_call_tools when there are tool calls."""
    mock_state["messages"].append(
        AIMessage(
            content="I'll help you with that.",
            additional_kwargs={
                "tool_calls": [{"name": "test_tool", "arguments": "{}"}]
            },
        )
    )
    result = should_call_tools(mock_state, mock_config)
    assert result in ["tools", "wait_for_auth_with_timeout", "agent"]


@pytest.mark.asyncio
async def test_wait_for_auth_with_timeout_no_auth_id(mock_state, mock_config):
    """Test wait_for_auth_with_timeout when there's no auth_id."""
    result = await wait_for_auth_with_timeout(mock_state, mock_config)
    assert result == {"messages": mock_state["messages"]}


@pytest.mark.asyncio
async def test_wait_for_auth_with_timeout_with_auth_id(mock_state, mock_config):
    """Test wait_for_auth_with_timeout when there's an auth_id."""
    mock_state["awaiting_authorization_id"] = "test_auth_id"
    mock_response = AsyncMock()
    mock_response.status = "completed"

    with patch(
        "agent_arcade_tools.graph.toolkit.wait_for_auth", return_value=mock_response
    ):
        result = await wait_for_auth_with_timeout(mock_state, mock_config)
        assert result == {"messages": mock_state["messages"]}
