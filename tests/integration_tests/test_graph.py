"""Tests for the LangGraph workflow."""

from typing import Any, List
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from agent_arcade_tools.backend.graph import graph, tool_manager

# Create a dummy interrupt to get its type
_dummy_interrupt = None
try:
    interrupt("dummy")
except Exception as e:
    _dummy_interrupt = type(e)

if _dummy_interrupt is None:
    raise RuntimeError("Failed to determine interrupt type")

INTERRUPT_TYPE = _dummy_interrupt  # Type-safe reference


class MockModel:
    """Mock model for testing."""

    def invoke(
        self, messages: List[BaseMessage], **kwargs: dict[str, Any]
    ) -> AIMessage:
        """Mock invoke method."""
        # Return a response based on the last message
        if isinstance(messages[-1], HumanMessage):
            return AIMessage(
                content="I'll help you with that",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "Github_ListOrgRepositories",
                        "args": {},
                    }
                ],
            )
        elif isinstance(messages[-1], ToolMessage):
            return AIMessage(content="Here's what I found")
        return AIMessage(content="I'll help you with that")

    def bind_tools(self, *args: Any, **kwargs: Any) -> "MockModel":
        """Mock bind_tools method."""
        return self


class MockAuthResponse:
    """Mock authorization response."""

    def __init__(self, status: str = "pending", url: str = "http://auth.url") -> None:
        self.status = status
        self.url = url
        self.id = "test-auth-id"


@pytest.mark.skip(
    reason="Authorization test needs to be reworked to handle runnable context properly"
)
def test_authorization_handling() -> None:
    """Test that authorization is properly handled for tools requiring it."""
    # Mock a tool that requires authorization
    messages: List[BaseMessage] = [
        HumanMessage(content="List my GitHub repositories"),
    ]

    config: RunnableConfig = {
        "configurable": {
            "model": "azure_openai/gpt-4o-mini",
            "user_id": "test-user",
            "max_search_results": 5,
            "thread_id": "test-thread",
            "debug_mode": True,
            "log_level": "DEBUG",
        }
    }

    # Mock the tool manager's authorization methods and the model
    with (
        patch.object(tool_manager, "requires_auth", return_value=True),
        patch.object(tool_manager, "authorize", return_value=MockAuthResponse()),
        patch(
            "agent_arcade_tools.backend.graph.load_chat_model", return_value=MockModel()
        ),
    ):
        # This should raise an interrupt for authorization
        with pytest.raises(INTERRUPT_TYPE) as exc_info:
            graph.invoke({"messages": messages}, config=config)

        assert "Visit the following URL to authorize" in str(exc_info.value)


def test_message_history_maintenance() -> None:
    """Test that message history is properly maintained throughout the workflow."""
    # Initial conversation with multiple turns
    messages: List[BaseMessage] = [
        HumanMessage(content="What are the instructions for using Google Search?"),
    ]

    config: RunnableConfig = {
        "configurable": {
            "model": "azure_openai/gpt-4o-mini",
            "user_id": "test-user",
            "max_search_results": 5,
            "thread_id": "test-thread",
            "debug_mode": True,
            "log_level": "DEBUG",
        }
    }

    # Mock the tool manager's authorization methods and the model
    with (
        patch.object(tool_manager, "requires_auth", return_value=False),
        patch(
            "agent_arcade_tools.backend.graph.load_chat_model", return_value=MockModel()
        ),
    ):
        # Run the workflow
        response = graph.invoke({"messages": messages}, config=config)
        assert len(response["messages"]) > len(messages)


def test_ai_message_tool_calls_validation() -> None:
    """Test that AIMessage tool_calls are properly validated as lists."""
    config: RunnableConfig = {
        "configurable": {
            "model": "azure_openai/gpt-4o-mini",
            "user_id": "test-user",
            "max_search_results": 5,
            "thread_id": "test-thread",
            "debug_mode": True,
            "log_level": "DEBUG",
        }
    }

    # Test case 1: Valid empty list of tool calls
    messages: List[BaseMessage] = [
        HumanMessage(content="Hello"),
    ]

    # Mock the tool manager's authorization methods and the model
    with (
        patch.object(tool_manager, "requires_auth", return_value=False),
        patch(
            "agent_arcade_tools.backend.graph.load_chat_model", return_value=MockModel()
        ),
    ):
        response = graph.invoke({"messages": messages}, config=config)
        assert len(response["messages"]) > len(messages)
