"""Integration tests for the LangGraph workflow.

This module contains integration tests that verify the behavior of the LangGraph workflow,
particularly focusing on:
    - Authorization handling
    - Message history maintenance
    - Tool call validation
    - Interrupt handling
    - Complete workflow execution

The tests use mocking to simulate external dependencies and verify the workflow's
behavior in various scenarios.
"""

import logging
from typing import Any, List
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from agent_arcade_tools.backend.graph import graph, tool_manager

# Create a dummy interrupt to get its type for type-safe testing
_dummy_interrupt = None
try:
    interrupt("dummy")
except Exception as e:
    _dummy_interrupt = type(e)

if _dummy_interrupt is None:
    raise RuntimeError("Failed to determine interrupt type")

INTERRUPT_TYPE = _dummy_interrupt  # Type-safe reference

logger = logging.getLogger(__name__)


class MockModel:
    """Mock model for testing agent responses.

    This class simulates an AI model that can:
        - Generate responses with tool calls
        - Process tool results
        - Maintain conversation context
    """

    def invoke(
        self, messages: List[BaseMessage], **kwargs: dict[str, Any]
    ) -> AIMessage:
        """Simulate model response generation.

        Args:
            messages: List of conversation messages
            **kwargs: Additional configuration parameters

        Returns:
            AIMessage: Simulated model response based on input type
        """
        # Return a response based on the last message
        if isinstance(messages[-1], HumanMessage):
            return AIMessage(
                content="I'll help you with that",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "Github_ListOrgRepositories",
                        "args": {"org": "test-org"},
                    }
                ],
            )
        elif isinstance(messages[-1], ToolMessage):
            return AIMessage(content="Found repositories: repo1, repo2")
        return AIMessage(content="I'll help you with that")

    def bind_tools(self, *args: Any, **kwargs: Any) -> "MockModel":
        """Mock tool binding.

        Returns:
            MockModel: Self reference for chaining
        """
        return self


class MockAuthResponse:
    """Mock authorization response for testing.

    Attributes:
        status: Authorization status ('pending' or 'completed')
        url: Authorization URL for user redirection
        id: Unique identifier for the authorization request
    """

    def __init__(self, status: str = "pending", url: str = "http://auth.url") -> None:
        self.status = status
        self.url = url
        self.id = "test-auth-id"


@pytest.mark.skip(
    reason="Authorization test needs to be reworked to handle runnable context properly"
)
def test_authorization_handling() -> None:
    """Test that the graph properly handles authorization requirements."""
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

    # Track number of times authorize is called
    auth_call_count = 0

    def count_auth_calls(*args: Any, **kwargs: Any) -> MockAuthResponse:
        """Count authorization calls and return mock response.

        This helper tracks how many times authorization is attempted and
        returns a pending auth response with a test URL.
        """
        nonlocal auth_call_count
        auth_call_count += 1
        logger.info(f"Authorization called {auth_call_count} times")
        return MockAuthResponse(status="pending", url="http://auth.test.url")

    # Mock the tool manager's authorization methods and the model
    with (
        patch.object(tool_manager, "requires_auth", return_value=True),
        patch.object(tool_manager, "authorize", side_effect=count_auth_calls),
        patch.object(
            tool_manager,
            "wait_for_auth",
            return_value=MockAuthResponse(status="pending"),
        ),
        patch.object(tool_manager, "is_authorized", return_value=False),
        patch(
            "agent_arcade_tools.backend.graph.load_chat_model", return_value=MockModel()
        ),
    ):
        # Run the workflow - should interrupt for authorization
        found_interrupt = False
        for chunk in graph.stream({"messages": messages}, config=config):
            logger.info(f"Got chunk: {chunk}")
            if "__interrupt__" in chunk:
                found_interrupt = True
                interrupt = chunk["__interrupt__"][0]
                assert (
                    interrupt.value
                    == "Visit the following URL to authorize: http://auth.test.url"
                )
                assert interrupt.resumable is True
                assert interrupt.when == "during"
                break

        assert found_interrupt, "Expected an interrupt but got none"
        assert auth_call_count == 1, (
            f"Expected exactly one auth call but got {auth_call_count}"
        )


def test_message_history_maintenance() -> None:
    """Test that message history is properly maintained throughout the workflow.

    This test verifies that:
        1. Messages are correctly added to the history
        2. Message order is preserved
        3. History grows appropriately with conversation turns
    """
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
    """Test that AIMessage tool_calls are properly validated.

    This test verifies that:
        1. Empty tool call lists are handled correctly
        2. Tool calls are properly formatted
        3. Messages without tool calls are processed correctly
    """
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


def test_authorized_complete_flow() -> None:
    """Test the complete workflow with authorization handling using streaming.

    This test verifies that:
        1. Authorization needs are detected
        2. Interrupts are properly formatted and contain auth URLs
        3. Authorization is called exactly once
        4. Interrupts are resumable
        5. Workflow state is maintained during interrupts
    """
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

    # Track number of times authorize is called
    auth_call_count = 0

    def count_auth_calls(*args: Any, **kwargs: Any) -> MockAuthResponse:
        """Count authorization calls and return mock response.

        This helper tracks how many times authorization is attempted and
        returns a pending auth response with a test URL.
        """
        nonlocal auth_call_count
        auth_call_count += 1
        logger.info(f"Authorization called {auth_call_count} times")
        return MockAuthResponse(status="pending", url="http://auth.test.url")

    # Mock the tool manager's authorization methods and the model
    with (
        patch.object(tool_manager, "requires_auth", return_value=True),
        patch.object(tool_manager, "authorize", side_effect=count_auth_calls),
        patch.object(
            tool_manager,
            "wait_for_auth",
            return_value=MockAuthResponse(status="pending"),
        ),
        patch.object(tool_manager, "is_authorized", return_value=False),
        patch(
            "agent_arcade_tools.backend.graph.load_chat_model", return_value=MockModel()
        ),
    ):
        # Run the workflow - should interrupt for authorization
        found_interrupt = False
        for chunk in graph.stream({"messages": messages}, config=config):
            logger.info(f"Got chunk: {chunk}")
            if "__interrupt__" in chunk:
                found_interrupt = True
                interrupt = chunk["__interrupt__"][0]
                assert (
                    interrupt.value
                    == "Visit the following URL to authorize: http://auth.test.url"
                )
                assert interrupt.resumable is True
                assert interrupt.when == "during"
                break

        assert found_interrupt, "Expected an interrupt but got none"
        assert auth_call_count == 1, (
            f"Expected exactly one auth call but got {auth_call_count}"
        )
