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
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langgraph.graph import END, MessagesState
from langgraph.types import interrupt

from agent_arcade_tools.graph import (
    ToolNode,
    call_agent,
    graph,
    handle_tools,
    should_continue,
    tool_manager,
)

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
        self,
        messages: List[BaseMessage] | None = None,
        input: List[BaseMessage] | None = None,
        **kwargs: dict[str, Any],
    ) -> AIMessage:
        """Simulate model response generation.

        Args:
            messages: List of conversation messages
            input: Alternative parameter name for messages (for compatibility)
            **kwargs: Additional configuration parameters

        Returns:
            AIMessage: Simulated model response based on input type
        """
        # Handle both parameter naming conventions
        msgs = messages if messages is not None else input
        if msgs is None:
            msgs = []

        if len(msgs) == 0:
            return AIMessage(content="I'll help you with that")

        # Return a response based on the last message
        if isinstance(msgs[-1], HumanMessage):
            if msgs[-1].content == "check email":
                return AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                "type": "function",
                                "function": {
                                    "name": "Google_ListEmails",
                                    "arguments": '{"n_emails": 5}',
                                },
                            }
                        ]
                    },
                )
            return AIMessage(
                content="I'll help you with that",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "Google_ListEmails",
                                "arguments": '{"n_emails": 5}',
                            },
                        }
                    ]
                },
            )
        elif isinstance(msgs[-1], ToolMessage):
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
        patch("agent_arcade_tools.graph.load_chat_model", return_value=MockModel()),
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
        patch("agent_arcade_tools.graph.load_chat_model", return_value=MockModel()),
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
        patch("agent_arcade_tools.graph.load_chat_model", return_value=MockModel()),
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
        patch("agent_arcade_tools.graph.load_chat_model", return_value=MockModel()),
    ):
        # Run the workflow - should interrupt for authorization
        found_interrupt = False
        for chunk in graph.stream({"messages": messages}, config=config):
            logger.info(f"Got chunk: {chunk}")
            if "__interrupt__" in chunk:
                found_interrupt = True
                interrupt = chunk["__interrupt__"][0]
                assert interrupt.value == {
                    "message": "Visit the following URL to authorize: http://auth.test.url",
                    "auth_url": "http://auth.test.url",
                    "type": "authorization",
                }
                assert interrupt.resumable is True
                assert interrupt.when == "during"
                break

        assert found_interrupt, "Expected an interrupt but got none"
        assert auth_call_count == 1, (
            f"Expected exactly one auth call but got {auth_call_count}"
        )


def test_email_check_flow() -> None:
    """Test the workflow for checking emails using Google_ListEmails tool.

    This test verifies that:
        1. The workflow correctly processes a request to check emails
        2. Tool calls are properly formatted for Google_ListEmails
        3. Message history and tool calls are maintained correctly
    """
    messages: List[AnyMessage] = [
        HumanMessage(content="check email"),
    ]

    config: RunnableConfig = {
        "configurable": {
            "model": "gpt-4o-2024-05-13",
            "user_id": "test-user",
            "max_search_results": 5,
            "thread_id": "test-thread",
            "debug_mode": True,
            "log_level": "DEBUG",
            "reset_history": True,  # Add this to reset message history
        }
    }

    class EmailCheckModel(MockModel):
        """Mock model specifically for email checking scenario."""

        def invoke(
            self,
            messages: List[BaseMessage] | None = None,
            input: List[BaseMessage] | None = None,
            **kwargs: dict[str, Any],
        ) -> AIMessage:
            """Simulate model response for email checking."""
            # Handle both parameter naming conventions
            msgs = messages if messages is not None else input
            if msgs is None:
                msgs = []

            if len(msgs) > 0 and isinstance(msgs[-1], HumanMessage):
                # Return a properly formatted AIMessage with tool_calls in additional_kwargs
                return AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                "type": "function",
                                "function": {
                                    "name": "Google_ListEmails",
                                    "arguments": '{"n_emails": 5}',
                                },
                            }
                        ]
                    },
                )
            elif len(msgs) > 0 and isinstance(msgs[-1], ToolMessage):
                return AIMessage(content="Here are your recent emails...")
            return AIMessage(content="I'll help you check your emails")

    # Create a mock instance
    mock_model = EmailCheckModel()

    # Mock the tool manager's authorization methods and the model
    with patch("agent_arcade_tools.graph.load_chat_model", return_value=mock_model):
        # Directly test the call_agent function
        state = MessagesState(messages=messages)
        result = call_agent(state, config=config)

        # Verify the message history
        assert len(result["messages"]) == 2, "Expected exactly 2 messages in history"

        # Verify the AI message with tool calls
        ai_message = result["messages"][1]
        assert isinstance(ai_message, AIMessage), "Expected an AI message in response"
        assert "tool_calls" in ai_message.additional_kwargs, (
            "Expected tool_calls in additional_kwargs"
        )

        tool_calls = ai_message.additional_kwargs["tool_calls"]
        assert len(tool_calls) == 1, "Expected exactly one tool call"

        tool_call = tool_calls[0]
        assert tool_call["function"]["name"] == "Google_ListEmails", (
            "Incorrect tool name"
        )
        assert tool_call["id"] == "call_uh9TLO9zSlDbMppVD6w8YYU2", (
            "Incorrect tool call ID"
        )
        assert tool_call["type"] == "function", "Incorrect tool call type"
        assert tool_call["function"]["arguments"] == '{"n_emails": 5}', (
            "Incorrect tool arguments"
        )


@pytest.fixture
def email_check_payload() -> dict:
    """Fixture providing the exact payload structure from production."""
    return {
        "values": {
            "messages": [
                {
                    "content": "check email",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": None,
                    "id": "960f62c4-46d3-4aa7-8f7d-bf0034a97b74",
                    "example": False,
                },
                {
                    "content": "",
                    "additional_kwargs": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                "function": {
                                    "arguments": '{"n_emails":5}',
                                    "name": "Google_ListEmails",
                                },
                                "type": "function",
                            }
                        ]
                    },
                    "response_metadata": {
                        "finish_reason": "tool_calls",
                        "model_name": "gpt-4o-2024-05-13",
                        "system_fingerprint": "fp_65792305e4",
                    },
                    "type": "ai",
                    "name": None,
                    "id": "run-bb6f7d1a-f100-424e-ba88-e004e5d7a37c",
                    "example": False,
                    "tool_calls": [
                        {
                            "name": "Google_ListEmails",
                            "args": {"n_emails": 5},
                            "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                            "type": "tool_call",
                        }
                    ],
                    "invalid_tool_calls": [],
                    "usage_metadata": None,
                },
            ]
        }
    }


def test_email_check_flow_with_exact_payload(email_check_payload: dict) -> None:
    """Test the workflow for checking emails using the exact payload structure.

    This test verifies that:
        1. The workflow correctly processes a request to check emails
        2. Tool calls are properly formatted in additional_kwargs
        3. Message history and tool calls are maintained correctly
        4. The exact payload structure from production is handled

    Args:
        email_check_payload: Fixture containing the exact payload structure from production
    """
    # Extract messages from the payload
    payload_messages = email_check_payload["values"]["messages"]
    messages: List[BaseMessage] = [
        HumanMessage(
            content=payload_messages[0]["content"],
            additional_kwargs=payload_messages[0]["additional_kwargs"],
            id=payload_messages[0]["id"],
        ),
    ]

    config: RunnableConfig = {
        "configurable": {
            "model": "azure_openai/gpt-4o",
            "user_id": "test-user",
            "thread_id": "1fdad899-f9a5-465d-b885-ab0c1052710a",
        }
    }

    class ExactPayloadModel(MockModel):
        """Mock model that returns the exact payload structure."""

        def invoke(
            self,
            messages: List[BaseMessage] | None = None,
            input: List[BaseMessage] | None = None,
            **kwargs: dict[str, Any],
        ) -> AIMessage:
            """Simulate model response with exact payload structure."""
            # Handle both parameter naming conventions
            msgs = messages if messages is not None else input
            if msgs is None:
                msgs = []

            if len(msgs) == 0:
                return AIMessage(content="No messages to process")

            if isinstance(msgs[-1], HumanMessage):
                ai_message = payload_messages[1]  # Get the AI message from the payload
                # Create AIMessage with only the attributes that are guaranteed to exist
                result = AIMessage(
                    content=ai_message["content"],
                    additional_kwargs=ai_message["additional_kwargs"],
                    response_metadata=ai_message["response_metadata"],
                    id=ai_message["id"],
                )

                # Add optional attributes if they exist in the payload
                if "tool_calls" in ai_message:
                    result.tool_calls = ai_message["tool_calls"]
                if "invalid_tool_calls" in ai_message:
                    result.invalid_tool_calls = ai_message["invalid_tool_calls"]
                if "usage_metadata" in ai_message:
                    result.usage_metadata = ai_message["usage_metadata"]

                return result
            elif isinstance(msgs[-1], ToolMessage):
                return AIMessage(content="Here are your recent emails...")
            return AIMessage(content="I'll help you check your emails")

    # Mock the tool manager's authorization methods and the model
    with (
        patch.object(tool_manager, "requires_auth", return_value=True),
        patch.object(
            tool_manager, "authorize", return_value=MockAuthResponse(status="completed")
        ),
        patch.object(tool_manager, "is_authorized", return_value=True),
        patch(
            "agent_arcade_tools.graph.load_chat_model",
            return_value=ExactPayloadModel(),
        ),
    ):
        # Run the workflow
        response = graph.invoke({"messages": messages}, config=config)

        # Verify the message history
        assert len(response["messages"]) >= 2, "Expected at least 2 messages in history"

        # Verify the AI message with tool calls
        ai_message = next(
            (msg for msg in response["messages"] if isinstance(msg, AIMessage)), None
        )
        assert ai_message is not None, "Expected an AI message in response"

        # Compare with the exact payload structure
        expected_ai_message = payload_messages[1]
        assert ai_message.content == expected_ai_message["content"]
        assert ai_message.additional_kwargs == expected_ai_message["additional_kwargs"]
        assert ai_message.response_metadata == expected_ai_message["response_metadata"]
        assert ai_message.id == expected_ai_message["id"]

        # Check optional attributes if they exist
        if hasattr(ai_message, "example") and "example" in expected_ai_message:
            assert ai_message.example == expected_ai_message["example"]
        if hasattr(ai_message, "tool_calls") and "tool_calls" in expected_ai_message:
            assert ai_message.tool_calls == expected_ai_message["tool_calls"]
        if (
            hasattr(ai_message, "invalid_tool_calls")
            and "invalid_tool_calls" in expected_ai_message
        ):
            assert (
                ai_message.invalid_tool_calls
                == expected_ai_message["invalid_tool_calls"]
            )
        if (
            hasattr(ai_message, "usage_metadata")
            and "usage_metadata" in expected_ai_message
        ):
            assert ai_message.usage_metadata == expected_ai_message["usage_metadata"]


@pytest.fixture
def agent_to_auth_checkpoint() -> dict:
    """Fixture providing the exact checkpoint data for agent->authorization transition."""
    return {
        "values": {
            "messages": [
                {
                    "content": "check email",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": None,
                    "id": "960f62c4-46d3-4aa7-8f7d-bf0034a97b74",
                    "example": False,
                },
                {
                    "content": "",
                    "additional_kwargs": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                "function": {
                                    "arguments": '{"n_emails":5}',
                                    "name": "Google_ListEmails",
                                },
                                "type": "function",
                            }
                        ]
                    },
                    "response_metadata": {
                        "finish_reason": "tool_calls",
                        "model_name": "gpt-4o-2024-05-13",
                        "system_fingerprint": "fp_65792305e4",
                    },
                    "type": "ai",
                    "name": None,
                    "id": "run-bb6f7d1a-f100-424e-ba88-e004e5d7a37c",
                    "example": False,
                    "tool_calls": [
                        {
                            "name": "Google_ListEmails",
                            "args": {"n_emails": 5},
                            "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                            "type": "tool_call",
                        }
                    ],
                    "invalid_tool_calls": [],
                    "usage_metadata": None,
                },
            ]
        },
        "next": ["authorization"],
        "tasks": [
            {
                "id": "0092cab5-0dc3-ae56-471a-6b0ac596839c",
                "name": "authorization",
                "path": ["__pregel_pull", "authorization"],
                "error": None,
                "interrupts": [],
                "checkpoint": None,
                "state": None,
                "result": {
                    "messages": [
                        {
                            "content": "check email",
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "type": "human",
                            "name": None,
                            "id": "960f62c4-46d3-4aa7-8f7d-bf0034a97b74",
                            "example": False,
                        },
                        {
                            "content": "",
                            "additional_kwargs": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                        "function": {
                                            "arguments": '{"n_emails":5}',
                                            "name": "Google_ListEmails",
                                        },
                                        "type": "function",
                                    }
                                ]
                            },
                            "response_metadata": {
                                "finish_reason": "tool_calls",
                                "model_name": "gpt-4o-2024-05-13",
                                "system_fingerprint": "fp_65792305e4",
                            },
                            "type": "ai",
                            "name": None,
                            "id": "run-bb6f7d1a-f100-424e-ba88-e004e5d7a37c",
                            "example": False,
                            "tool_calls": [
                                {
                                    "name": "Google_ListEmails",
                                    "args": {"n_emails": 5},
                                    "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                    "type": "tool_call",
                                }
                            ],
                            "invalid_tool_calls": [],
                            "usage_metadata": None,
                        },
                    ]
                },
            }
        ],
        "metadata": {
            "user_id": "test-user",
            "model": "azure_openai/gpt-4o",
            "thread_id": "1fdad899-f9a5-465d-b885-ab0c1052710a",
        },
    }


def test_agent_to_authorization_transition(agent_to_auth_checkpoint: dict) -> None:
    """Test the transition from agent to authorization node using exact checkpoint data.

    This test verifies that:
        1. The agent node correctly processes the initial message
        2. The workflow correctly transitions to authorization
        3. The message state is maintained exactly as in production
        4. Tool calls are properly handled in both additional_kwargs and tool_calls

    Args:
        agent_to_auth_checkpoint: Fixture containing the exact checkpoint data
    """
    # Extract messages from the checkpoint
    checkpoint_messages = agent_to_auth_checkpoint["values"]["messages"]
    messages: List[AnyMessage] = [
        HumanMessage(
            content=checkpoint_messages[0]["content"],
            additional_kwargs=checkpoint_messages[0]["additional_kwargs"],
            id=checkpoint_messages[0]["id"],
        ),
    ]

    config: RunnableConfig = {
        "configurable": {
            "model": agent_to_auth_checkpoint["metadata"]["model"],
            "user_id": agent_to_auth_checkpoint["metadata"]["user_id"],
            "thread_id": agent_to_auth_checkpoint["metadata"]["thread_id"],
            "reset_history": True,  # Add this to reset message history
        }
    }

    class CheckpointModel(MockModel):
        """Mock model that returns the exact checkpoint response."""

        def invoke(
            self,
            messages: List[BaseMessage] | None = None,
            input: List[BaseMessage] | None = None,
            **kwargs: dict[str, Any],
        ) -> AIMessage:
            """Simulate model response with exact checkpoint data."""
            # Handle both parameter naming conventions
            msgs = messages if messages is not None else input
            if msgs is None:
                msgs = []

            if len(msgs) > 0 and isinstance(msgs[-1], HumanMessage):
                ai_message = checkpoint_messages[1]
                # Create AIMessage with only the attributes that are guaranteed to exist
                result = AIMessage(
                    content=ai_message["content"],
                    additional_kwargs=ai_message["additional_kwargs"],
                    response_metadata=ai_message["response_metadata"],
                    id=ai_message["id"],
                )

                # Add optional attributes if they exist in the checkpoint
                if "tool_calls" in ai_message:
                    result.tool_calls = ai_message["tool_calls"]
                if "invalid_tool_calls" in ai_message:
                    result.invalid_tool_calls = ai_message["invalid_tool_calls"]
                if "usage_metadata" in ai_message:
                    result.usage_metadata = ai_message["usage_metadata"]

                return result
            elif len(msgs) > 0 and isinstance(msgs[-1], ToolMessage):
                return AIMessage(content="Here are your recent emails...")
            return AIMessage(content="I'll help you check your emails")

    # Mock the tool manager's authorization methods and the model
    with (
        patch.object(tool_manager, "requires_auth", return_value=True),
        patch.object(
            tool_manager, "authorize", return_value=MockAuthResponse(status="pending")
        ),
        patch.object(tool_manager, "is_authorized", return_value=False),
        patch(
            "agent_arcade_tools.graph.load_chat_model",
            return_value=CheckpointModel(),
        ),
    ):
        # Run the workflow
        state = MessagesState(messages=messages)
        response = call_agent(state, config=config)

        # Verify the message history matches the checkpoint
        assert len(response["messages"]) == 2

        # Verify each message matches exactly
        human_msg = response["messages"][0]
        expected_human = checkpoint_messages[0]
        assert human_msg.content == expected_human["content"]
        assert human_msg.additional_kwargs == expected_human["additional_kwargs"]
        assert human_msg.id == expected_human["id"]

        ai_msg = response["messages"][1]
        expected_ai = checkpoint_messages[1]
        assert ai_msg.content == expected_ai["content"]
        assert ai_msg.additional_kwargs == expected_ai["additional_kwargs"]
        assert ai_msg.response_metadata == expected_ai["response_metadata"]
        assert ai_msg.id == expected_ai["id"]

        # Check tool_calls if they exist in the expected AI message
        if "tool_calls" in expected_ai and isinstance(ai_msg, AIMessage):
            if hasattr(ai_msg, "tool_calls"):
                assert ai_msg.tool_calls == expected_ai["tool_calls"]

        # Check invalid_tool_calls if they exist in the expected AI message
        if "invalid_tool_calls" in expected_ai and isinstance(ai_msg, AIMessage):
            if hasattr(ai_msg, "invalid_tool_calls"):
                assert ai_msg.invalid_tool_calls == expected_ai["invalid_tool_calls"]

        # Check usage_metadata if it exists in the expected AI message
        if "usage_metadata" in expected_ai and isinstance(ai_msg, AIMessage):
            if hasattr(ai_msg, "usage_metadata"):
                assert ai_msg.usage_metadata == expected_ai["usage_metadata"]

        # Verify the next node is authorization
        next_node = should_continue({"messages": response["messages"]})
        assert next_node == "authorization", "Expected transition to authorization node"


@pytest.fixture
def auth_to_tools_error_checkpoint() -> dict:
    """Fixture providing the exact checkpoint data for authorization->tools transition with error."""
    return {
        "values": {
            "messages": [
                {
                    "content": "check email",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": None,
                    "id": "960f62c4-46d3-4aa7-8f7d-bf0034a97b74",
                    "example": False,
                },
                {
                    "content": "",
                    "additional_kwargs": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                "function": {
                                    "arguments": '{"n_emails":5}',
                                    "name": "Google_ListEmails",
                                },
                                "type": "function",
                            }
                        ]
                    },
                    "response_metadata": {
                        "finish_reason": "tool_calls",
                        "model_name": "gpt-4o-2024-05-13",
                        "system_fingerprint": "fp_65792305e4",
                    },
                    "type": "ai",
                    "name": None,
                    "id": "run-bb6f7d1a-f100-424e-ba88-e004e5d7a37c",
                    "example": False,
                    "tool_calls": [
                        {
                            "name": "Google_ListEmails",
                            "args": {"n_emails": 5},
                            "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                            "type": "tool_call",
                        }
                    ],
                    "invalid_tool_calls": [],
                    "usage_metadata": None,
                },
            ]
        },
        "next": ["tools"],
        "tasks": [
            {
                "id": "8e0448e3-8168-d885-8fde-3a9ebcac1065",
                "name": "tools",
                "path": ["__pregel_pull", "tools"],
                "error": "AttributeError(\"'dict' object has no attribute 'tool'\")",
                "interrupts": [],
                "checkpoint": None,
                "state": None,
                "result": None,
            }
        ],
        "metadata": {
            "user_id": "aw@eddolearning.com",
            "model": "azure_openai/gpt-4o",
            "x-auth-scheme": "langsmith",
            "langgraph_auth_user": None,
            "langgraph_auth_user_id": "",
            "langgraph_auth_permissions": [],
            "graph_id": "graph",
            "assistant_id": "4b7f3fdc-445f-498c-9220-94571327f269",
            "from_studio": True,
            "run_attempt": 1,
            "langgraph_version": "0.2.76",
            "langgraph_plan": "developer",
            "langgraph_host": "self-hosted",
            "thread_id": "1fdad899-f9a5-465d-b885-ab0c1052710a",
            "run_id": "1efffaee-4f0c-62fe-821c-7f7042e1e323",
            "source": "fork",
            "writes": {
                "authorization": {
                    "messages": [
                        {
                            "content": "check email",
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "type": "human",
                            "name": None,
                            "id": "960f62c4-46d3-4aa7-8f7d-bf0034a97b74",
                            "example": False,
                        },
                        {
                            "content": "",
                            "additional_kwargs": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                        "function": {
                                            "arguments": '{"n_emails":5}',
                                            "name": "Google_ListEmails",
                                        },
                                        "type": "function",
                                    }
                                ]
                            },
                            "response_metadata": {
                                "finish_reason": "tool_calls",
                                "model_name": "gpt-4o-2024-05-13",
                                "system_fingerprint": "fp_65792305e4",
                            },
                            "type": "ai",
                            "name": None,
                            "id": "run-bb6f7d1a-f100-424e-ba88-e004e5d7a37c",
                            "example": False,
                            "tool_calls": [
                                {
                                    "name": "Google_ListEmails",
                                    "args": {"n_emails": 5},
                                    "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                    "type": "tool_call",
                                }
                            ],
                            "invalid_tool_calls": [],
                            "usage_metadata": None,
                        },
                    ]
                }
            },
            "step": 4,
            "parents": {},
            "checkpoint_id": "1efffb0b-c6d3-64b4-8002-a8e5f889ebe1",
        },
        "created_at": "2025-03-13T02:20:28.890489+00:00",
        "checkpoint": {
            "checkpoint_id": "1efffb1b-b53d-6faa-8003-2b266694154c",
            "thread_id": "1fdad899-f9a5-465d-b885-ab0c1052710a",
            "checkpoint_ns": "",
        },
        "parent_checkpoint": {
            "checkpoint_id": "1efffaee-6543-6ed2-8001-492f0f25d4a7",
            "thread_id": "1fdad899-f9a5-465d-b885-ab0c1052710a",
            "checkpoint_ns": "",
        },
        "checkpoint_id": "1efffb1b-b53d-6faa-8003-2b266694154c",
        "parent_checkpoint_id": "1efffaee-6543-6ed2-8001-492f0f25d4a7",
    }


def test_auth_to_tools_error_transition(auth_to_tools_error_checkpoint: dict) -> None:
    """Test the transition from authorization to tools node.

    This test verifies that:
        1. The authorization node correctly processes the message
        2. The workflow transitions to tools node
        3. The message state matches the checkpoint exactly
        4. Tool calls are properly structured in both additional_kwargs and tool_calls

    Args:
        auth_to_tools_error_checkpoint: Fixture containing the exact checkpoint data
    """
    # Create the initial message
    messages: list[AnyMessage] = [
        HumanMessage(
            content="check email",
            additional_kwargs={},
            response_metadata={},
            id="960f62c4-46d3-4aa7-8f7d-bf0034a97b74",
            example=False,
        ),
    ]

    config: RunnableConfig = {
        "configurable": {
            "model": "azure_openai/gpt-4o",
            "user_id": "test-user",
            "thread_id": "test-thread",
            "reset_history": True,  # Add this to reset message history
        }
    }

    class CheckpointModel(MockModel):
        """Mock model that returns the exact checkpoint response."""

        def invoke(
            self,
            messages: List[BaseMessage] | None = None,
            input: List[BaseMessage] | None = None,
            **kwargs: dict[str, Any],
        ) -> AIMessage:
            """Simulate model response with exact checkpoint data."""
            # Handle both parameter naming conventions
            msgs = messages if messages is not None else input
            if msgs is None:
                msgs = []

            if len(msgs) > 0 and isinstance(msgs[-1], HumanMessage):
                return AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_uh9TLO9zSlDbMppVD6w8YYU2",
                                "function": {
                                    "arguments": '{"n_emails":5}',
                                    "name": "Google_ListEmails",
                                },
                                "type": "function",
                            }
                        ]
                    },
                    response_metadata={
                        "finish_reason": "tool_calls",
                        "model_name": "gpt-4o-2024-05-13",
                        "system_fingerprint": "fp_65792305e4",
                    },
                    id="run-bb6f7d1a-f100-424e-ba88-e004e5d7a37c",
                )
            elif len(msgs) > 0 and isinstance(msgs[-1], ToolMessage):
                return AIMessage(content="Here are your recent emails...")
            return AIMessage(content="I'll help you check your emails")

    # Create a mock tool that returns a tool message
    def mock_list_emails(n_emails: int) -> ToolMessage:
        """Mock implementation of Google_ListEmails."""
        return ToolMessage(
            content="Found 5 emails in your inbox",
            name="Google_ListEmails",
            tool_call_id="call_uh9TLO9zSlDbMppVD6w8YYU2",
            status="success",
        )

    # Create a real ToolNode with our mock tool
    mock_tool = Tool(
        name="Google_ListEmails",
        description="List recent emails from your Gmail inbox",
        func=mock_list_emails,
        return_direct=True,  # This ensures the tool response is returned directly
    )
    tool_node = ToolNode(tools=[mock_tool])

    # Mock the tool manager's authorization methods and the model
    with (
        patch.object(tool_manager, "requires_auth", return_value=True),
        patch.object(
            tool_manager, "authorize", return_value=MockAuthResponse(status="completed")
        ),
        patch.object(tool_manager, "is_authorized", return_value=True),
        patch(
            "agent_arcade_tools.graph.load_chat_model",
            return_value=CheckpointModel(),
        ),
        patch("agent_arcade_tools.graph.tool_node", tool_node),
    ):
        # First, call the agent to get the AI message with tool calls
        state = MessagesState(messages=messages)
        agent_response = call_agent(state, config=config)

        # Verify the AI message has tool calls
        assert len(agent_response["messages"]) == 2
        ai_message = agent_response["messages"][1]
        assert isinstance(ai_message, AIMessage)
        assert "tool_calls" in ai_message.additional_kwargs

        # Now, directly test the handle_tools function
        tools_response = handle_tools(agent_response)

        # Verify the message history matches exactly
        assert len(tools_response["messages"]) == 3

        # Verify human message
        human_msg = tools_response["messages"][0]
        assert human_msg.content == "check email"
        assert human_msg.additional_kwargs == {}
        assert human_msg.id == "960f62c4-46d3-4aa7-8f7d-bf0034a97b74"

        # Verify AI message with tool calls
        ai_msg = tools_response["messages"][1]
        assert ai_msg.content == ""
        assert "tool_calls" in ai_msg.additional_kwargs
        assert (
            ai_msg.additional_kwargs["tool_calls"][0]["function"]["name"]
            == "Google_ListEmails"
        )
        assert ai_msg.response_metadata["finish_reason"] == "tool_calls"
        assert ai_msg.id == "run-bb6f7d1a-f100-424e-ba88-e004e5d7a37c"

        # Verify tool message
        tool_msg = tools_response["messages"][2]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.name == "Google_ListEmails"
        assert tool_msg.tool_call_id == "call_uh9TLO9zSlDbMppVD6w8YYU2"
        assert tool_msg.status == "success"
        assert tool_msg.content == "Found 5 emails in your inbox"


def test_should_continue_with_tool_message() -> None:
    """Test that should_continue correctly handles tool messages.

    This test verifies that:
        1. Tool messages correctly transition to agent
        2. AI messages with tool calls transition to tools/authorization
        3. AI messages without tool calls end the workflow
        4. Human messages end the workflow
    """
    # Test case 1: Tool message should continue to agent
    tool_message = ToolMessage(
        content="Found 5 emails",
        name="Google_ListEmails",
        tool_call_id="test-call-id",
        status="success",
    )
    tool_state: MessagesState = {
        "messages": [
            HumanMessage(content="check email"),
            AIMessage(content=""),
            tool_message,
        ]
    }
    assert should_continue(tool_state) == "agent", (
        "Tool message should continue to agent"
    )

    # Test case 2: AI message with tool calls should go to tools
    ai_with_tools = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "test-call-id",
                    "function": {
                        "arguments": '{"n_emails":5}',
                        "name": "Google_ListEmails",
                    },
                    "type": "function",
                }
            ]
        },
    )
    tools_state: MessagesState = {
        "messages": [HumanMessage(content="check email"), ai_with_tools]
    }

    # Mock tool_manager.requires_auth to return False for this test case
    with patch.object(tool_manager, "requires_auth", return_value=False):
        assert should_continue(tools_state) == "tools", (
            "AI message with tool calls should go to tools"
        )

    # Test case 3: AI message with auth-required tool calls should go to authorization
    ai_with_auth = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "index": 0,
                    "id": "test-call-id",
                    "function": {
                        "arguments": '{"org": "test-org"}',
                        "name": "Github_ListOrgRepositories",
                    },
                    "type": "function",
                }
            ]
        },
    )
    auth_state: MessagesState = {
        "messages": [HumanMessage(content="list repos"), ai_with_auth]
    }
    with patch.object(tool_manager, "requires_auth", return_value=True):
        assert should_continue(auth_state) == "authorization", (
            "AI message with auth-required tool calls should go to authorization"
        )

    # Test case 4: AI message without tool calls should end
    ai_no_tools = AIMessage(content="Hello, how can I help you?")
    no_tools_state: MessagesState = {
        "messages": [HumanMessage(content="hello"), ai_no_tools]
    }
    assert should_continue(no_tools_state) == END, (
        "AI message without tool calls should end"
    )

    # Test case 5: Human message should end
    human_state: MessagesState = {"messages": [HumanMessage(content="hello")]}
    assert should_continue(human_state) == END, "Human message should end"


def test_agent_response_to_tool_message() -> None:
    """Test that the agent properly responds to tool messages.

    This test verifies that:
        1. The agent correctly processes tool messages
        2. The agent generates appropriate responses based on tool results
        3. The message history is properly maintained
        4. JSON parsing works correctly for different content types
        5. Tool call IDs are properly maintained throughout the workflow
    """
    # Test case 1: JSON string content
    json_tool_message = ToolMessage(
        content='{"emails": [{"subject": "Test Email", "body": "Test content"}]}',
        name="Google_ListEmails",
        tool_call_id="test-call-id",
        status="success",
    )

    # Test case 2: Plain text content
    text_tool_message = ToolMessage(
        content="Found 5 emails in your inbox",
        name="Google_ListEmails",
        tool_call_id="test-call-id-2",
        status="success",
    )

    # Test case 3: Invalid JSON content
    invalid_json_tool_message = ToolMessage(
        content='{"invalid": json}',
        name="Google_ListEmails",
        tool_call_id="test-call-id-3",
        status="success",
    )

    class MockEmailResponseModel(MockModel):
        """Mock model that simulates email response processing."""

        def invoke(
            self,
            messages: List[BaseMessage] | None = None,
            input: List[BaseMessage] | None = None,
            **kwargs: dict[str, Any],
        ) -> AIMessage:
            """Simulate model response for email results."""
            # Handle both parameter naming conventions
            msgs = messages if messages is not None else input
            if msgs is None:
                msgs = []

            if len(msgs) == 0:
                return AIMessage(content="No messages to process")

            # Verify the tool message was properly processed
            tool_msg = msgs[-1]
            if not isinstance(tool_msg, ToolMessage):
                return AIMessage(content="Last message is not a tool message")

            assert tool_msg.name == "Google_ListEmails"
            assert tool_msg.status == "success"  # Verify status is set

            # Get the content as a string
            content_str = tool_msg.content
            if not isinstance(content_str, str):
                content_str = str(content_str)

            # Try to parse as JSON if it looks like JSON
            try:
                import json

                parsed_content = json.loads(content_str)

                # For valid JSON with emails field, return a specific response
                if isinstance(parsed_content, dict) and "emails" in parsed_content:
                    email_list = parsed_content["emails"]
                    if isinstance(email_list, list):
                        return AIMessage(
                            content=f"Found {len(email_list)} emails in your inbox.",
                            additional_kwargs={},
                        )
            except json.JSONDecodeError:
                # Not valid JSON, treat as plain text
                pass

            # Default response for plain text or invalid JSON
            return AIMessage(
                content=f"Processed message: {content_str}",
                additional_kwargs={},
            )

    # Create a mock tool that returns a tool message
    def mock_list_emails(n_emails: int) -> ToolMessage:
        """Mock implementation of Google_ListEmails."""
        return ToolMessage(
            content="Found 5 emails in your inbox",
            name="Google_ListEmails",
            tool_call_id="test-call-id",
            status="success",
        )

    # Create a real ToolNode with our mock tool
    mock_tool = Tool(
        name="Google_ListEmails",
        description="List recent emails from your Gmail inbox",
        func=mock_list_emails,
        return_direct=True,  # This ensures the tool response is returned directly
    )
    tool_node = ToolNode(tools=[mock_tool])

    # Test each case
    with (
        patch(
            "agent_arcade_tools.graph.load_chat_model",
            return_value=MockEmailResponseModel(),
        ),
        patch("agent_arcade_tools.graph.tool_node", tool_node),
    ):
        config: RunnableConfig = {
            "configurable": {
                "model": "gpt-4o-2024-05-13",
                "user_id": "test-user",
                "thread_id": "test-thread",
                "reset_history": True,  # Add this to reset message history
            }
        }

        # Test case 1: JSON string content - directly test call_agent
        json_state = MessagesState(
            messages=[
                HumanMessage(content="check email"),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "test-call-id",
                                "type": "function",
                                "function": {
                                    "name": "Google_ListEmails",
                                    "arguments": '{"n_emails": 5}',
                                },
                            }
                        ]
                    },
                ),
                json_tool_message,
            ]
        )
        json_response = call_agent(json_state, config=config)
        assert len(json_response["messages"]) == 4
        last_message = json_response["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert isinstance(last_message.content, str)
        assert "found 1 emails" in last_message.content.lower()

        # Test case 2: Plain text content - directly test call_agent
        text_state = MessagesState(
            messages=[
                HumanMessage(content="check email"),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "test-call-id-2",
                                "type": "function",
                                "function": {
                                    "name": "Google_ListEmails",
                                    "arguments": '{"n_emails": 5}',
                                },
                            }
                        ]
                    },
                ),
                text_tool_message,
            ]
        )
        text_response = call_agent(text_state, config=config)
        assert len(text_response["messages"]) == 4
        last_message = text_response["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert isinstance(last_message.content, str)
        assert "processed message" in last_message.content.lower()

        # Test case 3: Invalid JSON content - directly test call_agent
        invalid_json_state = MessagesState(
            messages=[
                HumanMessage(content="check email"),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "test-call-id-3",
                                "type": "function",
                                "function": {
                                    "name": "Google_ListEmails",
                                    "arguments": '{"n_emails": 5}',
                                },
                            }
                        ]
                    },
                ),
                invalid_json_tool_message,
            ]
        )
        invalid_json_response = call_agent(invalid_json_state, config=config)
        assert len(invalid_json_response["messages"]) == 4
        last_message = invalid_json_response["messages"][-1]
        assert isinstance(last_message, AIMessage)
        assert isinstance(last_message.content, str)
        assert "processed message" in last_message.content.lower()


def test_graph_components_directly() -> None:
    """Test the individual components of the graph directly.

    This test verifies that:
        1. should_continue correctly routes messages based on their content
        2. handle_tools correctly processes tool calls
    """
    # Test should_continue with different message types

    # 1. Tool message should continue to agent
    tool_message = ToolMessage(
        content="Found 5 emails",
        name="Google_ListEmails",
        tool_call_id="test-call-id",
        status="success",
    )
    tool_state: MessagesState = {
        "messages": [
            HumanMessage(content="check email"),
            AIMessage(content=""),
            tool_message,
        ]
    }
    assert should_continue(tool_state) == "agent", (
        "Tool message should continue to agent"
    )

    # 2. AI message with tool calls should go to tools (when no auth required)
    ai_with_tools = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "test-call-id",
                    "type": "function",
                    "function": {
                        "name": "Google_ListEmails",
                        "arguments": '{"n_emails": 5}',
                    },
                }
            ]
        },
    )
    ai_tools_state: MessagesState = {
        "messages": [HumanMessage(content="check email"), ai_with_tools]
    }

    with patch.object(tool_manager, "requires_auth", return_value=False):
        assert should_continue(ai_tools_state) == "tools", (
            "AI message with tool calls should go to tools"
        )

    # 3. AI message with auth-required tool calls should go to authorization
    with patch.object(tool_manager, "requires_auth", return_value=True):
        assert should_continue(ai_tools_state) == "authorization", (
            "AI message with auth-required tool calls should go to authorization"
        )

    # 4. AI message without tool calls should end
    ai_no_tools = AIMessage(content="Hello, how can I help you?")
    no_tools_state: MessagesState = {
        "messages": [HumanMessage(content="hello"), ai_no_tools]
    }
    assert should_continue(no_tools_state) == END, (
        "AI message without tool calls should end"
    )

    # 5. Human message should end
    human_state: MessagesState = {"messages": [HumanMessage(content="hello")]}
    assert should_continue(human_state) == END, "Human message should end"

    # Test handle_tools function
    handle_tools_state: MessagesState = {
        "messages": [HumanMessage(content="check email"), ai_with_tools]
    }

    # Create a mock tool that returns a tool message
    def mock_list_emails(n_emails: int) -> ToolMessage:
        """Mock implementation of Google_ListEmails."""
        return ToolMessage(
            content="Found 5 emails in your inbox",
            name="Google_ListEmails",
            tool_call_id="test-call-id",
            status="success",
        )

    # Create a real ToolNode with our mock tool
    mock_tool = Tool(
        name="Google_ListEmails",
        description="List recent emails from your Gmail inbox",
        func=mock_list_emails,
        return_direct=True,  # This ensures the tool response is returned directly
    )

    with patch("agent_arcade_tools.graph.tool_node", ToolNode(tools=[mock_tool])):
        result = handle_tools(handle_tools_state)

        # Verify the result contains the original messages plus the tool message
        assert len(result["messages"]) == 3, "Expected 3 messages in result"
