"""Tests for the Streamlit app."""

from typing import Sequence, cast
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import (
    Block,
    ChatInput,
    ChatMessage,
    Expander,
    WidgetList,
)


def test_stream_handling() -> None:
    """Test that the stream handler correctly processes and displays messages."""
    at: AppTest = AppTest.from_file("streamlit_app.py")

    # Mock LangGraph SDK responses with None values to trigger initial message
    mock_thread_state = {"values": None}

    with patch("langgraph_sdk.client.SyncLangGraphClient") as mock_client:
        # Configure mock client
        mock_client_instance = MagicMock()
        mock_client_instance.threads.get_state.return_value = mock_thread_state
        mock_client.return_value = mock_client_instance

        # Run the app
        at.run()

        # Check that the initial message is displayed
        messages: Sequence[ChatMessage] = at.chat_message
        assistant_message: ChatMessage = next(
            m for m in messages if m.name == "assistant"
        )
        assert assistant_message is not None
        message_content = cast(str, assistant_message.markdown[0].value)
        assert "Hi there! What's going on?" in message_content


def test_chat_input() -> None:
    """Test that the chat input correctly handles user messages."""
    at = AppTest.from_file("streamlit_app.py")

    # Mock LangGraph SDK responses
    mock_thread_state = {"values": {"messages": []}}  # Empty initial state

    with patch("langgraph_sdk.client.SyncLangGraphClient") as mock_client:
        # Configure mock client
        mock_client_instance = MagicMock()
        mock_client_instance.threads.get_state.return_value = mock_thread_state
        mock_client.return_value = mock_client_instance

        # Run the app
        at.run()

        # Simulate user input
        test_message = "Hello, how are you?"
        inputs: WidgetList[ChatInput] = at.chat_input
        chat_input: ChatInput = inputs[0]  # Get the first chat input
        assert chat_input is not None
        chat_input.set_value(test_message)
        at.run()

        # Check that the user message is displayed
        messages: Sequence[ChatMessage] = at.chat_message
        user_message: ChatMessage = next(m for m in messages if m.name == "user")
        assert user_message is not None
        message_content: str = cast(str, user_message.markdown[0].value)
        assert test_message in message_content


def test_error_handling() -> None:
    """Test that errors are properly displayed."""
    at: AppTest = AppTest.from_file("streamlit_app.py")

    # Mock LangGraph SDK responses
    mock_thread_state = {"values": {"messages": []}}  # Empty initial state

    with patch("langgraph_sdk.client.SyncLangGraphClient") as mock_client:
        # Configure mock client
        mock_client_instance = MagicMock()
        mock_client_instance.threads.get_state.return_value = mock_thread_state
        mock_client.return_value = mock_client_instance

        # Run the app
        at.run()

        # Force an error by providing invalid input
        inputs: WidgetList[ChatInput] = at.chat_input
        chat_input: ChatInput = inputs[0]  # Get the first chat input
        assert chat_input is not None

        # Set an empty string as input (which should be handled gracefully)
        chat_input.set_value("")
        at.run()

        # Check that the app is still running and ready for input
        assert len(at.chat_input) > 0


def test_initial_state() -> None:
    """Test that the app initializes with correct state and UI elements."""
    at: AppTest = AppTest.from_file("streamlit_app.py")

    # Mock LangGraph SDK responses
    mock_thread_state = {"values": {"messages": []}}  # Empty initial state

    with patch("langgraph_sdk.client.SyncLangGraphClient") as mock_client:
        # Configure mock client
        mock_client_instance = MagicMock()
        mock_client_instance.threads.get_state.return_value = mock_thread_state
        mock_client.return_value = mock_client_instance

        # Run the app
        at.run()

        # Check title
        assert "Agent Arcade Tools Chat" in at.title[0].value

        # Check that chat input exists
        assert len(at.chat_input) > 0

        # Check that thread details expander exists in sidebar
        sidebar: Block = at.sidebar
        expanders: Sequence[Expander] = sidebar.expander
        assert any(e.label == "Thread details" for e in expanders)


def test_message_roles_with_mock_langgraph() -> None:
    """Test that messages are displayed properly with their roles when using mocked LangGraph SDK."""
    at: AppTest = AppTest.from_file("streamlit_app.py")

    # Mock LangGraph SDK responses
    mock_thread_state = {
        "values": {
            "messages": [
                {"type": "user", "content": "Hello!", "response_metadata": {}},
                {
                    "type": "ai",
                    "content": "Hi! How can I help?",
                    "response_metadata": {
                        "model_name": "test-model",
                        "token_usage": {"total": 10},
                    },
                },
                {
                    "type": "user",
                    "content": "What tools do you have?",
                    "response_metadata": {},
                },
                {
                    "type": "ai",
                    "content": "I have several tools available.",
                    "response_metadata": {
                        "model_name": "test-model",
                        "token_usage": {"total": 15},
                        "tools_used": ["search", "code"],
                    },
                },
            ]
        }
    }

    with patch("langgraph_sdk.client.SyncLangGraphClient") as mock_client:
        # Configure mock client
        mock_client_instance = MagicMock()
        mock_client_instance.threads.get_state.return_value = mock_thread_state
        mock_client.return_value = mock_client_instance

        # Run the app
        at.run()

        # Verify messages are displayed with correct roles
        messages: Sequence[ChatMessage] = at.chat_message

        # Check user messages
        user_messages = [m for m in messages if m.name == "user"]
        assert len(user_messages) == 2
        assert "Hello!" in cast(str, user_messages[0].markdown[0].value)
        assert "What tools do you have?" in cast(
            str, user_messages[1].markdown[0].value
        )

        # Check assistant messages
        assistant_messages = [m for m in messages if m.name == "assistant"]
        assert len(assistant_messages) == 2
        assert "Hi! How can I help?" in cast(
            str, assistant_messages[0].markdown[0].value
        )
        assert "I have several tools available." in cast(
            str, assistant_messages[1].markdown[0].value
        )

        # Check metadata expanders
        expanders: Sequence[Expander] = at.expander
        metadata_expanders = [
            e for e in expanders if e.label == "Debug: Message Metadata"
        ]
        assert len(metadata_expanders) == 2  # One for each AI message

        # Verify metadata content
        # Note: Streamlit's json element returns a string, so we need to check the string content
        first_metadata_str = cast(str, metadata_expanders[0].json[0].value)
        assert '"model_name": "test-model"' in first_metadata_str
        assert '"total": 10' in first_metadata_str

        second_metadata_str = cast(str, metadata_expanders[1].json[0].value)
        assert '"model_name": "test-model"' in second_metadata_str
        assert '"total": 15' in second_metadata_str
        assert '"tools_used": ["search", "code"]' in second_metadata_str
