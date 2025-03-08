"""Streamlit frontend for the LangGraph backend."""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, cast
from uuid import UUID

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from streamlit.delta_generator import DeltaGenerator

from agent_arcade_tools.backend.graph import graph

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "api_calls" not in st.session_state:
    st.session_state.api_calls = []

# Page config
st.set_page_config(
    page_title="Agent Arcade Tools Chat",
    page_icon="🤖",
    layout="wide",
)


class APICallbackHandler(BaseCallbackHandler):
    """Custom callback handler to track API calls with Streamlit integration."""

    def __init__(self, parent_container: DeltaGenerator) -> None:
        """Initialize the handler with a Streamlit container."""
        super().__init__()
        self.parent_container = parent_container
        self.current_run_id: Optional[str] = None
        self.st_callback = StreamlitCallbackHandler(
            parent_container=parent_container,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
            max_thought_containers=10,
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Track chain start with run ID."""
        self.current_run_id = str(run_id)
        self.st_callback.on_chain_start(serialized, inputs, run_id=run_id, **kwargs)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log LLM API calls with run ID."""
        self.current_run_id = str(run_id)
        if "invocation_params" in kwargs:
            st.session_state.api_calls.append(
                {
                    "type": "llm",
                    "timestamp": datetime.now().isoformat(),
                    "endpoint": kwargs["invocation_params"].get(
                        "deployment_name", "default"
                    ),
                    "model": kwargs["invocation_params"].get("model_name", "unknown"),
                    "run_id": str(run_id),
                }
            )
        self.st_callback.on_llm_start(serialized, prompts, run_id=run_id, **kwargs)

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        """Pass through LLM end event."""
        self.st_callback.on_llm_end(*args, **kwargs)

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        """Pass through LLM error event."""
        self.st_callback.on_llm_error(*args, **kwargs)

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        """Pass through chain end event."""
        self.st_callback.on_chain_end(*args, **kwargs)

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        """Pass through chain error event."""
        self.st_callback.on_chain_error(*args, **kwargs)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Log tool API calls."""
        st.session_state.api_calls.append(
            {
                "type": "tool",
                "timestamp": datetime.now().isoformat(),
                "name": serialized.get("name", "unknown"),
                "input": input_str,
            }
        )
        self.st_callback.on_tool_start(serialized, input_str, **kwargs)

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        """Pass through tool end event."""
        self.st_callback.on_tool_end(*args, **kwargs)

    def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        """Pass through tool error event."""
        self.st_callback.on_tool_error(*args, **kwargs)


def render_api_debug_info() -> None:
    """Render API endpoint and call information."""
    st.markdown("### 🔌 API Configuration")

    # Display endpoint configuration
    with st.expander("Endpoint Configuration", expanded=True):
        endpoints = {
            "Azure OpenAI": os.getenv("AZURE_OPENAI_ENDPOINT", "Not configured"),
            "Azure AI Search": os.getenv("AZURE_AI_SEARCH_ENDPOINT", "Not configured"),
            "Arcade API": "https://api.arcade.software",
        }

        for name, endpoint in endpoints.items():
            st.markdown(f"**{name}**: `{endpoint}`")

    # Display API call history
    if st.session_state.api_calls:
        st.markdown("### 📡 API Call History")
        for call in st.session_state.api_calls:
            with st.expander(
                f"🔄 {call['type'].upper()} Call - {datetime.fromisoformat(call['timestamp']).strftime('%H:%M:%S')}",
                expanded=False,
            ):
                if call["type"] == "llm":
                    st.markdown(f"**Endpoint**: `{call['endpoint']}`")
                    st.markdown(f"**Model**: `{call['model']}`")
                else:
                    st.markdown(f"**Tool**: `{call['name']}`")
                    st.markdown("**Input:**")
                    st.code(str(call["input"]), language="json")
    else:
        st.info("No API calls recorded yet")


def render_message_details(message: dict) -> None:
    """Render detailed message information in an expander."""
    with st.expander(f"🔍 {message['role'].title()} Message Details", expanded=False):
        st.markdown("#### Content")
        st.markdown(message["content"])

        if "tool_calls" in message:
            st.markdown("#### Tool Calls")
            for tool_call in message["tool_calls"]:
                st.markdown(f"**Tool**: `{tool_call['name']}`")
                st.markdown("**Input:**")
                st.code(tool_call["arguments"], language="json")

        if "run_id" in message:
            st.markdown("#### Debugging")
            st.markdown(
                f"🔍 [View in LangSmith](https://smith.langchain.com/runs/{message['run_id']})"
            )


def render_thread_history(messages: list) -> None:
    """Render the thread history with detailed analysis."""
    st.markdown("### Thread Analysis")

    # Message type statistics
    message_types = {
        "user": len([m for m in messages if m["role"] == "user"]),
        "assistant": len([m for m in messages if m["role"] == "assistant"]),
        "tool": len([m for m in messages if m["role"] == "tool"]),
    }

    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("User Messages", message_types["user"])
    with col2:
        st.metric("Assistant Messages", message_types["assistant"])
    with col3:
        st.metric("Tool Calls", message_types["tool"])

    # Timeline view
    st.markdown("### Message Timeline")
    for idx, message in enumerate(messages, 1):
        with st.container():
            # Create a visual timeline with lines connecting messages
            if idx > 1:
                st.markdown("↓")

            # Message container with role-based styling
            role_colors = {"user": "🟦", "assistant": "🟩", "tool": "🟨"}

            st.markdown(
                f"{role_colors[message['role']]} **Message #{idx}** ({message['role'].title()})"
            )
            render_message_details(message)


# Sidebar for configuration and thread management
with st.sidebar:
    st.title("Configuration")

    # View Mode Toggle
    st.subheader("View Mode")
    show_history = st.toggle(
        "Show Thread History",
        value=st.session_state.show_history,
        help="Switch between chat and history analysis view",
    )
    if show_history != st.session_state.show_history:
        st.session_state.show_history = show_history
        st.rerun()

    # Debug Configuration Section
    st.subheader("Debug Settings")
    debug_mode = st.toggle("Debug Mode", value=False, help="Enable detailed logging")
    if not debug_mode:
        log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=1,  # Default to INFO
            help="Select logging verbosity",
        )

    # Thread Management Section
    st.subheader("Thread Management")

    # Create New Thread
    col1, col2 = st.columns([3, 1])
    with col1:
        new_thread_name = st.text_input(
            "Thread Name (optional)", placeholder="Enter thread name"
        )
    with col2:
        if st.button("New Thread", type="primary"):
            thread_id = str(uuid.uuid4())
            created_at = datetime.now()
            thread_name = (
                new_thread_name
                if new_thread_name
                else created_at.strftime("Thread %Y-%m-%d %H:%M")
            )
            st.session_state.threads[thread_id] = {
                "name": thread_name,
                "created_at": created_at.isoformat(),
                "messages": [],
            }
            st.session_state.current_thread_id = thread_id
            st.session_state.messages = []
            st.rerun()

    # Thread Selection
    if st.session_state.threads:
        st.subheader("Select Thread")
        thread_options = {
            f"{data['name']} ({id[:8]})": id
            for id, data in st.session_state.threads.items()
        }
        selected_thread = st.selectbox(
            "Choose a thread",
            options=list(thread_options.keys()),
            index=(
                list(thread_options.values()).index(st.session_state.current_thread_id)
                if st.session_state.current_thread_id in thread_options.values()
                else 0
            ),
        )

        if selected_thread:
            thread_id = thread_options[selected_thread]
            if thread_id != st.session_state.current_thread_id:
                st.session_state.current_thread_id = thread_id
                st.session_state.messages = st.session_state.threads[thread_id][
                    "messages"
                ]
                st.rerun()

    # Model and User Configuration
    st.subheader("Model Settings")
    model = st.selectbox(
        "Select Model",
        ["azure_openai/gpt-4o-mini", "openai/gpt-4", "openai/gpt-3.5-turbo"],
        index=0,
    )
    user_id = st.text_input("User ID (for tool authorization)", "test-user")
    max_search_results = st.slider("Max Search Results", 1, 20, 10)

# Main interface
if st.session_state.current_thread_id:
    thread_data = st.session_state.threads[st.session_state.current_thread_id]
    st.title(f"Chat: {thread_data['name']} 🤖")
    st.caption(
        f"Thread ID: {st.session_state.current_thread_id[:8]} • Created: {datetime.fromisoformat(thread_data['created_at']).strftime('%Y-%m-%d %H:%M')}"
    )
else:
    st.title("Agent Arcade Tools Chat 🤖")
    st.info("Create a new thread to start chatting")

# Show either chat interface or history view
if st.session_state.show_history and st.session_state.messages:
    render_thread_history(st.session_state.messages)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if st.session_state.current_thread_id and (
        prompt := st.chat_input("What would you like to know?")
    ):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create a container for the assistant's response
        with st.chat_message("assistant"):
            # Create two containers: one for intermediate steps and one for the final response
            thoughts_container = st.container()
            response_placeholder = st.empty()

            # Initialize the StreamlitCallbackHandler with expanded thoughts
            st_callback = APICallbackHandler(
                parent_container=thoughts_container,
            )

            try:
                # Prepare configuration with callbacks
                config: RunnableConfig = cast(
                    RunnableConfig,
                    {
                        "configurable": {
                            "model": model,
                            "user_id": user_id,
                            "max_search_results": max_search_results,
                            "thread_id": st.session_state.current_thread_id,
                            "debug_mode": debug_mode,
                            "log_level": log_level if not debug_mode else "DEBUG",
                        },
                        "callbacks": [st_callback],
                    },
                )

                # Get response from LangGraph
                messages = [
                    (
                        HumanMessage(content=msg["content"])
                        if msg["role"] == "user"
                        else AIMessage(
                            content=msg["content"],
                            tool_calls=msg.get("tool_calls"),
                        )
                    )
                    for msg in st.session_state.messages
                ]
                response = graph.invoke(
                    {"messages": messages},
                    config=config,
                )

                # Extract the last message (should be the AI response)
                if response and "messages" in response and response["messages"]:
                    ai_message = response["messages"][-1]
                    if isinstance(ai_message, AIMessage):
                        # Update the response placeholder with the final content
                        content = ai_message.content or "I'm processing your request..."
                        response_placeholder.markdown(content)
                        # Store the final response in chat history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": getattr(ai_message, "tool_calls", None),
                                "run_id": st_callback.current_run_id,
                            }
                        )
                        # Update thread history
                        st.session_state.threads[st.session_state.current_thread_id][
                            "messages"
                        ] = st.session_state.messages
                    else:
                        st.error("Unexpected response format")
                else:
                    st.error("No response received")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Thread management buttons in the sidebar
with st.sidebar:
    if st.session_state.current_thread_id:
        if st.button("Clear Current Thread"):
            st.session_state.threads[st.session_state.current_thread_id][
                "messages"
            ] = []
            st.session_state.messages = []
            st.session_state.api_calls = []  # Clear API call history
            st.rerun()

        if st.button("Delete Current Thread", type="secondary"):
            del st.session_state.threads[st.session_state.current_thread_id]
            st.session_state.current_thread_id = None
            st.session_state.messages = []
            st.session_state.api_calls = []  # Clear API call history
            st.rerun()

# Add environment variable check and API debug info
with st.sidebar:
    st.subheader("Environment Check")
    required_vars = [
        "AZURE_AI_SEARCH_ENDPOINT",
        "AZURE_AI_SEARCH_API_KEY",
        "AZURE_AI_SEARCH_INDEX_NAME",
        "OPENAI_API_KEY",
        "ARCADE_API_KEY",
    ]
    for var in required_vars:
        if os.getenv(var):
            st.success(f"✓ {var} is set")
        else:
            st.error(f"✗ {var} is not set")

    # Show API debug info when debug mode is enabled
    if debug_mode:
        st.markdown("---")
        render_api_debug_info()
