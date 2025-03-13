"""Streamlit frontend for the LangGraph backend."""

from typing import Any, Dict, Iterator, Literal, Protocol, TypedDict, cast

import streamlit as st
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk.client import (
    LangGraphClient,
    SyncLangGraphClient,
    get_client,
    get_sync_client,
)
from langgraph_sdk.schema import ThreadState

# Page config
st.set_page_config(
    page_title="LangGraph Debug",
    page_icon="🔍",
    layout="wide",
)
url: str = st.secrets.get("LANGGRAPH_URL", "http://localhost:2024")
with st.sidebar:
    st.write(f"LangGraph URL: {url}")
# Prepare configuration with callbacks
runnable_config: RunnableConfig = {
    "configurable": {
        "user_id": st.session_state.get("user_id", "aw+test@eddolearning.com"),
        "thread_id": st.session_state.get("thread_id", None),
    }
}
graph_name: str = "graph"
client: LangGraphClient = get_client(
    url=url, api_key=st.secrets.get("LANGGRAPH_API_KEY")
)
sync_client: SyncLangGraphClient = get_sync_client(url=url)
remote_graph: RemoteGraph = RemoteGraph(
    graph_name, client=client, sync_client=sync_client
)

# Initialize session state
with st.sidebar:
    if "thread_id" not in st.session_state:
        thread: Dict[str, Any] = cast(Dict[str, Any], sync_client.threads.create())
        st.session_state.thread_id = thread["thread_id"]
    with st.expander("Thread details"):
        st.caption(f"Thread ID: `{st.session_state.thread_id}`")
        thread_state: ThreadState = sync_client.threads.get_state(
            st.session_state.thread_id
        )
        st.write(thread_state)

# Main chat interface
st.title("Agent Arcade Tools Chat 🤖")

# Initialize messages
current_thread_state: ThreadState = sync_client.threads.get_state(
    st.session_state.thread_id
)
values = cast(Dict[str, Any], current_thread_state.get("values"))
if values is not None:
    messages = values.get("messages", [])
    # Display messages from thread state
    if messages:
        for message in messages:
            role = message.get("type", "")
            content = message.get("content", "")
            metadata = message.get("response_metadata", {})

            if role == "user":
                st.chat_message("user").write(content)
            elif role == "ai":
                with st.chat_message("assistant"):
                    st.write(content)
                    if metadata:
                        with st.expander("Debug: Message Metadata", expanded=False):
                            st.json(metadata)
    else:
        st.chat_message("assistant").write("Hi there! What's going on?")


# Stream event types
class StreamEvent(Protocol):
    """Protocol for stream events from LangGraph."""


class StreamUpdate(TypedDict):
    """Type definition for stream updates."""

    type: Literal["content", "state"]
    data: str
    metadata: Dict[str, Any]


def handle_stream(response: Iterator[Any]) -> None:
    """Process the stream response from LangGraph.

    Args:
        response: Iterator of stream events from LangGraph
    """
    # Create message container and placeholders
    message_container = st.chat_message("assistant")
    message_placeholder = message_container.empty()
    metadata_container = message_container.container()

    current_content = ""
    current_metadata: Dict[str, Any] = {}

    def stream_generator() -> Iterator[StreamUpdate]:
        """Generate stream of content and metadata updates."""
        nonlocal current_content, current_metadata

        for event in response:
            # Handle message tuples (token-by-token streaming)
            if event.event == "messages":
                if isinstance(event.data, tuple) and len(event.data) == 2:
                    message, metadata = event.data
                    if isinstance(message, AIMessageChunk):
                        current_content += str(message.content)
                        if metadata:
                            current_metadata.update(metadata)
                        yield StreamUpdate(
                            type="content",
                            data=current_content,
                            metadata=current_metadata,
                        )

            # Handle values (complete state updates)
            elif event.event == "values":
                messages = event.data.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if last_message.get("type") == "ai":
                        content = last_message.get("content", "")
                        metadata = {
                            "token_usage": last_message.get(
                                "response_metadata", {}
                            ).get("token_usage", {}),
                            "model_name": last_message.get("response_metadata", {}).get(
                                "model_name", ""
                            ),
                            "usage_metadata": last_message.get("usage_metadata", {}),
                        }
                        current_content = content
                        current_metadata = metadata
                        yield StreamUpdate(
                            type="state", data=content, metadata=metadata
                        )

    # Stream both content and metadata using write_stream
    for update in st.write_stream(stream_generator()):
        update_typed = cast(StreamUpdate, update)
        # Update message content
        message_placeholder.markdown(update_typed["data"])

        # Update metadata if present
        if update_typed["metadata"]:
            with metadata_container:
                with st.expander("Debug: Message Metadata", expanded=False):
                    st.json(update_typed["metadata"])


# Chat input
if st.session_state.get("thread_id") and (
    prompt := st.chat_input("How can I help you today?")
):
    # Add user message
    st.chat_message("user").write(prompt)
    user_message = HumanMessage(content=prompt)
    try:
        # Get response from LangGraph
        # Only send the new message to the graph, since it maintains its own state
        input_state: list[BaseMessage] = [user_message]

        # Stream the response using the correct method
        response = sync_client.runs.stream(
            thread_id=st.session_state.thread_id,
            assistant_id=graph_name,
            input={"messages": input_state},
            stream_mode=["values", "messages"],  # Enable both streaming modes
            config=runnable_config,
        )

        handle_stream(response)

    except Exception as e:
        st.error(f"Error: {str(e)}")
