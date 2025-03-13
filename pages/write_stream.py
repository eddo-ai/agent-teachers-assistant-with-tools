"""Streamlit page demonstrating st.write_stream with mock LangGraph streaming."""

import random
import time
from typing import Iterator

import streamlit as st
from langchain_core.messages import AIMessageChunk

st.title("Write Stream Demo")

st.write(
    """
This demo shows how to use `st.write_stream` with different types of streaming data that might come from a LangGraph.
Choose a streaming mode to see how different types of data are handled.
"""
)


def mock_message_stream() -> Iterator[tuple[AIMessageChunk, dict]]:
    """Simulate a message stream from an LLM."""
    message = "I am thinking about your request... Let me process that step by step..."
    words = message.split()

    for i, word in enumerate(words):
        time.sleep(0.5)  # Simulate LLM thinking time
        chunk = AIMessageChunk(content=word + " ")
        metadata = {"node": "agent", "step": i, "confidence": random.random()}
        yield (chunk, metadata)


def mock_value_stream() -> Iterator[dict]:
    """Simulate a value stream showing full graph state."""
    states = [
        {"agent": {"thought": "Initial analysis"}, "memory": []},
        {"agent": {"thought": "Gathering data"}, "memory": ["context_1"]},
        {
            "agent": {"thought": "Processing information"},
            "memory": ["context_1", "context_2"],
        },
        {
            "agent": {"thought": "Finalizing response"},
            "memory": ["context_1", "context_2", "result"],
        },
    ]

    for state in states:
        time.sleep(1)  # Simulate processing time
        yield state


def mock_update_stream() -> Iterator[dict]:
    """Simulate an update stream showing only changes."""
    updates = [
        {"node": "agent", "update": {"status": "starting"}},
        {"node": "tool", "update": {"action": "search"}},
        {"node": "memory", "update": {"stored": "new_data"}},
        {"node": "agent", "update": {"status": "complete"}},
    ]

    for update in updates:
        time.sleep(0.8)  # Simulate processing time
        yield update


def mock_debug_stream() -> Iterator[dict]:
    """Simulate a debug stream with detailed events."""
    events = [
        {"type": "node_enter", "node": "agent", "timestamp": time.time()},
        {
            "type": "tool_call",
            "tool": "search",
            "args": {"query": "test"},
            "timestamp": time.time(),
        },
        {
            "type": "memory_access",
            "operation": "read",
            "keys": ["context"],
            "timestamp": time.time(),
        },
        {
            "type": "node_exit",
            "node": "agent",
            "result": "success",
            "timestamp": time.time(),
        },
    ]

    for event in events:
        time.sleep(1.2)  # Simulate processing time
        yield event


# Stream type selector
stream_type = st.selectbox(
    "Select stream type", ["Messages", "Values", "Updates", "Debug"]
)

if st.button("Start Streaming"):
    # Create placeholder for streaming output
    with st.container():
        if stream_type == "Messages":
            st.write("### Message Stream")
            st.write("Showing token-by-token output with metadata:")

            for chunk, metadata in mock_message_stream():
                st.write_stream({"content": chunk.content, "metadata": metadata})

        elif stream_type == "Values":
            st.write("### Value Stream")
            st.write("Showing complete graph state at each step:")

            for state in mock_value_stream():
                st.write_stream(state)

        elif stream_type == "Updates":
            st.write("### Update Stream")
            st.write("Showing only state changes:")

            for update in mock_update_stream():
                st.write_stream(update)

        else:  # Debug
            st.write("### Debug Stream")
            st.write("Showing detailed debug events:")

            for event in mock_debug_stream():
                st.write_stream(event)

st.write(
    """
### How it works

This demo shows four different types of streams you might encounter when working with LangGraph:

1. **Messages Stream**: Shows token-by-token output from an LLM with associated metadata
2. **Values Stream**: Shows the complete state of the graph at each step
3. **Updates Stream**: Shows only the changes in state, more efficient than values stream
4. **Debug Stream**: Shows detailed debug information about graph execution

Each stream type uses `st.write_stream()` to display the data in real-time as it's generated.
"""
)
