"""Streamlit frontend for the LangGraph backend."""

import os
import uuid
from datetime import datetime
from typing import cast

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent_arcade_tools.backend.graph import graph

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None

# Page config
st.set_page_config(
    page_title="Agent Arcade Tools Chat",
    page_icon="🤖",
    layout="wide",
)

# Sidebar for configuration and thread management
with st.sidebar:
    st.title("Configuration")

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

# Main chat interface
if st.session_state.current_thread_id:
    thread_data = st.session_state.threads[st.session_state.current_thread_id]
    st.title(f"Chat: {thread_data['name']} 🤖")
    st.caption(
        f"Thread ID: {st.session_state.current_thread_id[:8]} • Created: {datetime.fromisoformat(thread_data['created_at']).strftime('%Y-%m-%d %H:%M')}"
    )
else:
    st.title("Agent Arcade Tools Chat 🤖")
    st.info("Create a new thread to start chatting")

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
        response_container = st.container()

        # Initialize the StreamlitCallbackHandler with expanded thoughts
        st_callback = StreamlitCallbackHandler(
            parent_container=response_container,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
            max_thought_containers=10,
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
                    },
                    "callbacks": [st_callback],
                },
            )

            # Get response from LangGraph
            response = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )

            # Extract the last message (should be the AI response)
            if response and "messages" in response and response["messages"]:
                ai_message = response["messages"][-1]
                if isinstance(ai_message, AIMessage):
                    # Store the final response in chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": ai_message.content}
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
            st.rerun()

        if st.button("Delete Current Thread", type="secondary"):
            del st.session_state.threads[st.session_state.current_thread_id]
            st.session_state.current_thread_id = None
            st.session_state.messages = []
            st.rerun()

# Add environment variable check
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
