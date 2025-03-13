"""Define the workflow graph and control flow for the agent.

This module implements a LangGraph-based workflow for handling conversational AI interactions
with tool usage and authorization management. The workflow is structured as a directed graph
with nodes for agent calls, authorization checks, and tool execution.

Key Components:
    - StateGraph: Manages the flow between different states (agent, authorization, tools)
    - ToolNode: Handles the execution of various tools (Google, Github, Search APIs)
    - Authorization: Manages tool access permissions and user authorization flows
    - Message State: Maintains conversation history and tool responses

Flow:
    1. Agent receives input and generates response
    2. If tools are needed, checks for authorization
    3. If authorization needed, interrupts with auth URL
    4. Once authorized, executes tools
    5. Continues conversation with tool results

Environment Variables:
    ARCADE_API_KEY: API key for Arcade Tool Manager
    OPENAI_API_KEY: API key for OpenAI services
"""

import json
import logging
import os
from typing import Any, Callable, Dict, Sequence, Union, cast

from arcadepy.types.shared.authorization_response import AuthorizationResponse
from langchain_arcade import ArcadeToolManager
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent_arcade_tools.configuration import AgentConfigurable
from agent_arcade_tools.tools import retrieve_instructional_materials

# Configure logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)  # Set default to DEBUG


def set_log_level(level: str) -> None:
    """Set the log level for the graph module.

    Args:
        level: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    logger.setLevel(getattr(logging, level.upper()))


# Initialize the Arcade Tool Manager with your API key
arcade_api_key: str | None = os.getenv("ARCADE_API_KEY")
openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

tool_manager = ArcadeToolManager(api_key=cast(Dict[str, Any], arcade_api_key))
configuration = AgentConfigurable()

# Retrieve tools compatible with LangGraph
tools: Sequence[Union[BaseTool, Callable[..., Any]]] = [
    retrieve_instructional_materials,
    *tool_manager.get_tools(toolkits=["Google", "Github", "Search"]),
]
tool_node = ToolNode(tools)

# Initialize the model
model = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0,
    streaming=True,
)


def process_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Process messages to ensure they are in the correct format for the agent.

    Args:
        messages: The messages to process

    Returns:
        The processed messages
    """
    processed_messages: list[BaseMessage] = []

    logger.debug(f"Processing {len(messages)} messages")
    for msg in messages:
        if isinstance(msg, ToolMessage):
            logger.debug(f"Processing tool message: {msg.name}")
            # Ensure tool message content is a string
            content: str = ""
            msg_content: Any = msg.content

            # Handle different content types
            if isinstance(msg_content, dict):
                content = json.dumps(msg_content, indent=2)
            else:
                content = str(msg_content)

            processed_messages.append(
                ToolMessage(
                    content=content,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    status=msg.status,
                )
            )
        else:
            # For non-tool messages, preserve the original format
            processed_messages.append(msg)

    return processed_messages


def call_agent(
    state: MessagesState, *, config: RunnableConfig | None = None
) -> MessagesState:
    """Call the agent with the current conversation state.

    Args:
        state: Current conversation state containing messages
        config: Optional configuration containing model info

    Returns:
        Updated conversation state with agent's response
    """
    messages = state["messages"]
    processed_messages = process_messages(messages)

    try:
        # Use model from config if available, otherwise use default model
        configurable = config.get("configurable", {}) if config else {}
        model_name = configurable.get("model")
        current_model = load_chat_model(model_name) if model_name else model

        # Invoke the model with the processed messages
        # Following LangGraph SDK best practices
        response = current_model.invoke(
            input=processed_messages,
            config=RunnableConfig(
                configurable={"tools": tool_manager.get_tools(), "stream": False}
            ),
        )

        logger.debug(f"Model response: {response}")
        return MessagesState(
            messages=cast(list[AnyMessage], processed_messages + [response])
        )
    except Exception as e:
        logger.error(f"Error calling model: {e}")
        raise


def should_continue(state: MessagesState) -> str:
    """Determine the next step in the workflow based on the last message.

    This function analyzes the last message to decide the next action:
    - If last message is a tool message, continue to agent
    - If last message is not an AI message, end the workflow
    - If contains tool calls, proceed to tools
    - If any tool requires auth, go to authorization
    - Otherwise, end the workflow

    Args:
        state: Current conversation state

    Returns:
        str: Next node identifier ('authorization', 'tools', 'agent', or END)
    """
    last_message = state["messages"][-1]

    # If last message is a tool message, continue to agent
    if isinstance(last_message, ToolMessage):
        logger.debug("🔄 Continuing to agent: Last message is a tool message")
        return "agent"

    # If not an AI message, end the workflow
    if not isinstance(last_message, AIMessage):
        logger.debug("🔄 Ending workflow: Last message is not an AI message")
        return END

    # Check for tool calls in additional_kwargs
    tool_calls = last_message.additional_kwargs.get("tool_calls", [])
    if tool_calls:
        logger.info(f"🔧 Found {len(tool_calls)} tool calls")
        # First check if any tool requires authorization
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            if tool_manager.requires_auth(tool_name):
                logger.info(f"🔐 Tool {tool_name} requires authorization")
                return "authorization"
        # If no tool requires authorization, proceed to tools
        logger.info("🔧 Proceeding to tool execution")
        return "tools"

    logger.debug("🔄 Ending workflow: No tool calls present")
    return END  # End the workflow if no tool calls are present


def authorize(
    state: MessagesState, *, config: RunnableConfig | None = None
) -> dict[str, list[Any]]:
    """Handle authorization for tools that require it.

    Args:
        state: Current conversation state
        config: Optional configuration containing user info

    Returns:
        dict: Updated state

    Raises:
        ValueError: If user ID is missing or no auth URL is returned
        interrupt: When authorization is needed (contains auth URL)
    """
    configurable = config.get("configurable", {}) if config else {}
    user_id = configurable.get("user_id")
    if not user_id:
        raise ValueError("User ID not found in configuration")
    if not isinstance(state["messages"][-1], AIMessage):
        return {"messages": state["messages"]}

    # Track if any tool needs authorization
    needs_auth = False
    auth_url = None

    # Get tool calls from additional_kwargs
    last_message = state["messages"][-1]
    tool_calls = last_message.additional_kwargs.get("tool_calls", [])

    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        if not tool_manager.requires_auth(tool_name):
            continue
        needs_auth = True
        auth_response: AuthorizationResponse = tool_manager.authorize(
            tool_name, user_id
        )
        if auth_response.status != "completed":
            if not auth_response.url:
                raise ValueError("No authorization URL returned")
            auth_url = auth_response.url
            break

    if needs_auth and auth_url:
        # Send interrupt with authorization URL
        raise interrupt(
            {
                "message": f"Visit the following URL to authorize: {auth_url}",
                "auth_url": auth_url,
                "type": "authorization",
            }
        )

    # If we get here, all tools are authorized
    logger.info("🔐 All required tools are authorized")
    return {"messages": state["messages"]}


def handle_tools(state: MessagesState) -> dict[str, Sequence[BaseMessage]]:
    """Execute tools and process their responses.

    This function:
    1. Validates the last message is from the AI
    2. Executes any tool calls in the message
    3. Adds tool responses to the message history

    Args:
        state: Current conversation state

    Returns:
        dict: Updated state with tool responses
    """
    messages = state["messages"]
    if not isinstance(messages[-1], AIMessage):
        return {"messages": messages}

    # Get tool calls from additional_kwargs
    tool_calls = messages[-1].additional_kwargs.get("tool_calls", [])
    if not tool_calls:
        return {"messages": messages}

    # Convert tool calls to the format expected by ToolNode
    converted_message = AIMessage(
        content=messages[-1].content,
        tool_calls=[
            {
                "name": tool_call["function"]["name"],
                "args": json.loads(
                    tool_call["function"]["arguments"]
                ),  # Use json.loads instead of eval
                "id": tool_call["id"],
                "type": "tool_call",  # Use "tool_call" type consistently
            }
            for tool_call in tool_calls
        ],
    )

    # Create a new state with the converted message
    tool_state = {"messages": [*messages[:-1], converted_message]}

    # Execute tools using the global ToolNode
    tool_response = tool_node.invoke(tool_state)

    # Combine existing messages with tool responses
    return {"messages": [*messages, *tool_response["messages"]]}


# Build the workflow graph using StateGraph
workflow = StateGraph(MessagesState, AgentConfigurable)

# Add nodes (steps) to the graph
workflow.add_node("agent", call_agent)
workflow.add_node("authorization", authorize)
workflow.add_node(
    "tools", handle_tools
)  # Use handle_tools instead of tool_node directly

# Define the edges and control flow between nodes
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", should_continue, ["authorization", "tools", END]
)
workflow.add_edge("authorization", "tools")
workflow.add_edge("tools", "agent")

# Set up memory for checkpointing the state
memory = MemorySaver()

# Compile the graph with the checkpointer
graph: CompiledStateGraph = workflow.compile(checkpointer=memory)


def load_chat_model(model_name: str) -> ChatOpenAI:
    """Load and configure a chat model.

    Args:
        model_name: Name of the model to load

    Returns:
        Configured chat model
    """
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        streaming=True,
    )


if __name__ == "__main__":
    # Generate visual representations of the graph
    graph.get_graph().draw_mermaid_png()
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
