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

import logging
import os
from typing import Any, Callable, Dict, Sequence, Union, cast

from arcadepy.types.shared.authorization_response import AuthorizationResponse
from langchain_arcade import ToolManager
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent_arcade_tools.configuration import AgentConfigurable
from agent_arcade_tools.tools import retrieve_instructional_materials
from agent_arcade_tools.utils import load_chat_model

# Configure logging
logger = logging.getLogger(__name__)


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors and structure to log messages."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: f"{grey}🔍 %(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.INFO: f"{blue}ℹ️  %(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.WARNING: f"{yellow}⚠️  %(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.ERROR: f"{red}❌ %(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.CRITICAL: f"{bold_red}🚨 %(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with the appropriate color and emoji."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


formatter = ColoredFormatter()
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

tool_manager = ToolManager(api_key=cast(Dict[str, Any], arcade_api_key))
tool_manager.init_tools(
    toolkits=["Google", "Github", "Search", "Web", "CodeSandbox", "X"]
)
configuration = AgentConfigurable()

# Retrieve tools compatible with LangGraph
tools: Sequence[Union[BaseTool, Callable[..., Any]]] = [
    retrieve_instructional_materials,
    *tool_manager.to_langchain(use_interrupts=True),
]
logger.debug(f"🔧 Tools: {tools}")
tool_node = ToolNode(tools)


def call_agent(
    state: MessagesState, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Process the current state and generate an agent response.

    This function:
    1. Extracts the model configuration
    2. Loads and configures the model with available tools
    3. Processes the message history
    4. Generates and logs the agent's response

    Args:
        state: Current conversation state containing message history
        config: Configuration including model settings and user info

    Returns:
        dict: Updated state with the agent's response appended
    """
    if len(state["messages"]) == 0:
        return {"messages": [AIMessage(content="Hello! How can I help you today?")]}
    configurable: AgentConfigurable = AgentConfigurable.from_runnable_config(config)
    model: str = configurable.model
    logger.info(f"🤖 Using model: {model}")

    # Log the current state
    logger.debug(f"📥 Input messages: {[msg.content for msg in state['messages']]}")

    model_with_tools: Runnable[Sequence[BaseMessage], BaseMessage] = load_chat_model(
        model
    ).bind_tools(tools)
    logger.debug(
        f"🔧 Model configured with tools: {[getattr(tool, 'name', str(tool)) for tool in tools]}"
    )

    messages: Sequence[BaseMessage] = state["messages"]
    logger.debug("🎯 Invoking model with messages and streaming config")
    response: BaseMessage = model_with_tools.invoke(
        messages,
        config=config,  # Pass through the config for streaming callbacks
    )

    logger.info(f"📤 Model response type: {type(response).__name__}")
    logger.debug(f"📤 Model response content: {response.content}")

    # Return all messages including the new response
    return {"messages": [*messages, response]}


def should_continue(state: MessagesState) -> str:
    """Determine the next step in the workflow based on the last message.

    This function analyzes the last message to decide the next action:
    - If it's a ToolMessage, continue to agent
    - If not an AI message, end the workflow
    - If contains tool calls requiring auth, go to authorization
    - If contains tool calls (no auth needed), proceed to tools
    - Otherwise, end the workflow

    Args:
        state: Current conversation state

    Returns:
        str: Next node identifier ('agent', 'authorization', 'tools', or END)
    """
    last_message = state["messages"][-1]

    # If the last message is a ToolMessage, continue to agent
    if hasattr(last_message, "name") and hasattr(last_message, "tool_call_id"):
        logger.info("🔧 Tool message received, continuing to agent")
        return "agent"

    # If not an AI message, end the workflow
    if not isinstance(last_message, AIMessage):
        return END

    # Check for tool calls in AI message
    tool_calls = last_message.additional_kwargs.get("tool_calls", [])

    if tool_calls:
        logger.info(f"🔧 Found {len(tool_calls)} tool calls")
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            if tool_name and tool_manager.requires_auth(tool_name):
                logger.info(f"🔐 Tool {tool_name} requires authorization")
                return "authorization"
        logger.info(" Proceeding to tool execution")
        return "tools"  # Proceed to tool execution if no authorization is needed
    return END  # End the workflow if no tool calls are present


def authorize(
    state: MessagesState, *, config: RunnableConfig | None = None
) -> dict[str, list[Any]]:
    """Handle authorization for tools that require it.

    This function:
    1. Checks if tools in the last message require authorization
    2. Attempts to authorize each tool
    3. If authorization is pending, interrupts with auth URL
    4. If all tools are authorized, continues the workflow

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

    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        if not tool_manager.requires_auth(tool_name):
            continue
        auth_response: AuthorizationResponse = tool_manager.authorize(
            tool_name, user_id
        )
        if auth_response.status != "completed":
            # Immediately interrupt with the authorization URL
            if not auth_response.url:
                raise ValueError("No authorization URL returned")
            interrupt(
                {
                    "message": f"Visit the following URL to authorize: {auth_response.url}",
                    "auth_url": auth_response.url,
                    "type": "authorization",
                }
            )

    # If we get here, all tools are authorized
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
    if not isinstance(state["messages"][-1], AIMessage):
        return {"messages": state["messages"]}

    # Get the tool responses from the tool node
    tool_response = tool_node.invoke(state)

    # Combine existing messages with tool responses
    return {"messages": [*state["messages"], *tool_response["messages"]]}


# Build the workflow graph using StateGraph
workflow = StateGraph(MessagesState, AgentConfigurable)

# Add nodes (steps) to the graph
workflow.add_node("agent", call_agent)
workflow.add_node("authorization", authorize)
workflow.add_node("tools", tool_node)

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
graph = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    # Generate visual representations of the graph
    graph.get_graph().draw_mermaid_png()
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
