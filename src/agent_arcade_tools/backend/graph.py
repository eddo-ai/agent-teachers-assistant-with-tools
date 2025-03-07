"""Define the workflow graph and control flow for the agent."""

import logging
import os
from typing import Any, Callable, Dict, Sequence, Union, cast

from arcadepy.types.shared.authorization_response import AuthorizationResponse
from langchain_arcade import ArcadeToolManager
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent_arcade_tools.backend.configuration import AgentConfigurable
from agent_arcade_tools.backend.tools import retrieve_instructional_materials
from agent_arcade_tools.backend.utils import load_chat_model

logger: logging.Logger = logging.getLogger(__name__)

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


def call_agent(
    state: MessagesState, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the agent and get a response."""
    configurable: AgentConfigurable = AgentConfigurable.from_runnable_config(config)
    model: str = configurable.model
    logger.info(f"Using model: {model}")
    model_with_tools: Runnable = load_chat_model(model).bind_tools(tools)
    logger.debug(f"Model with tools: {model_with_tools}")
    messages: Sequence[BaseMessage] = state["messages"]
    response: BaseMessage = model_with_tools.invoke(messages)
    # Return all messages including the new response
    return {"messages": [*messages, response]}


def should_continue(state: MessagesState) -> str:
    """Determine the next step in the workflow based on the last message."""
    if not isinstance(state["messages"][-1], AIMessage):
        return END
    if state["messages"][-1].tool_calls:
        for tool_call in state["messages"][-1].tool_calls:
            if tool_manager.requires_auth(tool_call["name"]):
                return "authorization"
        return "tools"  # Proceed to tool execution if no authorization is needed
    return END  # End the workflow if no tool calls are present


# Function to handle authorization for tools that require it
def authorize(
    state: MessagesState, *, config: RunnableConfig | None = None
) -> dict[str, list[Any]]:
    """Handle authorization for tools that require it."""
    configurable = config.get("configurable", {}) if config else {}
    user_id = configurable.get("user_id")
    if not user_id:
        raise ValueError("User ID not found in configuration")
    if not isinstance(state["messages"][-1], AIMessage):
        return {"messages": []}
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        if not tool_manager.requires_auth(tool_name):
            continue
        auth_response: AuthorizationResponse = tool_manager.authorize(
            tool_name, user_id
        )
        if auth_response.status != "completed":
            # Prompt the user to visit the authorization URL
            interrupt(f"Visit the following URL to authorize: {auth_response.url}")

            # Wait for the user to complete the authorization
            # and then check the authorization status again
            if not auth_response.id:
                raise ValueError(
                    "No authorization ID returned from authorization request"
                )
            if response := tool_manager.wait_for_auth(auth_response.id):
                if response.status != "completed":
                    raise ValueError("Authorization request not completed")
                if not tool_manager.is_authorized(auth_response.id):
                    # This stops execution if authorization fails
                    raise ValueError("Tool authorization failed")

    return {"messages": []}


# Function to handle tool responses
def handle_tools(state: MessagesState) -> dict[str, Sequence[BaseMessage]]:
    """Execute tools and add their responses to the message history."""
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
    graph.get_graph().draw_mermaid_png()
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
