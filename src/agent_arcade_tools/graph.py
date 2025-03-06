"""Define the workflow graph and control flow for the agent."""

import os
from ast import Constant
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from arcadepy.types.shared.authorization_response import AuthorizationResponse
from langchain_arcade import ArcadeToolManager
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from agent_arcade_tools.configuration import Configuration
from agent_arcade_tools.tools import retrieve_instructional_materials
from agent_arcade_tools.utils import load_chat_model

# Initialize the Arcade Tool Manager with your API key
arcade_api_key: str | None = os.getenv("ARCADE_API_KEY")
openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

toolkit = ArcadeToolManager(api_key=cast(Dict[str, Any], arcade_api_key))

# Retrieve tools compatible with LangGraph
tools: Sequence[Union[BaseTool, Callable[..., Any]]] = [
    retrieve_instructional_materials,
    *toolkit.get_tools(toolkits=["Google"]),
]
tool_node = ToolNode(tools)


class AgentState(MessagesState):
    authorization_id: Optional[str] = None
    tool_call_name: Optional[str] = None


async def call_model(
    state: AgentState, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration: Configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(tools)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message: str = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response: AIMessage = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state["messages"]], config
        ),
    )
    return {"messages": [response]}


def request_auth(tool_name: str, user_id: str) -> Command:
    """Command node to request authorization for a tool."""
    auth_response: AuthorizationResponse = toolkit.authorize(tool_name, user_id)
    return Command(
        goto="tools",
        update=ToolMessage(
            f"Please authorize the application in your browser:\n\n {auth_response.url}"
        ),
    )


def _route_model_output(state: AgentState, config: RunnableConfig) -> str | Constant:
    if (
        not isinstance(state["messages"][-1], AIMessage)
        or not state["messages"][-1].tool_calls
    ):
        return END
    else:
        return "tools"


def call_tools(state: AgentState, config: RunnableConfig) -> Command | AgentState:
    """Command node to call a tool."""
    if (
        not isinstance(state["messages"][-1], AIMessage)
        or not state["messages"][-1].tool_calls
    ):
        return state
    for tool_call in state["messages"][-1].tool_calls:
        requires_auth: bool = toolkit.requires_auth(tool_call["name"])
        if requires_auth:
            user_id: Optional[str] = Configuration.from_runnable_config(config).user_id
            if user_id is None:
                raise ValueError("User ID is required for authorization.")
            should_continue: bool = interrupt(request_auth(tool_call["name"], user_id))
            if should_continue:
                return Command(goto="tools")
            else:
                return Command(
                    goto="agent",
                    update=ToolMessage(content="Authorization is not complete."),
                    resume={"authorization_id": auth_response.id},
                )
    else:
        return Command(
            goto="tools",
            update=ToolMessage(f"Yep, I called the {tool_call['name']} tool. Trust."),
        )


# Build the workflow graph
workflow = StateGraph(MessagesState, Configuration)

# Add nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("call_tools", call_tools)
workflow.add_node("tools", tool_node)

# Add edges with conditional routing
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", _route_model_output, {"tools": "call_tools", END: END}
)
workflow.add_edge("call_tools", "tools")
workflow.add_edge("tools", "agent")
workflow.add_edge("agent", END)

# Set the entry point
workflow.set_entry_point("agent")

# Compile the graph
graph: CompiledStateGraph = workflow.compile()
