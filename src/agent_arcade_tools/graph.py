import os
from datetime import datetime, timezone
from typing import cast, Dict, Any, List, Union, Sequence, Callable

from agent_arcade_tools.configuration import AgentConfigurable
from langchain_arcade import ArcadeToolManager
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.tools import BaseTool
from agent_arcade_tools.state import InputState, State
from agent_arcade_tools.utils import load_chat_model
from agent_arcade_tools.tools import retrieve_instructional_materials

# Initialize the Arcade Tool Manager with your API key
arcade_api_key = os.getenv("ARCADE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

toolkit = ArcadeToolManager(api_key=cast(Dict[str, Any], arcade_api_key))
configuration = AgentConfigurable()

# Retrieve tools compatible with LangGraph
tools: Sequence[Union[BaseTool, Callable[..., Any]]] = [
    retrieve_instructional_materials,
    *toolkit.get_tools(),
]
tool_node = ToolNode(tools)


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = AgentConfigurable.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(tools)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if (
        state.is_last_step
        and isinstance(response, AIMessage)
        and hasattr(response, "tool_calls")
        and response.tool_calls
    ):
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # If there are tool calls, return them directly without placeholder messages
    if (
        isinstance(response, AIMessage)
        and hasattr(response, "tool_calls")
        and response.tool_calls
    ):
        return {"messages": [response]}

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


def should_continue(state: State, config: Dict[str, Any]) -> str:
    """Function to determine the next step based on the model's response"""
    last_message = state.messages[-1]
    if (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        return "check_auth"
    # If no tool calls are present, end the workflow
    return END


async def check_auth(state: State, config: Dict[str, Any]) -> State:
    """Check if the user has authorized the tool."""
    user_id = config["configurable"].get("user_id")
    last_message = state.messages[-1]
    if (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        tool_name = last_message.tool_calls[0]["name"]
        auth_response = toolkit.authorize(tool_name, user_id)
        if auth_response.status != "completed":
            state.auth_url = auth_response.url
        else:
            state.auth_url = None
    return state


async def authorize(state: State, config: Dict[str, Any]) -> State:
    """Function to handle tool authorization"""
    user_id = config["configurable"].get("user_id")
    last_message = state.messages[-1]
    if (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        tool_name = last_message.tool_calls[0]["name"]
        auth_response = toolkit.authorize(tool_name, user_id)
        if auth_response.status != "completed":
            auth_message = f"Please authorize the application in your browser:\n\n {state.auth_url}"
            raise NodeInterrupt(auth_message)
    return state


# Build the workflow graph
workflow = StateGraph(InputState, State)

# Add nodes to the graph
workflow.add_node("agent", cast(Runnable[Any, Any], call_model))
workflow.add_node("tools", tool_node)
workflow.add_node("authorization", cast(Runnable[Any, Any], authorize))
workflow.add_node("check_auth", cast(Runnable[Any, Any], check_auth))

# Define the edges and control flow
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["check_auth", END])
workflow.add_edge("check_auth", "authorization")
workflow.add_edge("authorization", "tools")
workflow.add_edge("tools", "agent")

# Compile the graph with an interrupt after the authorization node
# so that we can prompt the user to authorize the application
graph = workflow.compile()
