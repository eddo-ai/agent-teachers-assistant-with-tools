import os
from datetime import datetime, timezone
from typing import cast

from agent_arcade_tools.configuration import AgentConfigurable
from langchain_arcade import ArcadeToolManager
from langgraph.errors import NodeInterrupt
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from typing import Annotated, Any, Dict, List, Optional
from langchain_community.vectorstores import AzureSearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from agent_arcade_tools.state import InputState, State
from agent_arcade_tools.utils import load_chat_model
from agent_arcade_tools.tools import retrieve_instructional_materials

# Initialize the Arcade Tool Manager with your API key
arcade_api_key = os.getenv("ARCADE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

toolkit = ArcadeToolManager(api_key=arcade_api_key)
configuration = AgentConfigurable()

# Retrieve tools compatible with LangGraph
tools = [retrieve_instructional_materials, *toolkit.get_tools()]
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
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # If there are tool calls, return them directly without placeholder messages
    if response.tool_calls:
        return {"messages": [response]}

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


def should_continue(state: State, config: dict):
    """Function to determine the next step based on the model's response"""
    last_message = state.messages[-1]
    if last_message.tool_calls:
        return "check_auth"
    # If no tool calls are present, end the workflow
    return END


def check_auth(state: State, config: dict):
    user_id = config["configurable"].get("user_id")
    tool_name = state.messages[-1].tool_calls[0]["name"]
    auth_response = toolkit.authorize(tool_name, user_id)
    if auth_response.status != "completed":
        state.auth_url = auth_response.url
    else:
        state.auth_url = None
    return state


def authorize(state: State, config: dict):
    """Function to handle tool authorization"""
    user_id = config["configurable"].get("user_id")
    tool_name = state.messages[-1].tool_calls[0]["name"]
    auth_response = toolkit.authorize(tool_name, user_id)
    if auth_response.status != "completed":
        auth_message = (
            f"Please authorize the application in your browser:\n\n {state.auth_url}"
        )
        raise NodeInterrupt(auth_message)
    return state


async def retrieve_instructional_materials(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> Optional[List[Dict[str, Any]]]:
    """Search for instructional materials from OpenSciEd that are relevant to the user's query.

    Args:
        query: The search query to find relevant instructional materials
            (e.g. "What materials are available for teaching about forces and motion?").
        config: Configuration for the runnable.

    Returns:
        A list of relevant documents if found, None if configuration is missing.

    Raises:
        ValueError: If required Azure Search configuration is missing.
    """
    # Get required environment variables
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    semantic_config = os.getenv(
        "AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME", "unit_lesson"
    )

    # Validate required environment variables
    if not all([endpoint, api_key, index_name]):
        raise ValueError(
            "Missing required Azure Search configuration. Please set AZURE_AI_SEARCH_ENDPOINT, "
            "AZURE_AI_SEARCH_API_KEY, and AZURE_AI_SEARCH_INDEX_NAME environment variables."
        )

    # Initialize Azure Search client
    embeddings = OpenAIEmbeddings()
    retriever = AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=api_key,
        index_name=index_name,
        embedding_function=embeddings,
    ).as_retriever(
        search_type="semantic_hybrid",
        semantic_configuration_name=semantic_config,
    )
    return retriever.get_relevant_documents(query)


# Build the workflow graph
workflow = StateGraph(InputState, State)

# Add nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("authorization", authorize)
workflow.add_node("check_auth", check_auth)

# Define the edges and control flow
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["check_auth", END])
workflow.add_edge("check_auth", "authorization")
workflow.add_edge("authorization", "tools")
workflow.add_edge("tools", "agent")

# Compile the graph with an interrupt after the authorization node
# so that we can prompt the user to authorize the application
graph = workflow.compile()
