"""Define the workflow graph for the agent backend."""

from langgraph.graph import END, START, MessagesState, StateGraph

from agent_arcade_tools.backend.configuration import AgentConfigurable


def create_workflow() -> StateGraph:
    """Create and return the workflow graph.

    This is a simplified version of the graph for testing purposes.

    Returns:
        StateGraph: The workflow graph
    """
    # Build the workflow graph using StateGraph
    graph = StateGraph(MessagesState, AgentConfigurable)

    # Add placeholder nodes for testing
    graph.add_node("agent", lambda x: x)
    graph.add_node("authorization", lambda x: x)
    graph.add_node("tools", lambda x: x)

    # Define the edges and control flow between nodes
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "authorization")
    graph.add_edge("authorization", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("agent", END)

    return graph


# Create a workflow instance for testing
workflow = create_workflow()
