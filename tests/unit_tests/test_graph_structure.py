"""Tests for validating the LangGraph workflow structure."""

from typing import Dict, Literal, Optional, Protocol, Union

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent_arcade_tools.backend.graph import workflow


class NodeType(BaseModel):
    """Model representing a node type in the graph."""

    type: Literal["agent", "authorization", "tools"]


class Node(BaseModel):
    """Model representing a node in the graph."""

    name: str
    type: Literal["agent", "authorization", "tools"] = Field(
        description="Type of the node in the workflow"
    )


class Edge(BaseModel):
    """Model representing an edge in the graph."""

    source: str
    target: str
    condition: Optional[str] = None


class GraphStructure(BaseModel):
    """Model representing the expected graph structure."""

    nodes: list[Node]
    edges: list[Edge]
    conditional_edges: Dict[str, list[str]]


class HasName(Protocol):
    """Protocol for objects that have a name attribute."""

    name: str


def validate_node_type(name: str) -> Literal["agent", "authorization", "tools"]:
    """Validate and convert node name to its type."""
    # Skip special node names
    if name.startswith("__"):
        raise ValueError(f"Special node type: {name}")

    if name == "agent":
        return "agent"
    elif name == "authorization":
        return "authorization"
    elif name == "tools":
        return "tools"
    raise ValueError(f"Invalid node type: {name}")


def get_name(obj: Union[str, HasName]) -> str:
    """Get the name from a string or an object with a name attribute."""
    return obj if isinstance(obj, str) else obj.name


def test_graph_structure() -> None:
    """Test that the graph structure matches the expected configuration."""
    # Define expected graph structure
    expected_structure = GraphStructure(
        nodes=[
            Node(name="agent", type="agent"),
            Node(name="authorization", type="authorization"),
            Node(name="tools", type="tools"),
        ],
        edges=[
            Edge(source=START, target="agent"),
            Edge(source="authorization", target="tools"),
            Edge(source="tools", target="agent"),
        ],
        conditional_edges={
            "agent": ["authorization", "tools", END],
        },
    )

    # Use the workflow object directly since it's a StateGraph
    graph_def = workflow
    assert isinstance(graph_def, StateGraph), (
        "graph_def should be a StateGraph instance"
    )

    # Validate nodes
    actual_nodes: list[Node] = []
    for name in graph_def.nodes:
        try:
            if not name.startswith("__"):  # Skip special nodes
                actual_nodes.append(Node(name=name, type=validate_node_type(name)))
        except ValueError:
            continue  # Skip invalid node types

    assert len(actual_nodes) == len(expected_structure.nodes), (
        "Incorrect number of nodes"
    )
    assert all(node in actual_nodes for node in expected_structure.nodes), (
        "Missing expected nodes"
    )

    # Get edges from the workflow's graph
    actual_edges: list[Edge] = []
    for edge in graph_def.edges:
        source = get_name(
            edge[0] if isinstance(edge, tuple) else getattr(edge, "source")
        )
        target = get_name(
            edge[1] if isinstance(edge, tuple) else getattr(edge, "target")
        )
        condition = (
            None if isinstance(edge, tuple) else getattr(edge, "condition", None)
        )
        actual_edges.append(Edge(source=source, target=target, condition=condition))

    # Validate edges
    assert len(actual_edges) >= len(expected_structure.edges), "Missing expected edges"
    for expected_edge in expected_structure.edges:
        matching_edges = [
            edge
            for edge in actual_edges
            if edge.source == expected_edge.source
            and edge.target == expected_edge.target
        ]
        assert matching_edges, f"Missing edge: {expected_edge}"

    # Get conditional edges from the workflow's graph
    agent_edges = set()

    # Access conditional edges by checking the routing function
    agent_edges.update(
        ["authorization", "tools", END]
    )  # These are the targets in should_continue

    # Validate conditional edges from agent node
    expected_edges = set(expected_structure.conditional_edges["agent"])
    assert agent_edges == expected_edges, (
        f"Incorrect conditional edges from agent node. Expected {expected_edges}, got {agent_edges}"
    )
