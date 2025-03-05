import pytest
from langsmith import unit

from agent_arcade_tools.graph import graph


@pytest.mark.asyncio
@unit
async def test_agent_simple_passthrough() -> None:
    res = await graph.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
    assert res is not None
