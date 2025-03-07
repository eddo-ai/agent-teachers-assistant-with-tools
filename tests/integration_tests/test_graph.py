from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import unit

from agent_arcade_tools.graph import graph


@pytest.mark.asyncio
@unit
async def test_agent_simple_passthrough() -> None:
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "test-thread",
            "checkpoint_ns": "test-namespace",
            "checkpoint_id": "test-checkpoint",
        }
    }

    # Mock the OpenAI API response
    mock_response = AIMessage(content="Hello! How can I help you today?")

    with patch("agent_arcade_tools.graph.load_chat_model") as mock_model:
        # Configure the mock
        mock_model.return_value.bind_tools.return_value.invoke.return_value = (
            mock_response
        )

        res = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "Hello"}]}, config=config
        )
        assert res is not None
        assert len(res["messages"]) == 2
        assert isinstance(res["messages"][0], HumanMessage)
        assert res["messages"][0].content == "Hello"
        assert isinstance(res["messages"][1], AIMessage)
        assert res["messages"][1].content == "Hello! How can I help you today?"
