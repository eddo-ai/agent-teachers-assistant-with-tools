from agent_arcade_tools.configuration import AgentConfigurable


def test_configuration_empty() -> None:
    AgentConfigurable.from_runnable_config({})
