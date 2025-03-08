"""Define the configurable parameters for the agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Annotated, Literal, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from agent_arcade_tools.backend import prompts

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass(kw_only=True)
class AgentConfigurable:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    user_id: Optional[str] = field(
        default=None,
        metadata={
            "description": "The user ID to use for tool authorization. Required for tools that need authentication."
        },
    )

    debug_mode: bool = field(
        default=False,
        metadata={"description": "Enable debug mode for detailed logging."},
    )

    log_level: LogLevel = field(
        default="INFO",
        metadata={
            "description": "The logging level to use. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> AgentConfigurable:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable", {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        instance = cls(**{k: v for k, v in configurable.items() if k in _fields})

        # Set log level based on configuration
        log_level = (
            logging.DEBUG
            if instance.debug_mode
            else getattr(logging, instance.log_level)
        )
        logging.getLogger("agent_arcade_tools").setLevel(log_level)

        return instance
