"""Utility & helper functions."""

from typing import Any, Dict, List, Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]] = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return str(content.get("text", ""))
    else:
        txts = [c if isinstance(c, str) else str(c.get("text", "")) for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)
