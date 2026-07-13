import importlib
from typing import Any, Protocol


class LLMProvider(Protocol):
    """Swappable LLM backend. Implement this to add a new provider."""

    def complete(self, system: str, user: str) -> str:
        """Send a prompt and return the response text. Blocking."""
        ...


def require_sdk(module_name: str, extra: str) -> Any:
    """Import an optional provider SDK, or raise a helpful ImportError
    pointing at the extra that installs it."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(f"pip install 'harel-agents[{extra}]'")
