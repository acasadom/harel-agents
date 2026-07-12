from typing import Protocol


class LLMProvider(Protocol):
    """Swappable LLM backend. Implement this to add a new provider."""

    def complete(self, system: str, user: str) -> str:
        """Send a prompt and return the response text. Blocking."""
        ...
