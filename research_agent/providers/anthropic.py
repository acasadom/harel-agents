from research_agent.providers.base import LLMProvider, require_sdk


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Args:
        model: Claude model id, default "claude-sonnet-5".
        max_tokens: max output tokens, default 2048.
        api_key: if None, reads from ANTHROPIC_API_KEY env var.
    """

    DEFAULT_MODEL = "claude-sonnet-5"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
    ) -> None:
        anthropic = require_sdk("anthropic", "anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
