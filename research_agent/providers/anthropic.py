from research_agent.providers.base import LLMProvider, require_sdk


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Args:
        model: Claude model id, default "claude-sonnet-5".
        max_tokens: max output tokens, default 2048.
        api_key: if None, reads from ANTHROPIC_API_KEY env var.
        timeout: request timeout in seconds, default 120 (the SDK's own
            default is 600s — too long for an interactive CLI to hang on).
    """

    DEFAULT_MODEL = "claude-sonnet-5"
    DEFAULT_TIMEOUT = 120.0

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        anthropic = require_sdk("anthropic", "anthropic")
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
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

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list())
