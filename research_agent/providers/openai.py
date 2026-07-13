from research_agent.providers.base import LLMProvider, require_sdk


class OpenAIProvider(LLMProvider):
    """
    OpenAI ChatCompletion provider.

    Args:
        model: model id, default "gpt-4o".
        max_tokens: max output tokens, default 2048.
        api_key: if None, reads from OPENAI_API_KEY env var.
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
    ) -> None:
        openai = require_sdk("openai", "openai")
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
