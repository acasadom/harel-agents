import os

from research_agent.providers.base import LLMProvider, require_sdk


class GroqProvider(LLMProvider):
    """
    Groq provider — has a genuinely free tier (no credit card required), so
    it's the easiest way to exercise `pytest -m live` without paying.

    Groq's API is OpenAI-compatible, so this reuses the `openai` SDK pointed
    at Groq's base URL instead of adding a separate SDK dependency.

    Args:
        model: Groq-hosted model id, default "llama-3.3-70b-versatile".
        max_tokens: max output tokens, default 2048.
        api_key: if None, reads from GROQ_API_KEY env var.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        api_key: str | None = None,
    ) -> None:
        openai = require_sdk("openai", "groq")
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
        self._client = openai.OpenAI(api_key=api_key, base_url=self.BASE_URL)
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
