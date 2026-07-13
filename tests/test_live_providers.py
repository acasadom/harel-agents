"""
Live smoke tests against the real Anthropic/OpenAI APIs — real network,
real cost, not deterministic. Excluded by default (see pyproject.toml's
`addopts`); opt in with:

    uv run pytest -m live

Each test skips on its own if its SDK isn't installed or its API key isn't
set, so this file is safe to collect in any environment.
"""

import importlib.util
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="anthropic not installed (uv sync --extra anthropic)",
)
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_provider_completes_a_real_prompt():
    from research_agent.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model="claude-haiku-4-5-20251001", max_tokens=16)
    result = provider.complete(
        "Reply with exactly one word, no punctuation.",
        "What is the capital of France?",
    )

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="anthropic not installed (uv sync --extra anthropic)",
)
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_provider_default_model_is_valid():
    # No model= override — this is what --provider anthropic actually uses,
    # and the only test that would catch DEFAULT_MODEL regressing to a
    # stale/invalid id (as it once did).
    from research_agent.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(max_tokens=16)
    result = provider.complete(
        "Reply with exactly one word, no punctuation.",
        "What is the capital of France?",
    )

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.skipif(
    importlib.util.find_spec("openai") is None,
    reason="openai not installed (uv sync --extra openai)",
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_openai_provider_completes_a_real_prompt():
    from research_agent.providers.openai import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o-mini", max_tokens=16)
    result = provider.complete(
        "Reply with exactly one word, no punctuation.",
        "What is the capital of France?",
    )

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.skipif(
    importlib.util.find_spec("openai") is None,
    reason="openai not installed (uv sync --extra openai)",
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_openai_provider_default_model_is_valid():
    # No model= override — same rationale as the Anthropic variant above.
    from research_agent.providers.openai import OpenAIProvider

    provider = OpenAIProvider(max_tokens=16)
    result = provider.complete(
        "Reply with exactly one word, no punctuation.",
        "What is the capital of France?",
    )

    assert isinstance(result, str)
    assert result.strip()


# GroqProvider reuses the openai SDK (pointed at Groq's base URL), so its
# guard is the same "openai" import — but Groq has a free tier, so this is
# the cheapest of the three to actually run with a real key.
@pytest.mark.skipif(
    importlib.util.find_spec("openai") is None,
    reason="openai not installed (uv sync --extra groq)",
)
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_groq_provider_completes_a_real_prompt():
    from research_agent.providers.groq import GroqProvider

    provider = GroqProvider(max_tokens=16)
    result = provider.complete(
        "Reply with exactly one word, no punctuation.",
        "What is the capital of France?",
    )

    assert isinstance(result, str)
    assert result.strip()
