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

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="anthropic not installed (pip install 'harel-agents[anthropic]')",
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
    importlib.util.find_spec("openai") is None,
    reason="openai not installed (pip install 'harel-agents[openai]')",
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
