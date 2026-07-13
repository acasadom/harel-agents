"""
Unit tests for provider implementations. No network calls, no API keys —
the Anthropic/OpenAI SDKs are faked via sys.modules so complete() request
shaping and response parsing can be checked without the packages installed.
"""

import importlib
import sys
import types

import pytest

from research_agent.providers.mock import MockProvider


def test_mock_provider_returns_scripted_responses_in_order():
    provider = MockProvider(["first", "second"])

    assert provider.complete("sys", "u1") == "first"
    assert provider.complete("sys", "u2") == "second"
    assert provider.calls == [("sys", "u1"), ("sys", "u2")]


def test_mock_provider_raises_when_exhausted():
    provider = MockProvider(["only"])
    provider.complete("sys", "u1")

    with pytest.raises(AssertionError):
        provider.complete("sys", "u2")


def test_mock_provider_raises_scripted_exception():
    provider = MockProvider([RuntimeError("boom")])

    with pytest.raises(RuntimeError, match="boom"):
        provider.complete("sys", "u1")


@pytest.mark.parametrize(
    "module_name,provider_module,provider_class,extra",
    [
        ("anthropic", "research_agent.providers.anthropic", "AnthropicProvider", "anthropic"),
        ("openai", "research_agent.providers.openai", "OpenAIProvider", "openai"),
    ],
)
def test_provider_missing_sdk_raises_import_error(
    monkeypatch, module_name, provider_module, provider_class, extra
):
    monkeypatch.setitem(sys.modules, module_name, None)
    mod = importlib.import_module(provider_module)
    provider_cls = getattr(mod, provider_class)

    with pytest.raises(ImportError, match=rf"harel-agents\[{extra}\]"):
        provider_cls()


def test_anthropic_provider_complete_sends_system_and_user(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = types.SimpleNamespace(text="anthropic response")
            return types.SimpleNamespace(content=[block])

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    from research_agent.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(model="claude-test", max_tokens=42, api_key="k")
    result = provider.complete("system prompt", "user prompt")

    assert result == "anthropic response"
    assert captured["model"] == "claude-test"
    assert captured["max_tokens"] == 42
    assert captured["system"] == "system prompt"
    assert captured["messages"] == [{"role": "user", "content": "user prompt"}]


def test_openai_provider_complete_sends_system_and_user(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content="openai response")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = FakeChat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    from research_agent.providers.openai import OpenAIProvider

    provider = OpenAIProvider(model="gpt-test", max_tokens=42, api_key="k")
    result = provider.complete("system prompt", "user prompt")

    assert result == "openai response"
    assert captured["model"] == "gpt-test"
    assert captured["max_tokens"] == 42
    assert captured["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
