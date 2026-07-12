from __future__ import annotations

from typing import Union

from research_agent.providers.base import LLMProvider

Response = Union[str, BaseException]


class MockProvider(LLMProvider):
    """
    Scripted response provider for tests.

    Pass a list of responses; each call to complete() pops the next one.
    Raises AssertionError if more calls are made than responses provided.
    If a response is a BaseException instance, it is raised instead of
    returned — script a failing call by putting an exception in the list.

    Usage:
        p = MockProvider([
            '["topic A", "topic B", "topic C"]',   # plan_research
            "Summary of topic A.",                  # research_topic (child 0)
            "Summary of topic B.",                  # research_topic (child 1)
            "Summary of topic C.",                  # research_topic (child 2)
            "complete",                             # grade_research
            "Final answer synthesizing all topics.", # draft_answer
        ])
    """

    def __init__(self, responses: list[Response]) -> None:
        self._responses = list(responses)
        self._calls: list[tuple[str, str]] = []  # (system, user) log

    def complete(self, system: str, user: str) -> str:
        self._calls.append((system, user))
        if not self._responses:
            raise AssertionError(
                f"MockProvider exhausted after {len(self._calls) - 1} calls. "
                f"Last prompt: {user[:120]!r}"
            )
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    @property
    def calls(self) -> list[tuple[str, str]]:
        return list(self._calls)
