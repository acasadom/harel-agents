"""
harel action and selector functions for the research agent.

Actions mutate stm.execution_ctx; selectors return a routing key string.
Signature is (stm, event, **inputs) — the engine always passes these.

The LLMProvider is bound per-action via bind_actions(), not carried in
execution_ctx: durable stores JSON-serialize the context on every commit,
and a provider (an API client) isn't serializable. The .stm files declare a
`bind {}` block pointing each handler at its real function too, so
`harel validate`/`harel render` work standalone from the CLI — but
bind_actions()'s closures are what actually run (see run.py).
"""

from __future__ import annotations

import functools
import json
from typing import Any, Callable, Union


def bind_actions(provider: Any) -> dict[str, Union[str, Callable]]:
    """The actions= dict for definition_from_dsl_file(): the 4 handlers that
    call the provider, with it baked in via closure. Handlers that don't need
    a provider resolve through each .stm file's own `bind {}` block instead."""
    return {
        "plan_research": functools.partial(plan_research, provider=provider),
        "research_topic": functools.partial(research_topic, provider=provider),
        "grade_research": functools.partial(grade_research, provider=provider),
        "draft_answer": functools.partial(draft_answer, provider=provider),
    }


class ProviderError(Exception):
    """A provider call, or its JSON response, didn't produce usable output."""


def _complete(provider, system: str, user: str) -> str:
    try:
        return provider.complete(system, user)
    except Exception as exc:
        raise ProviderError(str(exc)) from exc


def _strip_code_fence(text: str) -> str:
    """LLMs commonly wrap JSON in a ```json ... ``` fence despite being told
    not to. Strip it so json.loads() sees just the payload."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop the opening fence (with optional language tag)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _complete_json(provider, system: str, user: str) -> Any:
    raw = _strip_code_fence(_complete(provider, system, user))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid JSON from provider: {exc}") from exc


def _normalize_topics(parsed: Any) -> list[str]:
    """Coerce a few shapes an LLM might reasonably return instead of a flat
    list[str] — a dict wrapping the array, or a list of single-field objects
    like {"topic": "..."} — instead of failing on a harmless formatting
    choice the prompt didn't rule out explicitly enough."""
    if isinstance(parsed, dict):
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if len(list_values) == 1:
            parsed = list_values[0]
    if not isinstance(parsed, list) or not parsed:
        raise ProviderError(f"expected a JSON array of strings, got: {parsed!r}")
    topics = []
    for item in parsed:
        if isinstance(item, str):
            topics.append(item)
            continue
        if isinstance(item, dict):
            if len(item) == 1:
                (value,) = item.values()
            else:
                value = next(
                    (item[k] for k in ("topic", "title", "name", "subtopic") if k in item),
                    None,
                )
            if isinstance(value, str):
                topics.append(value)
                continue
        raise ProviderError(f"couldn't read a topic string from entry: {item!r}")
    return topics


def _summaries(stm) -> list[str]:
    """Extract sub-researcher summaries from region_results."""
    results = stm.execution_ctx.get("region_results", {})
    return [
        v.get("summary", "")
        for v in results.values()
        if v.get("outcome") == "success"
    ]


def _combined_summaries(stm) -> str:
    return "\n\n---\n\n".join(
        f"Sub-topic {i + 1}:\n{s}" for i, s in enumerate(_summaries(stm))
    )


# ---------------------------------------------------------------------------
# actions (on enter — run before automatic transition)
# ---------------------------------------------------------------------------


def plan_research(stm, event, provider, **kwargs) -> None:
    """
    Break the research question into sub-topics.

    Reads:  context["question"], context.get("num_topics", 3)
    Writes: context["sub_topics"] on success, context["plan_error"] on failure
    """
    question = stm.execution_ctx["question"]
    n = stm.execution_ctx.get("num_topics", 3)
    system = (
        "You are a research planner. Given a question, return a JSON array of "
        f"exactly {n} distinct sub-topics to research, as plain strings (not "
        'objects). Example: ["Sub-topic one", "Sub-topic two", "Sub-topic three"]. '
        "Output ONLY that JSON array, no explanation, no markdown code fences."
    )
    user = f"Question: {question}"
    try:
        parsed = _normalize_topics(_complete_json(provider, system, user))
    except ProviderError as exc:
        stm.execution_ctx["plan_error"] = str(exc)
        return
    stm.execution_ctx["sub_topics"] = parsed


def route_plan(stm, event, **kwargs) -> str:
    """Route after planning. Returns "failed" if plan_research recorded an
    error, else "ok"."""
    return "failed" if stm.execution_ctx.get("plan_error") else "ok"


def research_topic(stm, event, provider, **kwargs) -> None:
    """
    Research a single sub-topic (runs inside the fan-out child execution).

    Reads:  context["topic"], context["question"], context.get("feedback", "")
    Writes: context["summary"] on success, context["research_error"] on failure

    Must not let a provider error propagate — an unhandled exception fails
    this child outright and the parent join never sees it. route_research
    turns the failure into a modeled transition instead.
    """
    topic = stm.execution_ctx["topic"]
    question = stm.execution_ctx["question"]
    feedback = stm.execution_ctx.get("feedback", "")
    feedback_clause = f"\n\nAdditional guidance: {feedback}" if feedback else ""
    system = (
        "You are a research specialist. Write a concise but thorough summary "
        f"(3-5 paragraphs) on the given sub-topic as it relates to the main question."
        f"{feedback_clause}"
    )
    user = f"Main question: {question}\nSub-topic: {topic}"
    try:
        stm.execution_ctx["summary"] = _complete(provider, system, user)
    except ProviderError as exc:
        stm.execution_ctx["research_error"] = str(exc)


def route_research(stm, event, **kwargs) -> str:
    """
    Route after researching a single sub-topic.

    Returns: "failed" if research_topic recorded an error, else "ok"
    """
    return "failed" if stm.execution_ctx.get("research_error") else "ok"


def grade_research(stm, event, provider, **kwargs) -> None:
    """
    Evaluate whether the collected research answers the question sufficiently.

    Reads:  context["question"], context["region_results"]
    Writes: context["grade"]          ("complete"|"insufficient"|"escalate"|"failed")
             context["grade_feedback"] (string, guidance for next attempt)
    """
    question = stm.execution_ctx["question"]
    system = (
        "You are a research quality judge. Evaluate the research summaries and "
        "respond with a JSON object:\n"
        '  {"grade": "complete"|"insufficient"|"escalate", "feedback": "<string>"}\n'
        '"complete": the summaries together answer the question well.\n'
        '"insufficient": gaps exist; feedback describes what is missing.\n'
        '"escalate": the question requires human expertise (ambiguous, sensitive, or '
        "out of scope). Output ONLY valid JSON."
    )
    user = f"Question: {question}\n\nResearch collected:\n{_combined_summaries(stm)}"
    try:
        result = _complete_json(provider, system, user)
        grade = result["grade"]
    except (ProviderError, KeyError, TypeError) as exc:
        stm.execution_ctx["grade"] = "failed"
        stm.execution_ctx["grade_feedback"] = f"grading failed: {exc}"
        return
    stm.execution_ctx["grade"] = grade
    stm.execution_ctx["grade_feedback"] = result.get("feedback", "")


def prepare_retry(stm, event, **kwargs) -> None:
    """
    Increment the retry counter. If this exhausts max_retries, mark the grade
    as "escalate" — the same signal a direct escalation uses, so Drafting's
    route_draft routes the resulting best-effort answer to HumanReview.

    Reads:  context.get("retries", 0), context.get("max_retries", 2)
    Writes: context["retries"], and context["grade"] if retries are exhausted
    """
    retries = stm.execution_ctx.get("retries", 0) + 1
    stm.execution_ctx["retries"] = retries
    if retries >= stm.execution_ctx.get("max_retries", 2):
        stm.execution_ctx["grade"] = "escalate"


def draft_answer(stm, event, provider, **kwargs) -> None:
    """
    Synthesize the final answer from all research summaries.

    Reads:  context["question"], context["region_results"]
    Writes: context["draft"] on success, context["draft_error"] on failure
    """
    question = stm.execution_ctx["question"]
    system = (
        "You are a research synthesizer. Write a clear, well-structured answer "
        "to the question using the research summaries provided. Be comprehensive "
        "but concise."
    )
    user = f"Question: {question}\n\nResearch:\n{_combined_summaries(stm)}"
    try:
        stm.execution_ctx["draft"] = _complete(provider, system, user)
    except ProviderError as exc:
        stm.execution_ctx["draft_error"] = str(exc)


# ---------------------------------------------------------------------------
# selectors (return a routing key — no side effects beyond context reads)
# ---------------------------------------------------------------------------


def route_grade(stm, event, **kwargs) -> str:
    """
    Route after grading. Returns the grade set by grade_research.

    Returns: "complete" | "insufficient" | "escalate" | "failed"
    """
    return stm.execution_ctx.get("grade", "escalate")


def should_retry(stm, event, **kwargs) -> str:
    """
    Decide whether to retry or escalate after refinement.

    Returns: "retry" if retries < max_retries, else "escalate"
    """
    retries = stm.execution_ctx.get("retries", 0)
    max_retries = stm.execution_ctx.get("max_retries", 2)
    return "retry" if retries < max_retries else "escalate"


def route_draft(stm, event, **kwargs) -> str:
    """
    Route after drafting. A failed draft_answer call goes to Failed; a draft
    produced on the escalation path (context["grade"] == "escalate") needs a
    human to review it before Done; otherwise (the normal "complete" grade
    happy path) it's done.

    Returns: "failed" | "needs_review" | "ok"
    """
    if stm.execution_ctx.get("draft_error"):
        return "failed"
    if stm.execution_ctx.get("grade") == "escalate":
        return "needs_review"
    return "ok"
