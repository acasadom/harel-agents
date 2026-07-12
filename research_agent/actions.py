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
    """The actions= dict for definition_from_dsl_file(): handler name -> this
    provider baked in via closure."""
    return {
        "plan_research": functools.partial(plan_research, provider=provider),
        "research_topic": functools.partial(research_topic, provider=provider),
        "route_research": route_research,
        "grade_research": functools.partial(grade_research, provider=provider),
        "prepare_retry": prepare_retry,
        "draft_answer": functools.partial(draft_answer, provider=provider),
        "route_grade": route_grade,
        "should_retry": should_retry,
    }


def _summaries(stm) -> list[str]:
    """Extract sub-researcher summaries from region_results."""
    results = stm.execution_ctx.get("region_results", {})
    return [
        v.get("summary", "")
        for v in results.values()
        if v.get("outcome") == "success"
    ]


# ---------------------------------------------------------------------------
# actions (on enter — run before automatic transition)
# ---------------------------------------------------------------------------


def plan_research(stm, event, provider, **kwargs) -> None:
    """
    Break the research question into sub-topics.

    Reads:  context["question"], context.get("num_topics", 3)
    Writes: context["sub_topics"]  (list[str])
    """
    question = stm.execution_ctx["question"]
    n = stm.execution_ctx.get("num_topics", 3)
    system = (
        "You are a research planner. Given a question, return a JSON array of "
        f"exactly {n} distinct sub-topics to research. Output ONLY valid JSON, "
        "no explanation."
    )
    user = f"Question: {question}"
    raw = provider.complete(system, user)
    stm.execution_ctx["sub_topics"] = json.loads(raw)


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
        stm.execution_ctx["summary"] = provider.complete(system, user)
    except Exception as exc:
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
    Writes: context["grade"]          ("complete" | "insufficient" | "escalate")
             context["grade_feedback"] (string, guidance for next attempt)
    """
    question = stm.execution_ctx["question"]
    summaries = _summaries(stm)
    combined = "\n\n---\n\n".join(
        f"Sub-topic {i + 1}:\n{s}" for i, s in enumerate(summaries)
    )
    system = (
        "You are a research quality judge. Evaluate the research summaries and "
        "respond with a JSON object:\n"
        '  {"grade": "complete"|"insufficient"|"escalate", "feedback": "<string>"}\n'
        '"complete": the summaries together answer the question well.\n'
        '"insufficient": gaps exist; feedback describes what is missing.\n'
        '"escalate": the question requires human expertise (ambiguous, sensitive, or '
        "out of scope). Output ONLY valid JSON."
    )
    user = f"Question: {question}\n\nResearch collected:\n{combined}"
    raw = provider.complete(system, user)
    result = json.loads(raw)
    stm.execution_ctx["grade"] = result["grade"]
    stm.execution_ctx["grade_feedback"] = result.get("feedback", "")


def prepare_retry(stm, event, **kwargs) -> None:
    """
    Increment retry counter and set retry_feedback for the next research pass.

    Reads:  context["grade_feedback"], context.get("retries", 0)
    Writes: context["retries"], context["retry_feedback"]
    """
    stm.execution_ctx["retries"] = stm.execution_ctx.get("retries", 0) + 1
    stm.execution_ctx["retry_feedback"] = stm.execution_ctx.get("grade_feedback", "")


def draft_answer(stm, event, provider, **kwargs) -> None:
    """
    Synthesize the final answer from all research summaries.

    Reads:  context["question"], context["region_results"]
    Writes: context["draft"]
    """
    question = stm.execution_ctx["question"]
    summaries = _summaries(stm)
    combined = "\n\n---\n\n".join(
        f"Sub-topic {i + 1}:\n{s}" for i, s in enumerate(summaries)
    )
    system = (
        "You are a research synthesizer. Write a clear, well-structured answer "
        "to the question using the research summaries provided. Be comprehensive "
        "but concise."
    )
    user = f"Question: {question}\n\nResearch:\n{combined}"
    stm.execution_ctx["draft"] = provider.complete(system, user)


# ---------------------------------------------------------------------------
# selectors (return a routing key — no side effects beyond context reads)
# ---------------------------------------------------------------------------


def route_grade(stm, event, **kwargs) -> str:
    """
    Route after grading. Returns the grade set by grade_research.

    Returns: "complete" | "insufficient" | "escalate"
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
