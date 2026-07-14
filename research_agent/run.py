"""
CLI entry point.

Usage:
    python -m research_agent.run --question "..." [--provider anthropic|openai|groq|mock] [--db PATH] [-v]
    python -m research_agent.run --approve <execution_id> [--db PATH] [-v]
    python -m research_agent.run --revise  <execution_id> [--provider ...] [--db PATH] [-v]
    python -m research_agent.run --sweep-timers [--db PATH]

--db persists state across invocations (each CLI call is a fresh process);
required for --approve/--revise to find an execution created earlier.

-v / --verbose prints the plan, each sub-topic's research, and the grade —
not just the final answer. For the full step-by-step execution timeline
(every transition, action, and context change) with the statechart drawn
live and the active state highlighted, run `harel monitor` against the same
--db instead (see README) — every execution here records a trace (trace=True)
for exactly that.

--sweep-timers fires any due durable timers (e.g. HumanReview's 24h
escalation timeout) and exits. harel only delivers a Timeout event when
something calls this — run it periodically (e.g. from cron) against the same
--db for the HumanReview timeout to actually take effect.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from harel import DictResolver, DictStore, DurableRunner, Event, SqliteStore
from harel import definition_from_dsl_file

from research_agent import actions
from research_agent.providers.mock import MockProvider

load_dotenv()

MACHINES_DIR = Path(__file__).parent / "machines"


def _make_provider(provider_name: str, model: str | None = None):
    if provider_name == "mock":
        return MockProvider(
            [
                '["Background", "Current state", "Open questions"]',
                "Mock summary of Background.",
                "Mock summary of Current state.",
                "Mock summary of Open questions.",
                '{"grade": "complete", "feedback": ""}',
                "Mock synthesized answer.",
            ]
        )
    if provider_name == "anthropic":
        from research_agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**({"model": model} if model else {}))
    if provider_name == "openai":
        from research_agent.providers.openai import OpenAIProvider

        return OpenAIProvider(**({"model": model} if model else {}))
    if provider_name == "groq":
        from research_agent.providers.groq import GroqProvider

        return GroqProvider(**({"model": model} if model else {}))
    raise ValueError(f"Unknown provider: {provider_name}")


def _make_store(db_path: str | None):
    return SqliteStore(db_path) if db_path else DictStore()


def _load_runner(
    store, provider, *, include_sub_researcher: bool = True, clock=time.time
) -> tuple[DurableRunner, object]:
    """Build a DurableRunner with agent.stm (and sub_researcher.stm unless
    include_sub_researcher=False — HumanReview's edges never re-enter
    Researching, so --approve/--revise/--sweep-timers don't need it).
    `clock` is injectable so tests can fast-forward past a durable timeout."""
    bindings = actions.bind_actions(provider) if provider is not None else {}
    agent_defn = definition_from_dsl_file(
        MACHINES_DIR / "agent.stm", "research_agent", actions=bindings
    )
    resolver = None
    if include_sub_researcher:
        sub_defn = definition_from_dsl_file(
            MACHINES_DIR / "sub_researcher.stm", "sub_researcher", actions=bindings
        )
        resolver = DictResolver({"sub_researcher": sub_defn})
    runner = DurableRunner(
        store, {agent_defn.id: agent_defn}, clock=clock, resolver=resolver, trace=True
    )
    return runner, agent_defn


def _print_result(exe, verbose: bool = False) -> None:
    print(f"Execution id: {exe.id}")
    print(f"Status: {exe.status.name} / {exe.outcome}")
    if verbose:
        _print_verbose_breakdown(exe)
    if exe.context.get("draft") is not None:
        print("\n--- ANSWER ---")
        print(exe.context["draft"])
    elif exe.active_path == "HumanReview":
        print("\nWaiting for human review.")
        print(f"  Approve: python -m research_agent.run --approve {exe.id}")
        print(f"  Revise:  python -m research_agent.run --revise  {exe.id}")
    elif exe.outcome == "failed":
        _print_failure_reason(exe)


def _print_verbose_breakdown(exe) -> None:
    """Print what happened at each phase — plan, each sub-topic's research,
    grading — not just the final answer/failure. For the full step-by-step
    timeline (event in, transition, actions, context before->after) with the
    statechart drawn live, use `harel monitor` instead (see README)."""
    ctx = exe.context
    if ctx.get("sub_topics"):
        print("\n--- PLAN ---")
        for i, topic in enumerate(ctx["sub_topics"]):
            print(f"  {i + 1}. {topic}")
    region_results = ctx.get("region_results", {})
    if region_results:
        print("\n--- RESEARCH ---")
        for key, result in region_results.items():
            if result.get("outcome") == "success":
                print(f"  [{key}] {result.get('summary', '')}")
            else:
                print(f"  [{key}] FAILED: {result.get('research_error', '(no error recorded)')}")
    if ctx.get("grade"):
        print("\n--- GRADING ---")
        print(f"  grade: {ctx['grade']}")
        if ctx.get("grade_feedback"):
            print(f"  feedback: {ctx['grade_feedback']}")
    if ctx.get("retries"):
        print(f"\n--- RETRIES: {ctx['retries']} ---")


def _print_failure_reason(exe) -> None:
    """Surface *why* a run reached Failed — plan_research/grade_research/
    draft_answer record a "<step>_error" (or grade="failed") in context
    instead of raising, and a failed sub-topic carries its own
    research_error back in region_results. None of that is visible unless
    something prints it."""
    print("\n--- WHY IT FAILED ---")
    found = False
    for key in ("plan_error", "draft_error"):
        if exe.context.get(key):
            print(f"{key}: {exe.context[key]}")
            found = True
    if exe.context.get("grade") == "failed" and exe.context.get("grade_feedback"):
        print(f"grade_feedback: {exe.context['grade_feedback']}")
        found = True
    for key, result in exe.context.get("region_results", {}).items():
        if result.get("outcome") == "failed":
            print(f"{key}: {result.get('research_error', '(no error recorded)')}")
            found = True
    if exe.error:
        print(f"engine error: {exe.error}")
        found = True
    if not found:
        print("(no error recorded in context — re-run with a store you can "
              "inspect, e.g. --db, and check the execution's context by hand)")


def _require_execution(store, execution_id: str):
    """Load an execution or exit with a clean error instead of letting a
    KeyError from runner.process() leak an internal traceback."""
    existing = store.load(execution_id)
    if existing is None:
        print(
            f"No execution found with id {execution_id!r} — check --db path.",
            file=sys.stderr,
        )
        sys.exit(1)
    return existing


def cmd_ask(
    question: str,
    provider_name: str | None,
    db_path: str | None,
    verbose: bool = False,
    model: str | None = None,
) -> None:
    """Create and run a research agent for the given question."""
    if not db_path:
        print(
            "Warning: no --db given, using an in-memory store. "
            "--approve/--revise from another process will not find this "
            "execution.",
            file=sys.stderr,
        )
    provider_name = provider_name or "mock"
    store = _make_store(db_path)
    runner, agent_defn = _load_runner(store, _make_provider(provider_name, model))
    exe = runner.create(
        agent_defn.id,
        context={"question": question, "provider_name": provider_name, "model": model},
    )
    _print_result(exe, verbose=verbose)


def cmd_approve(execution_id: str, db_path: str | None, verbose: bool = False) -> None:
    """Send an Approved event to an execution in HumanReview. Needs no
    provider — that transition triggers no action."""
    store = _make_store(db_path)
    _require_execution(store, execution_id)
    runner, _ = _load_runner(store, None, include_sub_researcher=False)
    exe = runner.process(execution_id, Event(kind="Approved"))
    _print_result(exe, verbose=verbose)


def cmd_revise(
    execution_id: str,
    provider_name: str | None,
    db_path: str | None,
    verbose: bool = False,
    model: str | None = None,
) -> None:
    """Send a RequestRevision event to an execution in HumanReview.

    Needs a provider: RequestRevision re-runs Drafting's draft_answer. If
    --provider/--model weren't given, reuses whichever ones created this
    execution.
    """
    store = _make_store(db_path)
    existing = _require_execution(store, execution_id)
    provider_name = provider_name or existing.context.get("provider_name", "mock")
    model = model or existing.context.get("model")
    runner, _ = _load_runner(
        store, _make_provider(provider_name, model), include_sub_researcher=False
    )
    exe = runner.process(execution_id, Event(kind="RequestRevision"))
    _print_result(exe, verbose=verbose)


def cmd_sweep_timers(db_path: str | None) -> None:
    """Fire any due durable timers (e.g. HumanReview's 24h timeout)."""
    store = _make_store(db_path)
    runner, _ = _load_runner(store, None, include_sub_researcher=False)
    fired = runner.fire_due_timers()
    print(f"Fired {fired} due timer(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Research Agent")
    parser.add_argument("--question", help="Research question")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "groq", "mock"],
        default=None,
        help="Defaults to 'mock' for a new question; reuses the original "
        "run's provider for --revise if omitted",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Overrides the provider's default model id (e.g. a Groq model "
        "hit its daily rate limit — swap to another one without touching "
        "code). For --revise, reuses the original run's model if omitted.",
    )
    parser.add_argument("--approve", metavar="EXEC_ID")
    parser.add_argument("--revise", metavar="EXEC_ID")
    parser.add_argument(
        "--sweep-timers",
        action="store_true",
        help="Fire any due durable timers (e.g. HumanReview's 24h timeout) and exit",
    )
    parser.add_argument(
        "--db", metavar="PATH", help="SQLite file for durable storage across runs"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the plan, each sub-topic's research, and the grade — not "
        "just the final answer/failure. For the full step-by-step timeline "
        "with the statechart drawn live, use `harel monitor` instead.",
    )
    args = parser.parse_args()

    if args.sweep_timers:
        cmd_sweep_timers(args.db)
    elif args.approve:
        cmd_approve(args.approve, args.db, verbose=args.verbose)
    elif args.revise:
        cmd_revise(
            args.revise, args.provider, args.db, verbose=args.verbose, model=args.model
        )
    elif args.question:
        cmd_ask(
            args.question, args.provider, args.db, verbose=args.verbose, model=args.model
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
