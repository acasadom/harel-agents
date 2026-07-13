"""
CLI entry point.

Usage:
    python -m research_agent.run --question "..." [--provider anthropic|openai|mock] [--db PATH]
    python -m research_agent.run --approve <execution_id> [--db PATH]
    python -m research_agent.run --revise  <execution_id> [--db PATH]

--db persists state across invocations (each CLI call is a fresh process);
required for --approve/--revise to find an execution created earlier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from harel import DictResolver, DictStore, DurableRunner, Event, SqliteStore
from harel import definition_from_dsl_file

from research_agent import actions
from research_agent.providers.mock import MockProvider

load_dotenv()

MACHINES_DIR = Path(__file__).parent / "machines"


def _make_provider(provider_name: str):
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

        return AnthropicProvider()
    if provider_name == "openai":
        from research_agent.providers.openai import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {provider_name}")


def _make_store(db_path: str | None):
    return SqliteStore(db_path) if db_path else DictStore()


def _load_runner(store, provider) -> tuple[DurableRunner, object]:
    """Build a DurableRunner with both machines loaded, actions bound to provider."""
    bindings = actions.bind_actions(provider)
    agent_defn = definition_from_dsl_file(
        MACHINES_DIR / "agent.stm", "research_agent", actions=bindings
    )
    sub_defn = definition_from_dsl_file(
        MACHINES_DIR / "sub_researcher.stm", "sub_researcher", actions=bindings
    )
    resolver = DictResolver({"sub_researcher": sub_defn})
    runner = DurableRunner(store, {agent_defn.id: agent_defn}, resolver=resolver)
    return runner, agent_defn


def _print_result(exe) -> None:
    print(f"Execution id: {exe.id}")
    print(f"Status: {exe.status.name} / {exe.outcome}")
    if exe.context.get("draft"):
        print("\n--- ANSWER ---")
        print(exe.context["draft"])
    elif exe.active_path == "HumanReview":
        print("\nWaiting for human review.")
        print(f"  Approve: python -m research_agent.run --approve {exe.id}")
        print(f"  Revise:  python -m research_agent.run --revise  {exe.id}")


def cmd_ask(question: str, provider_name: str, db_path: str | None) -> None:
    """Create and run a research agent for the given question."""
    if db_path is None:
        print(
            "Warning: no --db given, using an in-memory store. "
            "--approve/--revise from another process will not find this "
            "execution.",
            file=sys.stderr,
        )
    store = _make_store(db_path)
    runner, agent_defn = _load_runner(store, _make_provider(provider_name))
    exe = runner.create(agent_defn.id, context={"question": question})
    _print_result(exe)


def cmd_approve(execution_id: str, provider_name: str, db_path: str | None) -> None:
    """Send an Approved event to an execution in HumanReview."""
    store = _make_store(db_path)
    runner, _ = _load_runner(store, _make_provider(provider_name))
    exe = runner.process(execution_id, Event(kind="Approved"))
    _print_result(exe)


def cmd_revise(execution_id: str, provider_name: str, db_path: str | None) -> None:
    """Send a RequestRevision event to an execution in HumanReview.

    Needs a provider: RequestRevision re-runs Drafting's draft_answer.
    """
    store = _make_store(db_path)
    runner, _ = _load_runner(store, _make_provider(provider_name))
    exe = runner.process(execution_id, Event(kind="RequestRevision"))
    _print_result(exe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Research Agent")
    parser.add_argument("--question", help="Research question")
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "mock"], default="mock"
    )
    parser.add_argument("--approve", metavar="EXEC_ID")
    parser.add_argument("--revise", metavar="EXEC_ID")
    parser.add_argument(
        "--db", metavar="PATH", help="SQLite file for durable storage across runs"
    )
    args = parser.parse_args()

    if args.approve:
        cmd_approve(args.approve, args.provider, args.db)
    elif args.revise:
        cmd_revise(args.revise, args.provider, args.db)
    elif args.question:
        cmd_ask(args.question, args.provider, args.db)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
