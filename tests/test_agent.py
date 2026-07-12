"""
Full flow tests for the research agent, using MockProvider only.
No network calls. No API keys needed.

Setup (building the runner, creating the execution) lives entirely in
fixtures. Each test only supplies its scripted `provider` responses (and
`context`, when it needs to override a default) via indirect parametrize,
then asserts. `provider`/`context`/`runner`/`exe` are plain fixtures when a
test doesn't need to vary them.
"""

from pathlib import Path

import pytest
from harel import DictResolver, DictStore, DurableRunner, Event, definition_from_dsl_file

from research_agent import actions
from research_agent.providers.mock import MockProvider

MACHINES_DIR = Path(__file__).resolve().parent.parent / "research_agent" / "machines"

PLAN = '["topic A", "topic B", "topic C"]'
GRADE_COMPLETE = '{"grade": "complete", "feedback": ""}'

HAPPY_PATH_RESPONSES = [
    PLAN,
    "Summary of topic A.", "Summary of topic B.", "Summary of topic C.",
    GRADE_COMPLETE,
    "Final answer synthesizing all topics.",
]

TWO_INSUFFICIENT_ROUNDS = [
    PLAN,
    "Summary A1", "Summary B1", "Summary C1",
    '{"grade": "insufficient", "feedback": "f1"}',
    "Summary A2", "Summary B2", "Summary C2",
    '{"grade": "insufficient", "feedback": "f2"}',
]


@pytest.fixture
def provider(request):
    """The MockProvider under test. Override per-test with
    @pytest.mark.parametrize("provider", [[...responses...]], indirect=True)."""
    responses = getattr(request, "param", HAPPY_PATH_RESPONSES)
    return MockProvider(responses)


@pytest.fixture
def context(request):
    """Initial execution context. Override with indirect parametrize to add
    keys like max_retries."""
    extra = getattr(request, "param", {})
    return {"question": "What is the capital of France?", **extra}


@pytest.fixture
def runner(provider):
    """A fresh DurableRunner + agent Definition, actions bound to `provider`."""
    bindings = actions.bind_actions(provider)
    agent_defn = definition_from_dsl_file(
        MACHINES_DIR / "agent.stm", "research_agent", actions=bindings
    )
    sub_defn = definition_from_dsl_file(
        MACHINES_DIR / "sub_researcher.stm", "sub_researcher", actions=bindings
    )
    resolver = DictResolver({"sub_researcher": sub_defn})
    return DurableRunner(DictStore(), {agent_defn.id: agent_defn}, resolver=resolver), agent_defn


@pytest.fixture
def exe(runner, context):
    """The result of creating an execution — the starting point for every test."""
    runner_obj, agent_defn = runner
    return runner_obj.create(agent_defn.id, context=context)


def test_happy_path_complete(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "success"
    assert exe.context["draft"]


@pytest.mark.parametrize(
    "provider",
    [
        [
            PLAN,
            "Summary A1", "Summary B1", "Summary C1",
            '{"grade": "insufficient", "feedback": "need more depth"}',
            "Summary A2", "Summary B2", "Summary C2",
            GRADE_COMPLETE,
            "Final answer.",
        ]
    ],
    indirect=True,
)
def test_retry_then_complete(exe):
    assert exe.status.name == "DONE"
    assert exe.context["retries"] == 1


@pytest.mark.parametrize("provider", [TWO_INSUFFICIENT_ROUNDS], indirect=True)
@pytest.mark.parametrize("context", [{"max_retries": 2}], indirect=True)
def test_max_retries_escalates_to_human_review(exe):
    # max_retries=2 escalates after 2 rounds — no 3rd research round happens.
    assert exe.active_path == "HumanReview"
    assert exe.status.name == "RUNNING"


@pytest.mark.parametrize("provider", [TWO_INSUFFICIENT_ROUNDS], indirect=True)
@pytest.mark.parametrize("context", [{"max_retries": 2}], indirect=True)
def test_human_approve(runner, exe):
    assert exe.active_path == "HumanReview"

    runner_obj, _ = runner
    exe = runner_obj.process(exe.id, Event(kind="Approved"))

    assert exe.status.name == "DONE"
    assert exe.outcome == "success"


@pytest.mark.parametrize("provider", [TWO_INSUFFICIENT_ROUNDS + ["Revised final answer."]], indirect=True)
@pytest.mark.parametrize("context", [{"max_retries": 2}], indirect=True)
def test_human_revise(runner, exe):
    assert exe.active_path == "HumanReview"

    runner_obj, _ = runner
    exe = runner_obj.process(exe.id, Event(kind="RequestRevision"))

    assert exe.status.name == "DONE"
    assert exe.context["draft"] == "Revised final answer."


@pytest.mark.parametrize(
    "provider",
    [[PLAN, "Summary A", "Summary B", "Summary C", '{"grade": "escalate", "feedback": "requires human judgement"}']],
    indirect=True,
)
def test_grade_escalate_goes_directly_to_human_review(exe):
    assert exe.active_path == "HumanReview"


@pytest.mark.parametrize(
    "provider",
    [[PLAN, "Summary A", RuntimeError("simulated provider outage"), "Summary C"]],
    indirect=True,
)
def test_fan_out_failure_routes_to_failed(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "failed"


@pytest.mark.parametrize(
    "provider,expected_draft",
    [
        (HAPPY_PATH_RESPONSES, "Final answer synthesizing all topics."),
        (
            [
                '["First angle", "Second angle", "Third angle"]',
                "Alt summary A.", "Alt summary B.", "Alt summary C.",
                GRADE_COMPLETE,
                "Alt final answer.",
            ],
            "Alt final answer.",
        ),
    ],
    indirect=["provider"],
)
def test_provider_is_swappable(exe, expected_draft):
    """Demonstrates that the provider abstraction is the only thing that
    changes: two differently-scripted MockProviders drive the same machine
    to the same successful shape."""
    assert exe.status.name == "DONE"
    assert exe.outcome == "success"
    assert exe.context["draft"] == expected_draft
