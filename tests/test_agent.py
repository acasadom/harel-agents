"""
Full flow tests for the research agent, using MockProvider only.
No network calls. No API keys needed.

Setup (building the runner, creating the execution) lives entirely in
fixtures. Each test only supplies its scripted `provider` responses (and
`context`, when it needs to override a default) via indirect parametrize,
then asserts. `provider`/`context`/`runner`/`exe` are plain fixtures when a
test doesn't need to vary them.
"""

import pytest
from harel import DictStore, Event
from harel.definition.events import ContextError

from research_agent.providers.mock import MockProvider
from research_agent.run import _load_runner as _build_runner

PLAN = '["topic A", "topic B", "topic C"]'
GRADE_COMPLETE = '{"grade": "complete", "feedback": ""}'
GRADE_ESCALATE = '{"grade": "escalate", "feedback": "requires human judgement"}'


def _grade_insufficient(feedback: str) -> str:
    return f'{{"grade": "insufficient", "feedback": "{feedback}"}}'


HAPPY_PATH_RESPONSES = [
    PLAN,
    "Summary of topic A.", "Summary of topic B.", "Summary of topic C.",
    GRADE_COMPLETE,
    "Final answer synthesizing all topics.",
]

# max_retries=2 allows 2 retries: the 3rd insufficient grade is the one that
# gives up. Drafting still produces a best-effort answer before parking at
# HumanReview, since the grade never reached "complete".
RETRIES_EXHAUSTED = [
    PLAN,
    "Summary A1", "Summary B1", "Summary C1",
    _grade_insufficient("f1"),
    "Summary A2", "Summary B2", "Summary C2",
    _grade_insufficient("f2"),
    "Summary A3", "Summary B3", "Summary C3",
    _grade_insufficient("f3"),
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
    return _build_runner(DictStore(), provider)


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
            _grade_insufficient("need more depth"),
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


@pytest.mark.parametrize("provider", [RETRIES_EXHAUSTED + ["draft after retries"]], indirect=True)
@pytest.mark.parametrize("context", [{"max_retries": 2}], indirect=True)
def test_max_retries_escalates_to_human_review(exe):
    # Escalation produces a best-effort draft before parking, so a human
    # always has something concrete to review.
    assert exe.active_path == "HumanReview"
    assert exe.status.name == "RUNNING"
    assert exe.context["retries"] == 2
    assert exe.context["draft"] == "draft after retries"


@pytest.mark.parametrize(
    "provider",
    [[PLAN, "Summary A", "Summary B", "Summary C", _grade_insufficient("f"), "draft"]],
    indirect=True,
)
@pytest.mark.parametrize("context", [{"max_retries": 0}], indirect=True)
def test_max_retries_zero_never_retries(exe):
    assert exe.active_path == "HumanReview"
    assert exe.context["retries"] == 0


@pytest.mark.parametrize(
    "provider",
    [[PLAN, "Summary A", "Summary B", "Summary C", '{"grade": "unsure"}', "draft"]],
    indirect=True,
)
def test_unexpected_grade_goes_to_human_review(exe):
    # Anything but "complete" needs a human — including a grade the prompt
    # never asked for.
    assert exe.active_path == "HumanReview"
    assert exe.context["draft"] == "draft"


@pytest.mark.parametrize("provider", [RETRIES_EXHAUSTED + ["draft after retries"]], indirect=True)
@pytest.mark.parametrize("context", [{"max_retries": 2}], indirect=True)
def test_human_approve(runner, exe):
    assert exe.active_path == "HumanReview"

    runner_obj, _ = runner
    exe = runner_obj.process(exe.id, Event(kind="Approved"))

    assert exe.status.name == "DONE"
    assert exe.outcome == "success"
    assert exe.context["draft"] == "draft after retries"


@pytest.mark.parametrize(
    "provider",
    [RETRIES_EXHAUSTED + ["draft after retries", "revised draft"]],
    indirect=True,
)
@pytest.mark.parametrize("context", [{"max_retries": 2}], indirect=True)
def test_human_revise(runner, exe):
    assert exe.active_path == "HumanReview"

    runner_obj, _ = runner
    exe = runner_obj.process(exe.id, Event(kind="RequestRevision"))

    # Revising parks back at HumanReview (not Done) — the point is a human
    # gets to look at the new draft too, not just the first one.
    assert exe.active_path == "HumanReview"
    assert exe.context["draft"] == "revised draft"


def test_human_review_loop_supports_multiple_revisions():
    provider = MockProvider(
        [
            PLAN, "Summary A", "Summary B", "Summary C",
            GRADE_ESCALATE,
            "draft 1", "draft 2", "draft 3",
        ]
    )
    runner, agent_defn = _build_runner(DictStore(), provider)
    exe = runner.create(agent_defn.id, context={"question": "Q"})
    assert exe.active_path == "HumanReview"
    assert exe.context["draft"] == "draft 1"

    exe = runner.process(exe.id, Event(kind="RequestRevision"))
    assert exe.active_path == "HumanReview"
    assert exe.context["draft"] == "draft 2"

    exe = runner.process(exe.id, Event(kind="RequestRevision"))
    assert exe.active_path == "HumanReview"
    assert exe.context["draft"] == "draft 3"

    exe = runner.process(exe.id, Event(kind="Approved"))
    assert exe.status.name == "DONE"
    assert exe.outcome == "success"
    assert exe.context["draft"] == "draft 3"


@pytest.mark.parametrize(
    "provider",
    [[PLAN, "Summary A", "Summary B", "Summary C", GRADE_ESCALATE, "best-effort draft"]],
    indirect=True,
)
def test_grade_escalate_produces_draft_then_parks_at_human_review(exe):
    assert exe.active_path == "HumanReview"
    assert exe.context["draft"] == "best-effort draft"


@pytest.mark.parametrize(
    "provider",
    [[PLAN, RuntimeError("a"), RuntimeError("b"), RuntimeError("c")]],
    indirect=True,
)
def test_fan_out_all_failures_routes_to_failed(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "failed"
    errors = [r["_error"]["message"] for r in exe.context["region_results"].values()]
    assert sorted(errors) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "provider",
    [
        [
            PLAN,
            "Summary A", RuntimeError("simulated provider outage"), "Summary C",
            GRADE_COMPLETE,
            "Final answer from partial research.",
        ]
    ],
    indirect=True,
)
def test_fan_out_partial_failure_survives(exe):
    # join any: one failed sub-topic doesn't sink research that otherwise
    # succeeded — grading/drafting proceed with whatever came back.
    assert exe.status.name == "DONE"
    assert exe.outcome == "success"
    assert exe.context["draft"] == "Final answer from partial research."


@pytest.mark.parametrize("provider", [["not valid json"]], indirect=True)
def test_plan_research_failure_routes_to_failed(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "failed"
    assert exe.context["_error"]["type"] == "ProviderError"


@pytest.mark.parametrize(
    "provider",
    [
        [
            # a real model returned this shape for plan_research once —
            # single-field objects instead of plain strings
            '[{"topic": "topic A"}, {"topic": "topic B"}, {"topic": "topic C"}]',
            "Summary A", "Summary B", "Summary C",
            GRADE_COMPLETE,
            "Final answer.",
        ]
    ],
    indirect=True,
)
def test_plan_research_tolerates_list_of_topic_objects(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "success"
    assert exe.context["sub_topics"] == ["topic A", "topic B", "topic C"]


@pytest.mark.parametrize(
    "provider",
    [
        [
            '{"sub_topics": ["topic A", "topic B", "topic C"]}',
            "Summary A", "Summary B", "Summary C",
            GRADE_COMPLETE,
            "Final answer.",
        ]
    ],
    indirect=True,
)
def test_plan_research_tolerates_dict_wrapped_array(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "success"
    assert exe.context["sub_topics"] == ["topic A", "topic B", "topic C"]


@pytest.mark.parametrize(
    "provider",
    [
        [
            '```json\n["topic A", "topic B", "topic C"]\n```',
            "Summary A", "Summary B", "Summary C",
            GRADE_COMPLETE,
            "Final answer.",
        ]
    ],
    indirect=True,
)
def test_plan_research_tolerates_markdown_fenced_json(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "success"
    assert exe.context["sub_topics"] == ["topic A", "topic B", "topic C"]


@pytest.mark.parametrize(
    "provider",
    [[PLAN, "Summary A", "Summary B", "Summary C", "not valid json"]],
    indirect=True,
)
def test_grade_research_failure_routes_to_failed(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "failed"


@pytest.mark.parametrize(
    "provider",
    [[PLAN, "Summary A", "Summary B", "Summary C", '{"feedback": "no grade here"}']],
    indirect=True,
)
def test_grade_without_grade_field_routes_to_failed(exe):
    assert exe.outcome == "failed"
    assert "no grade in provider response" in exe.context["_error"]["message"]


@pytest.mark.parametrize(
    "provider",
    [[PLAN, "Summary A", "Summary B", "Summary C", GRADE_COMPLETE, RuntimeError("draft boom")]],
    indirect=True,
)
def test_draft_answer_failure_routes_to_failed(exe):
    assert exe.status.name == "DONE"
    assert exe.outcome == "failed"
    assert exe.context["_error"]["message"] == "draft boom"


def test_context_schema_rejects_a_missing_question(runner):
    runner_obj, agent_defn = runner
    with pytest.raises(ContextError):
        runner_obj.create(agent_defn.id, context={})


def test_human_review_timeout_fires_via_sweep():
    fake_time = [1_000_000.0]

    def clock() -> float:
        return fake_time[0]

    provider = MockProvider(
        [PLAN, "Summary A", "Summary B", "Summary C", GRADE_ESCALATE, "draft"]
    )
    store = DictStore()
    runner, agent_defn = _build_runner(store, provider, clock=clock)
    exe = runner.create(agent_defn.id, context={"question": "Q"})
    assert exe.active_path == "HumanReview"

    fake_time[0] += 86400 + 1  # past the 24h timeout
    fired = runner.fire_due_timers()

    assert fired == 1
    updated = store.load(exe.id)
    assert updated.status.name == "DONE"
    assert updated.outcome == "failed"


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
