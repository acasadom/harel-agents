"""
Tests for the CLI's result printing — specifically that a failed run tells
the user *why*, not just "Status: DONE / failed".
"""

from harel import DictStore

from research_agent.providers.mock import MockProvider
from research_agent.run import _load_runner, _print_result

PLAN = '["A", "B", "C"]'


def _run(provider, **extra_context):
    runner, agent_defn = _load_runner(DictStore(), provider)
    return runner.create(agent_defn.id, context={"question": "Q", **extra_context})


def test_plan_research_failure_prints_the_reason(capsys):
    exe = _run(MockProvider(["not valid json"]))

    _print_result(exe)

    out = capsys.readouterr().out
    assert "WHY IT FAILED" in out
    assert "plan_error" in out


def test_grade_research_failure_prints_the_reason(capsys):
    exe = _run(MockProvider([PLAN, "s1", "s2", "s3", "not valid json"]))

    _print_result(exe)

    out = capsys.readouterr().out
    assert "WHY IT FAILED" in out
    assert "grade_feedback" in out


def test_fan_out_all_failures_prints_each_sub_topics_error(capsys):
    exe = _run(
        MockProvider([PLAN, RuntimeError("rate limited"), RuntimeError("b"), RuntimeError("c")])
    )

    _print_result(exe)

    out = capsys.readouterr().out
    assert "WHY IT FAILED" in out
    assert "rate limited" in out


def test_happy_path_does_not_print_failure_section(capsys):
    exe = _run(
        MockProvider(
            [PLAN, "s1", "s2", "s3", '{"grade": "complete", "feedback": ""}', "final answer"]
        )
    )

    _print_result(exe)

    out = capsys.readouterr().out
    assert "WHY IT FAILED" not in out
    assert "final answer" in out
