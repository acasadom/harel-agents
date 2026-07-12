# harel vs LangGraph

Both let you build multi-step LLM agent workflows with branching, retries, and
parallel fan-out. The difference is in what the *model* is: a statechart you
can read and validate, or Python code you have to run to find out what it
does.

## 1. Declarative vs imperative

A `.stm` file is a spec, not a program. `research_agent/machines/agent.stm`
in this repo is the entire orchestration logic for the research agent —
states, transitions, retry policy, human-in-the-loop escalation — in ~40
lines a non-engineer can read top to bottom. It's diffable in code review,
renders as a diagram (`harel viz`), and never executes arbitrary code at
definition time.

LangGraph models the same thing as a `StateGraph` built up imperatively:
`add_node`, `add_edge`, `add_conditional_edges` calls scattered through a
Python module, often across multiple files. The graph's actual shape only
exists once you run the code — there's no artifact you can hand a
non-engineer, or diff meaningfully in a PR.

## 2. Static validation

`harel validate agent.stm` catches unreachable states, non-deterministic
transitions, unresolved selector targets, and missing terminal verdicts —
before a single execution runs. Try to grade `"unknown"` and there's no
`"unknown" to X` branch: validation fails at definition time, not three weeks
into production when an LLM returns something you didn't anticipate.

LangGraph has no equivalent. A missing conditional edge, an unreachable node,
or a `Send` targeting a node that doesn't exist surfaces at runtime — if it
surfaces at all, rather than silently looping or dead-ending.

## 3. Fan-out

The parallel research step — spawn one child per sub-topic, wait for all of
them, then continue — is one line of harel DSL:

```
state Researching {
  invoke sub_researcher for topic in sub_topics
  with { topic: topic question: question feedback: retry_feedback }
}

from Researching join all to Grading else to Failed
```

The equivalent in LangGraph requires manually constructing `Send` objects in
a conditional edge function, wiring a reducer on the shared state key that
collects results, and reasoning about the graph's own recursion limit:

```python
def fan_out(state):
    return [Send("research_topic", {"topic": t, "question": state["question"]})
            for t in state["sub_topics"]]

graph.add_conditional_edges("plan", fan_out, ["research_topic"])
graph.add_edge("research_topic", "grade")
# collecting results into state["region_results"] is on you —
# a reducer function on the annotated state type
```

## 4. Durability

harel executions persist to any of its stores — SQLite, Postgres, Redis,
MongoDB — with the same code path used in this repo (`DictStore` for the
demo, `SqliteStore` for a durable one). A process restart mid-execution picks
up exactly where it left off, including durable timers (the `HumanReview`
24h timeout survives a crash).

LangGraph's open-source checkpointer covers single-process resumption;
crash-safe, distributed durability across restarts and workers is a
LangGraph Platform (cloud) feature.

## 5. Testability

harel's engine is pure — `start`/`process` are generators that describe
effects, with no I/O of its own. That's what makes the `MockProvider` pattern
in this repo work: every test in `tests/test_agent.py` drives the full
statechart, including the fan-out and retry loop, with zero network calls and
zero API keys, because the LLM calls are the *only* side effect and they're
injected through `context["__provider__"]`.

LangGraph nodes are just Python functions, so the same discipline is
possible — but it's a convention you have to impose yourself, not something
the framework gives you.

## 6. Visualization

`harel viz agent.stm` renders the machine to Mermaid or PlantUML — the
diagram at the top of this repo's README is generated straight from the
`.stm` file, so it can never drift out of sync with what actually runs.

LangGraph can render its compiled graph too, but the diagram is a projection
of the imperative graph-building code, not the source of truth itself.

## Fan-out, side by side

**harel:**

```
state Researching {
  invoke sub_researcher for topic in sub_topics
  with { topic: topic question: question feedback: retry_feedback }
}

from Researching join all to Grading else to Failed
```

**LangGraph:**

```python
from langgraph.constants import Send

def fan_out(state: AgentState) -> list[Send]:
    return [
        Send("research_topic", {"topic": t, "question": state["question"]})
        for t in state["sub_topics"]
    ]

def research_topic(state: TopicState) -> dict:
    summary = llm.invoke(...)
    return {"region_results": [{"topic": state["topic"], "summary": summary}]}

graph.add_conditional_edges("plan", fan_out, ["research_topic"])
graph.add_edge("research_topic", "grade")
```

Three lines vs. wiring a `Send` function, a node, an edge, and a reducer on
the annotated state type by hand.
