"""
Step 10 — tests.

Two tiers, because the Gemini free tier allows only ~20 requests per model
per day and a full end-to-end run of every agent would burn most of it:

  OFFLINE (default, free, no API calls)
      Graph topology, tool bindings, and the tool-loop condition.
      Run with:  pytest tests/ -v

  LIVE (opt-in, costs quota)
      Calls the router directly — ONE model call per route, instead of the
      4+ a full writer run costs. Proves all three routes classify right.
      Run with:  pytest tests/ -v -m live

  END-TO-END (opt-in, costs the most)
      One full graph run plus a memory check.
      Run with:  pytest tests/ -v -m e2e
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from src.graph import content_graph
from src.nodes import (
    SEO_TOOLS,
    X_TOOLS,
    route_decision,
    router,
    should_continue,
)

# ═══════════════════════════════════════════════════════════
#  TIER 1 — offline structure (free)
# ═══════════════════════════════════════════════════════════

EXPECTED_NODES = {
    "router",
    "seo_blog_writer",
    "seo_tools",
    "x_writer",
    "x_tools",
    "general_handler",
}


def test_all_nodes_present():
    """Every node in the architecture diagram is wired into the graph."""
    actual = set(content_graph.get_graph().nodes)
    missing = EXPECTED_NODES - actual
    assert not missing, f"missing nodes: {missing}"


def test_router_fans_out_to_all_three_agents():
    """The router must be able to reach each of the three agents."""
    edges = content_graph.get_graph().edges
    targets = {e.target for e in edges if e.source == "router"}
    assert {"seo_blog_writer", "x_writer", "general_handler"} <= targets, targets


def test_both_tool_loops_are_cycles():
    """Each writer reaches its tools node, and that node loops back.

    A one-way writer -> tools edge would run tools once and stop; the return
    edge is what makes it an agentic loop.
    """
    edges = {(e.source, e.target) for e in content_graph.get_graph().edges}
    for writer, tools in (("seo_blog_writer", "seo_tools"), ("x_writer", "x_tools")):
        assert (writer, tools) in edges, f"missing {writer} -> {tools}"
        assert (tools, writer) in edges, f"missing {tools} -> {writer} (loop never closes)"


def test_general_handler_terminates():
    """general_handler goes straight to END — it has no tool loop."""
    edges = {(e.source, e.target) for e in content_graph.get_graph().edges}
    assert ("general_handler", "__end__") in edges


def test_checkpointer_attached():
    """Persistence requirement: the compiled graph must have a checkpointer."""
    assert content_graph.checkpointer is not None


def test_tool_bindings():
    """Each writer gets only the tools it should have."""
    seo = {t.name for t in SEO_TOOLS}
    x = {t.name for t in X_TOOLS}
    assert "research_tool" in seo, "SEO writer needs the deep research tool"
    assert "internet_search_tool" in seo
    assert "internet_search_tool" in x
    # The expensive deep-research tool must NOT be reachable from the X writer.
    assert "research_tool" not in x, "X writer must not be able to call research_tool"


def test_should_continue_routes_to_tools_when_tool_calls_present():
    """The tool-loop condition: tool_calls -> run them."""
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "internet_search_tool", "args": {"query": "x"}, "id": "1"}],
    )
    assert should_continue({"messages": [msg]}) == "tools"


def test_should_continue_ends_when_no_tool_calls():
    """The tool-loop condition: no tool_calls -> the agent is finished."""
    assert should_continue({"messages": [AIMessage(content="done")]}) == END


def test_route_decision_reads_state():
    assert route_decision({"route": "x_writer"}) == "x_writer"


def test_route_decision_falls_back_when_route_missing():
    """A missing route degrades to general_handler instead of KeyError."""
    assert route_decision({}) == "general_handler"


# ── Served page (offline) ────────────────────────────────────

def test_page_javascript_has_no_unterminated_strings():
    """Guard the served page's inline JS against broken string literals.

    A single unterminated string is a SyntaxError that disables the ENTIRE
    <script> block — the page still renders, so it looks fine, but nothing
    works: no button handlers, no fetch, no output. It shipped once, caused by
    a "\\n" escape being interpreted by the Python string holding the page
    rather than surviving into the JavaScript.

    Checking that quotes balance per line catches it: the bug left a line
    ending mid-string.
    """
    import re

    from src.web import PAGE

    script = re.search(r"<script>(.*?)</script>", PAGE, re.S)
    assert script, "no <script> block found in the page"

    for lineno, line in enumerate(script.group(1).splitlines(), 1):
        code = line.split("//", 1)[0]  # strip trailing comments
        for quote in ('"', "'"):
            # Ignore escaped quotes; every real one must be paired on its own line.
            count = len(re.findall(r'(?<!\\)' + quote, code))
            assert count % 2 == 0, (
                f"unbalanced {quote} on script line {lineno}: {line.strip()!r}"
            )


def test_page_renders_and_posts_to_chat():
    """The page must exist, be non-trivial, and wire itself to POST /chat."""
    from src.web import PAGE

    assert len(PAGE) > 2000, "page suspiciously small"
    assert '"/chat"' in PAGE, "page never calls the /chat endpoint"
    assert "session_id" in PAGE, "page must send a session_id so visitors get separate threads"


# ═══════════════════════════════════════════════════════════
#  TIER 2 — live routing (opt-in: 1 model call per case)
# ═══════════════════════════════════════════════════════════

ROUTING_CASES = [
    ("Write a blog post about vector databases", "seo_blog_writer"),
    ("Write me a 1000 word article on prompt engineering", "seo_blog_writer"),
    ("Write a tweet about LangGraph", "x_writer"),
    ("Draft a short X post announcing our launch 🚀", "x_writer"),
    ("hello there", "general_handler"),
    ("What was my last request?", "general_handler"),
    ("What makes a good blog post?", "general_handler"),
]


@pytest.mark.live
@pytest.mark.parametrize("text,expected", ROUTING_CASES)
def test_router_picks_correct_agent(text, expected):
    """Call the router node directly — one model call, no writer run."""
    result = router({"messages": [HumanMessage(content=text)], "user_input": text})
    assert result["route"] == expected, (
        f"{text!r} routed to {result['route']}, expected {expected}"
    )


# ═══════════════════════════════════════════════════════════
#  TIER 3 — end to end + memory (opt-in: costs the most)
# ═══════════════════════════════════════════════════════════

@pytest.mark.e2e
def test_x_writer_end_to_end_and_memory():
    """One full graph run, then a memory recall on the same thread.

    Uses the X writer because it is the cheapest writer path, and asserts the
    280-character contract that the tweet format requires.
    """
    config = {"configurable": {"thread_id": "pytest-e2e-thread"}}

    first = "Write a tweet about LangGraph"
    r1 = content_graph.invoke(
        {"messages": [HumanMessage(content=first)], "user_input": first}, config
    )
    assert r1["route"] == "x_writer"
    assert isinstance(r1["output"], str), "output must be a string, not content blocks"
    assert r1["output"].strip(), "output must not be empty"
    assert len(r1["output"]) <= 280, f"tweet is {len(r1['output'])} chars"

    # Same thread_id -> the checkpointer replays turn 1 into turn 2.
    second = "What was my last request?"
    r2 = content_graph.invoke(
        {"messages": [HumanMessage(content=second)], "user_input": second}, config
    )
    assert r2["route"] == "general_handler"
    assert "tweet" in r2["output"].lower(), (
        f"memory failed — no reference to the earlier tweet request: {r2['output']!r}"
    )
