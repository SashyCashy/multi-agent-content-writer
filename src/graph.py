"""
Build and compile the LangGraph StateGraph.

    START -> router --+--> seo_blog_writer <-> seo_tools
                      |
                      +--> x_writer        <-> x_tools
                      |
                      +--> general_handler -> END

The router fans out three ways. Each writer sits in a cycle with its own
tools node; when the writer stops requesting tools, that branch ends.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from src.config import get_logger
from src.nodes import (
    general_handler,
    route_decision,
    router,
    seo_blog_writer,
    seo_tools,
    should_continue,
    x_tools,
    x_writer,
)
from src.state import ContentState

logger = get_logger("graph")

def build_graph():
    """Wire and compile the content-assistant graph."""
    builder = StateGraph(ContentState)
    # ── Nodes ────────────────────────────────────────────
    builder.add_node("router", router)
    builder.add_node("seo_blog_writer", seo_blog_writer)
    builder.add_node("seo_tools", seo_tools)
    builder.add_node("x_writer", x_writer)
    builder.add_node("x_tools", x_tools)
    builder.add_node("general_handler", general_handler)

    # ── Entry ────────────────────────────────────────────
    builder.add_edge(START, "router")
    # ── Router: 3-way conditional fan-out ────────────────
    # route_decision returns one of the dict's keys; the value is the node
    # it maps to. Keys match the Literal in RouteDecision, so an invalid
    # route is impossible by construction.
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
          "seo_blog_writer": "seo_blog_writer",
          "x_writer": "x_writer",
          "general_handler": "general_handler",  
        }
    )

    # ── SEO tool loop ────────────────────────────────────
    # should_continue returns "tools" (run them) or END (finished).
    # The edge back from seo_tools is what closes the cycle.
    builder.add_conditional_edges(
        "seo_blog_writer",
        should_continue,
        { "tools": "seo_tools", END:END }
    )

    builder.add_edge("seo_tools", "seo_blog_writer")

    # ── X tool loop ──────────────────────────────────────
    builder.add_conditional_edges(
        "x_writer",
        should_continue,
        { "tools":"x_tools", END:END }
    )

    builder.add_edge("x_tools", "x_writer");

    # ── General handler terminates immediately ───────────
    builder.add_edge("general_handler", END)

    # ── STEP 9 — Persistence ─────────────────────────────
    # MemorySaver snapshots the state after every node. Combined with a
    # stable thread_id in the config, it replays the whole conversation on
    # the next invoke — which is what gives the system memory.
    # Note: MemorySaver is in-process only; history is lost when Python
    # exits. Swap in SqliteSaver to survive restarts (see README).
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Graph compiled (6 nodes, MemorySaver checkpointer)")
    return graph

def render_graph(graph=None) -> str:
    """Render the topology for terminal display.
    
    Prefer ASCII (needs gandalf) and falls back to Mermaid source.
    """

    drawable = (graph or content_graph).get_graph()
    try:
        return drawable.draw_ascii()
    except (ImportError, ValueError) as exc:
        logger.debug("ASCII unavailable (%s) - using Mermaid", exc)
        return drawable.draw_mermaid()

# Module-level singleton — one compiled graph, one checkpointer, shared by
# every caller. Building a second one would give you a second, empty memory.
content_graph = build_graph()
