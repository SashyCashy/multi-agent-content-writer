"""
LangGraph state definitions.

ContentState  – the shared state every node reads and writes.
RouteDecision – the router's structured output schema.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# The three destinations the router can choose from. Defined once here so
# the schema below, the node names in graph.py, and the router prompt can
# never drift apart.
Route = Literal["seo_blog_writer", "x_writer", "general_handler"]

class ContentState(TypedDict):
    """Shared state that flows through the entire graph.

    Nodes return PARTIAL updates — only the keys they changed. How each
    key merges is decided per field:

      • user_input / route / output  → no reducer, so they OVERWRITE.
        These are per-turn scratch space, recomputed on every request.

      • messages → the add_messages reducer, so it ACCUMULATES.
        This is the conversation history, and it is what makes memory
        work: every turn appends under the same thread_id, so the
        checkpointer replays the whole conversation on the next turn.
        That is why "what was my last request?" can be answered.

    add_messages is used instead of operator.add because it upserts by
    message id (no duplicates on re-emit), coerces dicts/strings into
    real message objects, and understands RemoveMessage for trimming.
    """

    # Raw request for this turn. Nodes read this instead of digging into
    # messages[-1], which stops being the user's message once a tool loop runs.
    user_input: NotRequired[str]

    # Which agent the router selected this turn.
    route: NotRequired[str]

    # The user-facing reply produced by whichever agent ran.
    output: NotRequired[str]

    # Conversation + all tool traffic. The memory.
    messages: Annotated[list[AnyMessage], add_messages]

class RouteDecision(BaseModel):
    """What the router must return.

    Passed to llm.with_structured_output(RouteDecision), which needs a
    Pydantic model (a TypedDict will not do). The Field descriptions are
    sent to the model as part of the JSON schema, so treat them as prompt
    text, not comments.
    """

    route: Route = Field(
        description=(
            "Which agent handles this request. "
            "seo_blog_writer for long-form blog posts and articles; "
            "x_writer for tweets, X posts, and other short social copy; "
            "general_handler for greetings, questions, follow-ups, and anything else."
        )
    )

    reasoning:str = Field(
        description="One short sentence explaining why this route was chosen."
    )