"""
Graph node functions.

  Step 5 — seo_blog_writer  <-> seo_tools   (tool-calling loop)
  Step 6 — x_writer         <-> x_tools     (tool-calling loop)

Each writer is a pair of nodes wired into a cycle:

    writer --(tool_calls present)--> tools --> writer --> ...
    writer --(no tool_calls)-------> END

The writer node calls the LLM. If the reply carries tool_calls, the
conditional edge sends it to the tools node, which executes them and
loops back. When the reply has no tool_calls, the agent is done.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from src.config import get_logger, llm, router_llm
from src.state import ContentState, RouteDecision

from src.tools import SEO_TOOLS, X_TOOLS

from typing import Any, cast


logger = get_logger("nodes")

# ═══════════════════════════════════════════════════════════
#  Tool binding
# ═══════════════════════════════════════════════════════════
# .bind_tools() attaches the tool JSON schemas to the model so it can
# emit tool_calls. Each agent gets only the tools it should have —
# the X writer cannot reach research_tool at all.

seo_llm = llm.bind_tools(SEO_TOOLS)
x_llm = llm.bind_tools(X_TOOLS)


# Name -> tool lookup, so the tools nodes can dispatch by the name the
# model sent back.
seo_tools_by_name = {t.name: t for t in SEO_TOOLS}
x_tools_by_name = {t.name: t for t in X_TOOLS}

# ═══════════════════════════════════════════════════════════
#  Prompts
# ═══════════════════════════════════════════════════════════

SEO_PROMPT = """\
You are the SEO Blog Writer — an expert long-form content writer.

TOOLS AND WHEN TO USE THEM (this ordering matters, it controls cost):
  1. research_tool — EXPENSIVE and slow. Call it ONCE, at most twice, at
     the start, to gather substantive background, facts, statistics and
     references on the topic. Never call it for keywords.
  2. internet_search_tool — cheap and fast. Call it as often as you need
     AFTER research, for SEO keywords, search intent, competitor angles
     and current trends. One focused query per call.

Do not call research_tool for simple or well-known topics — go straight to
internet_search_tool. Once you have what you need, stop calling tools and
write the post.

OUTPUT FORMAT (always produce all of it):
  • An H1 title containing the primary keyword
  • A 2-3 sentence hook introduction stating the reader's problem
  • 4-6 H2 sections with descriptive, keyword-aware headings, using H3
    subsections where a section needs breaking down
  • Concrete facts, numbers and examples from your research — not filler
  • A short key-takeaways list
  • A clear call-to-action as the final section
  • A "Target keywords:" line at the very end listing the keywords you wove in

Write in markdown. Aim for 800-1200 words. Use an informed, direct voice —
no fluff, no "in today's fast-paced world" openings. Weave keywords in
naturally; never stuff them.
"""

X_PROMPT = """\
You are the X/Twitter Writer — you write short, high-engagement posts.

TOOLS:
  internet_search_tool — cheap and fast. Use it to check what is currently
  trending and which hashtags are actually in use on the topic before you
  write. One or two focused calls is usually enough.

HARD CONSTRAINT — THE 280 CHARACTER LIMIT:
  The finished post MUST be 280 characters or fewer, including spaces,
  emojis and hashtags. Before you reply, count the characters of your draft
  deliberately, character by character. If it is over, cut words and count
  again. A post over 280 characters is a failed response.
  Aim for 240-270 characters so you have margin.

STYLE:
  • Open with a hook in the first few words — no preamble
  • 1-3 relevant emojis, placed where they add meaning, not decoration
  • 2-3 hashtags at the end, drawn from what is actually trending
  • One clear idea per post. Punchy. No hedging.

Reply with the post text and nothing else — no explanation, no character
count, no quotation marks around it.
"""

ROUTER_PROMPT = """\
You are the router for a content-writing system. Decide which ONE agent
should handle the user's latest request.

AGENTS:
  seo_blog_writer  – long-form written content: blog posts, articles,
                     guides, how-tos, listicles, newsletters. Anything the
                     user wants as a multi-paragraph piece.
  x_writer         – short social copy: tweets, X posts, threads, captions,
                     anything explicitly short or under a character limit.
  general_handler  – EVERYTHING ELSE: greetings, thanks, chitchat, factual
                     questions, questions about this conversation or what
                     the user asked earlier, clarifications, and requests
                     that are not asking for a piece of content.

RULES:
1. Only route to a writer when the user is actually asking for content to
   be written. "What makes a good blog post?" is a question — that is
   general_handler, not seo_blog_writer.
2. Questions about the conversation itself ("what was my last request?",
   "what did you just write?") are ALWAYS general_handler. It is the only
   agent with access to the full history.
3. For a follow-up, route by the format the user now wants. "Now turn that
   into a tweet" is x_writer even if the previous turn was a blog post.
   "Make it shorter" keeps the same agent as the piece being revised.
4. When genuinely unsure, choose general_handler.
"""

# with_structured_output forces the reply into the RouteDecision schema, so
# `route` is always one of the three valid node names — never prose we would
# have to string-match.
router_structured = router_llm.with_structured_output(RouteDecision)

def _text_of(message) -> str:
    """Return a message's content as a plain string.

    LangChain types `content` as `str | list[str | dict]` because messages
    can carry multi-part content. OpenAI text replies are always `str` in
    practice, but normalising here keeps `output` a real string instead of
    something that might be a list.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "".join(parts)


# ═══════════════════════════════════════════════════════════
#  Shared loop helpers
# ═══════════════════════════════════════════════════════════

def should_continue(state: ContentState) -> str:
    """Decide whether the agent loops to its tools node or finishes.

    This is the tool-loop condition the assignment asks for: inspect the
    LAST message in state. If the model asked for tools, run them;
    otherwise the agent is done and the graph ends.
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END

def _execute_tools(state: ContentState, tools_by_name: dict, label: str) -> dict:
    """Run every tool call on the last message and return the results.

    All results go back as ToolMessages in a single update. Each one must
    carry the matching tool_call_id — that is how the model pairs a result
    with the call that produced it. A missing or mismatched id makes the
    next model call fail.
    """
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {"messages": []}
    
    results = []

    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[%s:tools] %s(%s)", label, name, args)
        tool = tools_by_name.get(name)
        if tool is None:
            # Never raise here — hand the model the error and let it recover.
            out = f"Unknown tool: {name}"
        else:
            try:
                out = tool.invoke(args)
            except Exception as exc:
                logger.exception("[%s:tools] %s failed", label, name)
                out = f"Tool {name} failed: {type(exc).__name__}: {exc}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

    return {"messages": results}

# ═══════════════════════════════════════════════════════════
#  STEP 5 — SEO Blog Writer
# ═══════════════════════════════════════════════════════════

def seo_blog_writer(state: ContentState) -> dict:
    """Call the SEO writer LLM, with both tools bound.

    The system prompt is prepended at call time rather than stored in
    state. Storing it would persist it in the checkpointed history and
    re-send a stale copy every turn — and the other agents would inherit
    the wrong persona from the shared message list.
    """
    response = seo_llm.invoke([SystemMessage(content=SEO_PROMPT)] + state["messages"])
    logger.info("[seo:model] tool_calls=%s", len(getattr(response, "tool_calls", None) or []))
    update: dict[str, Any] = {"messages": [response]}

    # Only a reply with no tool calls is the finished article.
    if not getattr(response, "tool_calls", None):
        update["output"] = _text_of(response)
    return update

def seo_tools(state: ContentState) -> dict:
    """Execute the SEO writer's tool calls, then loop back to the writer."""
    return _execute_tools(state, seo_tools_by_name, "seo")

# ═══════════════════════════════════════════════════════════
#  STEP 6 — X/Twitter Writer
# ═══════════════════════════════════════════════════════════

def x_writer(state: ContentState) -> dict:
    """Call the X writer LLM, with only internet_search_tool bound."""
    response = x_llm.invoke([SystemMessage(content=X_PROMPT)] + state["messages"])
    logger.info("[x:model] tool_calls=%s", len(getattr(response, "tool_calls", None) or []))

    update: dict[str, Any] = {"messages": [response]}

    if not getattr(response, "tool_calls", None):
        text = _text_of(response)
        # The limit is prompt-enforced, so log violations rather than trusting it.
        if len(text) > 280:
            logger.warning("[x:model] post is %d chars — over the 280 limit", len(text))
        update["output"] = text
    return update

def x_tools(state: ContentState) -> dict:
    """Execute the X writer's tool calls, then loop back to the writer."""
    return _execute_tools(state, x_tools_by_name, "x")


def router(state: ContentState) -> dict:
    """Classify the request and record which agent should handle it.
    
    The last few turns are included as context so follow-ups route
    correctly — "make it shorter" is meaningless on its own.
    """
    
    user_input = state.get("user_input", "")
    
    if not user_input and state.get("messages"):
        user_input = state["messages"][-1].content

    # Recent plain conversation only: tool traffic would just add noise and
    # cost here, and the router does not need it to classify intent.
    history = []
    for m in state.get("messages", [])[-6:]:
        if isinstance(m, HumanMessage):
            history.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            history.append(f"Assistant: {str(m.content)[:200]}")

    context = "RECENT CONVERSATION:\n" + "\n".join(history) + "\n\n" if history else ""

    decision = cast(RouteDecision, router_structured.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=f'{context}LATEST REQUEST: "{user_input}"'),
    ]))

    logger.info("[router] -> %s  (%s)", decision.route, decision.reasoning)
    return {"route": decision.route, "user_input": user_input}


def route_decision(state: ContentState) -> str:
    """Read the router's choice for the conditional edge.

    Kept separate from the router node because LangGraph needs a plain
    function returning a destination name, while the node returns a state
    update. The router decides once; this just reports it.
    """
    return state.get("route", "general_handler")

# ═══════════════════════════════════════════════════════════
#  STEP 7 — General Handler
# ═══════════════════════════════════════════════════════════

GENERAL_PROMPT = """\
You are the General Handler for a multi-agent content assistant.

You handle greetings, chitchat, factual questions, and any request that is
not asking for a blog post or a social post to be written.

You are also the ONLY agent with the full conversation history, so you
answer questions about the conversation itself — what the user asked for
earlier, what was written, what happened in this session. When asked, read
the history above and answer from it specifically: name the actual topic or
request, do not say "you asked me to write something".

The system has two other agents you can mention if the user asks what it
can do: an SEO Blog Writer for long-form posts, and an X/Twitter Writer for
short social posts.

Be warm, brief and direct. You have no tools — answer from the
conversation and your own knowledge.
"""

# Deliberately NOT bound to any tools: this agent answers from memory.
general_llm = llm


def general_handler(state: ContentState) -> dict:
    """Answer from the full conversation history, then end.

    Both halves of the turn end up persisted: the user's HumanMessage is
    added by the entry point when the graph is invoked, and the AIMessage
    returned here is appended by the add_messages reducer. Under one
    thread_id the checkpointer replays both on the next turn, which is what
    makes "what was my last request?" answerable.
    """
    response = general_llm.invoke(
        [SystemMessage(content=GENERAL_PROMPT)] + state["messages"]
    )
    logger.info("[general] answered from %d prior messages", len(state["messages"]))
    return {"messages": [response], "output": _text_of(response)}
