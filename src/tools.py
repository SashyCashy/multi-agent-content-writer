"""
Agent tools — the concrete actions the writer agents can take.

Two tools, matching the cost split the assignment calls for:

  • research_tool         – HEAVY. Deep content gathering. Call once or twice.
  • internet_search_tool  – LIGHT. Quick lookups (keywords, trends, hashtags).
                            Cheap enough to call several times.

Each tool works against whichever search backend is configured:

  • Tavily (if TAVILY_API_KEY is set) — purpose-built search, and it lets us
    request "advanced" vs "basic" depth explicitly, which maps cleanly onto
    the heavy/light split above.
  • OpenAI's hosted web_search tool (the fallback) — needs no key beyond
    OPENAI_API_KEY, so the tools do real web research out of the box.


Both backends return the same plain-text shape, so the agents never need to
know which one.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.config import SEARCH_BACKEND, TAVILY_API_KEY, get_logger, openai_client

logger = get_logger("tools")

# Built lazily so a bad/absent Tavily key can never break import of this module.
_tavily = None

def _tavily_client():
    """Return a cached Tavily Client, constructing it on first use."""
    global _tavily
    if _tavily is None:
        from tavily import TavilyClient

        _tavily = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily


# ═══════════════════════════════════════════════════════════
#  Backend 1 — Tavily
# ═══════════════════════════════════════════════════════════

def _search_tavily(query:str, *, deep:bool) -> str:
    """Search via Tavily. `deep=True` requests advanced depth + raw page text."""
    client = _tavily_client()
    result = client.search(
        query=query,
        search_depth="advanced" if deep else "basic",
        max_results=5 if deep else 4,
        include_answer=True,
        include_raw_content=deep,
    )

    parts: list[str] = []
    if result.get("answer"):
        parts.append(f"Summary:\n{result['answer']}\n")

    for i, item in enumerate(result.get("results", []), 1):
        parts.append(f"[{i}] {item.get('title', 'Untitled')} \n {item.get('url', '')}")
        snippet = (item.get("content") or "").strip()
        if snippet:
            parts.append(f"    {snippet[:400]}")
        # raw_content is the full page body — valuable for deep research, but
        # capped hard so a single tool result can't swallow the context window.
        if deep and item.get("raw_content"):
            parts.append(f"    EXCERPT: {item['raw_content'][:800]}")

    return "\n".join(parts) if parts else "No results found."

# ═══════════════════════════════════════════════════════════
#  Backend 2 — OpenAI hosted web_search
# ═══════════════════════════════════════════════════════════

def _search_openai(instruction: str, *, deep:bool) -> str:
    """Search via OpenAI's hosted web_search tool on the Response API.
    
    Citations arrive as `url_citation` annotations on the message content
    blocks, not in the text itself, so they are pulled out and appended.
    """
    # config.py only builds openai_client when OPENAI_API_KEY is set, so this
    # is None on a Gemini-only setup. Return a message the agent can read
    # instead of raising AttributeError on None.
    if openai_client is None:
        return "No OpenAI search backend configured. Proceed without search results."

    response = openai_client.responses.create(
        # The heavy path gets the stronger model; quick lookups do not need it.
        model="gpt-4o" if deep else "gpt-4o-mini",
        tools=[{"type": "web_search"}],
        input=instruction,
    )

    body = (response.output_text or "").strip()

    # Collect sources, de-duplicated — the API repeats the same citation
    # once per sentence that draws on it.
    sources: list[str] = []
    seen: set[str] = set()
    for item in response.output:
        for block in getattr(item, "content", None) or []:
            for ann in getattr(block, "annotations", None) or []:
                url = getattr(ann, "url", None)
                if getattr(ann, "type", None) == "url_citation" and url and url not in seen:
                    seen.add(url)
                    sources.append(f"  - {getattr(ann, 'title', None) or 'Untitled'}: {url}")

    if sources:
        body += "\n\nSOURCES:\n" + "\n".join(sources)
    return body or "No results found."

def _run_search(query: str, instruction: str, *, deep: bool) -> str:
    """Dispatch to the configured backend.

    Never raises. A tool that throws inside a graph node takes the whole
    graph down with it; a tool that returns an error string lets the agent
    read the failure, decide what to do, and keep going.
    """

    try:
        if SEARCH_BACKEND == "tavily":
            return _search_tavily(query, deep=deep)
        return _search_openai(instruction, deep=deep)
    except Exception as exc:
        logger.exception("Search failed  backend=%s  deep=%s", SEARCH_BACKEND, deep)
        return f"Search failed ({type(exc).__name__}: {exc}). Proceed without these results."

# ═══════════════════════════════════════════════════════════
#  TOOL 1 — deep research (expensive)
# ═══════════════════════════════════════════════════════════

@tool
def research_tool(topic: str) -> str:
    """Gather in-depth research on a topic: key facts, statistics, expert
    viewpoints, current developments, and cited sources.

    This is an EXPENSIVE, slow call. Use it at most once or twice per piece
    of content, for substantive background you cannot write without. Do NOT
    use it for keyword lists, hashtags, or trending topics — use
    internet_search_tool for those.

    Args:
        topic: the subject to research, e.g. "impact of RAG on enterprise search"
    """

    logger.info("research_tool  topic=%r", topic)
    instruction = (
        f"Research this topic thoroughly using web search: {topic}\n\n"
        "Report back:\n"
        "1. The 4-6 most important facts, with concrete numbers where they exist\n"
        "2. Current developments and where the field is heading\n"
        "3. Notable expert or industry viewpoints, including disagreements\n"
        "4. Any statistics worth quoting directly\n\n"
        "Be specific and factual. Do not write prose for publication — this is "
        "raw material for a writer."
    )

    return _run_search(topic, instruction, deep=True)

# ═══════════════════════════════════════════════════════════
#  TOOL 2 — quick lookups (cheap)
# ═══════════════════════════════════════════════════════════

@tool
def internet_search_tool(query: str) -> str:
    """Run a quick web lookup. Best for SEO keywords, search volume and
    competition signals, trending topics, popular hashtags, current events,
    and fact-checking a single claim.

    This is CHEAP and fast — call it as many times as you need, with one
    focused query each time rather than several questions bundled together.

    Args:
        query: a focused search query, e.g. "trending AI hashtags on X this week"
    """
    logger.info("internet_search_tool  query=%r", query)
    instruction = (
        f"Search the web for: {query}\n\n"
        "Give a concise, factual answer — bullet points, no preamble. "
        "If the query asks for keywords, hashtags, or trending topics, return "
        "them as a plain list."
    )
    return _run_search(query, instruction, deep=False)

# ═══════════════════════════════════════════════════════════
#  TOOL 3 — character counter (free, local)
# ═══════════════════════════════════════════════════════════

@tool
def count_characters(text: str) -> str:
    """Count the characters in a draft tweet and report whether it fits
    within X's 280-character limit.

    Always call this on a finished tweet draft before replying. Language
    models cannot count characters reliably by inspection, so verify here
    rather than guessing — and if it is over, revise and check again.

    Args:
        text: the exact tweet text to measure
    """
    n = len(text)
    logger.info("count_characters  n=%d", n)
    
    if n <= 280:
        return f"{n} characters. OK — fits with {280 - n} to spare."
    
    return f"{n} characters. TOO LONG by {n - 280}. Cut it down and check again."

# Tool sets, bound to agents in nodes.py.
# The general handler deliberately gets no tools — it answers from
# conversation memory alone.
SEO_TOOLS = [research_tool, internet_search_tool]
X_TOOLS = [internet_search_tool, count_characters]