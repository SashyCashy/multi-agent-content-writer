# How a request flows through the system

This traces one request from keystroke to answer, naming the exact function at
each step. It assumes you've read the [README](../README.md); the block diagram
is in [`architecture.txt`](architecture.txt).

---

## The one idea everything else follows from

LangGraph nodes are plain functions that **receive the whole state and return
only the keys they changed**. Nothing calls anything else directly — a node
returns, and the graph engine decides what runs next based on the edges declared
in `graph.py`.

So "the agent calls a tool" is never a function call inside a node. It's the
graph moving between two nodes, with a decision point in between.

---

## Step 0 — Session start

```python
assistant = ContentAssistant()      # main.py
self.thread_id = uuid.uuid4().hex   # generated ONCE
```

This id is the memory key. It's created in `__init__` and never regenerated, so
every request in this session shares it. That single fact is what makes memory
work; a fresh id per request would silently reset history every time.

---

## Step 1 — The request enters

`ContentAssistant.query(text)` builds two things:

```python
config: RunnableConfig = {"configurable": {"thread_id": self.thread_id}}

content_graph.invoke(
    {"messages": [HumanMessage(content=text)], "user_input": text},
    config,
)
```

Note what is **not** passed: `route` and `output`. Both are `NotRequired` in
`ContentState`, so they simply don't exist yet — they get written later by the
nodes that produce them.

## Step 1a — The checkpointer restores history (before any node runs)

Because the graph was compiled with a checkpointer, LangGraph first asks it:
*is there saved state for this `thread_id`?* If yes, it loads that state and
merges the new input into it.

This is the moment memory happens. By the time the first node executes,
`messages` already contains every prior turn of the conversation — not because
any agent fetched it, but because the reducer on `messages` accumulates and the
checkpointer replayed the accumulated list.

---

## Step 2 — Routing

`router()` in `nodes.py` runs. Three things happen:

**It gathers recent context.** It walks the last few messages and formats the
plain conversational ones into text, skipping tool traffic:

```python
for m in state.get("messages", [])[-6:]:
    if isinstance(m, HumanMessage): ...
    elif isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None): ...
```

Without this, a follow-up like *"now make it shorter"* would be unclassifiable —
there'd be nothing to tell the router what "it" refers to.

**It calls a constrained model.**

```python
router_structured = router_llm.with_structured_output(RouteDecision)
decision = cast(RouteDecision, router_structured.invoke([...]))
```

`RouteDecision.route` is typed `Literal["seo_blog_writer", "x_writer",
"general_handler"]`. LangChain converts that into a JSON Schema `enum` sent to
the model, so an out-of-range route is rejected at the API level rather than
caught later in your graph.

**It degrades instead of crashing.** Every request passes through this node, so a
failure here would take down the whole system. The call is wrapped:

```python
except Exception:
    logger.exception("[router] classification failed — falling back to general_handler")
    return {"route": "general_handler", "user_input": user_input}
```

`general_handler` can answer anything, so a routing failure produces a worse
answer rather than no answer.

The node returns `{"route": ..., "user_input": ...}` — it does not dispatch.

## Step 2a — The edge dispatches

`graph.py` declared:

```python
builder.add_conditional_edges("router", route_decision, {
    "seo_blog_writer": "seo_blog_writer",
    "x_writer": "x_writer",
    "general_handler": "general_handler",
})
```

LangGraph calls `route_decision(state)`, which reads what the router wrote:

```python
return state.get("route", "general_handler")
```

The returned string is looked up in that dict, and the mapped node runs next. The
dict doubles as a whitelist — a route not listed raises immediately rather than
dispatching somewhere wrong.

---

## Step 3 — The writer path (the tool loop)

Taking the SEO writer; the X writer is identical with its own model, prompt, and
tools.

### Pass 1 — the model decides

```python
response = seo_llm.invoke([SystemMessage(content=SEO_PROMPT)] + state["messages"])
```

Two details that matter:

- `seo_llm` is `llm.bind_tools(SEO_TOOLS)`. Binding sends each tool's schema —
  built by the `@tool` decorator from the function signature and **docstring** —
  so the model knows what it can request. The docstrings carry the cost guidance
  ("call at most once or twice"), which means that instruction reaches the model
  at the exact moment it's choosing.
- The system prompt is **prepended per call, never stored in state**. If it were
  appended to `messages`, it would be checkpointed and re-sent every turn, and
  since all three agents share one `messages` list, the X writer would inherit
  the SEO persona.

The node then writes `output` only if the model is finished:

```python
update: dict[str, Any] = {"messages": [response]}
if not getattr(response, "tool_calls", None):
    update["output"] = _text_of(response)
return update
```

A mid-loop reply is a tool *request*, not an article. Writing `output` on every
pass would leave a tool-call stub as the final answer.

### The loop condition

```python
def should_continue(state: ContentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END
```

Wired as a conditional edge after the writer, with the return edge that closes
the cycle:

```python
builder.add_conditional_edges("seo_blog_writer", should_continue,
                              {"tools": "seo_tools", END: END})
builder.add_edge("seo_tools", "seo_blog_writer")
```

That second line is what makes this a loop. Without it, the tools node would run
once and the graph would stop — the model would never get to read the result.

### Executing the tools

`seo_tools` delegates to `_execute_tools`, the only place in the codebase that
actually invokes a tool function:

```python
if not isinstance(last, AIMessage) or not last.tool_calls:
    return {"messages": []}

for tc in last.tool_calls:
    tool = tools_by_name.get(name)
    out = tool.invoke(args)
    results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
```

Three things to note:

- `tool_call_id=tc["id"]` must round-trip exactly. It's how the model pairs a
  result with the call that produced it; a mismatch makes the next request fail.
- The `isinstance` check narrows the `AnyMessage` union (only `AIMessage` carries
  `tool_calls`) and guards the loop.
- Failures become strings, never exceptions — see *Failure handling* below.

### Passes 2..n

The tool result is appended to `messages`, the graph follows the return edge, and
the writer runs **again** — now seeing its own earlier request and the result.
Each pass has strictly more information than the last, which is what makes it
converge. A real run:

```
[seo:model] tool_calls=1                      pass 1 — needs research
[seo:tools] research_tool(topic='...')
[seo:model] tool_calls=1                      pass 2 — now wants keywords
[seo:tools] internet_search_tool(query='...')
[seo:model] tool_calls=1                      pass 3 — one more lookup
[seo:tools] internet_search_tool(query='...')
[seo:model] tool_calls=0                      pass 4 — writes the article
```

Convergence is driven by the model plus prompt guidance, not a hard counter.
LangGraph's `recursion_limit` (25 transitions by default) is the backstop if a
model ever refuses to stop; it raises `GraphRecursionError` rather than looping
forever.

---

## Step 4 — The general handler path

No tools bound at all:

```python
response = general_llm.invoke([SystemMessage(content=GENERAL_PROMPT)] + state["messages"])
return {"messages": [response], "output": _text_of(response)}
```

One model call, straight to `END`. This is the node that answers *"what was my
last request?"* — and it can only do so because `state["messages"]` already holds
the whole conversation, restored by the checkpointer in Step 1a.

It's also why conversation-history questions are routed here explicitly in the
router prompt: it's the only agent that reads the full history without a task of
its own competing for attention.

---

## Step 5 — The answer returns

```python
answer = result.get("output") or ""
if not answer:
    answer = "Sorry, I couldn't produce a response. Try rephrasing?"
```

`query()` reads one field regardless of which of the three paths ran. That's the
point of having `output` in state: the caller doesn't need to know who handled it.

## Step 5a — The checkpointer saves

LangGraph snapshots state after every node, keyed by `thread_id`. So by the time
`query()` returns, the next request on this session already has the full
conversation waiting — including this turn's messages and all the tool traffic.

---

## Cross-cutting concerns

### Failure handling

| Failure | Behavior | Why |
|---|---|---|
| Search backend error | `_run_search` returns an error **string** | An exception in a node kills the whole graph run; a string becomes a `ToolMessage` the agent reads and works around |
| Unknown tool name | `"Unknown tool: {name}"` returned as the result | Same reason — let the model recover |
| Router classification fails | Falls back to `general_handler` | Every request passes through routing |
| `output` never set | `query()` substitutes a friendly message | Avoids returning an empty string to the user |
| Model never stops calling tools | `GraphRecursionError` at 25 transitions | Hard backstop behind the soft prompt guidance |

### The two search backends

Both search tools funnel through one dispatcher, which hands each backend the
input shape it expects:

```python
if SEARCH_BACKEND == "tavily":
    return _search_tavily(query, deep=deep)      # short keyword query
return _search_openai(instruction, deep=deep)     # full natural-language instruction
```

Tavily is a search engine, so it takes a keyword query. OpenAI's hosted
`web_search` is a model tool, so it takes an instruction describing how to
research and format the answer. Both forms are built at the call site; the
dispatcher stays a dispatcher. `deep=True` selects Tavily's `advanced` depth and
the stronger OpenAI model — that's the expensive/cheap distinction in practice.

### Cost control, three layers

1. **Structural** — `research_tool` is not in `X_TOOLS`, so the X writer cannot
   call it. A test asserts this stays true.
2. **Schema** — the cost guidance lives in the tool docstrings, which are sent to
   the model as the tool description.
3. **Prompt** — `SEO_PROMPT` states the ordering explicitly: research first, at
   most once or twice, then cheap search freely.

### Model split and the free tier

The router and the writers run on **different Gemini models** on purpose. The
free tier caps requests *per model per day* (~20), so splitting them gives each
its own quota pool instead of competing for one. Routing is trivial
classification, so the cheaper model costs nothing in quality there.

A `429 RESOURCE_EXHAUSTED` therefore means quota, not a broken key — and unlike a
rate limit, waiting minutes won't help if the daily cap is hit.

### Why `output` is normalised through `_text_of`

Gemini can return `content` as a list of content blocks rather than a plain
string. `_text_of` flattens either form to `str`, so `output` is always printable
and `len()` works for the 280-character check. Without it, `output` can end up
holding a raw list of dicts.
