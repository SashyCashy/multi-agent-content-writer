# Multi-Agent Content Writer

A LangGraph multi-agent system that reads a request, decides which specialist should
handle it, and hands off. Three agents, two of them with autonomous tool-calling loops,
and a checkpointer so follow-up questions resolve against real conversation history.

```
                 ┌→ SEO Blog Writer  ↔ tools (loop)
User → Router ───┼→ X/Twitter Writer ↔ tools (loop)
                 └→ General Handler  → END
```

| Agent | Handles | Tools |
|---|---|---|
| `seo_blog_writer` | Long-form posts, articles, guides | `research_tool`, `internet_search_tool` |
| `x_writer` | Tweets and short social copy (≤280 chars) | `internet_search_tool`, `count_characters` |
| `general_handler` | Greetings, questions, conversation recall | none — answers from memory |

<details>
<summary><b>Architecture block diagram</b> — click to expand</summary>

```

                            ┌─────────────────┐
                            │      USER       │
                            └────────┬────────┘
                                     │  text request
                                     ▼
                            ┌─────────────────┐
                            │     main.py     │
                            │    thread_id    │
                            └────────┬────────┘
                                     │  invoke(state, config)
                                     ▼
                            ┌─────────────────┐
              router_llm ──▶│     ROUTER      │
                            └──┬──────┬─────┬─┘
             ┌─────────────────┘      │     └─────────────────┐
             ▼                        ▼                       ▼
   ┌───────────────────┐    ┌──────────────────┐   ┌───────────────────┐
   │  SEO BLOG WRITER  │    │     X WRITER     │   │  GENERAL HANDLER  │
   └─────┬───────▲─────┘    └────┬───────▲─────┘   └─────────┬─────────┘
         │       │               │       │                   │
    tool │       │ result   tool │       │ result            │  no tools
    call │       │          call │       │                   │
         ▼       │               ▼       │                   │
   ┌───────────────────┐    ┌──────────────────┐             │
   │     SEO TOOLS     │    │     X TOOLS      │             │
   └─────────┬─────────┘    └────────┬─────────┘             │
             │                       │                       │
             └───────────┬───────────┘                       │
                         ▼                                   │
                ┌─────────────────┐                          │
                │    tools.py     │                          │
                │    research     │                          │
                │    search       │                          │
                │    count        │                          │
                └────────┬────────┘                          │
                         ▼                                   │
                ┌─────────────────┐                          │
                │ Tavily / Gemini │                          │
                └─────────────────┘                          │
                                                             │
         ┌───────────────────────────────────────────────────┘
         │              all paths converge
         ▼
   ┌─────────────────┐
   │     OUTPUT      │────▶ back to USER
   └─────────────────┘

   ┌───────────────────────────────────────────────────────────────────┐
   │  MemorySaver  —  snapshots state after every block, per thread_id │
   │  state.py     —  ContentState shape shared by every block         │
   └───────────────────────────────────────────────────────────────────┘
```

Full version with tool access and file dependencies: [`docs/architecture.txt`](docs/architecture.txt)

</details>

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then add your keys
```

You need one key, and one optional one:

| Key | Required | Where |
|---|---|---|
| `GOOGLE_API_KEY` | **yes** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free, no card |
| `TAVILY_API_KEY` | recommended | [tavily.com](https://tavily.com) — free tier covers 1,000 searches/month |

Without `TAVILY_API_KEY` the tools fall back to OpenAI's hosted `web_search`, which
needs an `OPENAI_API_KEY` with credit. Tavily is the better path: it's free and it
distinguishes deep from shallow searches natively.

```bash
python -m src.main                  # interactive chat
python -m src.main --query "..."    # single request
python -m src.main --show-graph     # print the topology (no API calls)
```

Try this sequence — it exercises all three routes plus memory:

```
Write a tweet about LangGraph
what can you do?
What was my last request?
```

---

## How it works

A step-by-step trace of one request through every function — including the
tool loop, failure handling, and where memory is restored — is in
[`docs/FLOW.md`](docs/FLOW.md). The summary below covers the three core ideas.

### Routing

The router calls a cheap model with `with_structured_output(RouteDecision)`, where
`route` is typed as a `Literal` of the three node names. The model is constrained to
the enum, so an invalid route is impossible by construction rather than caught at
runtime. The last few conversation turns are included so follow-ups classify
correctly — "make it shorter" is meaningless without them.

Routing is the one node every request passes through, so a failure there would break
the whole system. It catches exceptions and degrades to `general_handler`, which
handles anything.

### The tool-calling loop

Each writer is a pair of nodes wired into a cycle:

```
writer --(tool_calls present)--> tools --> writer --> ...
writer --(no tool_calls)-------> END
```

`should_continue` reads the last message in state. If the model emitted `tool_calls`,
the conditional edge routes to the tools node, which executes them, appends the
results as `ToolMessage`s, and loops back. When the model replies without tool calls,
that reply is the finished piece.

A real run looks like this — the model decides how many iterations it needs:

```
[router] -> x_writer
[x:model] tool_calls=1
[x:tools] internet_search_tool({'query': 'enterprise search RAG AI trends hashtags'})
[x:model] tool_calls=1
[x:tools] count_characters({'text': 'Stop treating enterprise search like...'})
[x:tools] count_characters  n=261
[x:model] tool_calls=0        ← done
```

### Persistence

Memory is not something the agents do. It emerges from two things working together:

1. `messages` uses the `add_messages` reducer, so node updates **accumulate** instead
   of overwriting.
2. `main.py` creates one `thread_id` per session and passes it on every call.

`MemorySaver` snapshots state per `thread_id`. Reuse it and turn 2 starts with turn 1
already in state. Generate a fresh one each call and you get a blank slate every time.

---

## Design decisions

**Tools are split by cost, and the discipline lives in the prompts.**
`research_tool` is deep and expensive; `internet_search_tool` is light and cheap. The
system prompts tell each agent to call research at most once or twice for substantive
background, then use the cheap tool freely for keywords and trends. The same guidance
is written into the tool **docstrings**, which LangChain sends to the model as the tool
schema — so it shapes behavior at the point of decision, not just in the preamble.
Access is also enforced structurally: `research_tool` is not bound to the X writer at
all, and a test asserts it stays that way.

**Tools return error strings; they never raise.** An exception inside a graph node
propagates up and kills the entire run. A returned error string becomes a
`ToolMessage` the agent reads, so it can note the failure and write the piece anyway.

**System prompts are prepended at call time, never stored in state.** All three agents
share one `messages` list. If a prompt were appended to it, it would be checkpointed,
re-sent every turn, and the X writer would inherit the SEO writer's persona.

**`add_messages`, not `operator.add`.** Both append, so ordering isn't the difference.
`add_messages` upserts by message id (no duplicates on re-emit), coerces dicts and
strings into real message objects, and understands `RemoveMessage` for trimming
history. `operator.add` gives you a list that can only grow.

**The router and the writers run on different models.** Not only for cost — the Gemini
free tier caps requests **per model per day**, so splitting them gives each its own
quota pool instead of competing for one. Routing is trivial classification, so the
lite model loses nothing there.

---

## Models and quota

| Role | Default | Why |
|---|---|---|
| Writers | `gemini-3.6-flash` | Best available quality; own quota pool |
| Router | `gemini-3.1-flash-lite` | Separate pool; classification needs no more |

Both are overridable from `.env`:

```bash
WRITER_MODEL=gemini-3.6-flash    # stronger; noticeably better blog quality
ROUTER_MODEL=gemini-3.1-flash-lite
```

**Free-tier models also return `503 UNAVAILABLE` ("high demand") without warning**,
and the lite tiers are hit hardest. This is separate from quota and usually clears
in minutes. Because `ChatGoogleGenerativeAI` defaults to `timeout=None` and
`max_retries=6`, an unconfigured client retries a 503 six times with backoff and
simply hangs — so `config.py` sets explicit timeouts and `max_retries=1`, keeping
the worst case inside Cloud Run's 300s request timeout. If requests start failing,
check which models are actually up before assuming the code broke.

**The free tier allows ~20 requests per model per day.** A blog post with three tool
iterations costs about four writer calls, so budget four or five posts per model per
day. A `429 RESOURCE_EXHAUSTED` means quota, not a broken key.

Note that Gemini 2.x models (`gemini-2.0-flash`, `gemini-2.5-flash`) return 404 for
keys created recently — they're retired for new users. Tutorials naming them won't work.

Gemini 3.x also uses fixed sampling defaults and **ignores `temperature`**, warning on
every call if you pass it. Output style is controlled entirely through the system
prompts in `nodes.py`.

---

## Tests

18 tests in three tiers, because a suite that ran every agent end-to-end would consume
most of a day's quota on each invocation.

```bash
pytest tests/ -v              # 10 offline tests — free, no API calls, ~1.5s
pytest tests/ -v -m live      # 7 routing tests — one model call each
pytest tests/ -v -m e2e       # 1 end-to-end test + memory — ~6 calls
```

`pytest.ini` deselects the paid tiers by default, so the bare command is always free
and re-runnable.

- **Offline** reads the compiled graph's real topology: all six nodes present, the
  router reaching all three agents, and both tool loops being genuine cycles — it
  asserts the `tools → writer` return edge specifically, since a one-way edge would
  run tools once and stop. Plus `should_continue` in both directions, the
  `route_decision` fallback, and tool isolation.
- **Live routing** calls the router node directly rather than running the graph, so
  each case costs one model call instead of four or more. Includes the cases that
  matter most: `"What was my last request?"` and `"What makes a good blog post?"` must
  both route to `general_handler` — the second mentions blog posts but is a question,
  not a request for content.
- **End-to-end** runs the graph, asserts `output` is a `str` and the tweet is ≤280
  characters, then sends a second turn on the same `thread_id` and asserts the reply
  references the earlier request. Persistence as a pass/fail assertion.

---

## Deploying

The CLI is the primary interface; `src/web.py` adds an HTTP one for hosting.

```bash
uvicorn src.web:app --reload --port 8080     # local, then open localhost:8080
```

Three endpoints: `GET /` (the page), `POST /chat`, `GET /health` (probe, makes no
model call).

**It is FastAPI, not Gradio, on purpose.** Cloud Run bills by vCPU-seconds while a
request is open, and Gradio holds a websocket for as long as the tab is open — an
idle visitor would consume the free allowance doing nothing. Plain HTTP POSTs only
burn CPU while the graph actually runs.

**Each browser gets its own `thread_id`.** The page generates a session id in
`localStorage` and sends it with every request, so visitors don't share one
conversation. On the CLI the id is per process; here it's per browser.

**Quota exhaustion returns a 429 with a readable message** rather than a stack
trace, since the free Gemini tier runs out most days.

Container build uses `requirements-prod.txt`, which drops `pytest`, `grandalf`,
and the optional OpenAI fallback — Artifact Registry's free tier is 0.5 GB.

```bash
docker build -t macw . && docker run -p 8080:8080 --env-file .env macw
```

Set `GOOGLE_API_KEY` and `TAVILY_API_KEY` as the host's secrets, never in the image.

---

## Documentation

| File | What it covers |
|---|---|
| [`docs/architecture.txt`](docs/architecture.txt) | Block diagram, per-agent tool access, file dependency direction |
| [`docs/FLOW.md`](docs/FLOW.md) | A request traced end to end, function by function, plus cross-cutting concerns |

---

## Project structure

```
src/
  config.py   Logger, model clients, search-backend detection.
              Imports nothing from this project — everything imports it.
  state.py    ContentState (the shared TypedDict) and RouteDecision.
  tools.py    research_tool, internet_search_tool, count_characters,
              plus the Tavily and OpenAI search backends.
  nodes.py    router, both writers, both tools nodes, general_handler,
              should_continue, and the system prompts.
  graph.py    Wires the StateGraph and compiles it with the checkpointer.
  main.py     CLI entry point. Owns the thread_id.
tests/
  test_routes.py
```

Swapping model providers touches `config.py` and `requirements.txt` only — the graph,
state, nodes, and tools are provider-agnostic. This project was moved from OpenAI to
Gemini that way.
