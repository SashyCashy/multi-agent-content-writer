"""
HTTP entry point — FastAPI + a single self-contained page.

Deliberately NOT Gradio. Cloud Run bills by vCPU-seconds while a request is
open, and Gradio holds a websocket for as long as the tab is open, so an idle
visitor would burn the monthly free allowance doing nothing. Plain HTTP POSTs
are short-lived: CPU is consumed while the graph runs, then drops to zero.

    Local:  uvicorn src.web:app --reload --port 8080
    Cloud:  the Dockerfile runs uvicorn on $PORT
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.config import ROUTER_MODEL, WRITER_MODEL, get_logger
from src.graph import content_graph

logger = get_logger("web")

app = FastAPI(title="Multi-Agent Content Writer", docs_url=None, redoc_url=None)

# Friendly labels for the route the graph actually took, shown in the UI so a
# visitor can see the router working rather than just getting text back.
ROUTE_LABELS = {
    "seo_blog_writer": "SEO Blog Writer",
    "x_writer": "X/Twitter Writer",
    "general_handler": "General Handler",
}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # The browser generates this once and reuses it, so each visitor gets their
    # own conversation. Without it every visitor would share one thread_id and
    # see each other's history.
    session_id: str = Field(default="", max_length=64)


def _is_quota_error(exc: Exception) -> bool:
    """Detect a Gemini free-tier exhaustion, as opposed to a real bug.

    The daily cap surfaces as 429 RESOURCE_EXHAUSTED. It is worth telling the
    visitor plainly, because unlike a rate limit, retrying shortly will not help.
    """
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower()


@app.get("/health")
def health() -> dict:
    """Cloud Run health probe. Must not call the model."""
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> JSONResponse:
    """Run one request through the graph and return the reply plus the route."""
    session_id = req.session_id or uuid.uuid4().hex
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    try:
        result = content_graph.invoke(
            {"messages": [HumanMessage(content=req.message)], "user_input": req.message},
            config,
        )
    except Exception as exc:
        if _is_quota_error(exc):
            logger.warning("Gemini quota exhausted: %s", exc)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "quota",
                    "reply": (
                        "The daily free Gemini quota for this demo is used up "
                        "(about 20 requests per model per day). It resets tomorrow — "
                        "please try again then."
                    ),
                },
            )
        logger.exception("Graph run failed")
        return JSONResponse(
            status_code=500,
            content={"error": "failed", "reply": f"Something went wrong: {type(exc).__name__}"},
        )

    route = result.get("route", "")
    return JSONResponse(
        content={
            "reply": result.get("output") or "No response produced. Try rephrasing?",
            "route": ROUTE_LABELS.get(route, route),
            "session_id": session_id,
        }
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Agent Content Writer</title>
<style>
  :root {
    --bg:#ffffff; --fg:#18181b; --muted:#71717a; --line:#e4e4e7;
    --card:#fafafa; --accent:#4f46e5; --accent-fg:#ffffff; --badge:#eef2ff;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#09090b; --fg:#fafafa; --muted:#a1a1aa; --line:#27272a;
      --card:#18181b; --accent:#818cf8; --accent-fg:#09090b; --badge:#1e1b4b;
    }
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--fg);
    font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  }
  .wrap { max-width:760px; margin:0 auto; padding:32px 16px 48px; }
  h1 { font-size:1.5rem; margin:0 0 4px; letter-spacing:-.02em; }
  .sub { color:var(--muted); font-size:.9rem; margin:0 0 24px; }
  .hints { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 20px; }
  .hint {
    background:var(--card); border:1px solid var(--line); border-radius:999px;
    padding:6px 12px; font-size:.82rem; color:var(--muted); cursor:pointer;
  }
  .hint:hover { color:var(--fg); border-color:var(--accent); }
  #log { display:flex; flex-direction:column; gap:16px; margin-bottom:20px; }
  .msg { border:1px solid var(--line); border-radius:12px; padding:14px 16px; background:var(--card); }
  .msg.you { background:transparent; }
  .who { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:6px; }
  .badge {
    display:inline-block; background:var(--badge); color:var(--accent);
    border-radius:999px; padding:2px 9px; font-size:.7rem; margin-left:6px;
    letter-spacing:.02em; text-transform:none;
  }
  .body { white-space:pre-wrap; word-wrap:break-word; }
  form { display:flex; gap:8px; }
  input[type=text] {
    flex:1; padding:12px 14px; border:1px solid var(--line); border-radius:10px;
    background:var(--bg); color:var(--fg); font-size:1rem; min-width:0;
  }
  input[type=text]:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  button {
    padding:12px 20px; border:0; border-radius:10px; background:var(--accent);
    color:var(--accent-fg); font-size:1rem; font-weight:600; cursor:pointer;
  }
  button:disabled { opacity:.5; cursor:not-allowed; }
  footer { margin-top:32px; color:var(--muted); font-size:.78rem; }
  a { color:var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Multi-Agent Content Writer</h1>
  <p class="sub">A router picks one of three agents. The writers call research
     and search tools on their own until they have what they need.</p>

  <div class="hints">
    <span class="hint">Write a blog post about vector databases</span>
    <span class="hint">Write a tweet about LangGraph</span>
    <span class="hint">What was my last request?</span>
  </div>

  <div id="log"></div>

  <form id="f">
    <input id="q" type="text" placeholder="Ask for a blog post, a tweet, or anything else…" autocomplete="off" required>
    <button id="b" type="submit">Send</button>
  </form>

  <footer>
    Gemini + Tavily · LangGraph ·
    <a href="https://github.com/SashyCashy/multi-agent-content-writer">source</a>
    <br>Running on a free daily quota, so it may be exhausted some days.
  </footer>
</div>
<script>
  // One session id per browser, so visitors get separate conversations.
  // Wrapped because localStorage throws in some privacy modes.
  let sid = "";
  try { sid = localStorage.getItem("sid") || ""; } catch (e) {}
  if (!sid) {
    sid = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random())).replace(/-/g, "");
    try { localStorage.setItem("sid", sid); } catch (e) {}
  }

  const log = document.getElementById("log");
  const form = document.getElementById("f");
  const input = document.getElementById("q");
  const btn = document.getElementById("b");

  function add(who, text, route) {
    const d = document.createElement("div");
    d.className = "msg" + (who === "You" ? " you" : "");
    const badge = route ? '<span class="badge">' + route + "</span>" : "";
    const head = document.createElement("div");
    head.className = "who";
    head.innerHTML = who + badge;
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = text;          // textContent, never innerHTML — model output is untrusted
    d.appendChild(head); d.appendChild(body);
    log.appendChild(d);
    d.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return body;
  }

  document.querySelectorAll(".hint").forEach(h =>
    h.addEventListener("click", () => { input.value = h.textContent.trim(); input.focus(); })
  );

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    add("You", text);
    input.value = "";
    btn.disabled = true;
    const pending = add("Assistant", "Thinking… the agent may call a few tools first.");
    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sid }),
      });
      const j = await r.json();
      pending.textContent = j.reply || "No response.";
      if (j.route) pending.previousSibling.innerHTML = "Assistant" + '<span class="badge">' + j.route + "</span>";
    } catch (err) {
      pending.textContent = "Network error — please try again.";
    } finally {
      btn.disabled = false;
      input.focus();
    }
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


logger.info("Web app ready  (writer=%s, router=%s, port=%s)",
            WRITER_MODEL, ROUTER_MODEL, os.environ.get("PORT", "8080"))
