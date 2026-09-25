"""
Configuration: logger, LLM clients, and search-backend detection.

This module is imported by everything else, so it must have zero
dependencies on other project modules.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Load .env ────────────────────────────────────────────────
load_dotenv()


# ── Logger ───────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger("config")

# ── Validate API keys ────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("your-"):
    logger.error("GOOGLE_API_KEY is missing. Get one free at https://aistudio.google.com/apikey")
    sys.exit(1)

# Optional — unlocks the Tavily search backend.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()

# Optional — only used as a search fallback if Tavily is absent.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# ── Clients ──────────────────────────────────────────────────
# Both models are gemini-3.6-flash: the router uses with_structured_output,
# which Gemini implements via function calling, so it needs a model with
# solid tool-calling support — not flash-lite.
# The free tier caps requests PER MODEL PER DAY (gemini-3.6-flash is only 20),
# so the router and the writers deliberately run on DIFFERENT models: each
# gets its own quota pool instead of competing for one. Routing is a trivial
# classification, so the cheaper lite model loses nothing there.
# For a final quality run, set WRITER_MODEL=gemini-3.6-flash in .env.
WRITER_MODEL = os.environ.get("WRITER_MODEL", "gemini-3.5-flash-lite")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gemini-3.1-flash-lite")

# Built ONLY from an OpenAI key, and left as None without one, so the
# search-backend check below and the guard in tools.py both work correctly.
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI

        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        # The production image (requirements-prod.txt) omits the openai package,
        # since Tavily is the search backend there. A stray OPENAI_API_KEY in the
        # environment must not crash startup.
        logger.warning("OPENAI_API_KEY is set but the openai package is not installed — ignoring.")

# Writers: creative, so a warmer temperature and plenty of output room.
# NOTE: gemini-3.x uses fixed sampling defaults and IGNORES temperature,
# warning on every call if you pass it. Output style is controlled through
# the system prompts in nodes.py instead.
llm = ChatGoogleGenerativeAI(
    model=WRITER_MODEL,
    max_output_tokens=8192,
)

# Router: a classification task. temperature=0 for stable, repeatable
# routing, and a small output cap since it only returns a route name.
router_llm = ChatGoogleGenerativeAI(
    model=ROUTER_MODEL,
    max_output_tokens=1024,
)


# ── Search backend ───────────────────────────────────────────
if TAVILY_API_KEY:
    SEARCH_BACKEND = "tavily"
elif openai_client is not None:
    SEARCH_BACKEND = "openai"
else:
    logger.error("No search backend. Set TAVILY_API_KEY (free at https://tavily.com).")
    sys.exit(1)

logger.info(
    "Clients ready  (writer=%s, router=%s, search backend=%s)",
    WRITER_MODEL, ROUTER_MODEL, SEARCH_BACKEND,
)
