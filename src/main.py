"""
Multi-agent content writer — entry point.

    python -m src.main                       # interactive chat
    python -m src.main --query "..."         # single request
    python -m src.main --show-graph          # print the topology
"""

from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig


from src.config import get_logger
from src.graph import content_graph, render_graph

logger = get_logger("main")


class ContentAssistant:
    """Wraps the compiled graph with one stable thread_id per session."""

    def __init__(self, thread_id: str | None = None):
        # ONE thread_id for the whole session. This is the memory key: the
        # checkpointer stores state per thread, so reusing it is what makes
        # turn 2 see turn 1. A fresh uuid per call would reset memory every
        # time and "what was my last request?" would fail.
        self.thread_id = thread_id or uuid.uuid4().hex
        logger.info("Session started  thread_id=%s", self.thread_id)

    def query(self, text: str) -> str:
        """Run one request through the graph and return the reply."""
        config: RunnableConfig = {"configurable": {"thread_id": self.thread_id}}

        result = content_graph.invoke(
            # The user's message goes in here, so add_messages appends it to
            # the persisted history alongside whatever the agent returns.
            {"messages": [HumanMessage(content=text)], "user_input": text},
            config,
        )

        answer = result.get("output") or ""
        if not answer:
            answer = "Sorry, I couldn't produce a response. Try rephrasing?"
        return answer

    def chat_loop(self) -> None:
        """Interactive REPL. Memory persists across turns in one session."""
        print("\n  Multi-Agent Content Writer   (type 'quit' to exit)")
        print("  Try: 'write a blog post about RAG', then 'now make it a tweet',")
        print("       then 'what was my last request?'\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye"):
                print("Goodbye!")
                break
            print(f"\nAssistant: {self.query(user_input)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Content Writer")
    parser.add_argument("--query", type=str, help="Run a single request and exit")
    parser.add_argument("--show-graph", action="store_true", help="Print the graph topology")
    args = parser.parse_args()

    if args.show_graph:
        print("\nGraph topology:\n")
        print(render_graph())
        print()

    assistant = ContentAssistant()

    if args.query:
        print(assistant.query(args.query))
    else:
        assistant.chat_loop()


if __name__ == "__main__":
    main()
