#!/usr/bin/env python3
"""miniClaude CLI — M2: agent loop with tool system.

Usage:
    python miniClaude.py

REPL commands:
    /exit       quit
    /clear      reset conversation history
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from agent import build_agent


def extract_text(content) -> str:
    """Pull plain text out of an AIMessage.content.

    Anthropic-style responses may be a list of content blocks
    (text / thinking / redacted_thinking / tool_use). We only show text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


def main() -> int:
    load_dotenv()
    console = Console()

    if not (
        os.getenv("ANTHROPIC_AUTH_TOKEN")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    ):
        console.print(
            "[red]Missing API key. Set one of "
            "ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY / OPENROUTER_API_KEY in .env.[/red]"
        )
        return 1

    agent = build_agent()
    messages: list = []

    console.print("[bold cyan]miniClaude[/bold cyan] — M2 (agent loop + tools)")
    console.print("[dim]/exit to quit, /clear to reset.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]>[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return 0

        if not user_input:
            continue
        if user_input == "/exit":
            return 0
        if user_input == "/clear":
            messages = []
            console.print("[dim]conversation cleared[/dim]\n")
            continue

        messages.append(HumanMessage(content=user_input))

        try:
            with console.status("[dim]thinking…[/dim]"):
                result = agent.invoke({"messages": messages})
        except Exception as e:
            console.print(f"[red]error: {e}[/red]\n")
            messages.pop()  # don't keep the failed user turn
            continue

        for m in result["messages"][len(messages):-1]:
            for tc in getattr(m, "tool_calls", None) or []:
                console.print(f"[dim]🔧 {tc['name']}({tc.get('args', {})})[/dim]")
        messages = result["messages"]
        reply = messages[-1].content

        console.print(Markdown(extract_text(reply)))
        console.print()


if __name__ == "__main__":
    sys.exit(main())
