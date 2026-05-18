#!/usr/bin/env python3
"""miniClaude CLI — M3: agent loop + tools + HITL approval.

Usage:
    python miniClaude.py

REPL commands:
    /exit       quit
    /clear      reset conversation history
"""
from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from agent import build_agent


def extract_text(content) -> str:
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


def ask_approvals(console: Console, pending: list) -> dict:
    """Prompt the user for each pending tool call. Returns {call_id: bool}."""
    decisions = {}
    for tc in pending:
        console.print(
            f"[yellow]agent wants to run:[/yellow] [bold]{tc['name']}[/bold]({tc.get('args', {})})"
        )
        ans = Prompt.ask("approve?", choices=["y", "n"], default="n")
        decisions[tc["id"]] = ans == "y"
    return decisions


def run_turn(agent, console: Console, payload, config) -> None:
    """Invoke the agent; if it interrupts for approval, resume until done."""
    result = agent.invoke(payload, config=config)

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        decisions = {}
        for itr in interrupts:
            pending = itr.value.get("pending", [])
            decisions.update(ask_approvals(console, pending))
        result = agent.invoke(Command(resume=decisions), config=config)

    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            console.print(f"[dim]🔧 {tc['name']}({tc.get('args', {})})[/dim]")

    reply = result["messages"][-1].content
    console.print(Markdown(extract_text(reply)))
    console.print()


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
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    console.print("[bold cyan]miniClaude[/bold cyan] — M3 (agent loop + tools + HITL)")
    console.print(f"[dim]thread: {thread_id}  ·  /exit · /clear[/dim]\n")

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
            thread_id = uuid.uuid4().hex
            config = {"configurable": {"thread_id": thread_id}}
            console.print(f"[dim]new thread: {thread_id}[/dim]\n")
            continue

        try:
            with console.status("[dim]thinking…[/dim]"):
                run_turn(agent, console, {"messages": [HumanMessage(user_input)]}, config)
        except Exception as e:
            console.print(f"[red]error: {e}[/red]\n")


if __name__ == "__main__":
    sys.exit(main())
