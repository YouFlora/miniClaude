#!/usr/bin/env python3
"""miniClaude CLI — M8: M7 + steering (Ctrl+C interrupts and redirects).

Usage:
    python miniClaude.py                       # new session
    python miniClaude.py --list-sessions       # show stored thread_ids
    python miniClaude.py --resume <thread_id>  # continue an old session

REPL commands:
    /exit       quit
    /clear      start a fresh thread (old one stays in the DB)
    /compact    force compaction of the current thread now

Steering: while the agent is running, press Ctrl+C. The agent task is
cancelled; the checkpointer has already persisted state up to the last
completed super-step. You're then prompted for a new direction, which
gets sent as a fresh user message — the LLM sees the full history plus
your redirect and continues from there.
"""
from __future__ import annotations

import argparse
import asyncio
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
from checkpointer import list_sessions
from compactor import (
    DEFAULT_THRESHOLD_TOKENS,
    needs_compact,
    replace_with_summary,
    summarize,
    usage_input_tokens,
)


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
    decisions = {}
    for tc in pending:
        console.print(
            f"[yellow]agent wants to run:[/yellow] [bold]{tc['name']}[/bold]({tc.get('args', {})})"
        )
        ans = Prompt.ask("approve?", choices=["y", "n"], default="n")
        decisions[tc["id"]] = ans == "y"
    return decisions


async def run_turn_async(agent, console: Console, payload, config) -> None:
    result = await agent.ainvoke(payload, config=config)

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        decisions = {}
        for itr in interrupts:
            pending = itr.value.get("pending", [])
            d = await asyncio.to_thread(ask_approvals, console, pending)
            decisions.update(d)
        result = await agent.ainvoke(Command(resume=decisions), config=config)

    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            console.print(f"[dim]🔧 {tc['name']}({tc.get('args', {})})[/dim]")

    reply = result["messages"][-1].content
    console.print(Markdown(extract_text(reply)))
    console.print()


async def steerable_turn(agent, console: Console, payload, config) -> None:
    """Run a turn; on Ctrl+C, cancel + ask user for a new direction, repeat."""
    while True:
        task = asyncio.create_task(run_turn_async(agent, console, payload, config))
        try:
            await task
            return
        except KeyboardInterrupt:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            console.print(
                "\n[yellow]interrupted. state saved to checkpoint. "
                "type a new direction (empty = stop):[/yellow]"
            )
            try:
                new = (await asyncio.to_thread(input, "↳ ")).strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not new:
                return
            payload = {"messages": [HumanMessage(new)]}


def maybe_compact(agent, console: Console, config, *, force: bool = False) -> None:
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    if not messages:
        return
    if not force and not needs_compact(messages):
        return
    used = usage_input_tokens(messages)
    console.print(
        f"[dim]compacting: input_tokens={used} (threshold {DEFAULT_THRESHOLD_TOKENS})…[/dim]"
    )
    summary = summarize(messages)
    replace_with_summary(agent, config, messages, summary)
    console.print(f"[dim]compacted → {len(summary)} chars summary[/dim]\n")


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="miniClaude")
    p.add_argument("--resume", metavar="THREAD_ID", help="resume an existing session")
    p.add_argument("--list-sessions", action="store_true", help="list stored sessions")
    return p.parse_args(argv)


async def amain(args: argparse.Namespace) -> int:
    load_dotenv()
    console = Console()

    if args.list_sessions:
        ids = list_sessions()
        if not ids:
            console.print("[dim](no sessions stored yet)[/dim]")
        else:
            for tid in ids:
                console.print(tid)
        return 0

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
    thread_id = args.resume or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    banner = "M8 (M7 + steering via Ctrl+C)"
    console.print(f"[bold cyan]miniClaude[/bold cyan] — {banner}")
    label = "resumed" if args.resume else "new"
    console.print(
        f"[dim]{label} thread: {thread_id}  ·  /exit · /clear · /compact"
        "  ·  Ctrl+C while running = steer[/dim]\n"
    )

    while True:
        try:
            user_input = (await asyncio.to_thread(input, "> ")).strip()
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
        if user_input == "/compact":
            maybe_compact(agent, console, config, force=True)
            continue

        try:
            await steerable_turn(
                agent, console, {"messages": [HumanMessage(user_input)]}, config
            )
            maybe_compact(agent, console, config)
        except Exception as e:
            console.print(f"[red]error: {e}[/red]\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(amain(args)) or 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
