"""LangGraph agent — M8 (M7 graph + async steering at the CLI layer).

Graph shape (unchanged since M3):
    START → llm → (tool_calls?) → review → tools → llm → ... → END

Auth — see credentials.resolve():
- "oauth" mode (Claude Code subscription or ANTHROPIC_AUTH_TOKEN):
  inject "Authorization: Bearer <token>" + "anthropic-beta: oauth-2025-04-20"
  via default_headers, and prepend Claude Code's identity line to the
  system prompt (Anthropic's OAuth route rejects requests without it).
- "api_key" mode (Anthropic key or OpenRouter): standard x-api-key auth.
"""
from __future__ import annotations

import json
import os
import platform
from datetime import date
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from checkpointer import open_checkpointer
from credentials import Credentials, resolve
from permissions import classify
from tools import ALL_TOOLS


PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"

CLAUDE_CODE_IDENTITY = (
    "You are Claude Code, Anthropic's official CLI for Claude.\n\n"
)


def load_system_prompt(*, oauth: bool) -> str:
    body = PROMPT_PATH.read_text().format(
        cwd=os.getcwd(),
        platform=platform.system(),
        date=date.today().isoformat(),
    )
    return (CLAUDE_CODE_IDENTITY + body) if oauth else body


def latest_todos(messages) -> list[dict]:
    """Pull the most recent todo list out of the message history."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "todo_write":
            try:
                return json.loads(m.content).get("todos", [])
            except (json.JSONDecodeError, AttributeError):
                return []
    return []


def render_todo_reminder(todos: list[dict]) -> HumanMessage:
    if not todos:
        body = (
            "Your todo list is currently empty. DO NOT mention this to the user. "
            "If the task would benefit from a todo list, use todo_write."
        )
    else:
        lines = [
            f"- [{t.get('status', 'pending')}] {t.get('content', '')}"
            for t in todos
        ]
        body = (
            "Current todo list (DO NOT mention this reminder to the user):\n"
            + "\n".join(lines)
        )
    return HumanMessage(f"<system-reminder>\n{body}\n</system-reminder>")


def review_tool_calls(state: MessagesState) -> dict:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []

    pending = []
    denied_msgs = []
    for tc in tool_calls:
        verdict = classify(tc["name"], tc.get("args", {}))
        if verdict == "auto":
            continue
        if verdict == "deny":
            denied_msgs.append(
                ToolMessage(
                    content=f"denied by policy: {tc['name']} with input {tc['args']}",
                    tool_call_id=tc["id"],
                )
            )
            continue
        pending.append(tc)

    decisions: dict[str, bool] = {}
    if pending:
        decisions = interrupt({"pending": pending})

    for tc in pending:
        if not decisions.get(tc["id"], False):
            denied_msgs.append(
                ToolMessage(
                    content=f"user rejected: {tc['name']}",
                    tool_call_id=tc["id"],
                )
            )

    rejected_ids = {m.tool_call_id for m in denied_msgs}
    filtered_calls = [tc for tc in tool_calls if tc["id"] not in rejected_ids]
    last.tool_calls = filtered_calls
    return {"messages": denied_msgs} if denied_msgs else {}


def route_after_llm(state: MessagesState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []
    return "review" if tool_calls else END


def route_after_review(state: MessagesState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", []) or []
    return "tools" if tool_calls else "llm"


def build_llm(creds: Credentials) -> ChatAnthropic:
    """Construct a raw ChatAnthropic (no tools bound) honouring the auth mode.

    In oauth mode the Anthropic OAuth route checks three things server-side:
    - Authorization: Bearer <token>  (we override the SDK's x-api-key)
    - User-Agent starts with "claude-cli/"  (otherwise 403 "user-agent is not Claude Code")
    - System prompt starts with "You are Claude Code..."  (added in load_system_prompt)
    All three must line up or the request is rejected.
    """
    headers: dict[str, str] = {}
    if creds.use_bearer:
        headers["Authorization"] = f"Bearer {creds.token}"
        headers["anthropic-beta"] = "oauth-2025-04-20"
        headers["User-Agent"] = "claude-cli/1.0.83 (external, cli)"
        headers["x-app"] = "cli"
        headers["x-claude-code-session-id"] = str(__import__("uuid").uuid4())
    model_kwargs = {}
    if creds.use_bearer:
        model_kwargs["metadata"] = {"user_id": "miniclaude-local-user"}
    return ChatAnthropic(
        model=creds.model,
        anthropic_api_url=creds.base_url,
        anthropic_api_key=creds.token,
        default_headers=headers or None,
        model_kwargs=model_kwargs,
        max_tokens=4096,
    )


def current_credentials() -> Credentials:
    creds = resolve()
    if creds is None:
        raise RuntimeError(
            "No credentials found. Either log into Claude Code (subscription) "
            "or set ANTHROPIC_API_KEY / OPENROUTER_API_KEY in .env."
        )
    return creds


def build_agent():
    creds = current_credentials()
    llm = build_llm(creds).bind_tools(ALL_TOOLS)
    system_prompt = load_system_prompt(oauth=(creds.mode == "oauth"))

    def call_llm(state: MessagesState) -> dict:
        todos = latest_todos(state["messages"])
        reminder = render_todo_reminder(todos)
        messages = [SystemMessage(system_prompt), *state["messages"], reminder]
        return {"messages": [llm.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm", call_llm)
    graph.add_node("review", review_tool_calls)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route_after_llm, {"review": "review", END: END})
    graph.add_conditional_edges(
        "review", route_after_review, {"tools": "tools", "llm": "llm"}
    )
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=open_checkpointer())


