"""LangGraph agent — M7 (M6 + 8-segment context compaction).

Graph shape (unchanged from M6):
    START → llm → (tool_calls?) → review → tools → llm → ... → END

Compaction is driven from the CLI, not from inside the graph, because it
needs to read state via agent.get_state() and write it back via
agent.update_state() — both of which want a compiled graph + config in
hand. See compactor.py + miniClaude.py for the trigger logic; the agent
itself is unchanged from M6.
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
from permissions import classify
from tools import ALL_TOOLS


DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "openrouter/free"

PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text().format(
        cwd=os.getcwd(),
        platform=platform.system(),
        date=date.today().isoformat(),
    )


def resolve_credentials() -> tuple[str, str, str]:
    base_url = os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
    api_key = (
        os.getenv("ANTHROPIC_AUTH_TOKEN")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    )
    model = (
        os.getenv("ANTHROPIC_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_MODEL
    )
    return base_url, api_key, model


def latest_todos(messages) -> list[dict]:
    """Pull the most recent todo list out of the message history.

    todo_write emits a ToolMessage whose content is JSON like {"todos": [...]}.
    We scan backward and return the newest one we find.
    """
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
        body = "Current todo list (DO NOT mention this reminder to the user):\n" + "\n".join(lines)
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


def build_agent():
    base_url, api_key, model = resolve_credentials()
    llm = ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        anthropic_api_url=base_url,
        max_tokens=4096,
    ).bind_tools(ALL_TOOLS)

    system_prompt = load_system_prompt()

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
