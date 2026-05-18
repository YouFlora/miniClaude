"""LangGraph agent — M3 (HITL permissions on top of M2 tool system).

Graph shape:
    START → llm → (tool_calls?) → review → tools → llm → ... → END
                                     ↓
                                interrupt() if any tool_call needs approval

- review node inspects every tool_call against permissions.classify():
  - "auto"  → pass through
  - "ask"   → call interrupt(payload); front-end resumes with allow/deny
  - "deny"  → replace tool_call with a rejection ToolMessage (no execution)
- A checkpointer (MemorySaver here, SqliteSaver in M5) is required for
  interrupt() to work: it persists state when the graph pauses.
"""
from __future__ import annotations

import os
import platform
from datetime import date
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

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


def review_tool_calls(state: MessagesState) -> dict:
    """Inspect the latest AIMessage's tool_calls and gate them.

    For each tool_call we either:
    - let it through (auto)
    - interrupt() so the user can approve / reject (ask)
    - skip execution and emit a ToolMessage saying it was denied (deny)

    interrupt() pauses the graph; the front-end calls invoke with
    Command(resume={call_id: True/False, ...}) to continue.
    """
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
        messages = [SystemMessage(system_prompt), *state["messages"]]
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

    return graph.compile(checkpointer=MemorySaver())
