"""LangGraph agent — M2 (agent loop + tool system).

Graph shape:
    START → llm → (tool_calls?) → tools → llm → ... → END

- bind_tools tells the LLM what tools exist; it emits tool_calls
- ToolNode runs the tool and pushes a ToolMessage back into state
- tools_condition routes: if last AI message has tool_calls → "tools", else END
- The system prompt is loaded from prompts/system.md with env vars filled in
"""
from __future__ import annotations

import os
import platform
from datetime import date
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")
    return graph.compile()
