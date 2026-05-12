"""LangGraph agent definition.

M1: minimal agent — one LLM node, no tools, no checkpointer.
The graph is just START → llm → END. Conversation history is
passed in by the caller each invocation (stateless graph).

Provider: configured to talk to OpenRouter via Anthropic SDK
(same convention as Claude Code's `.claude/settings.json`).
"""
from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, END, MessagesState, StateGraph


# Defaults match real Claude Code's OpenRouter setup.
DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "openrouter/free"


def _resolve_credentials() -> tuple[str, str, str]:
    """Pull base_url / api_key / model from env, with fallbacks."""
    base_url = os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
    api_key = (
        os.getenv("ANTHROPIC_AUTH_TOKEN")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")  # fallback if user named it this way
        or ""
    )
    model = (
        os.getenv("ANTHROPIC_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_MODEL
    )
    return base_url, api_key, model


def build_agent():
    """Compile and return the LangGraph agent."""
    base_url, api_key, model = _resolve_credentials()

    llm = ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        anthropic_api_url=base_url,
        max_tokens=4096,
    )

    def call_llm(state: MessagesState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm", call_llm)
    graph.add_edge(START, "llm")
    graph.add_edge("llm", END)

    return graph.compile()
