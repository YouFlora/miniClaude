"""SubAgent runner: spawn an ephemeral child graph for a delegated task.

Why this exists:
- Long context kills attention. Pushing an isolated investigation into a
  child agent means the parent only sees the final summary, not the
  intermediate tool calls / thinking. This is the "context isolation"
  half of context engineering (the "compression" half is M7).

Design:
- Each call builds a fresh StateGraph with its own MessagesState. No
  checkpointer — the sub-agent is stateless and one-shot.
- Tool sets are restricted by subagent_type. `task` itself is NEVER in
  the sub-agent's tool set, preventing infinite recursion.
- The sub-agent runs LLM ↔ ToolNode until it stops emitting tool_calls.
  Its final AI message text becomes the parent's ToolMessage payload.
- Auth is inherited from the parent (Claude Code subscription / API key
  / OpenRouter), via agent.build_llm + credentials.resolve.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent import build_llm, current_credentials, load_system_prompt
from tools import bash, edit_file, glob, grep, read_file, write_file


SUBAGENT_TOOL_SETS = {
    "general": [bash, read_file, write_file, edit_file, grep, glob],
    "code-explorer": [read_file, grep, glob],
}

SUBAGENT_PROMPT_SUFFIX = (
    "\n\n# Sub-agent mode\n"
    "You are running as a sub-agent. Complete the task autonomously and "
    "return a SINGLE concise summary as your final reply — the parent "
    "agent will see only that message, not your tool calls. Do not ask "
    "the user follow-up questions; if information is missing, report it "
    "in your summary."
)


def build_subagent_graph(tools: list):
    creds = current_credentials()
    llm = build_llm(creds).bind_tools(tools)
    sub_prompt = load_system_prompt(
        identity_prefix=creds.mode in ("oauth", "claude_cli")
    ) + SUBAGENT_PROMPT_SUFFIX

    def call_llm(state: MessagesState) -> dict:
        return {
            "messages": [
                llm.invoke([SystemMessage(sub_prompt), *state["messages"]])
            ]
        }

    g = StateGraph(MessagesState)
    g.add_node("llm", call_llm)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", tools_condition)
    g.add_edge("tools", "llm")
    return g.compile()


def run_subagent(subagent_type: str, prompt: str) -> str:
    tools = SUBAGENT_TOOL_SETS.get(subagent_type, SUBAGENT_TOOL_SETS["general"])
    graph = build_subagent_graph(tools)
    result = graph.invoke({"messages": [HumanMessage(prompt)]})
    last = result["messages"][-1]
    content = last.content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p) or "(empty)"
    return content or "(empty)"
