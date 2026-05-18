"""8-segment context compaction (Yuyz's reverse-engineered compact prompt).

Why this exists:
- Conversations grow without bound. Truncating loses early decisions; doing
  nothing blows the context window.
- Claude Code's answer: a structured summary that compresses everything into
  8 sections (background / concepts / files / errors / problem-solving /
  user messages / pending tasks / current work). Triggered at ~92% of the
  model's context budget.

What this module does:
- usage_input_tokens / needs_compact: cheap check based on the latest
  AIMessage's usage_metadata.input_tokens (LangChain populates this from the
  Anthropic API response on every llm.invoke).
- summarize: feed the conversation to an LLM with prompts/compact.md (copied
  verbatim from Yuyz). The model returns a single text block with an
  <analysis> ... <summary> ... 8-section structure.
- replace_with_summary: emit RemoveMessage ids for every existing message and
  append one HumanMessage holding the summary. LangGraph's add_messages
  reducer treats RemoveMessage as a deletion marker, so state ends up holding
  ONLY the summary.

Threshold default is intentionally low (8 000 tokens) so the trigger is
visible in short demos. Production Claude Code uses ~92 % of 200 k = ~184 k.
"""
from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, RemoveMessage

from agent import resolve_credentials


COMPACT_PROMPT_PATH = Path(__file__).parent / "prompts" / "compact.md"
DEFAULT_THRESHOLD_TOKENS = 8_000


def usage_input_tokens(messages) -> int:
    for m in reversed(messages):
        meta = getattr(m, "usage_metadata", None) or {}
        if "input_tokens" in meta:
            return int(meta["input_tokens"])
    return 0


def needs_compact(messages, threshold: int = DEFAULT_THRESHOLD_TOKENS) -> bool:
    return usage_input_tokens(messages) >= threshold


def _summarizer_llm() -> ChatAnthropic:
    base_url, api_key, model = resolve_credentials()
    return ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        anthropic_api_url=base_url,
        max_tokens=4096,
    )


def summarize(messages) -> str:
    """Compress the conversation into the 8-segment summary."""
    prompt = COMPACT_PROMPT_PATH.read_text()
    llm = _summarizer_llm()
    result = llm.invoke([*messages, HumanMessage(prompt)])
    content = result.content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return content or ""


def replace_with_summary(agent, config, messages, summary: str) -> None:
    """Atomically swap the entire message history for a single summary."""
    if not summary.strip():
        return
    update = [RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None)]
    update.append(
        HumanMessage(f"<conversation-summary>\n{summary}\n</conversation-summary>")
    )
    agent.update_state(config, {"messages": update})
