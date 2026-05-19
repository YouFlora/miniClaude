"""LLM backend that shells out to the local `claude` CLI.

Why: lets miniClaude reuse the user's existing Claude Code subscription
(no separate API bill, no OAuth header guessing). The CLI handles auth
itself; we just feed it a prompt and parse the response.

How: each `invoke()` is a stateless single-shot run of
  claude -p --model X --tools "" --output-format json
  --no-session-persistence --append-system-prompt <SYS>
with the conversation history rendered as the stdin user prompt.

Tool calls: `claude -p` does not expose Anthropic's native tool_use
protocol, so we instruct the model (in the appended system prompt) to
emit ONE of two JSON shapes:
  {"action":"final","text":"..."}
  {"action":"tool_call","calls":[{"name":"...","args":{...}}, ...]}
We parse that and synthesize a LangChain AIMessage with .tool_calls.

This is a duck-typed mini ChatModel — agent.py only calls .invoke() and
.bind_tools(), so inheriting BaseChatModel would just be ceremony.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_TIMEOUT_SEC = 180


_ACTION_INSTRUCTION = """\
You are the planning brain of a ReAct agent. On every turn you MUST respond \
with EXACTLY ONE line of compact JSON — no markdown, no code fence, no prose \
before or after.

Schema (pick one):
  {"action":"final","text":"<reply to the user>"}
  {"action":"tool_call","calls":[{"name":"<tool>","args":{...}}, ...]}

Rules:
- Use "tool_call" when you need to gather information or change state.
- You may emit multiple parallel calls in one turn if they are independent.
- Use "final" only when you have enough info to fully answer the user.
- Tool args MUST match the tool's schema below — extra keys will be ignored.

Available tools:
"""


def _resolve_claude_bin() -> str:
    found = shutil.which("claude")
    if found:
        return found
    for cand in ("/root/.local/bin/claude", "/usr/local/bin/claude"):
        if os.path.exists(cand):
            return cand
    raise RuntimeError(
        "claude CLI not found on PATH. Install Claude Code, run `claude /login`, "
        "and retry — or set MINICLAUDE_PREFER_API_KEY=1 to skip this backend."
    )


def is_claude_cli_available() -> bool:
    """Cheap probe for credentials.py — does `claude --version` exit 0?"""
    try:
        bin_path = _resolve_claude_bin()
    except RuntimeError:
        return False
    try:
        res = subprocess.run(
            [bin_path, "--version"],
            capture_output=True, text=True, timeout=3,
        )
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _render_tools_desc(tools: list) -> str:
    """Render LangChain @tool list as a text spec for the LLM.

    Includes the full docstring (not just the first line) because some tools
    document non-obvious arg shapes there — e.g. todo_write spells out each
    todo's expected keys (`id`/`content`/`status`) and the model needs to see
    that to produce well-formed args.
    """
    lines = []
    for t in tools:
        try:
            schema = t.args_schema.model_json_schema()
            props = schema.get("properties", {})
            args_sig = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        except (AttributeError, Exception):
            args_sig = ""
        # Indent each docstring line so the block is visually grouped.
        doc = (t.description or "").strip()
        indented = "\n".join(f"    {line}" for line in doc.splitlines())
        lines.append(f"- {t.name}({args_sig})\n{indented}")
    return "\n".join(lines)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    return str(content)


def _render_message(m: BaseMessage) -> str:
    if isinstance(m, HumanMessage):
        return f"User: {_extract_text(m.content)}"
    if isinstance(m, AIMessage):
        text = _extract_text(m.content)
        parts = []
        if text:
            parts.append(f"Assistant: {text}")
        for tc in (m.tool_calls or []):
            parts.append(
                f"Assistant (tool call): {tc['name']}({json.dumps(tc.get('args', {}))})"
            )
        return "\n".join(parts) or "Assistant: (silent)"
    if isinstance(m, ToolMessage):
        name = getattr(m, "name", "tool")
        return f"Tool[{name}] → {_extract_text(m.content)}"
    # SystemMessage handled separately by the caller; fall through for unknowns
    return f"{type(m).__name__}: {_extract_text(m.content)}"


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    return s.strip()


def _extract_first_json_object(s: str) -> dict | None:
    """Walk forward until we find a balanced {...} block — survives extra prose."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_action(raw: str) -> AIMessage:
    """Parse the CLI's text into an AIMessage. Falls back to plain text."""
    s = _strip_fences(raw)
    action: dict | None
    try:
        action = json.loads(s)
    except json.JSONDecodeError:
        action = _extract_first_json_object(s)
    if not isinstance(action, dict):
        return AIMessage(content=raw)

    if action.get("action") == "tool_call":
        tool_calls = []
        for c in action.get("calls", []):
            if not isinstance(c, dict) or "name" not in c:
                continue
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": c["name"],
                "args": c.get("args") or {},
            })
        if tool_calls:
            return AIMessage(content="", tool_calls=tool_calls)
    # "final" or anything unrecognized → treat as text answer
    return AIMessage(content=action.get("text") or raw)


def _spawn_claude(system: str, user: str, model: str, timeout: int) -> str:
    """Run claude -p once and return the inner result string."""
    bin_path = _resolve_claude_bin()
    # Strip API-key env vars so the CLI is forced through its OAuth session,
    # not a stale (or wrong-account) API key on this machine.
    clean_env = {
        k: v for k, v in os.environ.items()
        if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_API_KEY")
    }
    try:
        proc = subprocess.run(
            [
                bin_path, "-p",
                "--model", model,
                "--tools", "",
                "--output-format", "json",
                "--no-session-persistence",
                "--append-system-prompt", system,
            ],
            input=user,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),  # avoid loading any project CLAUDE.md
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude CLI timeout after {timeout}s")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()[:800]
        if "Not logged in" in msg or "Invalid bearer" in msg:
            raise RuntimeError(
                "claude CLI auth failed. Run `claude /login` once, or unset "
                "MINICLAUDE_PREFER_API_KEY and fall back to a configured API key."
            )
        raise RuntimeError(f"claude CLI exit {proc.returncode}: {msg}")

    try:
        outer = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"claude CLI: stdout was not JSON. First 300 chars:\n{proc.stdout[:300]}"
        )
    if outer.get("is_error"):
        raise RuntimeError(f"claude CLI error: {outer.get('result')}")
    result = outer.get("result")
    if not isinstance(result, str):
        raise RuntimeError("claude CLI: missing 'result' field in JSON output")
    return result


class ClaudeCliChatModel:
    """Duck-typed LangChain ChatModel that delegates to `claude -p`.

    Only implements what agent.py actually uses: bind_tools() and invoke().
    """

    def __init__(self, model: str | None = None, timeout: int = DEFAULT_TIMEOUT_SEC):
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.tools: list = []
        self._tools_desc: str = ""

    def bind_tools(self, tools: list) -> "ClaudeCliChatModel":
        self.tools = list(tools)
        self._tools_desc = _render_tools_desc(self.tools)
        return self

    def invoke(self, messages: list[BaseMessage], config=None, **kwargs) -> AIMessage:
        system_parts = []
        history_parts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_parts.append(_extract_text(m.content))
            else:
                history_parts.append(_render_message(m))

        appended_system = (
            "\n\n".join(system_parts).strip()
            + "\n\n"
            + _ACTION_INSTRUCTION
            + self._tools_desc
            + "\n\nRespond now with one JSON object only."
        )
        user_prompt = "\n\n".join(history_parts) or "(no prior messages)"

        raw = _spawn_claude(appended_system, user_prompt, self.model, self.timeout)
        return _parse_action(raw)
