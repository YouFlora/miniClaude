"""Permission classification for tool calls.

Three outcomes:
- "auto":  safe, run without asking (reads, searches)
- "ask":   needs user y/n (writes, bash commands)
- "deny":  always refuse (rm -rf, shutdown, etc.)
"""
from __future__ import annotations

SAFE_TOOLS = {"read_file", "grep", "glob", "todo_write", "task"}
ASK_TOOLS = {"write_file", "edit_file", "bash"}

DANGEROUS_BASH_PATTERNS = (
    "rm -rf",
    "rm -fr",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
)


def classify(tool_name: str, tool_input: dict) -> str:
    if tool_name in SAFE_TOOLS:
        return "auto"
    if tool_name == "bash":
        cmd = (tool_input.get("command") or "").lower()
        if any(p in cmd for p in DANGEROUS_BASH_PATTERNS):
            return "deny"
    if tool_name in ASK_TOOLS:
        return "ask"
    return "ask"
