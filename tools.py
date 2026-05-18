"""Six core tools for miniClaude: bash, read_file, write_file, edit_file, grep, glob.

Each function is decorated with @tool — LangGraph's ToolNode runs it when the
LLM emits a matching tool_call. The docstring becomes the description the LLM
sees in the tool schema, so it must teach the model when and how to use it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool


@tool
def bash(command: str) -> str:
    """Run a shell command and return its combined stdout/stderr.

    30-second timeout. Prefer grep/glob/read_file over invoking grep/find/cat.
    """
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr) or "(no output)"


@tool
def read_file(file_path: str) -> str:
    """Read a file and return its contents. Use instead of `cat`."""
    return Path(file_path).read_text()


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, overwriting if it exists.

    Prefer edit_file when changing existing files. Read the file first.
    """
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {file_path}"


@tool
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in a file.

    Fails if old_string is missing or appears more than once — provide
    enough surrounding context to make the match unique.
    """
    text = Path(file_path).read_text()
    count = text.count(old_string)
    if count == 0:
        return f"error: old_string not found in {file_path}"
    if count > 1:
        return f"error: old_string appears {count} times; needs to be unique"
    Path(file_path).write_text(text.replace(old_string, new_string))
    return f"edited {file_path}"


@tool
def grep(pattern: str, path: str = ".", file_glob: str = "*") -> str:
    """Search a regex over files. Returns matches as file:line:text."""
    regex = re.compile(pattern)
    hits = []
    for p in Path(path).rglob(file_glob):
        if not p.is_file():
            continue
        try:
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{p}:{i}:{line}")
        except (UnicodeDecodeError, PermissionError):
            continue
    return "\n".join(hits) or "(no matches)"


@tool
def glob(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern (e.g. '**/*.py')."""
    matches = [str(p) for p in Path(path).rglob(pattern) if p.is_file()]
    return "\n".join(matches) or "(no matches)"


ALL_TOOLS = [bash, read_file, write_file, edit_file, grep, glob]
