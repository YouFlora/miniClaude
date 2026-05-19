"""Detect and load credentials for the Anthropic API.

Priority order (first hit wins):
1. Claude CLI backend — if `claude --version` works on PATH, delegate
   each LLM call to `claude -p` and reuse the user's local subscription.
   No OAuth header guessing — the CLI handles auth itself.
2. Claude Code subscription OAuth (raw) — pull token from macOS Keychain
   or ~/.claude/.credentials.json and call api.anthropic.com directly.
   Bypassed if path 1 worked.
3. ANTHROPIC_AUTH_TOKEN env — any OAuth Bearer token.
4. ANTHROPIC_API_KEY env — direct Anthropic API billing (sk-ant-...).
5. OPENROUTER_API_KEY env — OpenRouter route (cheap / free models).

resolve() returns a Credentials object describing what mode + endpoint
to use. build_agent() in agent.py consumes this to configure the LLM.

Note: pulling OAuth tokens out of Claude Code's credential store and
using them from a third-party app sits in a grey zone wrt Anthropic
TOS. This is fine for personal demos; do NOT use this pattern in
production or shared deployments.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


CLAUDE_CRED_FILE = Path.home() / ".claude" / ".credentials.json"
KEYCHAIN_SERVICE = "Claude Code-credentials"

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api"


@dataclass
class Credentials:
    mode: str            # "claude_cli" | "oauth" | "api_key"
    token: str           # OAuth access token or API key (empty for claude_cli)
    base_url: str        # ignored in claude_cli mode
    model: str
    source: str          # short label for the CLI banner
    use_bearer: bool     # True → Authorization: Bearer; False → x-api-key


def _read_keychain() -> dict | None:
    """macOS only: pull Claude Code's stored credentials out of Keychain."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _read_cred_file() -> dict | None:
    """Linux / fallback: read ~/.claude/.credentials.json if present."""
    if not CLAUDE_CRED_FILE.exists():
        return None
    try:
        return json.loads(CLAUDE_CRED_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _claude_code_oauth() -> tuple[str, int] | None:
    """Return (access_token, expires_at_ms) from Claude Code, or None."""
    raw = _read_keychain() or _read_cred_file()
    if not raw:
        return None
    oauth = raw.get("claudeAiOauth") or raw
    token = oauth.get("accessToken")
    if not token:
        return None
    expires_at = int(oauth.get("expiresAt") or 0)
    if expires_at and expires_at < int(time.time() * 1000):
        return None
    return token, expires_at


def resolve() -> Credentials | None:
    # Escape hatch: skip both subscription paths (CLI and raw OAuth) and go
    # straight to API-key auth. Useful when also logged into Claude Code but
    # you want to exercise the OpenRouter / API-key code path.
    skip_subscription = os.getenv("MINICLAUDE_PREFER_API_KEY") == "1"

    if not skip_subscription:
        # Lazy import to avoid pulling subprocess machinery during help/list runs.
        from claude_cli_backend import is_claude_cli_available
        if is_claude_cli_available():
            return Credentials(
                mode="claude_cli",
                token="",
                base_url="",
                model=os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL,
                source="Claude CLI (local subscription)",
                use_bearer=False,
            )

    sub = None if skip_subscription else _claude_code_oauth()
    if sub:
        token, _ = sub
        return Credentials(
            mode="oauth",
            token=token,
            base_url=DEFAULT_ANTHROPIC_URL,
            model=os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL,
            source="Claude Code subscription",
            use_bearer=True,
        )

    if token := os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return Credentials(
            mode="oauth",
            token=token,
            base_url=os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_URL,
            model=os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL,
            source="ANTHROPIC_AUTH_TOKEN env",
            use_bearer=True,
        )

    if key := os.getenv("ANTHROPIC_API_KEY"):
        return Credentials(
            mode="api_key",
            token=key,
            base_url=os.getenv("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_URL,
            model=os.getenv("ANTHROPIC_MODEL") or DEFAULT_CLAUDE_MODEL,
            source="ANTHROPIC_API_KEY env",
            use_bearer=False,
        )

    if key := os.getenv("OPENROUTER_API_KEY"):
        return Credentials(
            mode="api_key",
            token=key,
            base_url=DEFAULT_OPENROUTER_URL,
            model=os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
            source="OPENROUTER_API_KEY env",
            use_bearer=False,
        )

    return None
