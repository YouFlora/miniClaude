# miniClaude

> A minimal reimplementation of Claude Code's core mechanisms — built in Python with LangGraph.

Eight progressive milestones, each tagged in git (`m1`–`m8`). Every milestone is an independently runnable version.

## Milestones

| Tag | Milestone | Capability |
|---|---|---|
| `m1` | Agent Loop | Minimal LangGraph agent that chats with an LLM |
| `m2` | Tool System | Bash / Read / Write / Edit / Grep / Glob |
| `m3` | HITL Approval | Ask the user before dangerous operations |
| `m4` | TodoList | Agent self-manages multi-step tasks |
| `m5` | Persistence | SQLite checkpointer; resume sessions across processes |
| `m6` | SubAgent | Dispatch concurrent sub-agents |
| `m7` | Context Compaction | 8-segment compaction so long conversations don't blow the token budget |
| `m8` | Steering | Interrupt and redirect the agent mid-execution |

## Quick start

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env          # fill in OPENROUTER_API_KEY (or ANTHROPIC_AUTH_TOKEN)
uv run python miniClaude.py
```

REPL commands: type to chat, `/clear` to reset history, `/exit` to quit.

## Project layout

```
miniClaude/
├── miniClaude.py    # CLI entry (REPL)
├── agent.py         # LangGraph main graph
├── requirements.txt # Python dependencies
├── .env.example     # Env var template
└── .gitignore
```

Later milestones will add `tools/`, `prompts/`, `permissions.py`, `checkpointer.py`, and so on.

## Auth

miniClaude auto-detects credentials in this order (first hit wins):

1. **Claude Code subscription** — if you're logged into Claude Code on this machine, miniClaude pulls the OAuth token from macOS Keychain or `~/.claude/.credentials.json` and uses your Pro/Max plan's Sonnet/Opus. Zero config.
2. `ANTHROPIC_AUTH_TOKEN` — any OAuth Bearer token in env.
3. `ANTHROPIC_API_KEY` — direct Anthropic API billing.
4. `OPENROUTER_API_KEY` — OpenRouter fallback for cheap / free models.

The startup banner prints which route is in use. To force a specific route, only set env vars for that route.

> ⚠️ Using Claude Code OAuth tokens from a third-party app sits in a grey zone w.r.t. Anthropic TOS. Fine for personal demos and learning; do not deploy this pattern.

## License

MIT
