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

## Model

Defaults to **OpenRouter** (Anthropic-compatible endpoint). The free model `openai/gpt-oss-120b:free` is enough to run M1. You can swap in the official Anthropic API or any OpenAI-compatible endpoint by editing `.env`.

## License

MIT
