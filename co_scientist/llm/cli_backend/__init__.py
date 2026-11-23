"""Subscription-backed LLM backends driven by local agent CLIs.

Instead of billing an API key per token, these backends shell out to an agent
harness the user already pays for a seat on — `claude -p` (Claude Code),
`codex exec` (Codex), or `agy` (Antigravity CLI) — and let it run its own
agentic loop. Structured output and tool provenance come back through the MCP
server in `co_scientist.mcp`.
"""

from .agy import AgyCliProvider
from .base import AgentCliProvider, CliInvocation, CliOutcome
from .claude_code import ClaudeCliProvider
from .codex import CodexCliProvider

__all__ = [
    "AgentCliProvider",
    "AgyCliProvider",
    "ClaudeCliProvider",
    "CliInvocation",
    "CliOutcome",
    "CodexCliProvider",
]
