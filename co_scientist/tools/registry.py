"""ToolRegistry — discovers and indexes all available tools.

Tools available to each agent are decided by `tools_for(agent, mode)` so we can
restrict what the LLM sees per call (smaller tool list = better tool-use quality).
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from .base import Tool, ToolCtx, ToolResult, to_anthropic_tool
from .builtins.arxiv import ArxivSearchTool
from .builtins.europe_pmc import EuropePMCSearchTool
from .builtins.pubmed import PubmedSearchTool
from .science_skills import ScienceSkillTool, discover_skills
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool

# Per-agent tool allowlists. Keys are tool names (or "literature_*" wildcards
# matched explicitly in the resolver).
AGENT_TOOLS: dict[str, set[str]] = {
    "generation": {
        "web_search", "web_fetch",
        "pubmed_search", "arxiv_search", "europe_pmc_search",
        "literature_*",   # any science-skills literature_* tools
    },
    "reflection": {
        "web_search", "web_fetch",
        "pubmed_search", "arxiv_search", "europe_pmc_search",
        "literature_*",
        # code_exec wired in M2
    },
    "ranking": set(),                # no tools mid-debate
    "evolution": {
        "web_search", "web_fetch",
        "pubmed_search", "arxiv_search", "europe_pmc_search",
        "literature_*",
    },
    "proximity": set(),
    "metareview": set(),
}


class ToolRegistry:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._tools: dict[str, Tool] = {}

    def discover(self) -> ToolRegistry:
        # Built-ins
        for t in (
            WebFetchTool(self._cfg),
            PubmedSearchTool(self._cfg),
            ArxivSearchTool(),
            EuropePMCSearchTool(),
        ):
            self._register(t)
        # web_search only registers if a backing search API key is set.
        # Otherwise the model would see a tool it can't actually use and
        # smaller models tend to abort the task instead of falling back to
        # PubMed / arxiv / Europe PMC.
        import os
        if (
            self._cfg.secrets.TAVILY_API_KEY or os.environ.get("TAVILY_API_KEY")
            or self._cfg.secrets.BRAVE_API_KEY or os.environ.get("BRAVE_API_KEY")
        ):
            self._register(WebSearchTool(self._cfg))
        # Science-skills
        for meta in discover_skills(self._cfg):
            self._register(ScienceSkillTool(self._cfg, meta))
        return self

    def _register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            # later registrations win; log a warning at use site if needed
            pass
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def tools_for(self, agent: str) -> list[Tool]:
        allow = AGENT_TOOLS.get(agent, set())
        out: list[Tool] = []
        for t in self._tools.values():
            if t.name in allow:
                out.append(t)
            else:
                for pattern in allow:
                    if pattern.endswith("*") and t.name.startswith(pattern[:-1]):
                        out.append(t)
                        break
        return out

    def anthropic_tools_for(self, agent: str) -> list[dict[str, Any]]:
        return [to_anthropic_tool(t) for t in self.tools_for(agent)]

    async def call(
        self, name: str, args: dict[str, Any], ctx: ToolCtx
    ) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(is_error=True, error_message=f"unknown tool: {name}")
        return await tool.call(args, ctx)

    def summary(self) -> list[dict[str, Any]]:
        """Used by `co-scientist tools list` and the UI."""
        return [
            {"name": t.name, "description": t.description[:200]}
            for t in sorted(self._tools.values(), key=lambda x: x.name)
        ]
