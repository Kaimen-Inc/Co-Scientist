"""Smoke tests for tool registry + science-skills bridge parsing."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from co_scientist.tools.registry import ToolRegistry
from co_scientist.tools.science_skills import discover_skills, parse_skill_md


def test_registry_discovers_builtins(tmp_cfg) -> None:
    """web_search needs a TAVILY/BRAVE key; the others are always available."""
    tmp_cfg.secrets.TAVILY_API_KEY = "sk-fake"
    reg = ToolRegistry(tmp_cfg).discover()
    names = {t.name for t in reg.all()}
    assert {"web_search", "web_fetch", "pubmed_search", "arxiv_search", "europe_pmc_search"} <= names


def test_web_search_skipped_when_no_search_api_key(tmp_cfg) -> None:
    """Without a Tavily/Brave key the model would only see a tool that returns
    errors; small models tend to abort instead of falling back to PubMed.
    Auto-skip the registration to remove that footgun."""
    tmp_cfg.secrets.TAVILY_API_KEY = ""
    tmp_cfg.secrets.BRAVE_API_KEY = ""
    reg = ToolRegistry(tmp_cfg).discover()
    names = {t.name for t in reg.all()}
    assert "web_search" not in names
    # Other literature tools still available.
    assert "pubmed_search" in names
    assert "europe_pmc_search" in names


def test_agent_allowlist_resolution(tmp_cfg) -> None:
    tmp_cfg.secrets.TAVILY_API_KEY = "sk-fake"
    reg = ToolRegistry(tmp_cfg).discover()
    assert len(reg.tools_for("ranking")) == 0
    assert len(reg.tools_for("proximity")) == 0
    # generation/reflection/evolution get all built-in literature tools
    for agent in ("generation", "reflection", "evolution"):
        ts = {t.name for t in reg.tools_for(agent)}
        assert "web_search" in ts
        assert "pubmed_search" in ts


def test_skill_md_parsing(tmp_path: Path, tmp_cfg, monkeypatch) -> None:
    skills_root = tmp_path / "skills"
    sk = skills_root / "my_test_skill"
    (sk / "scripts").mkdir(parents=True)
    (sk / "SKILL.md").write_text(
        dedent(
            """\
            ---
            name: my_test_skill
            description: A short description for the LLM
            entrypoint: scripts/run.py
            timeout_seconds: 30
            ---

            More detail follows.
            """
        )
    )
    (sk / "scripts" / "run.py").write_text("print('{}')\n")

    meta = parse_skill_md(sk)
    assert meta is not None
    assert meta.name == "my_test_skill"
    assert meta.description.startswith("A short description")
    assert meta.entrypoint is not None and meta.entrypoint.name == "run.py"
    assert meta.timeout_seconds == 30

    # discover_skills walks <science_skills.path>/skills
    monkeypatch.setattr(tmp_cfg.science_skills, "path", str(tmp_path))
    discovered = discover_skills(tmp_cfg)
    assert any(d.name == "my_test_skill" for d in discovered)


def test_skill_md_without_front_matter_still_parses(tmp_path: Path) -> None:
    sk = tmp_path / "raw_skill"
    (sk / "scripts").mkdir(parents=True)
    (sk / "SKILL.md").write_text("# Raw skill\n\nThis describes what it does.\n")
    (sk / "scripts" / "main.py").write_text("print('{}')\n")
    meta = parse_skill_md(sk)
    assert meta is not None
    assert meta.name == "raw_skill"
    assert meta.entrypoint is not None and meta.entrypoint.name == "main.py"
