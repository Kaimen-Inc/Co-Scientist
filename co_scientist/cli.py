"""Typer CLI entrypoint.

M0 surface: init, list, status, tools list, serve, version. Run/resume/report/feedback
land in M3+. All commands resolve config the same way; secrets come from env.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import has_llm_key, load_config, provider_key_env
from .logging import get_logger, setup_logging
from .storage import db as db_mod

VERSION = "0.1.0"

app = typer.Typer(
    help="AI Co-Scientist — multi-agent hypothesis generation, ranking, and synthesis.",
    no_args_is_help=True,
    add_completion=False,
)
tools_app = typer.Typer(help="Tool registry inspection and debug invocation.", no_args_is_help=True)
app.add_typer(tools_app, name="tools")

console = Console()
log = get_logger("cli")


def _common_setup(config_file: Path | None = None, verbose: bool = False) -> tuple:
    setup_logging("DEBUG" if verbose else "INFO")
    cfg = load_config(config_file)
    return cfg, log


@app.callback()
def _main(
    ctx: typer.Context,
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to an extra TOML config to overlay."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
) -> None:
    ctx.obj = _common_setup(config_file, verbose)


@app.command()
def version() -> None:
    """Print the co-scientist version."""
    console.print(f"co-scientist {VERSION}")


@app.command()
def init(ctx: typer.Context) -> None:
    """Create the data directory, apply migrations, sanity-check env."""
    cfg, _ = ctx.obj
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "artifacts").mkdir(exist_ok=True)
    (cfg.data_dir / "vectors").mkdir(exist_ok=True)
    (cfg.data_dir / "logs").mkdir(exist_ok=True)

    asyncio.run(db_mod.init_db(cfg))

    # Report
    env_var = provider_key_env(cfg)
    tbl = Table(title="Init complete", show_header=False, box=None)
    tbl.add_row("data dir", str(cfg.data_dir))
    tbl.add_row("database", str(cfg.db_path))
    tbl.add_row("LLM provider", cfg.llm.provider)
    tbl.add_row(
        f"{env_var or '(keyless)'} set",
        "yes" if has_llm_key(cfg) else "[red]no[/red]",
    )
    tbl.add_row("science-skills", cfg.science_skills.path)
    console.print(tbl)

    if not has_llm_key(cfg):
        console.print(
            f"[yellow]{env_var} is not set; set it (or change [llm] provider in your "
            f"config) before running a session. See .env.example.[/yellow]"
        )


@app.command(name="list")
def list_sessions(ctx: typer.Context) -> None:
    """List all sessions in the local database."""
    cfg, _ = ctx.obj

    async def _run() -> list[dict]:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute(
                """SELECT id, status, research_goal, created_at, updated_at,
                          budget_usd, budget_used_usd,
                          (SELECT COUNT(*) FROM hypotheses WHERE session_id = s.id) AS n_hyps,
                          (SELECT MAX(elo) FROM hypotheses WHERE session_id = s.id) AS top_elo
                     FROM sessions s
                     ORDER BY updated_at DESC"""
            ) as cur:
                rows = await cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    rows = asyncio.run(_run())
    if not rows:
        console.print("[dim]No sessions yet. Try:  co-scientist run \"your research goal\"[/dim]")
        return

    tbl = Table(title="Sessions")
    tbl.add_column("id")
    tbl.add_column("status")
    tbl.add_column("goal", overflow="fold", max_width=60)
    tbl.add_column("hyps", justify="right")
    tbl.add_column("top Elo", justify="right")
    tbl.add_column("$ used / $ budget", justify="right")
    tbl.add_column("updated")
    for r in rows:
        tbl.add_row(
            r["id"],
            r["status"],
            (r["research_goal"] or "")[:80],
            str(r["n_hyps"]),
            f"{r['top_elo']:.0f}" if r["top_elo"] is not None else "—",
            f"${r['budget_used_usd']:.2f} / ${r['budget_usd']:.2f}",
            r["updated_at"],
        )
    console.print(tbl)


@app.command()
def status(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Show detailed status for one session."""
    cfg, _ = ctx.obj

    async def _run() -> dict:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                return {}
            out = dict(row)
            for col in ("research_plan", "config_snapshot"):
                with contextlib.suppress(Exception):
                    out[col] = json.loads(out[col])
            async with conn.execute(
                """SELECT status, COUNT(*) AS n FROM tasks
                       WHERE session_id = ? GROUP BY status""",
                (session_id,),
            ) as cur:
                out["task_counts"] = {r["status"]: r["n"] for r in await cur.fetchall()}
            async with conn.execute(
                """SELECT state, COUNT(*) AS n FROM hypotheses
                       WHERE session_id = ? GROUP BY state""",
                (session_id,),
            ) as cur:
                out["hypothesis_states"] = {r["state"]: r["n"] for r in await cur.fetchall()}
            return out
        finally:
            await conn.close()

    out = asyncio.run(_run())
    if not out:
        console.print(f"[red]No session {session_id}[/red]")
        raise typer.Exit(1)
    console.print_json(data=out)


@app.command()
def run(
    ctx: typer.Context,
    goal: str = typer.Argument(..., help="Research goal in natural language."),
    preferences_file: Path | None = typer.Option(
        None, "--preferences-file", help="Path to a text file with extra preferences."
    ),
    n_initial: int = typer.Option(
        3, "--n", help="Number of initial Generation calls (parallel)."
    ),
    wall_clock: int | None = typer.Option(
        None, "--wall-clock", help="Override wall-clock cap in seconds."
    ),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="Override session USD budget."
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", help="Override worker concurrency."
    ),
) -> None:
    """Start a fresh research session. Generation → Reflection → Ranking tournament → Meta-review."""
    cfg, _ = ctx.obj
    if not has_llm_key(cfg):
        env_var = provider_key_env(cfg)
        console.print(f"[red]{env_var} is not set (LLM provider = {cfg.llm.provider}). See .env.example.[/red]")
        raise typer.Exit(1)

    if budget_usd is not None:
        cfg.run.budget_usd = budget_usd
    if concurrency is not None:
        cfg.run.concurrency = concurrency

    # Pre-flight cost estimate
    from .llm.estimator import estimate as _estimate

    est = _estimate(cfg)
    console.print(
        f"[dim]Pre-flight estimate: ${est.total_usd:.2f} "
        f"(budget ${cfg.run.budget_usd:.2f}, "
        f"max_ideas={cfg.run.max_ideas}, "
        f"max_matches_per_idea={cfg.run.max_matches_per_idea})[/dim]"
    )
    if est.warning:
        console.print(f"[yellow]{est.warning}[/yellow]")

    prefs = preferences_file.read_text() if preferences_file else None
    from .agents.supervisor import Supervisor

    sup = Supervisor(cfg)
    session_id = asyncio.run(
        sup.run_session(goal, preferences_text=prefs, n_initial=n_initial,
                        wall_clock_seconds=wall_clock)
    )
    console.print(f"[green]Done.[/green] session={session_id}")
    console.print(f"View report:  co-scientist report {session_id}")


@app.command()
def resume(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
) -> None:
    """Resume a paused or interrupted session."""
    cfg, _ = ctx.obj
    if not has_llm_key(cfg):
        env_var = provider_key_env(cfg)
        console.print(f"[red]{env_var} is not set (LLM provider = {cfg.llm.provider}). See .env.example.[/red]")
        raise typer.Exit(1)
    from .agents.supervisor import Supervisor

    sup = Supervisor(cfg)
    sid = asyncio.run(sup.run_session("", resume_session_id=session_id))
    console.print(f"[green]Done.[/green] session={sid}")


@app.command()
def pause(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Pause a running session. Workers drain; the loop sleeps until resume."""
    cfg, _ = ctx.obj

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            from .storage.repos import sessions as sess_repo

            await sess_repo.set_status(conn, session_id, "paused")
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[yellow]Paused[/yellow] session={session_id}")


@app.command()
def abort(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Abort a running session. The main loop exits at the next check."""
    cfg, _ = ctx.obj

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            from .storage.repos import sessions as sess_repo

            await sess_repo.set_status(conn, session_id, "aborted")
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[red]Aborted[/red] session={session_id}")


@app.command()
def feedback(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
    text: str = typer.Argument(..., help="Free-text feedback to inject."),
    kind: str = typer.Option(
        "directive", "--kind",
        help="directive | preference | rejection | pin",
    ),
    target: str | None = typer.Option(
        None, "--target", help="Hypothesis ID this feedback is about (optional)."
    ),
) -> None:
    """Inject researcher feedback into a running (or future) session."""
    cfg, _ = ctx.obj
    from . import ids as _ids
    from .models import SystemFeedback
    from .storage.repos import feedback as fb_repo
    from .storage.repos import hypotheses as hyp_repo

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            fb = SystemFeedback(
                id=_ids.feedback_id(), session_id=session_id,
                created_at=datetime.now(UTC),
                source="human", kind=kind,
                target_id=target, text=text, active=True,
            )
            await fb_repo.insert(conn, fb)
            # Pinning / rejection also flip the hypothesis state.
            if kind == "pin" and target:
                await hyp_repo.set_state(conn, target, "pinned")
            elif kind == "rejection" and target:
                await hyp_repo.set_state(conn, target, "rejected")
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[green]Feedback recorded[/green] for session={session_id}")


@app.command()
def report(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
    format: str = typer.Option("md", "--format", "-f", help="md or json"),
) -> None:
    """Print the final research overview for a session (when available)."""
    cfg, _ = ctx.obj

    async def _path() -> Path | None:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute(
                "SELECT final_overview FROM sessions WHERE id = ?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row or not row["final_overview"]:
                return None
            return cfg.data_dir / row["final_overview"]
        finally:
            await conn.close()

    p = asyncio.run(_path())
    if p is None:
        console.print(f"[red]No final overview yet for {session_id}[/red]")
        raise typer.Exit(1)
    text = p.read_text()
    if format == "json":
        console.print_json(data={"session_id": session_id, "path": str(p), "content": text})
    else:
        console.print(text)


@app.command()
def serve(
    ctx: typer.Context,
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Launch the FastAPI + htmx + SSE web UI."""
    cfg, _ = ctx.obj
    host = host or cfg.web_ui.host
    port = port or cfg.web_ui.port
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


@app.command()
def estimate(ctx: typer.Context) -> None:
    """Print the pre-flight cost estimate without launching a session."""
    cfg, _ = ctx.obj
    from .llm.estimator import estimate as _estimate

    est = _estimate(cfg)
    console.print_json(data=est.to_dict())


@app.command("eval")
def eval_cmd(
    ctx: typer.Context,
    agent: str | None = typer.Argument(None, help="generation|reflection|ranking|overview"),
    offline: bool = typer.Option(False, "--offline", help="Structural checks only; no judge."),
) -> None:
    """Run the eval rubric runner over the bundled fixtures."""
    cfg, _ = ctx.obj
    from .evals.runner import run_agent, run_all

    if agent:
        result = asyncio.run(run_agent(cfg, agent, offline=offline))
    else:
        result = asyncio.run(run_all(cfg, offline=offline))
    console.print_json(data=result)


@app.command("bench")
def bench_cmd(
    ctx: typer.Context,
    goal: str | None = typer.Argument(
        None, help="Research goal. Optional when --preset bundles a default goal."
    ),
    preset: str | None = typer.Option(
        None, "--preset",
        help=(
            "Use a built-in candidate list. Available: "
            "'paper' (Co-Scientist paper baselines + Haiku), "
            "'paper-aml' (same candidates + AML drug-repurposing goal + "
            "5-drug recall scoring against the paper's answer key). "
            "Mutually exclusive with --candidate."
        ),
    ),
    candidate: list[str] = typer.Option(
        None, "--candidate", "-c",
        help=(
            "Repeat: label=provider:model[@mode]. Mode is `pipeline` "
            "(default) or `direct` (single raw LM call, no tools). e.g. "
            "'gemini-flash=openrouter:google/gemini-3-flash-preview', "
            "'flash-raw=openrouter:google/gemini-3-flash-preview@direct'."
        ),
    ),
    n: int = typer.Option(2, "--n", help="Hypotheses per candidate."),
    matches: int = typer.Option(2, "--matches", help="Tournament matches per pair."),
    judge: str | None = typer.Option(
        None, "--judge",
        help="Judge as provider:model. Defaults to the preset's suggestion, else anthropic:claude-sonnet-4-6.",
    ),
    budget_per_candidate: float = typer.Option(
        2.0, "--budget-per-candidate", help="USD cap per candidate."
    ),
    judge_budget: float = typer.Option(
        5.0, "--judge-budget", help="USD cap for all judge calls combined."
    ),
) -> None:
    """Compare N models on the same goal via a cross-Elo tournament.

    Quick start (paper repro):
      co-scientist bench "Identify hypotheses about X" \\
        --preset paper \\
        --judge openrouter:google/gemini-3-flash-preview

    Custom candidates:
      co-scientist bench "Identify hypotheses about X" \\
        -c gemini-flash=openrouter:google/gemini-3-flash-preview \\
        -c gpt5=openai:gpt-5 \\
        -c opus=anthropic:claude-opus-4-7 \\
        --judge anthropic:claude-sonnet-4-6
    """
    cfg, _ = ctx.obj
    from .bench import BenchCandidate, get_preset, run_bench

    if preset and candidate:
        console.print("[red]--preset and --candidate are mutually exclusive[/red]")
        raise typer.Exit(2)

    candidates: list[BenchCandidate]
    goldset = None
    if preset:
        p = get_preset(preset)
        candidates = list(p.candidates)
        if judge is None:
            judge = p.suggested_judge
        if goal is None and p.default_goal is not None:
            goal = p.default_goal
        if p.goldset is not None:
            goldset = p.goldset
        console.print(f"[dim]Using preset '{p.name}': {p.description}[/dim]")
        for c in candidates:
            mode_suffix = f" [{c.mode}]" if c.mode != "pipeline" else ""
            console.print(f"[dim]  • {c.label}{mode_suffix}: {c.provider}:{c.model}[/dim]")
        if goldset:
            ent_list = ", ".join(e.name for e in goldset.entities)
            console.print(f"[dim]  gold set: {goldset.label} ({ent_list})[/dim]")
    else:
        if not candidate:
            console.print(
                "[red]Must provide either --preset or at least one --candidate[/red]"
            )
            raise typer.Exit(2)
        candidates = []
        for entry in candidate:
            if "=" not in entry or ":" not in entry.split("=", 1)[1]:
                console.print(
                    f"[red]--candidate must look like label=provider:model[@mode], got {entry!r}[/red]"
                )
                raise typer.Exit(2)
            label, rest = entry.split("=", 1)
            # Optional @mode suffix selects pipeline vs direct.
            mode = "pipeline"
            if "@" in rest:
                rest, mode = rest.rsplit("@", 1)
                mode = mode.strip().lower()
                if mode not in ("pipeline", "direct"):
                    console.print(
                        f"[red]unknown mode {mode!r} in {entry!r}; "
                        f"use `pipeline` or `direct`[/red]"
                    )
                    raise typer.Exit(2)
            provider, model = rest.split(":", 1)
            candidates.append(BenchCandidate(
                label=label, provider=provider, model=model, mode=mode,
            ))

    if goal is None:
        console.print(
            "[red]Must provide a research goal (positional argument) or a "
            "--preset that bundles one.[/red]"
        )
        raise typer.Exit(2)
    if judge is None:
        judge = "anthropic:claude-sonnet-4-6"
    if ":" not in judge:
        console.print(f"[red]--judge must look like provider:model, got {judge!r}[/red]")
        raise typer.Exit(2)
    judge_provider, judge_model = judge.split(":", 1)

    outcome = asyncio.run(
        run_bench(
            cfg, goal=goal, candidates=candidates,
            n_hyps_per_candidate=n,
            matches_per_pair=matches,
            judge_provider=judge_provider, judge_model=judge_model,
            per_candidate_budget_usd=budget_per_candidate,
            judge_budget_usd=judge_budget,
            goldset=goldset,
        )
    )

    has_gold = goldset is not None
    title = f"Bench {outcome.bench_id} — {outcome.matches_played} matches"
    if has_gold:
        title += f" • gold-set {goldset.label} (recall / {len(goldset.entities)})"
    tbl = Table(title=title)
    tbl.add_column("rank", justify="right")
    tbl.add_column("label", style="bold")
    tbl.add_column("mode")
    tbl.add_column("model")
    tbl.add_column("n_hyps", justify="right")
    tbl.add_column("W-L", justify="right")
    tbl.add_column("mean_elo", justify="right")
    if has_gold:
        tbl.add_column("gold hits", justify="right")
    tbl.add_column("$ spent", justify="right")
    tbl.add_column("p50_ms", justify="right")
    for i, row in enumerate(outcome.candidates, 1):
        row_cells = [
            str(i), row["label"],
            row.get("mode") or "pipeline",
            row["model"],
            str(row["n_hypotheses"]),
            f"{row['wins']}-{row['losses']}",
            f"{row['mean_elo']:.0f}" if row["mean_elo"] is not None else "—",
        ]
        if has_gold:
            n_hit = row.get("gold_hits") or 0
            row_cells.append(f"{n_hit}/{len(goldset.entities)}")
        row_cells.extend([
            f"{row['cost_usd']:.4f}",
            str(row["mean_latency_ms"] or "—"),
        ])
        tbl.add_row(*row_cells)
    console.print(tbl)
    console.print(f"[dim]Total cost: ${outcome.total_cost_usd:.4f}[/dim]")
    console.print(f"[dim]Artifact: {outcome.artifact_path}[/dim]")

    if has_gold:
        # Per-candidate hit detail so you can see which specific entities surfaced.
        for row in outcome.candidates:
            hits = row.get("gold_hit_names") or []
            if hits:
                console.print(
                    f"[dim]  {row['label']} surfaced: {', '.join(hits)}[/dim]"
                )


@tools_app.command("list")
def tools_list(ctx: typer.Context) -> None:
    """List registered tools (builtins + any discovered science-skills)."""
    cfg, _ = ctx.obj
    from .tools.registry import AGENT_TOOLS, ToolRegistry

    reg = ToolRegistry(cfg).discover()
    tbl = Table(title=f"Tools ({len(reg.all())})")
    tbl.add_column("name", style="bold")
    tbl.add_column("description", overflow="fold")
    for item in reg.summary():
        tbl.add_row(item["name"], item["description"])
    console.print(tbl)

    # Show per-agent allowlist resolution counts
    tbl2 = Table(title="Per-agent tool availability")
    tbl2.add_column("agent")
    tbl2.add_column("# tools", justify="right")
    tbl2.add_column("allowlist (patterns)")
    for agent, patterns in AGENT_TOOLS.items():
        ts = reg.tools_for(agent)
        tbl2.add_row(agent, str(len(ts)), ", ".join(sorted(patterns)) or "—")
    console.print(tbl2)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
