"""Shared machinery for CLI-driven LLM backends.

A backend supplies two things — how to build the command line
(`build_invocation`) and how to read the result (`parse_outcome`). Everything
else is common and lives here:

- budget admission and settlement (including releasing the reservation when a
  call raises, which otherwise leaks for the rest of the session)
- a per-call capture directory wired to the MCP server
- subprocess execution with a timeout and rate-limit-aware retries
- transcript row + on-disk artifact
- turning the CLI's text output plus captured records into an
  Anthropic-shaped response the agents can already read
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from ...config import Config
from ...ids import transcript_id
from ...logging import get_logger
from ...models import Transcript
from ...storage.artifacts import write_json
from ...storage.repos import sessions as sessions_repo
from ...storage.repos import transcripts as transcripts_repo
from ..budgets import TokenBudget
from ..retry import (
    CliBackendError,
    CliRetryableError,
    CliRetryPolicy,
    backoff_seconds,
)
from ..routing import estimate_cost_usd
from ..types import (
    AgentCallSpec,
    CallContext,
    CaptureBundle,
    LLMResponse,
    SynthMessage,
    TextBlock,
    ToolUseBlock,
    Usage,
    rough_token_count,
)

log = get_logger("llm.cli")

# Environment variables that would make an agent CLI bill a metered API key
# instead of the user's subscription. Stripped from every child process — the
# whole point of this backend is that no API key is involved.
BILLING_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)

@dataclass
class CliInvocation:
    """Everything needed to run one CLI process."""

    argv: list[str]
    stdin: str = ""
    env: dict[str, str] = field(default_factory=dict)
    """Extra env on top of the sanitized parent environment."""
    cwd: str | None = None


@dataclass
class CliOutcome:
    """Normalized result of one CLI process."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    num_turns: int = 1
    reported_cost_usd: float | None = None
    """Cost the CLI itself reported. Claude Code provides this; when absent we
    derive it from the price table. Under a subscription it is an
    equivalent-cost gauge, not money actually charged."""
    stop_reason: str = "end_turn"
    raw: dict[str, Any] = field(default_factory=dict)


class AgentCliProvider(ABC):
    """Base class implementing the `LLMProvider` protocol over a subprocess."""

    backend_name = "cli"
    server_label = "cosci"

    runs_own_loop = True
    """The CLI is itself an agent harness: it dispatches tools and iterates
    internally. `tool_loop.py` reads `LLMResponse.capture` instead of driving
    turns. The API clients leave this unset (falsy) and get driven."""

    def __init__(
        self,
        cfg: Config,
        *,
        db: aiosqlite.Connection | None,
        budget: TokenBudget,
        retry_policy: Any = None,       # accepted for call-site compatibility
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._budget = budget
        self._backend_cfg = self._resolve_backend_cfg(cfg)
        self._retry = retry_policy or CliRetryPolicy.from_config(cfg)
        self._semaphore = asyncio.Semaphore(max(1, self._backend_cfg.max_parallel))
        self._binary = self._resolve_binary()

    # ----------------------------- subclass API ----------------------------- #

    @abstractmethod
    def _resolve_backend_cfg(self, cfg: Config) -> Any:
        """Return the `[llm.<backend>]` config section."""

    @abstractmethod
    def build_invocation(
        self, spec: AgentCallSpec, ctx: CallContext, capture_dir: Path
    ) -> CliInvocation:
        """Build the command for one call."""

    @abstractmethod
    def parse_outcome(
        self, *, stdout: str, stderr: str, returncode: int, capture_dir: Path
    ) -> CliOutcome:
        """Turn process output into a `CliOutcome`, or raise CliBackendError."""

    # ----------------------------- main call ----------------------------- #

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> LLMResponse:
        est_in = est_input_tokens or rough_token_count(spec)
        est_out = spec.max_output_tokens
        est_cost = estimate_cost_usd(
            model=spec.route.model, input_tokens=est_in, output_tokens=est_out
        )
        await self._budget.admit(ctx.agent, est_tokens=est_in + est_out, est_usd=est_cost)

        started = datetime.now(UTC)
        t0 = time.monotonic()
        capture_root = Path(tempfile.mkdtemp(prefix="cosci-capture-"))
        try:
            outcome, invocation, capture = await self._run_with_retry(
                spec, ctx, capture_root
            )
        except BaseException:
            # Release the reservation; otherwise a failed call permanently
            # shrinks the agent's share of the budget.
            await self._budget.settle(
                ctx.agent,
                est_tokens=est_in + est_out, est_usd=est_cost,
                actual_input_tokens=0, actual_output_tokens=0, actual_usd=0.0,
            )
            raise
        finally:
            shutil.rmtree(capture_root, ignore_errors=True)

        finished = datetime.now(UTC)
        cost_usd = (
            outcome.reported_cost_usd
            if outcome.reported_cost_usd is not None
            else estimate_cost_usd(
                model=spec.route.model,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                cache_read=outcome.cache_read,
                cache_write=outcome.cache_write,
            )
        )

        await self._budget.settle(
            ctx.agent,
            est_tokens=est_in + est_out, est_usd=est_cost,
            actual_input_tokens=outcome.input_tokens,
            actual_output_tokens=outcome.output_tokens,
            actual_usd=cost_usd,
        )

        message = build_message(spec, outcome, capture)

        # One-off callers (the eval judge) have no session and no DB. Skip
        # persistence rather than forcing them to fabricate a session row.
        if self._db is None:
            return LLMResponse(
                raw=message, transcript_id="",
                cost_usd=cost_usd,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                cache_read=outcome.cache_read,
                cache_write=outcome.cache_write,
                num_turns=outcome.num_turns,
                capture=capture,
            )

        trn_id = transcript_id()
        artifact_path = await write_json(
            self._cfg, ctx.session_id, f"transcripts/{ctx.agent}", trn_id,
            {
                "backend": self.backend_name,
                "command": invocation.argv,
                "stdin": invocation.stdin,
                "response": message.model_dump(),
                "cli_raw": outcome.raw,
                "captured_records": [
                    {"tool": n, "input": p} for n, p in capture.records
                ],
                "tool_calls": capture.tool_calls,
                "seen_urls": sorted(capture.seen_urls),
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "duration_ms": int((time.monotonic() - t0) * 1000),
            },
        )

        await transcripts_repo.insert(self._db, Transcript(
            id=trn_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            agent=ctx.agent,
            action=ctx.action,
            model=spec.route.model,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cache_read=outcome.cache_read,
            cache_write=outcome.cache_write,
            cost_usd=cost_usd,
            started_at=started,
            finished_at=finished,
            artifact_path=artifact_path,
        ))
        await sessions_repo.add_usage(
            self._db, ctx.session_id,
            outcome.input_tokens + outcome.output_tokens, cost_usd,
        )

        return LLMResponse(
            raw=message,
            transcript_id=trn_id,
            cost_usd=cost_usd,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cache_read=outcome.cache_read,
            cache_write=outcome.cache_write,
            num_turns=outcome.num_turns,
            capture=capture,
        )

    # ----------------------------- execution ----------------------------- #

    async def _run_with_retry(
        self, spec: AgentCallSpec, ctx: CallContext, capture_root: Path
    ) -> tuple[CliOutcome, CliInvocation, CaptureBundle]:
        """Run the CLI until an attempt succeeds, returning only its capture.

        Each attempt gets its own capture directory. Sharing one would let a
        retry inherit the abandoned attempt's work: the MCP server numbers
        records per process from 0001, so a retry overwrites the low numbers and
        anything the failed attempt wrote beyond that survives into the result.
        """
        max_attempts = self._retry.max_attempts
        last: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            capture_dir = capture_root / f"attempt-{attempt}"
            capture_dir.mkdir(parents=True, exist_ok=True)
            invocation = self.build_invocation(spec, ctx, capture_dir)
            try:
                stdout, stderr, rc = await self._spawn(invocation)
                outcome = self.parse_outcome(
                    stdout=stdout, stderr=stderr, returncode=rc, capture_dir=capture_dir,
                )
                return outcome, invocation, read_capture(capture_dir)
            except CliRetryableError as e:
                last = e
                if attempt == max_attempts:
                    break
                delay = backoff_seconds(attempt, str(e), self._retry)
                log.warning(
                    "cli_retry",
                    backend=self.backend_name, agent=ctx.agent,
                    attempt=attempt, delay_s=round(delay, 1), error=str(e)[:200],
                )
                await asyncio.sleep(delay)
            except CliBackendError:
                raise                      # non-transient: surface immediately

        raise CliBackendError(
            f"{self.backend_name} failed after {max_attempts} attempts: {last}"
        )

    async def _spawn(self, inv: CliInvocation) -> tuple[str, str, int]:
        """Run the CLI, capturing output through files rather than pipes.

        Pipes would be the obvious choice, but they deadlock here. The CLI
        spawns our MCP server as its own child, which inherits the stdout
        write end. `communicate()` returns on EOF, and EOF requires *every*
        holder of that fd to close it — so an MCP server that outlives its
        parent by even a moment wedges the call until the timeout. Observed in
        practice: a ranking task sat idle with no CLI process running.

        Temporary files have no EOF semantics to wait on. We wait for the
        direct child and then read whatever it wrote.
        """
        env = sanitized_env(inv.env)
        timeout = float(self._backend_cfg.timeout_seconds)

        async with self._semaphore:
            with (
                tempfile.TemporaryFile() as out_f,
                tempfile.TemporaryFile() as err_f,
            ):
                proc = await asyncio.create_subprocess_exec(
                    *inv.argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=out_f,
                    stderr=err_f,
                    env=env,
                    cwd=inv.cwd,
                )
                try:
                    if proc.stdin is not None:
                        proc.stdin.write(inv.stdin.encode("utf-8"))
                        await proc.stdin.drain()
                        proc.stdin.close()
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                except TimeoutError as e:
                    await self._terminate(proc)
                    raise CliRetryableError(
                        f"{self.backend_name} timed out after {timeout:.0f}s"
                    ) from e
                except (BrokenPipeError, ConnectionResetError):
                    # CLI exited before consuming the prompt; its own output
                    # (below) explains why far better than this exception.
                    await asyncio.wait_for(proc.wait(), timeout=timeout)

                out_f.seek(0)
                err_f.seek(0)
                out = out_f.read()
                err = err_f.read()

        return (
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
            proc.returncode or 0,
        )

    @staticmethod
    async def _terminate(proc: asyncio.subprocess.Process) -> None:
        """SIGTERM, then SIGKILL if it doesn't go quietly."""
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()

    # ----------------------------- helpers ----------------------------- #

    def _resolve_binary(self) -> str:
        binary = self._backend_cfg.binary
        resolved = shutil.which(binary)
        if resolved is None:
            raise CliBackendError(
                f"{binary!r} is not on PATH. The {self.backend_name} backend drives "
                f"that CLI directly; install it and sign in with your subscription."
            )
        return resolved

    def mcp_config(
        self,
        *,
        capture_dir: Path,
        record_tools: list[str],
        agent: str,
        ctx: CallContext,
    ) -> dict[str, Any]:
        """MCP server definition handed to the CLI."""
        import sys

        return {
            "mcpServers": {
                self.server_label: {
                    "command": sys.executable,
                    "args": ["-m", "co_scientist.mcp.server"],
                    "cwd": str(_package_root()),
                    "env": {
                        "COSCI_CAPTURE_DIR": str(capture_dir),
                        "COSCI_RECORD_TOOLS": ",".join(record_tools),
                        "COSCI_AGENT": agent,
                        "COSCI_SESSION_ID": ctx.session_id,
                        "COSCI_TASK_ID": ctx.task_id or "",
                        "COSCI_TOOL_TIMEOUT": str(
                            self._cfg.tool_loop.tool_timeout_seconds
                        ),
                    },
                }
            }
        }


# --------------------------------------------------------------------------- #
# module-level helpers (kept out of the class so tests can use them directly)


def sanitized_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Parent environment minus anything that would trigger API-key billing."""
    env = dict(os.environ)
    for key in BILLING_ENV_VARS:
        env.pop(key, None)
    if extra:
        env.update(extra)
    return env


def read_capture(capture_dir: Path) -> CaptureBundle:
    """Read the MCP server's records and provenance log back."""
    bundle = CaptureBundle()
    for path in sorted(capture_dir.glob("record-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tool = payload.get("tool")
        data = payload.get("input")
        if isinstance(tool, str) and isinstance(data, dict):
            bundle.records.append((tool, data))

    tools_log = capture_dir / "tools.jsonl"
    if tools_log.exists():
        for line in tools_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            bundle.tool_calls.append({
                "name": entry.get("name", ""),
                "args": entry.get("args", {}),
                "is_error": bool(entry.get("is_error")),
                "duration_ms": int(entry.get("duration_ms", 0) or 0),
            })
            for url in entry.get("urls") or []:
                if isinstance(url, str):
                    bundle.seen_urls.add(url)
    return bundle


def build_message(
    spec: AgentCallSpec, outcome: CliOutcome, capture: CaptureBundle
) -> SynthMessage:
    """Assemble an Anthropic-shaped message from CLI text + captured records.

    `agents/base.py` finds structured output by scanning content blocks for a
    `tool_use` with a given name, so each captured record becomes exactly such
    a block. Text comes first so `_final_text` keeps working too.
    """
    content: list[Any] = []
    if outcome.text:
        content.append(TextBlock(text=outcome.text))
    for i, (name, payload) in enumerate(capture.records):
        content.append(ToolUseBlock(name=name, input=payload, id=f"capture_{i + 1}"))

    return SynthMessage(
        content=content,
        stop_reason=outcome.stop_reason,
        usage=Usage(
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cache_read_input_tokens=outcome.cache_read,
            cache_creation_input_tokens=outcome.cache_write,
        ),
        model=spec.route.model,
    )


def _package_root() -> Path:
    """Directory containing the `co_scientist` package (the MCP server's cwd)."""
    return Path(__file__).resolve().parents[3]
