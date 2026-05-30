"""JSON-on-disk helpers.

Artifacts live under `<data_dir>/artifacts/<session_id>/<kind>/<id>.json`.
The DB stores the path *relative* to `data_dir` for portability.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from ..config import Config

# Identifiers used as path components. We refuse anything outside this charset
# so a caller can't smuggle "../../etc/passwd" through `kind` or `id_`.
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def _validate_component(value: str, *, label: str) -> str:
    if not value or value in ("..", ".") or value.startswith("/"):
        raise ValueError(f"{label}={value!r} is not a valid path component")
    if not _SAFE_COMPONENT_RE.fullmatch(value):
        raise ValueError(f"{label}={value!r} contains disallowed characters")
    parts = [p for p in value.split("/") if p]
    if any(p == ".." for p in parts):
        raise ValueError(f"{label}={value!r} contains '..' segment")
    return value


def _resolved_under(base: Path, candidate: Path) -> Path:
    """Resolve `candidate` and assert it stays under `base` (after both are
    resolved). Raises ValueError on escape.
    """
    base_resolved = base.resolve()
    # We resolve without strict=True so non-existent leaves are allowed; the
    # call still normalizes "../" segments and any symlink hops.
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(base_resolved)
    except ValueError as e:
        raise ValueError(f"path {candidate} escapes {base}") from e
    return candidate_resolved


def _rel(cfg: Config, p: Path) -> str:
    return str(p.relative_to(cfg.data_dir))


def session_root(cfg: Config, session_id: str) -> Path:
    _validate_component(session_id, label="session_id")
    return cfg.data_dir / "artifacts" / session_id


def _write(p: Path, payload: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # atomic on POSIX


def _read(p: Path) -> Any:
    return json.loads(p.read_text())


async def write_json(cfg: Config, session_id: str, kind: str, id_: str, payload: Any) -> str:
    """Persist a JSON artifact; return its relative path."""
    _validate_component(kind, label="kind")
    _validate_component(id_, label="id")
    root = session_root(cfg, session_id)
    p = root / kind / f"{id_}.json"
    # Defence-in-depth: even if the regex misses some pathological case,
    # confirm the final path is inside the session root.
    _resolved_under(root.parent, p)
    await asyncio.to_thread(_write, p, payload)
    return _rel(cfg, p)


async def read_json(cfg: Config, rel_path: str) -> Any:
    p = cfg.data_dir / rel_path
    _resolved_under(cfg.data_dir, p)
    return await asyncio.to_thread(_read, p)


async def write_text(cfg: Config, session_id: str, kind: str, id_: str, suffix: str, body: str) -> str:
    _validate_component(kind, label="kind")
    _validate_component(id_, label="id")
    if "/" in suffix or ".." in suffix:
        raise ValueError(f"suffix={suffix!r} is not a valid filename suffix")
    root = session_root(cfg, session_id)
    p = root / kind / f"{id_}{suffix}"
    _resolved_under(root.parent, p)

    def _do() -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(p)

    await asyncio.to_thread(_do)
    return _rel(cfg, p)
