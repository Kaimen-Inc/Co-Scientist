"""ID generators.

- ULID-based IDs for tasks, transcripts, sessions, etc. (sortable, time-prefixed).
- Deterministic-hash IDs for hypotheses (dedup), reviews (kind+iteration), matches
  (sorted pair + round). Identical inputs → identical ID, so INSERT OR IGNORE
  collapses re-runs.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from ulid import ULID

# --------------------------------------------------------------------------- #
# ULID-prefixed IDs


def _ulid() -> str:
    return str(ULID())


def session_id() -> str:
    return f"ses_{_ulid()}"


def task_id() -> str:
    return f"tsk_{_ulid()}"


def transcript_id() -> str:
    return f"trn_{_ulid()}"


def feedback_id() -> str:
    return f"fb_{_ulid()}"


def span_id() -> str:
    return f"spn_{_ulid()}"


def trace_id() -> str:
    return f"trc_{_ulid()}"


def tool_run_id() -> str:
    return f"trn_{_ulid()}"


def bench_id() -> str:
    return f"bnc_{_ulid()}"


def bench_candidate_id() -> str:
    return f"bcd_{_ulid()}"


def bench_match_id() -> str:
    return f"bmt_{_ulid()}"


# --------------------------------------------------------------------------- #
# Deterministic-hash IDs


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, Unicode-NFKC, collapse whitespace. Used for stable hashing."""
    norm = unicodedata.normalize("NFKC", text).lower().strip()
    return _WHITESPACE_RE.sub(" ", norm)


def _short_sha(*parts: str, n: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:n]


def hypothesis_id(session_id_: str, origin: str, statement: str) -> str:
    """Deterministic: re-running Generation with the same statement → same ID."""
    return f"hyp_{_short_sha(session_id_, origin, normalize_text(statement))}"


def review_id(hypothesis_id_: str, kind: str, iteration: int = 0) -> str:
    return f"rev_{_short_sha(hypothesis_id_, kind, str(iteration))}"


def match_id(hyp_a: str, hyp_b: str, round_id: str) -> str:
    """Order-independent: match_id(A, B, R) == match_id(B, A, R)."""
    lo, hi = sorted((hyp_a, hyp_b))
    return f"mat_{_short_sha(lo, hi, round_id)}"


def embedding_id(hypothesis_id_: str, model: str) -> str:
    return f"emb_{_short_sha(hypothesis_id_, model)}"


def text_hash(text: str) -> str:
    """sha256 of normalized text; used as embedding cache key."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def url_hash(url: str) -> str:
    """sha1 of URL for the web-fetch cache filename."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()
