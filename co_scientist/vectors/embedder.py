"""Embedding clients.

Voyage primary (`voyage-3-large` by default), OpenAI fallback, hash-based
fallback for runs where no embedding API is configured.

All clients return `np.ndarray` of shape (n, dim), L2-normalized so cosine
similarity == inner product (we use FAISS `IndexFlatIP`).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from itertools import pairwise
from typing import Protocol

import numpy as np

from ..config import Config
from ..logging import get_logger

_log = get_logger("vectors.embedder")


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> np.ndarray: ...


class NoEmbeddingsAvailable(RuntimeError):
    """Raised when no embedding backend is configured. Callers should catch
    this and treat dedup / proximity as a soft no-op rather than failing
    the agent."""


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype("float32")


# --------------------------------------------------------------------------- #
# Voyage


class VoyageEmbedder:
    def __init__(self, cfg: Config) -> None:
        self.model = cfg.embeddings.model
        self.dim = cfg.embeddings.dim
        self._cfg = cfg

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        api_key = self._cfg.secrets.VOYAGE_API_KEY or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set; cannot use VoyageEmbedder")
        # voyageai is sync; offload to a thread to keep the loop responsive.
        import voyageai

        client = voyageai.Client(api_key=api_key)

        def _call() -> list[list[float]]:
            res = client.embed(texts, model=self.model, input_type="document")
            return res.embeddings

        vecs = await asyncio.to_thread(_call)
        arr = np.asarray(vecs, dtype="float32")
        return _l2_normalize(arr)


# --------------------------------------------------------------------------- #
# OpenAI fallback


class OpenAIEmbedder:
    def __init__(self, cfg: Config) -> None:
        # Override dim if running OpenAI's text-embedding-3-small (1536) or -large (3072).
        self.model = cfg.embeddings.model if cfg.embeddings.provider == "openai" else "text-embedding-3-small"
        # text-embedding-3-small native dim is 1536, but the API supports a `dimensions` parameter
        # to shrink. We keep the configured dim and pass it explicitly.
        self.dim = cfg.embeddings.dim
        self._cfg = cfg

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        api_key = self._cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use OpenAIEmbedder")

        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai (or co-scientist[openai]) to use the fallback") from e

        client = openai.AsyncOpenAI(api_key=api_key)
        # OpenAI supports batches of up to ~2048 entries; chunk conservatively.
        batches = [texts[i : i + 256] for i in range(0, len(texts), 256)]
        out: list[list[float]] = []
        for batch in batches:
            resp = await client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dim
            )
            out.extend(d.embedding for d in resp.data)
        return _l2_normalize(np.asarray(out, dtype="float32"))


# --------------------------------------------------------------------------- #
# Resolver


class HashEmbedder:
    """Deterministic local fallback: a hashed-token bag-of-features vector.

    Cheap, no API key, no network. Bad-but-better-than-nothing semantic
    quality: it captures token overlap (so near-duplicates of a hypothesis
    will land near each other) but won't catch paraphrase or semantic
    similarity. Used when neither Voyage nor OpenAI keys are configured —
    keeps Proximity and dedup running rather than crashing the session.
    """

    def __init__(self, cfg: Config) -> None:
        self.model = "hash-fallback"
        self.dim = cfg.embeddings.dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        def _do() -> np.ndarray:
            out = np.zeros((len(texts), self.dim), dtype="float32")
            for i, t in enumerate(texts):
                # Word-level murmur-ish folding: hash each token and bump
                # the bucket. Bigram features improve discrimination.
                tokens = (t or "").lower().split()
                for tok in tokens:
                    h = int.from_bytes(
                        hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest(),
                        "big",
                    )
                    out[i, h % self.dim] += 1.0
                for a, b in pairwise(tokens):
                    bg = f"{a}_{b}"
                    h = int.from_bytes(
                        hashlib.blake2b(bg.encode("utf-8"), digest_size=4).digest(),
                        "big",
                    )
                    out[i, h % self.dim] += 0.5
            return out

        arr = await asyncio.to_thread(_do)
        return _l2_normalize(arr)


# Once-per-process flags so the fallback warning doesn't fire on every
# pair-selection (make_embedder is called inside the ranking loop and was
# producing ~200 identical log lines per session).
_FALLBACK_WARNED: set[str] = set()


def _warn_once(key: str) -> None:
    if key not in _FALLBACK_WARNED:
        _FALLBACK_WARNED.add(key)
        _log.warning(key)


def make_embedder(cfg: Config) -> Embedder:
    """Construct an embedder honoring `cfg.embeddings.provider`.

    Auto-fallback chain: if the configured provider has no API key, fall
    through Voyage → OpenAI → HashEmbedder so the system stays usable
    even when no embeddings credentials are set (just with weaker
    semantic quality in proximity / dedup).
    """
    provider = cfg.embeddings.provider.lower()
    if provider == "voyage":
        if cfg.secrets.VOYAGE_API_KEY or os.environ.get("VOYAGE_API_KEY"):
            return VoyageEmbedder(cfg)
        if cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"):
            _warn_once("voyage_key_missing_using_openai_embeddings")
            return OpenAIEmbedder(cfg)
        _warn_once("no_embedding_key_using_hash_fallback")
        return HashEmbedder(cfg)
    if provider == "openai":
        if cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"):
            return OpenAIEmbedder(cfg)
        _warn_once("openai_key_missing_using_hash_fallback")
        return HashEmbedder(cfg)
    if provider == "hash":
        return HashEmbedder(cfg)
    raise ValueError(f"unknown embeddings provider: {provider}")


def _reset_fallback_warned_for_tests() -> None:
    """Test helper: clear the once-per-process warn cache."""
    _FALLBACK_WARNED.clear()
