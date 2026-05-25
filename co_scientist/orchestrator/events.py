"""In-memory event bus → SSE fanout.

Agents call `bus.publish(session_id, event)`. SSE handlers subscribe with
`bus.subscribe(session_id)` to get an async iterator of events. The bus is
in-process only; on restart, the UI snapshots from the `events` table and
reconnects for live updates.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    name: str                              # e.g. "match_complete", "hypothesis_created"
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


class EventBus:
    """One bus per process. Subscribers are per-session asyncio.Queues."""

    def __init__(self, max_buffer: int = 256) -> None:
        self._subs: dict[str, set[asyncio.Queue[Event]]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._max_buffer = max_buffer

    async def publish(self, session_id: str, name: str, payload: dict[str, Any] | None = None) -> None:
        ev = Event(name=name, session_id=session_id, payload=payload or {})
        async with self._lock:
            queues = list(self._subs.get(session_id, ()))
        for q in queues:
            # drop oldest if full so a slow subscriber can't block the bus
            if q.qsize() >= self._max_buffer:
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            await q.put(ev)

    async def subscribe(self, session_id: str) -> AsyncIterator[Event]:
        """Yield published events for `session_id`.

        The subscription is registered for the lifetime of the async generator.
        Use `async with contextlib.aclosing(bus.subscribe(...)) as gen:` to
        guarantee deterministic unregister; otherwise unregister happens on GC.
        """
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_buffer)
        async with self._lock:
            self._subs[session_id].add(q)
        try:
            while True:
                ev = await q.get()
                yield ev
        finally:
            async with self._lock:
                self._subs[session_id].discard(q)
                if not self._subs[session_id]:
                    del self._subs[session_id]

    def subscriber_count(self, session_id: str) -> int:
        return len(self._subs.get(session_id, ()))


# Module-level singleton for the running process. Web UI and Supervisor share it.
GLOBAL_BUS = EventBus()
