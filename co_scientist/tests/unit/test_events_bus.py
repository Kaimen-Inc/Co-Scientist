"""Tests for the in-memory event bus / SSE fanout."""

from __future__ import annotations

import asyncio

import pytest

from co_scientist.orchestrator.events import Event, EventBus


@pytest.mark.asyncio
async def test_subscribe_receives_publish() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def reader() -> None:
        async for ev in bus.subscribe("ses_a"):
            received.append(ev)
            if len(received) >= 2:
                break

    task = asyncio.create_task(reader())
    # Give the subscriber a moment to register
    await asyncio.sleep(0.05)
    await bus.publish("ses_a", "match_complete", {"x": 1})
    await bus.publish("ses_a", "match_complete", {"x": 2})
    await asyncio.wait_for(task, timeout=2.0)
    assert [ev.payload["x"] for ev in received] == [1, 2]


@pytest.mark.asyncio
async def test_publishes_isolated_by_session() -> None:
    bus = EventBus()
    got_a: list[Event] = []

    async def reader() -> None:
        async for ev in bus.subscribe("ses_a"):
            got_a.append(ev)
            if got_a:
                break

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.05)
    await bus.publish("ses_b", "match_complete", {"x": "other"})
    await bus.publish("ses_a", "match_complete", {"x": "mine"})
    await asyncio.wait_for(task, timeout=2.0)
    assert len(got_a) == 1
    assert got_a[0].payload["x"] == "mine"


@pytest.mark.asyncio
async def test_unsubscribe_via_aclosing() -> None:
    """Deterministic unsubscribe using contextlib.aclosing."""
    import contextlib

    bus = EventBus()

    async def reader() -> None:
        async with contextlib.aclosing(bus.subscribe("ses_x")) as gen:
            async for _ in gen:
                return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.05)
    assert bus.subscriber_count("ses_x") == 1
    await bus.publish("ses_x", "hello", {})
    await asyncio.wait_for(task, timeout=2.0)
    assert bus.subscriber_count("ses_x") == 0
