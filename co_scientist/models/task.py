"""Task queue model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskAgent = Literal["generation", "reflection", "ranking", "evolution", "proximity", "metareview"]
TaskAction = Literal[
    # Generation
    "CreateInitialHypotheses",
    "GenerateFromFeedback",
    "DirectGeneration",         # bench-only: single LM call, no tool loop
    # Reflection
    "ReviewHypothesis",
    # Ranking
    "AddToTournament",
    "RunTournamentBatch",
    # Evolution
    "EvolveTopHypotheses",
    # Proximity
    "UpdateProximityGraph",
    # Meta-review
    "GenerateSystemFeedback",
    "GenerateFinalResearchOverview",
]
TaskStatus = Literal["pending", "leased", "in_progress", "done", "failed", "dead", "cancelled"]


class Task(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    agent: TaskAgent
    action: TaskAction
    target_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = 100
    status: TaskStatus = "pending"
    lease_owner: str | None = None
    lease_expires_at: int | None = None
    attempts: int = 0
    last_error: str | None = None
    idempotency_key: str | None = None


# --------------------------------------------------------------------------- #
# Task results — return values from agent.execute()


TaskResultKind = Literal[
    "hypothesis_created",
    "review_completed",
    "added_to_tournament",
    "tournament_match_complete",
    "proximity_updated",
    "evolution_completed",
    "system_feedback_generated",
    "final_overview_generated",
    "noop",
]


class TaskResult(BaseModel):
    kind: TaskResultKind
    hypothesis_ids: list[str] = Field(default_factory=list)
    review_ids: list[str] = Field(default_factory=list)
    match_ids: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
