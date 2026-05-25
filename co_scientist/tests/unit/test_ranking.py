"""Tests for the ranking verdict parser and mode-selection logic."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from co_scientist.agents.ranking import RankingAgent, _parse_better_idea
from co_scientist.config import Config
from co_scientist.models import Hypothesis

# ----------------------------- verdict parser ----------------------------- #

def test_parse_better_idea_basic() -> None:
    assert _parse_better_idea("blah\nbetter idea: 1") == 1
    assert _parse_better_idea("blah\nbetter idea: 2") == 2


def test_parse_better_idea_trailing_marker_wins() -> None:
    text = "An earlier mention: better idea: 1\n\nFinal verdict.\nbetter idea: 2"
    assert _parse_better_idea(text) == 2


def test_parse_better_idea_handles_case_and_punctuation() -> None:
    assert _parse_better_idea("...\nBetter Idea: 2.") == 2
    assert _parse_better_idea("...\n**better idea**: 1") == 1


def test_parse_better_idea_returns_none_when_missing() -> None:
    assert _parse_better_idea("no verdict here") is None
    assert _parse_better_idea("") is None


# ----------------------------- mode selection ----------------------------- #

def _h(*, elo: float, matches: int, hid: str = "hyp_x") -> Hypothesis:
    return Hypothesis(
        id=hid, session_id="ses", created_at=datetime.now(UTC),
        created_by="generation", strategy="literature",
        title="t", summary="s", full_text="f",
        artifact_path=f"artifacts/ses/hypotheses/{hid}.json",
        elo=elo, matches_played=matches, state="in_tournament",
    )


def _agent() -> RankingAgent:
    deps = MagicMock()
    deps.cfg = Config()
    return RankingAgent(deps)


def test_mode_debate_when_either_player_has_few_matches() -> None:
    a = _h(hid="a", elo=1500, matches=0)
    b = _h(hid="b", elo=1500, matches=10)
    assert _agent()._select_mode(a, b) == "debate"


def test_mode_debate_when_elo_gap_is_small() -> None:
    a = _h(hid="a", elo=1500, matches=5)
    b = _h(hid="b", elo=1520, matches=5)
    assert _agent()._select_mode(a, b) == "debate"


def test_mode_pairwise_when_warm_and_large_gap() -> None:
    a = _h(hid="a", elo=1500, matches=10)
    b = _h(hid="b", elo=1300, matches=10)
    assert _agent()._select_mode(a, b) == "pairwise"


# ----------------------------- nearest-Elo helper ----------------------------- #

def test_nearest_elo_picks_closest() -> None:
    target = _h(hid="t", elo=1300, matches=0)
    pool = [
        _h(hid="a", elo=1000, matches=5),
        _h(hid="b", elo=1310, matches=5),    # closest
        _h(hid="c", elo=1500, matches=5),
    ]
    nearest = _agent()._nearest_elo(target, pool)
    assert nearest is not None and nearest.id == "b"


def test_nearest_elo_empty_pool() -> None:
    target = _h(hid="t", elo=1300, matches=0)
    assert _agent()._nearest_elo(target, []) is None
