"""Tests for core/mood.py — MoodState model."""

import pytest

from core.mood import MoodState


class TestMoodStateSerialization:
    """Round-trip serialization."""

    def test_to_dict_and_back(self):
        state = MoodState(
            mood_score=-3.0,
            energy=4.0,
            intimacy=7.0,
            dominant_emotion="irritable",
            active_tools=[{"name": "cold_violence", "expires_at": "2025-05-25T18:00:00"}],
            history=[{"timestamp": "2025-05-25T16:00:00", "event": "test", "mood_change": -2}],
            last_interaction="2025-05-25T16:00:00",
            consecutive_unpleasant=2,
        )
        d = state.to_dict()
        restored = MoodState.from_dict(d)
        assert restored.mood_score == -3.0
        assert restored.energy == 4.0
        assert restored.intimacy == 7.0
        assert restored.dominant_emotion == "irritable"
        assert restored.active_tools[0]["name"] == "cold_violence"
        assert len(restored.history) == 1
        assert restored.consecutive_unpleasant == 2

    def test_from_dict_none_returns_none(self):
        assert MoodState.from_dict(None) is None

    def test_from_dict_empty_returns_defaults(self):
        state = MoodState.from_dict({})
        assert state.mood_score == 0.0
        assert state.dominant_emotion == "calm"


class TestMoodStateHelpers:
    """Helper methods."""

    def test_is_tool_active(self):
        state = MoodState(active_tools=[{"name": "cold_violence"}])
        assert state.is_tool_active("cold_violence") is True
        assert state.is_tool_active("read_no_reply") is False

    def test_get_active_tool(self):
        state = MoodState(active_tools=[{"name": "cold_violence", "params": {}}])
        tool = state.get_active_tool("cold_violence")
        assert tool is not None
        assert tool["name"] == "cold_violence"
        assert state.get_active_tool("none") is None

    def test_expire_tools_removes_expired(self):
        state = MoodState(
            active_tools=[
                {"name": "cold_violence", "expires_at": "2025-05-25T10:00:00"},
                {"name": "read_no_reply", "rounds_left": 0},
                {"name": "cold_violence", "expires_at": "2025-05-25T20:00:00"},
            ]
        )
        expired = state.expire_tools("2025-05-25T15:00:00")
        assert len(expired) == 2
        assert len(state.active_tools) == 1
        assert state.active_tools[0]["expires_at"] == "2025-05-25T20:00:00"

    def test_expire_tools_keeps_valid(self):
        state = MoodState(
            active_tools=[{"name": "cold_violence", "expires_at": "2025-05-25T20:00:00"}]
        )
        expired = state.expire_tools("2025-05-25T15:00:00")
        assert len(expired) == 0
        assert len(state.active_tools) == 1

    def test_clamp(self):
        state = MoodState(mood_score=-20, energy=15, intimacy=-5)
        state.clamp()
        assert state.mood_score == -10.0
        assert state.energy == 10.0
        assert state.intimacy == 0.0

    def test_add_history_trims(self):
        state = MoodState()
        for i in range(15):
            state.add_history(event=f"evt{i}", max_length=5)
        assert len(state.history) == 5
        assert state.history[-1]["event"] == "evt14"

    def test_add_history_fields(self):
        state = MoodState()
        state.add_history(
            event="detection:offensive",
            mood_change=-3,
            tool_used="cold_violence",
            user_message="bad msg",
            max_length=10,
        )
        entry = state.history[0]
        assert entry["event"] == "detection:offensive"
        assert entry["mood_change"] == -3
        assert entry["tool_used"] == "cold_violence"
        assert entry["user_message"] == "bad msg"
        assert "timestamp" in entry
