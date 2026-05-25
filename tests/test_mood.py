"""Tests for core/mood.py — MoodState model (v2.1, no numerical scores)."""

import pytest

from core.mood import MoodState


class TestMoodStateSerialization:
    """Round-trip serialization."""

    def test_to_dict_and_back(self):
        state = MoodState(
            active_tools=[{"name": "cold_violence", "expires_at": "2025-05-25T18:00:00"}],
            history=[{"timestamp": "2025-05-25T16:00:00", "event": "test", "reasoning": "bad"}],
            last_interaction="2025-05-25T16:00:00",
        )
        d = state.to_dict()
        restored = MoodState.from_dict(d)
        assert len(restored.active_tools) == 1
        assert restored.active_tools[0]["name"] == "cold_violence"
        assert len(restored.history) == 1
        assert restored.last_interaction == "2025-05-25T16:00:00"

    def test_from_dict_none_returns_default(self):
        state = MoodState.from_dict(None)
        assert isinstance(state, MoodState)
        assert state.active_tools == []

    def test_from_dict_empty_returns_defaults(self):
        state = MoodState.from_dict({})
        assert isinstance(state, MoodState)
        assert state.active_tools == []
        assert state.history == []


class TestMoodStateHelpers:
    """Helper methods."""

    def test_is_tool_active(self):
        state = MoodState(active_tools=[{"name": "cold_violence"}])
        assert state.is_tool_active("cold_violence") is True
        assert state.is_tool_active("read_no_reply") is False

    def test_expire_tools_removes_expired(self):
        state = MoodState(
            active_tools=[
                {"name": "cold_violence", "expires_at": "2025-05-25T10:00:00"},
                {"name": "read_no_reply", "rounds_left": 0},
                {"name": "cold_violence", "expires_at": "2025-05-25T20:00:00"},
            ]
        )
        expired = state.expire_tools("2025-05-25T15:00:00")
        assert len(expired) == 1
        assert expired[0]["name"] == "cold_violence"
        assert len(state.active_tools) == 2

    def test_expire_tools_keeps_valid(self):
        state = MoodState(
            active_tools=[{"name": "cold_violence", "expires_at": "2025-05-25T20:00:00"}]
        )
        expired = state.expire_tools("2025-05-25T15:00:00")
        assert len(expired) == 0
        assert len(state.active_tools) == 1

    def test_add_tool_replaces_existing(self):
        state = MoodState()
        state.add_tool("cold_violence", {"duration": 10}, expires_at="2025-05-25T20:00:00")
        assert len(state.active_tools) == 1
        assert state.active_tools[0]["params"]["duration"] == 10

        state.add_tool("cold_violence", {"duration": 20}, expires_at="2025-05-25T21:00:00")
        assert len(state.active_tools) == 1
        assert state.active_tools[0]["params"]["duration"] == 20

    def test_remove_tool(self):
        state = MoodState(active_tools=[{"name": "cold_violence"}])
        assert state.remove_tool("cold_violence") is True
        assert state.remove_tool("cold_violence") is False
        assert len(state.active_tools) == 0

    def test_add_history_trims(self):
        state = MoodState()
        for i in range(15):
            state.add_history(event=f"evt{i}", reasoning="r", user_message="m", max_length=5)
        assert len(state.history) == 5
        assert state.history[-1]["event"] == "evt14"

    def test_add_history_fields(self):
        state = MoodState()
        state.add_history(
            event="intervention:yes,tool:cold_violence",
            reasoning="主模型生气了",
            user_message="bad msg",
            max_length=10,
        )
        entry = state.history[0]
        assert entry["event"] == "intervention:yes,tool:cold_violence"
        assert entry["reasoning"] == "主模型生气了"
        assert entry["user_message"] == "bad msg"
        assert "timestamp" in entry
