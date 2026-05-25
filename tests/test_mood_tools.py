"""Tests for core/mood_tools.py — MoodToolExecutor."""

import pytest

from core.mood import MoodState
from core.mood_tools import MoodToolExecutor


class TestToolValidation:
    """Parameter validation and clamping."""

    def test_validate_cold_violence_defaults(self):
        params = MoodToolExecutor.validate_params("cold_violence", {})
        assert params == {"duration": 30}

    def test_validate_cold_violence_clamped(self):
        params = MoodToolExecutor.validate_params("cold_violence", {"duration": 9999})
        assert params["duration"] == 1440

        params = MoodToolExecutor.validate_params("cold_violence", {"duration": 0})
        assert params["duration"] == 1

    def test_validate_unknown_tool_returns_empty(self):
        params = MoodToolExecutor.validate_params("unknown", {"foo": "bar"})
        assert params == {}

    def test_validate_str_options_fallback(self):
        params = MoodToolExecutor.validate_params("seek_comfort", {"type": "invalid"})
        assert params["type"] == "emotional"

    def test_validate_perfunctory_level(self):
        params = MoodToolExecutor.validate_params("perfunctory_reply", {"level": 5})
        assert params["level"] == 3

        params = MoodToolExecutor.validate_params("perfunctory_reply", {"level": 0})
        assert params["level"] == 1


class TestToolExecution:
    """Tool application to mood state."""

    def test_cold_violence_sets_expiry(self):
        state = MoodState()
        MoodToolExecutor.execute("cold_violence", {"duration": 30}, state)
        assert state.is_tool_active("cold_violence")
        tool = state.get_active_tool("cold_violence")
        assert "expires_at" in tool
        assert tool["initiated"] is False

    def test_read_no_reply_sets_rounds(self):
        state = MoodState()
        MoodToolExecutor.execute("read_no_reply", {"rounds": 5}, state)
        assert state.is_tool_active("read_no_reply")
        assert state.get_active_tool("read_no_reply")["rounds_left"] == 5

    def test_tool_replaces_existing(self):
        state = MoodState()
        MoodToolExecutor.execute("cold_violence", {"duration": 10}, state)
        MoodToolExecutor.execute("cold_violence", {"duration": 20}, state)
        assert len(state.active_tools) == 1
        assert state.active_tools[0]["params"]["duration"] == 20

    def test_non_intercept_tool_no_expiry(self):
        state = MoodState()
        MoodToolExecutor.execute("seek_comfort", {"type": "emotional"}, state)
        assert state.is_tool_active("seek_comfort")
        assert "expires_at" not in state.active_tools[0]


class TestPromptInjection:
    """Prompt injection strings."""

    def test_perfunctory_level_1(self):
        text = MoodToolExecutor.get_prompt_injection("perfunctory_reply", {"level": 1})
        assert "冷淡" in text

    def test_perfunctory_level_3(self):
        text = MoodToolExecutor.get_prompt_injection("perfunctory_reply", {"level": 3})
        assert "爱答不理" in text

    def test_seek_comfort_emotional(self):
        text = MoodToolExecutor.get_prompt_injection("seek_comfort", {"type": "emotional"})
        assert "索取情感安慰" in text

    def test_emotional_outburst_angry(self):
        text = MoodToolExecutor.get_prompt_injection("emotional_outburst", {"type": "angry"})
        assert "生气" in text

    def test_topic_shift(self):
        text = MoodToolExecutor.get_prompt_injection("topic_shift", {})
        assert "转移话题" in text

    def test_unknown_tool_returns_empty(self):
        assert MoodToolExecutor.get_prompt_injection("unknown", {}) == ""


class TestInitialMessages:
    """Pre-intercept messages."""

    def test_angry_then_silent_angry(self):
        msg = MoodToolExecutor.get_initial_message("angry_then_silent", "angry")
        assert "滚" in msg

    def test_angry_then_silent_playful(self):
        msg = MoodToolExecutor.get_initial_message("angry_then_silent", "playful")
        assert "哼" in msg

    def test_outburst_then_silent_depressed(self):
        msg = MoodToolExecutor.get_initial_message("outburst_then_silent", "depressed")
        assert "委屈" in msg or "为什么" in msg

    def test_silent_returns_empty(self):
        assert MoodToolExecutor.get_initial_message("silent", "angry") == ""
