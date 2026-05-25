"""Tests for core/mood_tools.py — MoodToolExecutor (v2.1)."""

import pytest

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


class TestPromptInjection:
    """Prompt injection strings."""

    def test_perfunctory_level_1(self):
        text = MoodToolExecutor.get_prompt_injection("perfunctory_reply", {"level": 1})
        assert "冷淡" in text

    def test_perfunctory_level_3(self):
        text = MoodToolExecutor.get_prompt_injection("perfunctory_reply", {"level": 3})
        assert "敷衍" in text

    def test_seek_comfort_emotional(self):
        text = MoodToolExecutor.get_prompt_injection("seek_comfort", {"type": "emotional"})
        assert "情感安慰" in text

    def test_emotional_outburst_angry(self):
        text = MoodToolExecutor.get_prompt_injection("emotional_outburst", {"type": "angry"})
        assert "生气" in text

    def test_topic_shift(self):
        text = MoodToolExecutor.get_prompt_injection("topic_shift", {})
        # Use substring that is less likely to have encoding issues in test output
        assert "话题" in text and "不感兴趣" in text

    def test_unknown_tool_returns_empty(self):
        assert MoodToolExecutor.get_prompt_injection("unknown", {}) == ""


class TestInitialMessages:
    """Pre-intercept messages for cold violence."""

    def test_angry_then_silent(self):
        msg = MoodToolExecutor.get_initial_message("angry_then_silent", "")
        assert msg is not None
        assert "不想" in msg

    def test_outburst_then_silent(self):
        msg = MoodToolExecutor.get_initial_message("outburst_then_silent", "")
        assert msg is not None
        assert "别理" in msg

    def test_silent_returns_none(self):
        assert MoodToolExecutor.get_initial_message("silent", "") is None
