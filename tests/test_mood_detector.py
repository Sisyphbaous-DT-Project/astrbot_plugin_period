"""Tests for core/mood_detector.py — MoodDetector."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from core.mood_detector import MoodDetector, DEFAULT_SYSTEM_PROMPT
from core.mood import MoodState
from core.engine import PhaseInfo


class TestParseJson:
    """JSON extraction from LLM responses."""

    def test_direct_json(self):
        text = '{"tool": {"name": "cold_violence"}}'
        result = MoodDetector._parse_json(text)
        assert result["tool"]["name"] == "cold_violence"

    def test_markdown_fenced_json(self):
        text = '```json\n{"tool": {"name": "none"}}\n```'
        result = MoodDetector._parse_json(text)
        assert result["tool"]["name"] == "none"

    def test_mixed_text_with_json(self):
        text = 'Here is my decision:\n```\n{"tool": {"name": "seek_comfort"}}\n```\nDone.'
        result = MoodDetector._parse_json(text)
        assert result["tool"]["name"] == "seek_comfort"

    def test_unbalanced_returns_none(self):
        text = 'I think {"name": "cold_violence"'  # missing closing brace
        result = MoodDetector._parse_json(text)
        assert result == {"tool": {"name": "none"}}

    def test_empty_string(self):
        result = MoodDetector._parse_json("")
        assert result == {"tool": {"name": "none"}}


class TestSystemPrompt:
    """System prompt construction."""

    def test_uses_default_when_config_empty(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        assert detector._system_prompt() == DEFAULT_SYSTEM_PROMPT

    def test_uses_custom_when_provided(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {"mood_detector_prompt": "CUSTOM PROMPT"})
        assert detector._system_prompt() == "CUSTOM PROMPT"


class TestUserPrompt:
    """User prompt construction."""

    def test_contains_phase_info(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=2, days_to_next=5, total_day=2)
        mood = MoodState(mood_score=-3, dominant_emotion="irritable")
        prompt = detector._build_user_prompt(phase, mood, [], "test msg")
        assert "月经期" in prompt
        assert "test msg" in prompt
        assert "心情值：-3/10" in prompt

    def test_contains_system_prompt(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=2, days_to_next=5, total_day=2)
        mood = MoodState()
        prompt = detector._build_user_prompt(phase, mood, [], "hi", "你是一个高冷的御姐")
        assert "系统提示词" in prompt
        assert "高冷的御姐" in prompt

    def test_system_prompt_truncated(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {"mood_detector_system_prompt_max_length": 20})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        mood = MoodState()
        long_sys = "这是一个非常长的系统提示词" * 10
        prompt = detector._build_user_prompt(phase, mood, [], "hi", long_sys)
        assert "…" in prompt

    def test_contains_history(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="follicular", day=1, days_to_next=10, total_day=6)
        mood = MoodState()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        prompt = detector._build_user_prompt(phase, mood, history, "how are you")
        assert "用户：hello" in prompt
        assert "助手：hi" in prompt

    def test_active_tools_summary(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="luteal", day=5, days_to_next=3, total_day=19)
        mood = MoodState(active_tools=[{"name": "cold_violence", "params": {"duration": 30}}])
        prompt = detector._build_user_prompt(phase, mood, [], "msg")
        assert "cold_violence" in prompt


class TestDetect:
    """End-to-end detection with mocked provider."""

    @pytest.mark.asyncio
    async def test_detect_returns_parsed_result(self):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text='{"tool": {"name": "perfunctory_reply"}, "new_mood_score": -2}'
        ))
        ctx.get_using_provider.return_value = mock_provider

        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        mood = MoodState()
        result = await detector.detect("test:umo", phase, mood, [], "hi")

        assert result["tool"]["name"] == "perfunctory_reply"
        assert result["new_mood_score"] == -2
        mock_provider.text_chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_detect_no_provider_fallback(self):
        ctx = MagicMock()
        ctx.get_using_provider.return_value = None
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        result = await detector.detect("test:umo", phase, MoodState(), [], "hi")
        assert result == {"tool": {"name": "none"}}

    @pytest.mark.asyncio
    async def test_detect_llm_error_fallback(self):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(side_effect=Exception("API down"))
        ctx.get_using_provider.return_value = mock_provider

        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        result = await detector.detect("test:umo", phase, MoodState(), [], "hi")
        assert result == {"tool": {"name": "none"}}
