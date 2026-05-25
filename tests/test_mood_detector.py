"""Tests for core/mood_detector.py — MoodDetector (v2.1 three-call architecture)."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from core.mood_detector import MoodDetector
from core.mood import MoodState
from core.engine import PhaseInfo


class TestParseJson:
    """JSON extraction from LLM responses."""

    def test_direct_json(self):
        text = '{"tool_name": "cold_violence", "tool_params": {"duration": 20}}'
        result = MoodDetector._parse_json(text)
        assert result["tool_name"] == "cold_violence"

    def test_markdown_fenced_json(self):
        text = '```json\n{"tool_name": "none"}\n```'
        result = MoodDetector._parse_json(text)
        assert result["tool_name"] == "none"

    def test_mixed_text_with_json(self):
        text = 'Here is my decision:\n```\n{"tool_name": "seek_comfort"}\n```\nDone.'
        result = MoodDetector._parse_json(text)
        assert result["tool_name"] == "seek_comfort"

    def test_unbalanced_returns_empty(self):
        text = 'I think {"name": "cold_violence"'  # missing closing brace
        result = MoodDetector._parse_json(text)
        assert result == {}

    def test_empty_string(self):
        result = MoodDetector._parse_json("")
        assert result == {}

    def test_non_dict_returns_empty(self):
        text = '"just a string"'
        result = MoodDetector._parse_json(text)
        assert result == {}


class TestScreenPrompt:
    """Screen call prompt construction."""

    def test_uses_default_when_config_empty(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        assert "筛选" in detector._screen_system_prompt()

    def test_uses_custom_when_provided(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {"mood_detector_screen_prompt": "CUSTOM SCREEN"})
        assert detector._screen_system_prompt() == "CUSTOM SCREEN"


class TestConsultPrompt:
    """Consult call prompt construction."""

    def test_contains_phase_info(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=2, days_to_next=5, total_day=2)
        mood = MoodState()
        prompt = detector._build_consult_user_prompt(phase, mood, [], "test msg")
        assert "月经期" in prompt
        assert "test msg" in prompt

    def test_contains_tools_summary(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="luteal", day=5, days_to_next=3, total_day=19)
        mood = MoodState(active_tools=[{"name": "cold_violence", "params": {"duration": 30}}])
        prompt = detector._build_consult_user_prompt(phase, mood, [], "msg")
        assert "cold_violence" in prompt

    def test_contains_history(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="follicular", day=1, days_to_next=10, total_day=6)
        mood = MoodState()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        prompt = detector._build_consult_user_prompt(phase, mood, history, "how are you")
        assert "用户：hello" in prompt
        assert "助手：hi" in prompt


class TestInterpretPrompt:
    """Interpret call prompt construction."""

    def test_contains_active_tools(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {})
        prompt = detector._interpret_system_prompt([{"name": "cold_violence"}])
        assert "cold_violence" in prompt

    def test_uses_custom_when_provided(self):
        ctx = MagicMock()
        detector = MoodDetector(ctx, {"mood_detector_interpret_prompt": "CUSTOM INTERPRET"})
        assert detector._interpret_system_prompt([]) == "CUSTOM INTERPRET"


class TestScreen:
    """Call 1: Screen with mocked provider."""

    @pytest.mark.asyncio
    async def test_screen_returns_parsed_result(self):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text='{"need_intervention": true, "reasoning": "用户态度恶劣"}'
        ))
        ctx.get_using_provider.return_value = mock_provider

        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        result = await detector.screen("test:umo", phase, MoodState(), [], "hi")

        assert result["need_intervention"] is True
        assert "恶劣" in result["reasoning"]
        mock_provider.text_chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_screen_no_provider_fallback(self):
        ctx = MagicMock()
        ctx.get_using_provider.return_value = None
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        result = await detector.screen("test:umo", phase, MoodState(), [], "hi")
        assert result["need_intervention"] is False

    @pytest.mark.asyncio
    async def test_screen_llm_error_fallback(self):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(side_effect=Exception("API down"))
        ctx.get_using_provider.return_value = mock_provider

        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        result = await detector.screen("test:umo", phase, MoodState(), [], "hi")
        assert result["need_intervention"] is False
        assert "API down" in result["reasoning"]


class TestConsultMainModel:
    """Call 2: Consult main model."""

    @pytest.mark.asyncio
    async def test_consult_returns_text(self):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text='我现在很生气，不想理他，冷暴力20分钟'
        ))
        ctx.get_using_provider.return_value = mock_provider

        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        result = await detector.consult_main_model("test:umo", phase, MoodState(), [], "hi")

        assert "冷暴力" in result
        mock_provider.text_chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_consult_no_provider_returns_empty(self):
        ctx = MagicMock()
        ctx.get_using_provider.return_value = None
        detector = MoodDetector(ctx, {})
        phase = PhaseInfo(phase="menstrual", day=1, days_to_next=0, total_day=1)
        result = await detector.consult_main_model("test:umo", phase, MoodState(), [], "hi")
        assert result == ""


class TestInterpret:
    """Call 3: Interpret main model reply."""

    @pytest.mark.asyncio
    async def test_interpret_returns_parsed_result(self):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text='{"tool_name": "cold_violence", "tool_params": {"duration": 20}, "lift_tools": [], "reasoning": "生气"}'
        ))
        ctx.get_using_provider.return_value = mock_provider

        detector = MoodDetector(ctx, {})
        result = await detector.interpret("test:umo", "不想理他", [])

        assert result["tool_name"] == "cold_violence"
        assert result["tool_params"]["duration"] == 20
        mock_provider.text_chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_interpret_empty_reply(self):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text='{"tool_name": null, "tool_params": {}, "lift_tools": [], "reasoning": "无"}'
        ))
        ctx.get_using_provider.return_value = mock_provider

        detector = MoodDetector(ctx, {})
        result = await detector.interpret("test:umo", "", [])

        assert result["tool_name"] is None

    @pytest.mark.asyncio
    async def test_interpret_no_provider_fallback(self):
        ctx = MagicMock()
        ctx.get_using_provider.return_value = None
        detector = MoodDetector(ctx, {})
        result = await detector.interpret("test:umo", "test", [])
        assert result["tool_name"] is None

    @pytest.mark.asyncio
    async def test_interpret_llm_error_fallback(self):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(side_effect=Exception("API down"))
        ctx.get_using_provider.return_value = mock_provider

        detector = MoodDetector(ctx, {})
        result = await detector.interpret("test:umo", "test", [])
        assert result["tool_name"] is None
