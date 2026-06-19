"""Tests for dynamic state inject location logic."""

import sys
from pathlib import Path

# Add parent dir so astrbot_plugin_period is importable as a package
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import pytest
from unittest.mock import AsyncMock, MagicMock

from astrbot_plugin_period.main import PeriodPlugin


class TestInjectLocation:
    """Test the four inject locations for dynamic state."""

    @pytest.fixture
    def plugin(self, sample_config, monkeypatch):
        from astrbot.api.star import Context, StarTools
        config = dict(sample_config)
        config["default_anchor_date"] = "2024-01-15"
        config["default_enabled"] = True
        config["inject_mode"] = "every_request"
        config["warmup_rounds"] = 0
        config["global_inject"] = True
        ctx = Context()
        ctx.get_using_provider = MagicMock(return_value=None)
        monkeypatch.setattr(StarTools, "get_data_dir", lambda _name=None: Path("/tmp"))
        return PeriodPlugin(ctx, config)

    @pytest.fixture
    def mock_event(self):
        ev = MagicMock()
        ev.unified_msg_origin = "test_platform:test_guild:test_user"
        ev.message_str = "你好"
        return ev

    @pytest.fixture
    def mock_req(self):
        from astrbot.api.provider import ProviderRequest
        req = ProviderRequest()
        req.prompt = "用户原始消息"
        req.system_prompt = "原始人设"
        return req

    @pytest.fixture
    def valid_cfg(self):
        return {
            "anchor_date": "2024-01-15",
            "cycle_length": 28,
            "period_length": 5,
            "ovulation_day": 14,
            "ovulation_window": 3,
            "enabled": True,
            "advance_days": 0,
        }

    @pytest.mark.asyncio
    async def test_default_extra_user_content_parts(self, plugin, mock_event, mock_req, valid_cfg):
        """Default inject_location appends dynamic state to extra_user_content_parts."""
        plugin._get_session_config = AsyncMock(return_value=valid_cfg)
        await plugin.on_llm_request(mock_event, mock_req)
        assert mock_req.prompt == "用户原始消息"
        assert len(mock_req.extra_user_content_parts) == 1
        assert "[当前生理状态]" in mock_req.extra_user_content_parts[0].text
        # Anchor should be in system_prompt
        assert "[身体感知系统]" in mock_req.system_prompt

    @pytest.mark.asyncio
    async def test_user_message_before(self, plugin, mock_event, mock_req, valid_cfg):
        """user_message_before prepends dynamic state to user message."""
        plugin.config["inject_location"] = "user_message_before"
        plugin._get_session_config = AsyncMock(return_value=valid_cfg)
        await plugin.on_llm_request(mock_event, mock_req)
        assert mock_req.prompt.startswith("[当前生理状态]")
        assert "用户原始消息" in mock_req.prompt

    @pytest.mark.asyncio
    async def test_system_prompt_append(self, plugin, mock_event, mock_req, valid_cfg):
        """system_prompt_append puts dynamic state into system_prompt."""
        plugin.config["inject_location"] = "system_prompt_append"
        plugin._get_session_config = AsyncMock(return_value=valid_cfg)
        await plugin.on_llm_request(mock_event, mock_req)
        assert "[当前生理状态]" in mock_req.system_prompt
        # User message should be untouched
        assert mock_req.prompt == "用户原始消息"

    @pytest.mark.asyncio
    async def test_extra_user_content_parts(self, plugin, mock_event, mock_req, valid_cfg):
        """extra_user_content_parts appends to extra_user_content_parts list."""
        plugin.config["inject_location"] = "extra_user_content_parts"
        plugin._get_session_config = AsyncMock(return_value=valid_cfg)
        await plugin.on_llm_request(mock_event, mock_req)
        assert len(mock_req.extra_user_content_parts) == 1
        assert "[当前生理状态]" in mock_req.extra_user_content_parts[0].text
        # User message should be untouched
        assert mock_req.prompt == "用户原始消息"

    @pytest.mark.asyncio
    async def test_fake_tool_call(self, plugin, mock_event, mock_req, valid_cfg):
        """fake_tool_call inserts assistant+tool messages into contexts."""
        plugin.config["inject_location"] = "fake_tool_call"
        plugin._get_session_config = AsyncMock(return_value=valid_cfg)
        await plugin.on_llm_request(mock_event, mock_req)
        assert len(mock_req.contexts) == 2
        assert mock_req.contexts[0]["role"] == "assistant"
        assert "tool_calls" in mock_req.contexts[0]
        assert mock_req.contexts[1]["role"] == "tool"
        assert "[当前生理状态]" in mock_req.contexts[1]["content"]

    @pytest.mark.asyncio
    async def test_fake_tool_call_gemini_fallback(self, plugin, mock_event, mock_req, valid_cfg):
        """fake_tool_call auto-falls back to user_message_before for Gemini."""
        gemini_provider = MagicMock()
        gemini_provider.provider_config = {"type": "googlegenai_chat_completion"}
        plugin.context.get_using_provider.return_value = gemini_provider
        plugin.config["inject_location"] = "fake_tool_call"
        plugin._get_session_config = AsyncMock(return_value=valid_cfg)
        await plugin.on_llm_request(mock_event, mock_req)
        # Should fall back to user_message_before
        assert mock_req.prompt.startswith("[当前生理状态]")
        assert len(mock_req.contexts) == 0  # No fake tool calls inserted

    @pytest.mark.asyncio
    async def test_anchor_injected_every_request(self, plugin, mock_event, mock_req, valid_cfg):
        """Anchor should be injected on every request, not just the first."""
        plugin._get_session_config = AsyncMock(return_value=valid_cfg)
        # First request
        await plugin.on_llm_request(mock_event, mock_req)
        first_system = mock_req.system_prompt
        assert "[身体感知系统]" in first_system
        # Second request (simulate fresh req object like AstrBot does)
        from astrbot.api.provider import ProviderRequest
        mock_req2 = ProviderRequest()
        mock_req2.prompt = "第二次消息"
        mock_req2.system_prompt = ""
        mock_req2.contexts = []
        mock_req2.extra_user_content_parts = []
        await plugin.on_llm_request(mock_event, mock_req2)
        assert "[身体感知系统]" in mock_req2.system_prompt
