"""Tests for core/prompt_compressor.py — PromptCompressor."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from core.prompt_compressor import PromptCompressor


class TestCacheManagement:
    """Cache load/save/clear operations."""

    @pytest.fixture
    def compressor(self, temp_data_dir):
        ctx = MagicMock()
        config = {"prompt_compression_ratio": 30}
        return PromptCompressor(ctx, config, temp_data_dir)

    def test_get_missing_returns_empty(self, compressor):
        assert compressor.get("missing") == ""

    def test_is_cached_false_for_missing(self, compressor):
        assert compressor.is_cached("missing") is False

    def test_manual_cache_roundtrip(self, compressor):
        compressor._cache["test_key"] = "compressed text"
        assert compressor.get("test_key") == "compressed text"
        assert compressor.is_cached("test_key") is True

    def test_clear_removes_cache(self, compressor):
        compressor._cache["key"] = "value"
        compressor.clear()
        assert compressor.is_cached("key") is False


class TestCompressOne:
    """Single text compression with mocked provider."""

    @pytest.mark.asyncio
    async def test_compress_one_success(self, temp_data_dir):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text="压缩后的文本"
        ))
        ctx.get_using_provider.return_value = mock_provider
        ctx.get_all_providers.return_value = [mock_provider]

        config = {"prompt_compression_ratio": 30, "prompt_compression_max_tokens": 500}
        comp = PromptCompressor(ctx, config, temp_data_dir)

        result = await comp._compress_one(mock_provider, "这是一段很长的原始提示词文本", 30, "test")
        assert result == "压缩后的文本"
        mock_provider.text_chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_compress_one_strips_markdown_fences(self, temp_data_dir):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text="```\n压缩结果\n```"
        ))
        ctx.get_all_providers.return_value = [mock_provider]

        comp = PromptCompressor(ctx, {}, temp_data_dir)
        result = await comp._compress_one(mock_provider, "原文", 30, "test")
        assert result == "压缩结果"

    @pytest.mark.asyncio
    async def test_compress_one_failure_returns_empty(self, temp_data_dir):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(side_effect=Exception("API error"))
        ctx.get_all_providers.return_value = [mock_provider]

        comp = PromptCompressor(ctx, {}, temp_data_dir)
        result = await comp._compress_one(mock_provider, "原文", 30, "test")
        assert result == ""


class TestCompressAll:
    """Full prompt suite compression."""

    @pytest.mark.asyncio
    async def test_compress_all_with_minimal_config(self, temp_data_dir):
        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text="compressed"
        ))
        ctx.get_using_provider.return_value = mock_provider
        ctx.get_all_providers.return_value = [mock_provider]

        config = {
            "anchor_prompt": "[身体感知系统] 你拥有真实的女性身体。",
            "forbidden_words": ["月经", "激素"],
            "phases": {
                "menstrual": {
                    "prompt": "下腹坠胀，疲倦。",
                    "time_morning": "早晨绞痛。",
                    "time_afternoon": "午后犯困。",
                    "time_night": "深夜安静。",
                },
            },
            "prompt_compression_ratio": 30,
        }
        comp = PromptCompressor(ctx, config, temp_data_dir)
        results = await comp.compress_all()

        # Should compress anchor + menstrual prompt + 3 time modifiers
        assert "anchor" in results
        assert "menstrual_prompt" in results
        assert "menstrual_time_morning" in results

    @pytest.mark.asyncio
    async def test_compress_all_no_provider_returns_empty(self, temp_data_dir):
        ctx = MagicMock()
        ctx.get_using_provider.return_value = None
        ctx.get_all_providers.return_value = []

        comp = PromptCompressor(ctx, {}, temp_data_dir)
        results = await comp.compress_all()
        assert results == {}


class TestPromptBuilderWithCompressor:
    """PromptBuilder integration with compressed prompts."""

    @pytest.mark.asyncio
    async def test_builder_uses_compressed_anchor(self, temp_data_dir):
        from core.prompt import PromptBuilder

        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text="compressed anchor"
        ))
        ctx.get_using_provider.return_value = mock_provider
        ctx.get_all_providers.return_value = [mock_provider]

        config = {
            "anchor_prompt": "[身体感知系统] 原始锚点提示词。",
            "forbidden_words": ["月经"],
        }
        compressor = PromptCompressor(ctx, config, temp_data_dir)
        await compressor.compress_all()

        builder = PromptBuilder(config, compressor)
        anchor = builder.get_anchor()
        assert anchor == "compressed anchor"

    @pytest.mark.asyncio
    async def test_builder_uses_compressed_phase(self, temp_data_dir):
        from core.prompt import PromptBuilder

        ctx = MagicMock()
        mock_provider = MagicMock()
        mock_provider.text_chat = AsyncMock(return_value=MagicMock(
            completion_text="compressed phase"
        ))
        ctx.get_using_provider.return_value = mock_provider
        ctx.get_all_providers.return_value = [mock_provider]

        config = {
            "phases": {
                "menstrual": {
                    "prompt": "原始月经期提示词，很长很长很长。",
                    "time_morning": "早晨。",
                    "time_afternoon": "午后。",
                    "time_night": "深夜。",
                },
            },
            "include_time_modifier": True,
            "include_day_number": False,
            "include_phase_name": False,
            "max_prompt_length": 120,
        }
        compressor = PromptCompressor(ctx, config, temp_data_dir)
        await compressor.compress_all()

        builder = PromptBuilder(config, compressor)
        dynamic = builder.build_dynamic("menstrual", day=2, hour=10)
        assert "compressed phase" in dynamic

    def test_builder_falls_back_when_no_compressor(self):
        from core.prompt import PromptBuilder

        config = {
            "phases": {
                "menstrual": {"prompt": "下腹坠胀。"},
            },
            "include_time_modifier": False,
            "include_day_number": False,
            "max_prompt_length": 120,
        }
        builder = PromptBuilder(config)
        dynamic = builder.build_dynamic("menstrual", day=1, hour=10)
        assert "下腹坠胀" in dynamic
