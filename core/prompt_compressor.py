"""Prompt compression using LLM to reduce token usage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.star import Context


COMPRESSION_SYSTEM_PROMPT = """你是一名提示词优化专家。请将用户提供的系统提示词压缩为精华版。

压缩要求：
1. 保留所有关键约束和行为规则（禁止词、行为边界等）
2. 保留人格设定的核心特征和语气风格
3. 去除冗余修辞、重复表达、过渡语句和装饰性词汇
4. 用极简中文表达，严格控制在目标长度以内
5. 不要改变原意，不要添加原文没有的内容
6. 只输出压缩后的文本，不要任何解释和 markdown 标记"""


class PromptCompressor:
    """Compresses prompt texts using the active LLM provider.

    Compressed results are cached in memory and persisted to disk
    so they survive plugin reloads.
    """

    def __init__(self, context: Context, config: dict, data_dir: Path) -> None:
        self.context = context
        self.config = config
        self.data_dir = data_dir
        self._cache: dict[str, str] = {}
        self._cache_file = data_dir / "compressed_prompts.json"
        self._load_cache()

    # ------------------------------------------------------------------ #
    #  Cache management
    # ------------------------------------------------------------------ #

    def _load_cache(self) -> None:
        """Load compressed prompts from disk."""
        if not self._cache_file.exists():
            return
        try:
            text = self._cache_file.read_text(encoding="utf-8")
            data = json.loads(text) if text.strip() else {}
            if isinstance(data, dict):
                self._cache = data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[PromptCompressor] Failed to load cache: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save compressed prompts to disk."""
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"[PromptCompressor] Failed to save cache: {e}")

    def get(self, key: str, fallback: str = "") -> str:
        """Retrieve a compressed prompt by key."""
        return self._cache.get(key, fallback)

    def is_cached(self, key: str) -> bool:
        """Check whether a given key has been compressed."""
        return key in self._cache and bool(self._cache[key])

    def clear(self) -> None:
        """Clear all cached compressed prompts."""
        self._cache.clear()
        try:
            if self._cache_file.exists():
                self._cache_file.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    #  Compression
    # ------------------------------------------------------------------ #

    async def compress_all(self) -> dict[str, str]:
        """Compress all plugin-managed prompts.

        Returns a dict of {cache_key: compressed_text}.
        """
        logger.info("[PromptCompressor] 开始压缩提示词...")
        provider = self._get_provider()
        if not provider:
            logger.warning("[PromptCompressor] 无可用 LLM 提供商，跳过压缩")
            return {}

        ratio = self.config.get("prompt_compression_ratio", 30)
        logger.info(f"[PromptCompressor] 压缩比例目标: {ratio}%")
        results: dict[str, str] = {}

        # 1. Anchor prompt
        anchor = self._build_raw_anchor()
        if anchor:
            compressed = await self._compress_one(provider, anchor, ratio, "锚点提示词")
            if compressed:
                results["anchor"] = compressed

        # 2. Phase prompts
        phases = self.config.get("phases", {})
        for phase in ("menstrual", "follicular", "ovulatory", "luteal"):
            phase_cfg = phases.get(phase, {})
            for key in ("prompt", "time_morning", "time_afternoon", "time_night"):
                text = phase_cfg.get(key, "")
                if text:
                    cache_key = f"{phase}_{key}"
                    compressed = await self._compress_one(
                        provider, text, ratio, f"{phase}.{key}"
                    )
                    if compressed:
                        results[cache_key] = compressed

        # Update cache
        self._cache.update(results)
        self._save_cache()

        if results:
            total_original = sum(
                len(self._get_original_text(k)) for k in results
            )
            total_compressed = sum(len(v) for v in results.values())
            saved = total_original - total_compressed
            logger.info(
                f"[PromptCompressor] 压缩完成: {len(results)} 条提示词, "
                f"原文 {total_original} 字 → 压缩后 {total_compressed} 字, "
                f"节省 {saved} 字 ({saved / max(total_original, 1) * 100:.1f}%)"
            )
        else:
            logger.info("[PromptCompressor] 无可压缩的提示词")
        return results

    async def _compress_one(
        self,
        provider,
        text: str,
        ratio: int,
        label: str,
    ) -> str:
        """Compress a single text block using the LLM."""
        target_len = max(20, int(len(text) * ratio / 100))
        user_prompt = (
            f"请将以下提示词压缩为精华版，目标长度约 {target_len} 字"
            f"（当前 {len(text)} 字的 {ratio}%）。\n\n"
            f"原文：\n{text}\n\n"
            f"请只输出压缩后的文本："
        )

        try:
            max_tokens = self.config.get("prompt_compression_max_tokens", 1000)
            logger.info(f"[PromptCompressor] 正在压缩 [{label}]，原文 {len(text)} 字")
            resp = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=COMPRESSION_SYSTEM_PROMPT,
                max_tokens=max_tokens,
            )
            result = (resp.completion_text or "").strip()
            # Strip markdown fences if present
            if result.startswith("```"):
                lines = result.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                result = "\n".join(lines).strip()
            logger.info(f"[PromptCompressor] [{label}] 压缩完成: {len(text)} 字 → {len(result)} 字")
            return result
        except Exception as e:
            logger.warning(f"[PromptCompressor] 压缩失败 [{label}]: {e}")
            return ""

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_provider(self):
        """Get the active LLM provider (any umo works for provider selection)."""
        # Provider selection does not depend on umo for most setups,
        # but we try to get a fallback.
        try:
            return self.context.get_using_provider()
        except Exception:
            # Fallback: grab first available provider
            providers = self.context.get_all_providers()
            return providers[0] if providers else None

    def _build_raw_anchor(self) -> str:
        """Build the raw anchor prompt text via PromptBuilder."""
        from .prompt import PromptBuilder
        return PromptBuilder.build_raw_anchor(self.config)

    def _get_original_text(self, cache_key: str) -> str:
        """Get the original uncompressed text for a cache key."""
        if cache_key == "anchor":
            return self._build_raw_anchor()
        if "_" in cache_key:
            phase, key = cache_key.split("_", 1)
            phases = self.config.get("phases", {})
            return phases.get(phase, {}).get(key, "")
        return ""
