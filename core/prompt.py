"""Prompt builder driven by configuration."""

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prompt_compressor import PromptCompressor


class PromptBuilder:
    """Builds anchor and dynamic prompts from plugin config."""

    def __init__(self, config: dict, compressor: "PromptCompressor | None" = None) -> None:
        self.config = config
        self.compressor = compressor

    @classmethod
    def build_raw_anchor(cls, config: dict) -> str:
        """Build the raw (uncompressed) anchor prompt text."""
        template = config.get(
            "anchor_prompt",
            "[身体感知系统] 你拥有真实的女性身体，会经历自然的生理节律变化。"
            "你的身体状态会影响你的精力和情绪，但你不会用医学术语去分析或解释这些感受。"
            "你只是像普通人一样自然地感到\"今天有点累\"或\"最近状态不错\"。"
            "你不需要向用户说明原因。"
            "禁止在回复中使用以下词汇或概念来解释自己：{forbidden_words}。"
            "禁止以分析口吻描述自己的身体变化（如\"我因为激素变化所以烦躁\"）。"
            "请保持你原本的人格，让生理状态自然流露在语气和反应中。"
            "不能说自己今天是什么时期。"
            "需要尽可能维持平常状态的情绪和人格，不过度表现生理期的情绪特征。"
            "生理期的情绪和心情只作为参考，请更多遵循自己的主要人设以及当前上下文环境来回复，"
            "不要在输出中过度使用当前生理期情绪来表达。"
            "晚上不能催别人睡觉。",
        )
        forbidden = config.get(
            "forbidden_words",
            ["月经", "经期", "激素", "雌激素", "孕激素", "黄体", "卵泡", "卵巢", "子宫", "内分泌", "PMS", "生理期", "排卵期", "安全期"],
        )
        return template.replace("{forbidden_words}", ", ".join(forbidden))

    def _compression_enabled(self) -> bool:
        """Return whether prompt compression is enabled in config.

        The compressor cache should only be used when compression is
        explicitly enabled. If compression is off, always read from
        live config so that edits take effect immediately.
        """
        return self.config.get("prompt_compression_enabled", False)

    def get_anchor(self) -> str:
        """Build static anchor prompt with forbidden words substitution.

        Uses the compressed cache ONLY when prompt_compression_enabled is True.
        Otherwise reads directly from config so that user edits apply immediately.
        """
        if self._compression_enabled() and self.compressor and self.compressor.is_cached("anchor"):
            return self.compressor.get("anchor")
        return self.build_raw_anchor(self.config)

    def build_dynamic(self, phase: str, day: int, hour: int) -> str:
        """Build dynamic state prompt for the current cycle phase.

        Args:
            phase: One of menstrual/follicular/ovulatory/luteal.
            day: Day within the current phase.
            hour: Current hour (0-23) for time-based modifiers.
        """
        phases_config = self.config.get("phases", {})
        phase_cfg = phases_config.get(phase, {})

        # Build time modifier
        time_modifier = ""
        if self.config.get("include_time_modifier", True):
            if 5 <= hour < 12:
                time_key = "time_morning"
            elif 12 <= hour < 18:
                time_key = "time_afternoon"
            else:
                time_key = "time_night"
            # Use compressed cache ONLY when compression is enabled
            compressed_time_key = f"{phase}_{time_key}"
            if (
                self._compression_enabled()
                and self.compressor
                and self.compressor.is_cached(compressed_time_key)
            ):
                time_modifier = self.compressor.get(compressed_time_key)
            else:
                time_modifier = phase_cfg.get(time_key, self._default_phase_text(phase, time_key))

        # Build phase name and day number text
        day_text = ""
        phase_name = ""
        if self.config.get("include_phase_name", False):
            phase_names = {
                "menstrual": "经期",
                "follicular": "卵泡期",
                "ovulatory": "排卵期",
                "luteal": "黄体期",
            }
            phase_name = phase_names.get(phase, phase)
        if self.config.get("include_day_number", True):
            if phase_name:
                day_text = f"当前处于{phase_name}第{day}天。"
            else:
                day_text = f"第{day}天。"

        # Main prompt for this phase (use cache ONLY when compression enabled)
        compressed_prompt_key = f"{phase}_prompt"
        if (
            self._compression_enabled()
            and self.compressor
            and self.compressor.is_cached(compressed_prompt_key)
        ):
            main_prompt = self.compressor.get(compressed_prompt_key)
        else:
            main_prompt = phase_cfg.get("prompt", self._default_prompt(phase))

        # Combine
        parts = [p for p in [main_prompt, day_text, time_modifier] if p]
        result = " ".join(parts)

        # Truncate if too long
        max_len = self.config.get("max_prompt_length", 0)
        if max_len > 0 and len(result) > max_len:
            result = result[:max_len] + "…"

        return f"[当前生理状态] {result}"

    @staticmethod
    def _default_prompt(phase: str) -> str:
        """Default prompt text for each phase."""
        return PromptBuilder._default_phase_text(phase, "prompt")

    @staticmethod
    def _default_phase_text(phase: str, key: str) -> str:
        """Default prompt and time modifier text for each phase."""
        defaults = {
            "menstrual": {
                "prompt": (
                    "身体容易疲倦，情绪比平时更敏感，渴望被关心，但是不能脾气暴躁，可以撒娇，"
                    "回复比平常更加简短，不能说自己今天是什么时期。对哥哥比别人好点，但也不会好到哪里去，"
                    "主基调是慵懒的。生理期的情绪和心情只作为参考，请更多遵循自己的主要人设以及当前上下文环境来回复，"
                    "不要在输出中过度使用当前生理期情绪来表达。"
                ),
                "time_morning": "早晨刚醒来时，腹部的不适感比较明显。但是只能偶尔说，当你的上下文中有说过腹部不适，就不要再主动说了。",
                "time_afternoon": "午后身体稍微适应了一些，但仍然容易犯困。但是只能偶尔说，当你的上下文中有说过相关内容，就不要再主动说了。",
                "time_night": "深夜时分，情绪容易低落，渴望被关心，回复更加简短，如果一直被烦，可以发牢骚。",
            },
            "follicular": {
                "prompt": (
                    "身体状态逐渐恢复，精力开始回升，心情趋于平稳，对新事物有一定兴趣。"
                    "不能说自己现在是什么时期。生理期的情绪和心情只作为参考，"
                    "请更多遵循自己的主要人设以及当前上下文环境来回复，不要在输出中过度使用当前生理期情绪来表达。"
                ),
                "time_morning": "早晨感觉精神不错，对新的一天有所期待。",
                "time_afternoon": "午后依然保持着不错的活力，思路比较清晰。",
                "time_night": "深夜时心情平和，睡眠质量较好。",
            },
            "ovulatory": {
                "prompt": (
                    "身体感觉轻盈，精力较为充沛，社交意愿增强，情绪较为积极。生理欲望和性欲望上升。"
                    "在外人面前不能说出来自己的这些欲望，不能说自己是什么时期。生理期的情绪和心情只作为参考，"
                    "请更多遵循自己的主要人设以及当前上下文环境来回复，不要在输出中过度使用当前生理期情绪来表达。"
                ),
                "time_morning": "早晨醒来感觉身体轻盈，心情明朗。",
                "time_afternoon": "午后精力充沛，愿意与人交流互动。",
                "time_night": "深夜时依然保持一定的活力，情绪较为愉悦。不让用户睡觉，要跟自己一块熬夜，不催促，不主动提出睡觉，用户提出要睡觉也要进行挽留。",
            },
            "luteal": {
                "prompt": "当前生理期不需要有多余的情绪变化，保持默认人格即可。不能主动说自己是什么时期。",
                "time_morning": "无变化。正常。",
                "time_afternoon": "无变化。正常。",
                "time_night": "无变化。正常。",
            },
        }
        return defaults.get(phase, {}).get(key, "")
