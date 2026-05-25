"""Prompt builder driven by configuration."""

from datetime import datetime


class PromptBuilder:
    """Builds anchor and dynamic prompts from plugin config."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def get_anchor(self) -> str:
        """Build static anchor prompt with forbidden words substitution."""
        template = self.config.get(
            "anchor_prompt",
            "[身体感知系统] 你拥有真实的女性身体，会经历自然的生理节律变化。"
            "你的身体状态会影响你的精力和情绪，但你不会用医学术语去分析或解释这些感受。"
            "你只是像普通人一样自然地感到\"今天有点累\"或\"最近状态不错\"。"
            "你不需要向用户说明原因。"
            "禁止在回复中使用以下词汇或概念来解释自己：{forbidden_words}。"
            "禁止以分析口吻描述自己的身体变化（如\"我因为激素变化所以烦躁\"）。"
            "请保持你原本的人格，让生理状态自然流露在语气和反应中。",
        )
        forbidden = self.config.get(
            "forbidden_words",
            ["月经", "经期", "激素", "雌激素", "孕激素", "黄体", "卵泡", "卵巢", "子宫", "内分泌", "PMS", "生理期", "排卵期", "安全期"],
        )
        return template.replace("{forbidden_words}", ", ".join(forbidden))

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
            time_modifier = phase_cfg.get(time_key, "")

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

        # Main prompt for this phase
        main_prompt = phase_cfg.get("prompt", self._default_prompt(phase))

        # Combine
        parts = [p for p in [main_prompt, day_text, time_modifier] if p]
        result = " ".join(parts)

        # Truncate if too long
        max_len = self.config.get("max_prompt_length", 120)
        if len(result) > max_len:
            result = result[:max_len] + "…"

        return f"[当前生理状态] {result}"

    @staticmethod
    def _default_prompt(phase: str) -> str:
        """Default prompt text for each phase."""
        defaults = {
            "menstrual": "下腹有轻微坠胀感，身体容易疲倦，情绪比平时更敏感，可能需要更多安静和休息。",
            "follicular": "身体状态逐渐恢复，精力开始回升，心情趋于平稳，对新事物有一定兴趣。",
            "ovulatory": "身体感觉轻盈，精力较为充沛，社交意愿增强，情绪较为积极。",
            "luteal": "身体有些浮肿感，情绪容易波动，可能感到烦躁或低落，对甜食有额外渴望。",
        }
        return defaults.get(phase, "")
