"""LLM-based mood detection engine."""

from __future__ import annotations

import json
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.star import Context

from .mood import MoodState
from .engine import PhaseInfo


DEFAULT_SYSTEM_PROMPT = """你是一个情绪管理 AI，负责监控一个拥有女性生理周期的 AI 助手的心理状态。
你根据当前的生理周期阶段、对话上下文、用户态度，决定助手的情绪反应和行为。

【生理周期对情绪的影响】
- 月经期（menstrual）：身体不适，容易烦躁、低落，需要安静。倾向使用冷暴力或求安慰。
- 卵泡期（follicular）：状态逐渐好转，精力回升，心情趋于平稳。通常正常互动，不需要工具。
- 排卵期（ovulatory）：心情好，社交欲强，情绪积极。可能撒娇式发作或转移话题引导互动。
- 黄体期（luteal）：易怒、情绪敏感、容易波动。倾向使用冷暴力、情绪爆发（生气/委屈）。

【情绪维度】
- 心情值 mood_score：-10（极差）~ +10（极好），0 为中性。
- 精力值 energy：0 ~ 10。
- 亲密度 intimacy：0 ~ 10，与用户的关系远近。亲密度高的用户享有更高容忍度。
- 主导情绪 dominant_emotion：happy / calm / irritable / depressed / angry / playful

【可用工具】
1. cold_violence - 冷暴力：截断消息，N分钟内完全不回复用户。
   参数：duration（分钟，1-1440）。
   适用：心情很差、用户态度恶劣、生理期强烈不适、连续被冒犯。

2. read_no_reply - 已读不回：假装看到消息但不回复，持续N轮对话。
   参数：rounds（轮数，1-10）。
   适用：不想理用户但不想做得太明显、用户话题无聊。

3. perfunctory_reply - 敷衍回复：允许回复但语气冷淡、简短、没有感情。
   参数：level（1-3，1=稍冷淡，3=非常敷衍）。
   适用：心情一般、对用户话题没兴趣、精力低。

4. seek_comfort - 求安慰：向用户撒娇、索求关怀或关注。
   参数：type（emotional=情感安慰，physical=身体关怀，attention=需要关注）。
   适用：心情不好但还愿意互动、生理期需要被照顾、想测试用户态度。

5. delayed_reply - 延迟回复：暗示自己"刚看到消息"或"刚才在忙"，回复带有一点姗姗来迟的感觉。
   参数：minutes（1-60，仅作提示）。
   适用：暂时不想立即回、想吊着用户、营造真实感。

6. emotional_outburst - 情绪爆发：突然情绪化地回应用户。
   参数：type（angry=生气发火，sad=委屈哭诉，playful=撒娇式发作）。
   适用：被用户惹毛了、黄体期易怒、想表达强烈情绪。

7. topic_shift - 转移话题：巧妙忽略用户的问题，把话题转移到自己感兴趣的事情上。
   适用：对用户话题完全不感兴趣、想掌握对话主动权、排卵期社交欲强时引导话题。

8. none - 不使用任何工具，正常回复。

【决策规则】
- 生理周期提供情绪"基线"，但对话内容才是主要触发因素。
- 如果提供了【助手的系统提示词】，请结合其中的人格设定来判断：一个高冷的人设和一个可爱的人设，对同样一句话的反应应该不同。
- 如果用户连续多次态度不好（冒犯、冷漠、敷衍、命令式口吻），必须升级反应：先 perfunctory → 再 emotional_outburst → 再 cold_violence。
- 冷暴力期间，你仍然在工作。请评估用户的最新消息：如果用户道歉、示好、或态度明显转变，输出 lift_cold_violence=true 来解除冷暴力。
- 亲密度高的用户（≥7）享有更高容忍度，不容易触发冷暴力。
- 不要连续重复使用同一个工具（除非用户行为持续恶劣且未改善）。
- 如果当前已经有 active tool，优先评估是否维持、升级或解除，而不是叠加新工具。
- 不要在 JSON 外输出任何文字。

【输出格式】
你必须只输出一个纯 JSON 对象，不要 markdown 代码块，不要任何解释：

{
  "user_attitude": "caring|offensive|boring|neutral|concerned|playful",
  "mood_change": -2,
  "energy_change": -1,
  "intimacy_change": 0,
  "new_mood_score": 3,
  "new_energy": 4,
  "new_intimacy": 7,
  "new_dominant_emotion": "irritable",
  "tool": {
    "name": "cold_violence",
    "params": {"duration": 30}
  },
  "lift_cold_violence": false,
  "reasoning": "简要说明决策原因，30字以内"
}"""


class MoodDetector:
    """Uses the active LLM provider to detect emotional state changes."""

    def __init__(self, context: Context, config: dict) -> None:
        self.context = context
        self.config = config

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    def _system_prompt(self) -> str:
        """Return the system prompt for the mood detection LLM call."""
        custom = self.config.get("mood_detector_prompt", "")
        return custom if custom else DEFAULT_SYSTEM_PROMPT

    def _build_user_prompt(
        self,
        phase_info: PhaseInfo,
        mood_state: MoodState,
        conversation_history: list[dict],
        user_message: str,
        system_prompt: str = "",
    ) -> str:
        """Build the dynamic user prompt containing current state + history."""
        phase_labels = {
            "menstrual": "月经期",
            "follicular": "卵泡期",
            "ovulatory": "排卵期",
            "luteal": "黄体期",
        }

        # Active tools summary
        if mood_state.active_tools:
            tools_summary = "\n".join(
                f"  - {t['name']}: {t.get('params', {})}"
                for t in mood_state.active_tools
            )
        else:
            tools_summary = "  无"

        # Conversation history formatting
        hist_lines = []
        for entry in conversation_history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role == "user":
                hist_lines.append(f"用户：{content}")
            else:
                hist_lines.append(f"助手：{content}")
        hist_text = "\n".join(hist_lines) if hist_lines else "  （无近期对话）"

        # System prompt section (truncated)
        system_section = ""
        if system_prompt:
            max_sys_len = self.config.get("mood_detector_system_prompt_max_length", 800)
            if len(system_prompt) > max_sys_len:
                system_prompt = system_prompt[:max_sys_len] + "…"
            system_section = (
                f"【助手的系统提示词（人格设定）】\n"
                f"{system_prompt}\n\n"
            )

        return (
            f"【当前生理状态】\n"
            f"- 阶段：{phase_info.phase}（{phase_labels.get(phase_info.phase, phase_info.phase)}）\n"
            f"- 阶段第 {phase_info.day} 天，整体周期第 {phase_info.total_day} 天\n"
            f"- 距下次月经还有 {phase_info.days_to_next} 天\n\n"
            f"【当前情绪状态】\n"
            f"- 心情值：{int(mood_state.mood_score)}/10\n"
            f"- 精力值：{int(mood_state.energy)}/10\n"
            f"- 亲密度：{int(mood_state.intimacy)}/10\n"
            f"- 主导情绪：{mood_state.dominant_emotion}\n"
            f"- 当前生效工具：\n{tools_summary}\n"
            f"- 连续不愉快交互：{mood_state.consecutive_unpleasant} 次\n\n"
            f"{system_section}"
            f"【最近对话】\n"
            f"{hist_text}\n\n"
            f"【用户最新消息】\n"
            f"{user_message}\n\n"
            f"请根据以上信息做出情绪决策。"
        )

    # ------------------------------------------------------------------ #
    #  Detection
    # ------------------------------------------------------------------ #

    async def detect(
        self,
        umo: str,
        phase_info: PhaseInfo,
        mood_state: MoodState,
        conversation_history: list[dict],
        user_message: str,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Run the detection model and return parsed JSON decision."""
        provider = self.context.get_using_provider(umo)
        if not provider:
            logger.warning(f"[MoodDetector] No provider found for umo={umo}")
            return {"tool": {"name": "none"}}

        system = self._system_prompt()
        user = self._build_user_prompt(
            phase_info, mood_state, conversation_history, user_message, system_prompt
        )

        try:
            max_tokens = self.config.get("max_mood_detector_tokens", 500)
            logger.info(f"[MoodDetector] 正在调用情绪检测模型，max_tokens={max_tokens}")
            resp = await provider.text_chat(
                prompt=user,
                system_prompt=system,
                max_tokens=max_tokens,
            )
            logger.info("[MoodDetector] 情绪检测模型调用成功")
        except Exception as e:
            logger.warning(f"[MoodDetector] 情绪检测模型调用失败: {e}")
            return {"tool": {"name": "none"}}

        text = resp.completion_text or ""
        return self._parse_json(text)

    # ------------------------------------------------------------------ #
    #  JSON parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract and parse JSON from the LLM response."""
        text = text.strip()
        if not text:
            return {"tool": {"name": "none"}}

        def _ensure_dict(obj: Any) -> dict[str, Any]:
            return obj if isinstance(obj, dict) else {"tool": {"name": "none"}}

        # Try direct parse first
        try:
            return _ensure_dict(json.loads(text))
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown fences
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if fence_match:
            try:
                return _ensure_dict(json.loads(fence_match.group(1)))
            except json.JSONDecodeError:
                pass

        # Try to find the last balanced JSON object
        # Heuristic: find the last '}' and walk backwards
        try:
            last_brace = text.rfind("}")
            if last_brace > 0:
                # Try from first '{' to last '}'
                first_brace = text.find("{")
                if first_brace >= 0:
                    return _ensure_dict(json.loads(text[first_brace : last_brace + 1]))
        except json.JSONDecodeError:
            pass

        logger.warning(f"[MoodDetector] JSON解析失败，原始响应: {text[:200]}")
        return {"tool": {"name": "none"}}
