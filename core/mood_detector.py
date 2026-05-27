"""LLM-based mood detection engine - three-call architecture."""

from __future__ import annotations

import json
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.star import Context
from astrbot.api.provider import Provider

from .mood import MoodState
from .engine import PhaseInfo


# --------------------------------------------------------------------------- #
#  Default system prompts for the three calls
# --------------------------------------------------------------------------- #

DEFAULT_SCREEN_PROMPT = """你是一个情绪管理筛选助手。你的唯一任务是：判断当前对话情境是否需要让主模型（Bot）介入情绪决策。

【输入信息】
- 当前生理周期阶段
- 当前生效的情绪工具列表（如果有）
- 最近几轮对话历史
- 用户最新消息
- Bot 的系统提示词（人格设定，如有）

【判断规则】
1. 如果当前已有拦截型工具生效（冷暴力、已读不回），必须返回 need_intervention=true
   （因为每轮都要询问主模型是否解除）。
2. 如果用户消息包含明显的负面情绪词（滚、烦、闭嘴、垃圾、废物等），返回 true。
3. 如果用户消息包含道歉、示好、关怀词（对不起、抱歉、抱抱、还好吗），返回 true。
4. 如果用户消息非常无聊、敷衍、命令式口吻，返回 true。
5. 如果对话完全正常，没有异常，返回 false。

【输出格式】
只输出纯 JSON，不要 markdown：
{
  "need_intervention": true,
  "reasoning": "简要说明原因，20字以内"
}"""

DEFAULT_CONSULT_PROMPT = """【情绪管理插件私聊】

你现在不是在和用户对话，而是和情绪管理插件"私聊"。插件会根据你的回复决定接下来怎么调整你的行为。

【你的当前状态】
- 生理周期阶段：{phase}（{phase_label}）
- 阶段第 {day} 天

【当前生效的情绪工具】
{tools_summary}

【最近对话】
{history}

【用户最新消息】
{user_message}

【可用工具介绍】
1. cold_violence - 冷暴力：完全不理用户，期间每条消息插件都会来问你"要不要回复"。参数：duration（分钟）。
2. read_no_reply - 已读不回：假装看到但不回复，持续N轮。每轮都会问你"要不要回这条"。参数：rounds。
3. perfunctory_reply - 敷衍回复：允许回复但语气冷淡简短。参数：level（1-3）。
4. seek_comfort - 求安慰：向用户撒娇索求关怀。参数：type（emotional/physical/attention）。
5. delayed_reply - 延迟回复风格：回复带"姗姗来迟"的感觉。参数：minutes。
6. emotional_outburst - 情绪爆发：突然情绪化回应。参数：type（angry/sad/playful）。
7. topic_shift - 转移话题：忽略用户问题，聊自己感兴趣的。
8. none - 不调用任何工具，正常回复。

【请回答】
1. 你现在心情如何？（用你自己的话描述，不要受限于数值）
2. 是否需要调用工具？如果要，调用什么工具？参数是什么？
3. 如果当前有冷暴力/已读不回生效，你觉得要不要解除？

请用自然语言自由表达，不要输出 JSON。"""

DEFAULT_INTERPRET_PROMPT = """你是一个工具执行助手。主模型（Bot）用自然语言描述了它当前的心情和想做的事情。你需要从这段描述中提取具体的工具调用。

【当前生效的工具】
{active_tools_summary}

【可用工具定义】
1. cold_violence - 冷暴力：完全不理用户。参数：duration（分钟，1-1440）。
2. read_no_reply - 已读不回：假装看到但不回，持续N轮。参数：rounds（1-10）。
3. perfunctory_reply - 敷衍回复：语气冷淡简短。参数：level（1-3）。
4. seek_comfort - 求安慰：撒娇索求关怀。参数：type（emotional/physical/attention）。
5. delayed_reply - 延迟回复风格：回复带"姗姗来迟"感。参数：minutes（1-60）。
6. emotional_outburst - 情绪爆发：突然情绪化回应。参数：type（angry/sad/playful）。
7. topic_shift - 转移话题：忽略用户问题，聊自己的。
8. none - 不调用工具。

【任务】
1. 判断主模型是否想要调用某个工具（或保持静默）。
2. 如果提到"不想理""冷暴力""不理他"等 → cold_violence + duration。
3. 如果提到"假装没看到""已读不回"等 → read_no_reply + rounds。
4. 如果提到"敷衍""冷淡""简短"等 → perfunctory_reply + level。
5. 如果提到"安慰""撒娇""索求关怀"等 → seek_comfort + type。
6. 如果提到"晚点回""姗姗来迟"等 → delayed_reply + minutes。
7. 如果提到"爆发""发火""哭""发脾气"等 → emotional_outburst + type。
8. 如果提到"换个话题""不想聊这个"等 → topic_shift。
9. 如果说"算了""正常回""不调用"等 → tool_name 为 null。
10. 如果说"解除冷暴力""回复他""算了原谅他"等 → lift_tools 加入 "cold_violence"。
11. 如果说"解除已读不回""回他吧"等 → lift_tools 加入 "read_no_reply"。

【输出格式】
只输出纯 JSON：
{
  "tool_name": "cold_violence",
  "tool_params": {"duration": 20},
  "lift_tools": ["cold_violence"],
  "reasoning": "主模型说不想理他，提取冷暴力20分钟"
}"""


class _SafeDict(dict):
    """dict subclass that returns the original placeholder for missing keys,
    preventing KeyError when a custom template contains fewer placeholders
    than what the code passes to format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class MoodDetector:
    """Three-call mood detection engine.

    Call 1 (screen):  Small model decides if intervention is needed.
    Call 2 (consult): Small model "DMs" the main model for a natural-language decision.
    Call 3 (interpret): Small model parses the main model's reply into structured tool calls.
    """

    def __init__(self, context: Context, config: dict) -> None:
        self.context = context
        self.config = config

    # ------------------------------------------------------------------ #
    #  Provider helpers
    # ------------------------------------------------------------------ #

    def _get_provider(self, umo: str, prefer_main: bool = False):
        """Get LLM provider.

        - If mood_detector_provider_id is configured, use it as the "small model".
        - Otherwise fall back to the main model (context.get_using_provider).
        - If prefer_main=True (for consult call), force main model.
        - Validates that the returned provider supports text_chat (Chat Completion).
        """
        if prefer_main:
            return self.context.get_using_provider(umo)

        provider_id = self.config.get("mood_detector_provider_id", "")
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider and isinstance(provider, Provider):
                return provider
            if provider:
                logger.warning(
                    "[MoodDetector] 配置的模型 %s 不是文本生成模型，回退到主模型",
                    provider_id,
                )
            else:
                logger.warning(
                    "[MoodDetector] 配置的小模型 %s 未找到，回退到主模型", provider_id,
                )
        return self.context.get_using_provider(umo)

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    def _screen_system_prompt(self) -> str:
        return self.config.get("mood_detector_screen_prompt", DEFAULT_SCREEN_PROMPT)

    def _interpret_system_prompt(self, active_tools: list[dict]) -> str:
        custom = self.config.get("mood_detector_interpret_prompt", "")
        if custom:
            return custom
        if active_tools:
            tools_summary = "\n".join(
                f"  - {t['name']}: {t.get('params', {})}"
                for t in active_tools
            )
        else:
            tools_summary = "  无"
        return DEFAULT_INTERPRET_PROMPT.replace("{active_tools_summary}", tools_summary)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        lines = []
        for entry in history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}：{content}")
        return "\n".join(lines) if lines else "  （无近期对话）"

    def _build_consult_user_prompt(
        self,
        phase_info: PhaseInfo,
        mood_state: MoodState,
        conversation_history: list[dict],
        user_message: str,
    ) -> str:
        phase_labels = {
            "menstrual": "月经期",
            "follicular": "卵泡期",
            "ovulatory": "排卵期",
            "luteal": "黄体期",
        }

        if mood_state.active_tools:
            tools_summary = "\n".join(
                f"  - {t['name']}: {t.get('params', {})}"
                for t in mood_state.active_tools
            )
        else:
            tools_summary = "  无"

        custom_template = self.config.get("mood_detector_consult_prompt", "")
        template = custom_template if custom_template else DEFAULT_CONSULT_PROMPT

        return template.format_map(_SafeDict(
            phase=phase_info.phase,
            phase_label=phase_labels.get(phase_info.phase, phase_info.phase),
            day=phase_info.day,
            tools_summary=tools_summary,
            history=self._format_history(conversation_history),
            user_message=user_message,
        ))

    # ------------------------------------------------------------------ #
    #  JSON parsing helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        text = text.strip()
        if not text:
            return {}

        def _ensure_dict(obj: Any) -> dict[str, Any]:
            return obj if isinstance(obj, dict) else {}

        # Try direct parse first
        try:
            return _ensure_dict(json.loads(text))
        except json.JSONDecodeError:
            pass

        # Try markdown fences
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if fence_match:
            try:
                return _ensure_dict(json.loads(fence_match.group(1)))
            except json.JSONDecodeError:
                pass

        # Try first { to last }
        try:
            first = text.find("{")
            last = text.rfind("}")
            if first >= 0 and last > first:
                return _ensure_dict(json.loads(text[first : last + 1]))
        except json.JSONDecodeError:
            pass

        logger.warning("[MoodDetector] JSON解析失败，原始响应: %s", text[:200])
        return {}

    # ------------------------------------------------------------------ #
    #  Call ①: Screen
    # ------------------------------------------------------------------ #

    async def screen(
        self,
        umo: str,
        phase_info: PhaseInfo,
        mood_state: MoodState,
        conversation_history: list[dict],
        user_message: str,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Call 1: Small model screens whether intervention is needed.

        Returns: {"need_intervention": bool, "reasoning": str}
        """
        provider = self._get_provider(umo, prefer_main=False)
        if not provider:
            logger.warning("[MoodDetector][umo=%s] 无可用 provider，跳过筛选", umo)
            return {"need_intervention": False, "reasoning": "无可用模型"}

        phase_labels = {
            "menstrual": "月经期",
            "follicular": "卵泡期",
            "ovulatory": "排卵期",
            "luteal": "黄体期",
        }

        tools_summary = "\n".join(
            f"  - {t['name']}: {t.get('params', {})}"
            for t in mood_state.active_tools
        ) if mood_state.active_tools else "  无"

        user = (
            f"【当前生理状态】\n"
            f"- 阶段：{phase_info.phase}（{phase_labels.get(phase_info.phase, phase_info.phase)}）\n"
            f"- 阶段第 {phase_info.day} 天\n\n"
            f"【当前生效工具】\n{tools_summary}\n\n"
            f"【最近对话】\n{self._format_history(conversation_history)}\n\n"
            f"【用户最新消息】\n{user_message}\n"
        )

        system = self._screen_system_prompt()
        if system_prompt:
            max_len = self.config.get("mood_detector_system_prompt_max_length", 800)
            truncated = system_prompt[:max_len] + "…" if len(system_prompt) > max_len else system_prompt
            user += f"\n\n【Bot 系统提示词（人格设定）】\n{truncated}\n"

        logger.debug("[MoodDetector][umo=%s] 筛选调用开始", umo)
        try:
            resp = await provider.text_chat(
                prompt=user,
                system_prompt=system,
                # CR-1 fix: AstrBot does not support call-level max_tokens
            )
            logger.debug("[MoodDetector][umo=%s] 筛选调用成功", umo)
        except Exception as e:
            logger.warning("[MoodDetector][umo=%s] 筛选调用失败: %s", umo, e, exc_info=True)
            return {"need_intervention": False, "reasoning": f"调用失败: {e}"}

        text = resp.completion_text or ""
        result = self._parse_json(text)
        return {
            "need_intervention": bool(result.get("need_intervention", False)),
            "reasoning": str(result.get("reasoning", "")),
        }

    # ------------------------------------------------------------------ #
    #  Call ②: Consult main model
    # ------------------------------------------------------------------ #

    async def consult_main_model(
        self,
        umo: str,
        phase_info: PhaseInfo,
        mood_state: MoodState,
        conversation_history: list[dict],
        user_message: str,
        system_prompt: str = "",
    ) -> str:
        """Call 2: Small model DMs the main model for a natural-language decision.

        Returns: Natural language reply from the main model.
        """
        provider = self._get_provider(umo, prefer_main=True)
        if not provider:
            logger.warning("[MoodDetector][umo=%s] 无可用主模型，跳过决策", umo)
            return ""

        user = self._build_consult_user_prompt(
            phase_info, mood_state, conversation_history, user_message
        )

        # Use the bot's own system prompt (persona) so the main model answers
        # as itself, not as a generic assistant.
        main_system = system_prompt or ""

        logger.debug("[MoodDetector][umo=%s] 主模型决策调用开始", umo)
        try:
            resp = await provider.text_chat(
                prompt=user,
                system_prompt=main_system,
                # CR-1 fix: AstrBot does not support call-level max_tokens
            )
            logger.debug("[MoodDetector][umo=%s] 主模型决策调用成功", umo)
        except Exception as e:
            logger.warning("[MoodDetector][umo=%s] 主模型决策调用失败: %s", umo, e, exc_info=True)
            return ""

        return resp.completion_text or ""

    # ------------------------------------------------------------------ #
    #  Call ③: Interpret main model's reply
    # ------------------------------------------------------------------ #

    async def interpret(
        self,
        umo: str,
        main_model_reply: str,
        active_tools: list[dict],
    ) -> dict[str, Any]:
        """Call 3: Small model interprets the main model's natural-language reply.

        Returns: {
            "tool_name": str | null,
            "tool_params": dict,
            "lift_tools": list[str],
            "reasoning": str,
        }
        """
        provider = self._get_provider(umo, prefer_main=False)
        if not provider:
            logger.warning("[MoodDetector][umo=%s] 无可用 provider，跳过理解", umo)
            return {"tool_name": None, "tool_params": {}, "lift_tools": [], "reasoning": "无可用模型"}

        if not main_model_reply.strip():
            return {"tool_name": None, "tool_params": {}, "lift_tools": [], "reasoning": "主模型未回复"}

        system = self._interpret_system_prompt(active_tools)
        user = (
            f"【主模型的自然语言回复】\n"
            f"{main_model_reply}\n\n"
            f"请根据以上回复提取工具调用。"
        )

        logger.debug("[MoodDetector][umo=%s] 理解调用开始", umo)
        try:
            resp = await provider.text_chat(
                prompt=user,
                system_prompt=system,
                # CR-1 fix: AstrBot does not support call-level max_tokens
            )
            logger.debug("[MoodDetector][umo=%s] 理解调用成功", umo)
        except Exception as e:
            logger.warning("[MoodDetector][umo=%s] 理解调用失败: %s", umo, e, exc_info=True)
            return {"tool_name": None, "tool_params": {}, "lift_tools": [], "reasoning": f"调用失败: {e}"}

        text = resp.completion_text or ""
        result = self._parse_json(text)

        return {
            "tool_name": result.get("tool_name") if result.get("tool_name") not in (None, "none", "null", "") else None,
            "tool_params": result.get("tool_params", {}) if isinstance(result.get("tool_params"), dict) else {},
            "lift_tools": list(result.get("lift_tools", [])) if isinstance(result.get("lift_tools"), list) else [],
            "reasoning": str(result.get("reasoning", "")),
        }
