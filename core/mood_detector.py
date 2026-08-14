"""LLM-based mood detection engine - three-call architecture (vNext).

① 小模型筛选：判断是否需要情绪介入；
② 当前人格主模型：携带真实历史、完整人格、即时状态与已提交日记，用自然语言决策；
③ 小模型翻译：把②的自然语言忠实翻译为严格 JSON，再由动作矩阵校验。

隐私约束：② 的自然语言回答不记录完整内容；③ 不接触聊天历史、人格或日记；
reasoning_summary 一律视为脱敏摘要。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.star import Context
from astrbot.api.provider import Provider

from .engine import PhaseInfo
from .mood import CAUSE_CATEGORIES, HARD_ACTIONS, STATUS_VALUES, MoodState, RequestMoodDecision
from .mood_tools import MoodToolExecutor, validate_decision


# --------------------------------------------------------------------------- #
#  Default system prompts for the three calls
# --------------------------------------------------------------------------- #

DEFAULT_SCREEN_PROMPT = """你是一个情绪管理筛选助手。你的唯一任务是：判断当前对话是否需要让 Bot 的人格主模型做一次情绪决策。

【输入信息】
- Bot 当前生理周期阶段
- Bot 的即时内在状态（心境摘要、当前动作）
- 最近对话历史
- 用户最新消息
- Bot 人格摘要（如有，可能截断）

【判断规则】
1. 当前已有冷暴力/已读不回等动作生效时，返回 need_intervention=true（每轮都要询问是否解除）。
2. 用户消息包含明显负面情绪（滚、烦、闭嘴等）、道歉示好、或明显敷衍冷落时，返回 true。
3. 对话完全正常、没有情绪信号时，返回 false。
4. 拿不准时返回 false，避免无意义的额外调用。

【输出格式】
只输出纯 JSON，不要 markdown：
{
  "need_intervention": true,
  "reasoning": "20字以内的简要原因"
}"""

DEFAULT_CONSULT_PROMPT = """【情绪管理插件私聊】

你现在不是在和用户对话，而是在和情绪管理插件私聊。插件会把你的真实感受翻译成行为调整，请完全用你自己的口吻自然表达，不要输出 JSON。

【你的生理周期】
- 阶段：{phase}（{phase_label}），第 {day} 天

【你的即时内在状态】
{mood_snapshot}

【你的日记（长期记忆摘要）】
{diary_text}

【你可以选择的动作】
{actions_description}

冷暴力需要你同时决定沉默方式：
- immediate：从这条消息开始就不回应；
- after_expression：这条消息你按人格自然说出最后一句（比如表达边界），从下一条开始沉默。

【用户最新消息】
{user_message}

【请自然回答以下几点】
1. 你现在的心情是怎样的？最近是什么让你有这种感觉？
2. 相比之前有没有好转？是否已经完全恢复？如果恢复了，是因为什么？
3. 这条消息你想怎么回应？要不要用某个动作（给出参数）？要不要解除正在生效的动作？

记住：你在对话历史里就是"你"，用户最新消息在上面。请用自然语言自由表达。"""

DEFAULT_INTERPRET_PROMPT = """你是一个忠实的翻译器。Bot 的主模型用自然语言描述了自己的心情和想做的事（在下方用户消息中给出），你要把它严格翻译成 JSON，不得加入自己的判断，不得引用用户原话。

【当前已生效的动作】
{active_actions}

【允许的动作定义】
{actions_description}

【输出格式】只输出纯 JSON：
{
  "mood_update": {
    "status": "stable|active|recovering|recovered 之一",
    "summary": "自然语言内在心境，200字以内",
    "cause_category": "neutral|neglect|dismissive|conflict|insult|apology|care|boundary|other 之一",
    "latest_reason": "脱敏的最近原因，不引用用户原话",
    "improved": true或false,
    "fully_recovered": true或false,
    "recovery_reason": "恢复原因，未恢复则为空字符串"
  },
  "actions": [{"name": "动作名", "params": {...}}],
  "lift_actions": ["要解除的动作名"],
  "silence_mode": "none|immediate|after_expression",
  "reasoning_summary": "脱敏短摘要，50字以内"
}

【硬性规则】
1. 一轮可以同时有多个软动作（敷衍回复/求安慰/延迟回复/情绪爆发/转移话题）。
2. 冷暴力和已读不回互斥，一轮最多一个，且不能与任何软动作同时出现。
3. 冷暴力激活时 silence_mode 取 immediate（立刻沉默）或 after_expression（本轮自然表达最后一句后，下轮起沉默），按主模型的意思选择。
4. 已读不回激活时 silence_mode 只能是 immediate；rounds 表示含本轮在内共拦截的条数。
5. 没有新动作时 silence_mode 为 none。
6. 同一个动作不能同时出现在 actions 和 lift_actions。
7. 主模型没有明确表达的意思不要编造；不想动作就输出空 actions。
8. reasoning_summary 和 latest_reason 必须是脱敏摘要，禁止抄录用户原话。"""


class _SafeDict(dict):
    """dict subclass that returns the original placeholder for missing keys,
    preventing KeyError when a custom template contains fewer placeholders
    than what the code passes to format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class MoodDetector:
    """Three-call mood detection engine (vNext)."""

    def __init__(self, context: Context, config: dict) -> None:
        self.context = context
        self.config = config
        self._sig_cache: dict[int, set[str]] = {}

    # ------------------------------------------------------------------ #
    #  Provider helpers
    # ------------------------------------------------------------------ #

    def _get_provider(self, umo: str, prefer_main: bool = False):
        """获取 Provider。

        - prefer_main=True（调用②）强制使用当前正式主模型；
        - 否则优先 mood_detector_provider_id 配置的小模型，回退主模型。
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

    async def _text_chat(self, provider, umo: str, stage: str, **kwargs):
        """带签名探测与超时的 text_chat 包装。

        - 对不含目标参数的旧版 Provider 自动丢弃不支持的 kwargs（>=4.24 兼容）；
        - 超时由 mood_call_timeout_seconds 控制（默认 60 秒）。
        """
        supported = self._sig_cache.get(id(provider))
        if supported is None:
            try:
                params = inspect.signature(provider.text_chat).parameters
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    supported = {"*"}
                else:
                    supported = set(params)
            except (TypeError, ValueError):
                supported = {"*"}
            self._sig_cache[id(provider)] = supported

        if "*" not in supported:
            dropped = [k for k in kwargs if k not in supported]
            if dropped:
                logger.info(
                    "[MoodDetector][umo=%s] Provider 不支持参数 %s，已降级",
                    umo, dropped,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in supported}

        timeout = self.config.get("mood_call_timeout_seconds", 60)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = 60.0
        return await asyncio.wait_for(provider.text_chat(**kwargs), timeout=timeout)

    # ------------------------------------------------------------------ #
    #  配置辅助
    # ------------------------------------------------------------------ #

    def enabled_actions(self) -> set[str]:
        """当前启用的动作集合（enable_<name> 配置）。"""
        return {
            name for name in MoodToolExecutor.TOOLS
            if self.config.get(f"enable_{name}", True)
        }

    def _actions_description(self, names: set[str] | None = None) -> str:
        names = names if names is not None else self.enabled_actions()
        if not names:
            return "  （当前没有可用动作）"
        lines = []
        for name, definition in MoodToolExecutor.TOOLS.items():
            if name in names:
                lines.append(f"- {name}：{definition['description']}")
        return "\n".join(lines)

    @staticmethod
    def _compact_state_text(mood_state: MoodState) -> str:
        """给①/③的精简状态（不注入用，纯文本摘要）。"""
        parts = []
        if mood_state.summary:
            parts.append(f"心境：{mood_state.summary}")
        if mood_state.latest_reason:
            parts.append(f"原因：{mood_state.latest_reason}")
        for a in mood_state.persistent_actions:
            if a.name == "cold_violence":
                parts.append(f"动作：cold_violence(duration={a.params.get('duration')}分钟, 至{a.expires_at})")
            elif a.name == "read_no_reply":
                parts.append(f"动作：read_no_reply(剩余{a.remaining_replies}条)")
        return "；".join(parts) if parts else "（无特殊状态）"

    # ------------------------------------------------------------------ #
    #  JSON parsing helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _completion_text(resp: Any) -> str:
        """安全提取响应文本：resp 为 None、缺属性或文本非字符串时返回 ""。

        畸形 Provider 不得让属性访问/类型错误穿透请求钩子——
        已有硬动作时必须能落入保守沉默分支。只捕获 Exception：
        CancelledError 等取消语义必须照常传播。
        """
        try:
            text = getattr(resp, "completion_text", None)
        except Exception:
            # 自定义 Provider 返回对象的属性 getter 自身可能抛异常
            return ""
        return text if isinstance(text, str) else ""

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        if not isinstance(text, str):
            # 畸形 Provider 可能返回非字符串（dict/list 等）：按不可解析
            # 处理，交由调用方走 failed 保守分支，不得抛 AttributeError
            return {}
        text = text.strip()
        if not text:
            return {}

        def _ensure_dict(obj: Any) -> dict[str, Any]:
            return obj if isinstance(obj, dict) else {}

        try:
            return _ensure_dict(json.loads(text))
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if fence_match:
            try:
                return _ensure_dict(json.loads(fence_match.group(1)))
            except json.JSONDecodeError:
                pass

        try:
            first = text.find("{")
            last = text.rfind("}")
            if first >= 0 and last > first:
                return _ensure_dict(json.loads(text[first : last + 1]))
        except json.JSONDecodeError:
            pass

        # 隐私硬约束：解析失败只记录长度与类型，绝不记录模型原始输出
        # （可能包含用户原话转述或②的回答片段）
        logger.warning(
            "[MoodDetector] JSON解析失败（响应长度=%d）", len(text),
        )
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
        persona_summary: str = "",
    ) -> dict[str, Any]:
        """调用①：小模型筛选是否需要情绪介入。

        只接收周期、精简状态、规范化历史、当前消息与（可选）截断人格摘要；
        不接收完整日记。
        """
        provider = self._get_provider(umo, prefer_main=False)
        if not provider:
            logger.warning("[MoodDetector][umo=%s] 无可用 provider，跳过筛选", umo)
            return {"need_intervention": False, "failed": True, "reasoning": "无可用模型"}

        phase_labels = {
            "menstrual": "月经期",
            "follicular": "卵泡期",
            "ovulatory": "排卵期",
            "luteal": "黄体期",
        }

        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '助手'}：{h['content']}"
            for h in conversation_history
        ) or "  （无近期对话）"

        user = (
            f"【当前生理状态】\n"
            f"- 阶段：{phase_info.phase}（{phase_labels.get(phase_info.phase, phase_info.phase)}）"
            f"，第 {phase_info.day} 天\n\n"
            f"【Bot 即时状态】\n{self._compact_state_text(mood_state)}\n\n"
            f"【最近对话】\n{history_text}\n\n"
            f"【用户最新消息】\n{user_message}\n"
        )

        if persona_summary:
            max_len = self.config.get("mood_detector_system_prompt_max_length", 800)
            truncated = (
                persona_summary[:max_len] + "…"
                if len(persona_summary) > max_len else persona_summary
            )
            user += f"\n\n【Bot 人格摘要】\n{truncated}\n"

        system = self.config.get("mood_detector_screen_prompt", DEFAULT_SCREEN_PROMPT)

        logger.debug("[MoodDetector][umo=%s] 筛选调用开始", umo)
        try:
            resp = await self._text_chat(provider, umo, "screen", prompt=user, system_prompt=system)
        except Exception as e:
            # 只记异常类型：自定义 Provider 可能在异常消息里回显请求体
            # （含历史/人格），绝不能写日志
            logger.warning(
                "[MoodDetector][umo=%s] 筛选调用失败: %s", umo, type(e).__name__,
            )
            # failed 与“筛选为否”严格区分：调用方在已有硬动作时必须走保守沉默
            return {
                "need_intervention": False, "failed": True,
                "reasoning": f"调用失败: {type(e).__name__}",
            }

        result = self._parse_json(self._completion_text(resp))
        if not result:
            # 响应为空或不可解析 = 调用失败（而非"无需介入"），
            # 与 need_intervention=false 严格区分
            return {
                "need_intervention": False, "failed": True,
                "reasoning": "响应不可解析",
            }
        # need_intervention 必须是真布尔：缺失或类型错误（如字符串 "false"
        # 经 bool() 会变成 True）一律按调用失败处理，不得擅自解释
        need = result.get("need_intervention")
        if not isinstance(need, bool):
            return {
                "need_intervention": False, "failed": True,
                "reasoning": "need_intervention 字段缺失或类型非法",
            }
        reasoning = result.get("reasoning", "")
        return {
            "need_intervention": need,
            "reasoning": (reasoning if isinstance(reasoning, str) else "")[:50],
        }

    # ------------------------------------------------------------------ #
    #  Call ②: Consult main model
    # ------------------------------------------------------------------ #

    def _build_consult_user_prompt(
        self,
        phase_info: PhaseInfo,
        mood_state: MoodState,
        user_message: str,
        diary_text: str,
    ) -> str:
        phase_labels = {
            "menstrual": "月经期",
            "follicular": "卵泡期",
            "ovulatory": "排卵期",
            "luteal": "黄体期",
        }
        custom_template = self.config.get("mood_detector_consult_prompt", "")
        template = custom_template if custom_template else DEFAULT_CONSULT_PROMPT

        return template.format_map(_SafeDict(
            phase=phase_info.phase,
            phase_label=phase_labels.get(phase_info.phase, phase_info.phase),
            day=phase_info.day,
            mood_snapshot=mood_state.build_snapshot_text() or "（当前状态平稳）",
            diary_text=diary_text or "（暂无日记）",
            actions_description=self._actions_description(),
            user_message=user_message,
        ))

    async def consult_main_model(
        self,
        umo: str,
        phase_info: PhaseInfo,
        mood_state: MoodState,
        history_contexts: list[dict],
        user_message: str,
        system_prompt: str = "",
        diary_text: str = "",
        model: str | None = None,
        provider: Any = None,
    ) -> str:
        """调用②：当前人格主模型用自然语言决策。

        使用本轮实际 Provider（由调用方按 selected_provider/会话偏好解析）、
        req.model、完整原始人格 system_prompt 与规范化 contexts；
        返回自然语言（调用方不得记录完整内容）。
        """
        if provider is None:
            provider = self._get_provider(umo, prefer_main=True)
        if not provider:
            logger.warning("[MoodDetector][umo=%s] 无可用主模型，跳过决策", umo)
            return ""

        try:
            user = self._build_consult_user_prompt(phase_info, mood_state, user_message, diary_text)
        except Exception as e:
            # 自定义模板语法错误（如未转义的 {）按调用失败处理：
            # 配置错误不得让已有硬动作失去保守沉默路径
            logger.warning(
                "[MoodDetector][umo=%s] 决策提示词构建失败: %s", umo, type(e).__name__,
            )
            return ""

        logger.debug("[MoodDetector][umo=%s] 主模型决策调用开始", umo)
        try:
            kwargs: dict[str, Any] = {
                "prompt": user,
                "system_prompt": system_prompt or "",
            }
            if history_contexts:
                kwargs["contexts"] = history_contexts
            if model:
                kwargs["model"] = model
            resp = await self._text_chat(provider, umo, "consult", **kwargs)
        except Exception as e:
            # 只记异常类型，理由同筛选（异常消息可能回显人格/历史/日记）
            logger.warning(
                "[MoodDetector][umo=%s] 主模型决策调用失败: %s", umo, type(e).__name__,
            )
            return ""

        # 一次提取复用：非字符串/空响应按调用失败处理，
        # 调用方走"保持原状态"保守分支
        return self._completion_text(resp)

    # ------------------------------------------------------------------ #
    #  Call ③: Interpret main model's reply
    # ------------------------------------------------------------------ #

    async def interpret(
        self,
        umo: str,
        main_model_reply: str,
        mood_state: MoodState,
    ) -> RequestMoodDecision:
        """调用③：小模型把②的自然语言翻译为严格 JSON 并校验。

        只接收②的自然语言、当前状态与已启用动作定义；
        不接收聊天历史、人格或日记。校验失败时保持原状态（由调用方决定）。
        """
        if not main_model_reply.strip():
            return RequestMoodDecision(valid=False, reject_reason="empty_main_reply")

        provider = self._get_provider(umo, prefer_main=False)
        if not provider:
            logger.warning("[MoodDetector][umo=%s] 无可用 provider，跳过理解", umo)
            return RequestMoodDecision(valid=False, reject_reason="no_provider")

        enabled = self.enabled_actions()
        custom = self.config.get("mood_detector_interpret_prompt", "")
        template = custom if custom else DEFAULT_INTERPRET_PROMPT
        # 模板内含字面 JSON 花括号，必须用 replace 而非 format
        system = template.replace(
            "{active_actions}", self._compact_state_text(mood_state),
        ).replace(
            "{actions_description}", self._actions_description(enabled),
        ).replace("{main_reply}", main_model_reply)

        user = f"【主模型的自然语言】\n{main_model_reply}\n\n请严格翻译为 JSON。"

        logger.debug("[MoodDetector][umo=%s] 理解调用开始", umo)
        try:
            resp = await self._text_chat(provider, umo, "interpret", prompt=user, system_prompt=system)
        except Exception as e:
            logger.warning(
                "[MoodDetector][umo=%s] 理解调用失败: %s", umo, type(e).__name__,
            )
            return RequestMoodDecision(valid=False, reject_reason=f"call_failed:{type(e).__name__}")

        raw = self._parse_json(self._completion_text(resp))
        decision = validate_decision(raw, enabled)
        if not decision.valid:
            logger.info(
                "[MoodDetector][umo=%s] 决策校验失败: %s", umo, decision.reject_reason,
            )
        elif decision.actions_rejected:
            logger.info(
                "[MoodDetector][umo=%s] 动作组被拒绝: %s", umo, decision.reject_reason,
            )
        return decision
