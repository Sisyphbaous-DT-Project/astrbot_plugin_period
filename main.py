"""AstrBot plugin for physiological cycle simulation."""

import asyncio
import datetime

from quart import jsonify, request

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger
from astrbot.core.agent.message import TextPart

from .core.engine import CycleEngine
from .core.store import CycleStore
from .core.prompt import PromptBuilder
from .core.prompt_compressor import PromptCompressor
from .core.mood import MoodState
from .core.mood_store import MoodStore
from .core.mood_tools import MoodToolExecutor
from .core.mood_detector import MoodDetector


@register("astrbot_plugin_period", "C₂₂H₂₅NO₆", "生理周期模拟插件", "2.0.0",
          "https://github.com/Sisyphbaous-DT-Project/astrbot_plugin_period")
class PeriodPlugin(Star):
    """Plugin that simulates physiological cycles for female-persona bots."""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self.plugin_data_dir = StarTools.get_data_dir()
        self.engine = CycleEngine()
        self.store = CycleStore(self.plugin_data_dir)

        # Prompt compression
        self.prompt_compressor = PromptCompressor(self.context, self.config, self.plugin_data_dir)
        self.prompt_builder = PromptBuilder(self.config, self.prompt_compressor)

        self._anchored_sessions: set[str] = set()  # Sessions with anchor injected
        self._inject_counters: dict[str, int] = {}  # Interval injection counters
        self._warmup_counters: dict[str, int] = {}  # Warmup round counters

        # Mood / emotion system
        self.mood_store = MoodStore(self.plugin_data_dir)
        self.mood_detector = MoodDetector(self.context, self.config)
        self.mood_executor = MoodToolExecutor()
        self._mood_locks: dict[str, asyncio.Lock] = {}

        self._register_web_apis()

        # Auto-compress prompts on init if enabled
        if self.config.get("prompt_compression_enabled", False):
            if self.config.get("prompt_compression_auto_trigger", True):
                logger.info("[PeriodPlugin] 提示词压缩已启用，将在后台自动压缩...")
                try:
                    import asyncio
                    asyncio.create_task(self._auto_compress_prompts())
                except RuntimeError:
                    logger.warning("[PeriodPlugin] 当前无运行中的事件循环，跳过后台自动压缩，可手动执行 period compress")

        logger.info(
            f"[PeriodPlugin] 插件初始化完成，"
            f"数据目录={self.plugin_data_dir}, "
            f"情绪系统={'开启' if self.config.get('mood_system_enabled', True) else '关闭'}, "
            f"自动注入={'开启' if self.config.get('auto_inject', True) else '关闭'}"
        )

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _check_command_permission(self, cmd_name: str) -> tuple[bool, str]:
        """Check if a command is allowed under current commands_enabled setting.

        Returns (allowed, message).
        """
        mode = self.config.get("commands_enabled", "all")
        if mode == "all":
            return True, ""
        if mode == "none":
            return False, "当前会话的周期指令已关闭，如需调整请前往插件配置修改指令权限控制"
        if mode == "readonly":
            if cmd_name == "status":
                return True, ""
            return False, "当前仅允许查看状态，设置类指令已被关闭，如需调整请前往插件配置修改指令权限控制"
        return True, ""

    # ------------------------------------------------------------------ #
    #  Web API (for dashboard)
    # ------------------------------------------------------------------ #

    def _register_web_apis(self) -> None:
        """Register Web API routes for the dashboard page."""
        base = f"/{self.__class__.name}"
        logger.info(f"[PeriodPlugin] 注册 Web API 路由，前缀={base}")
        self.context.register_web_api(
            f"{base}/sessions",
            self._webapi_list_sessions,
            ["GET"],
            "List all session cycle statuses",
        )
        self.context.register_web_api(
            f"{base}/config",
            self._webapi_get_config,
            ["GET"],
            "Get global default config",
        )
        self.context.register_web_api(
            f"{base}/sessions/<umo>/toggle",
            self._webapi_toggle_session,
            ["POST"],
            "Toggle session enabled state",
        )
        self.context.register_web_api(
            f"{base}/sessions/<umo>/advance",
            self._webapi_advance_session,
            ["POST"],
            "Advance session days",
        )
        self.context.register_web_api(
            f"{base}/sessions/<umo>/anchor",
            self._webapi_set_anchor,
            ["POST"],
            "Set session anchor date",
        )
        self.context.register_web_api(
            f"{base}/sessions/<umo>/delete",
            self._webapi_delete_session,
            ["POST"],
            "Delete session data",
        )

    def _infer_source(self, cfg: dict) -> str:
        """Infer whether a session config originates from global defaults."""
        default_anchor = self.config.get("default_anchor_date", "")
        if not default_anchor:
            return "manual"
        cycle_settings = self.config.get("cycle_settings", {})
        expected = {
            "anchor_date": default_anchor,
            "cycle_length": self.config.get("default_cycle_length", 28),
            "period_length": self.config.get("default_period_length", 5),
            "ovulation_day": cycle_settings.get("ovulation_day", 14),
            "ovulation_window": cycle_settings.get("ovulation_window", 3),
            "enabled": True,
        }
        for key in expected:
            if cfg.get(key) != expected[key]:
                return "manual"
        return "global_default"

    def _serialize_session(self, umo: str, cfg: dict) -> dict | None:
        """Build session dict with live phase calculation."""
        if not cfg or "anchor_date" not in cfg:
            return None
        try:
            info = self.engine.get_phase(
                cfg["anchor_date"],
                cfg.get("cycle_length", 28),
                cfg.get("period_length", 5),
                cfg.get("ovulation_day", 14),
                cfg.get("ovulation_window", 3),
                cfg.get("advance_days", 0),
            )
        except Exception as e:
            logger.warning(f"[PeriodPlugin] Failed to serialize session {umo}: {e}")
            return None

        phase_labels = {
            "menstrual": "月经期",
            "follicular": "卵泡期",
            "ovulatory": "排卵期",
            "luteal": "黄体期",
        }
        return {
            "umo": umo,
            "source": self._infer_source(cfg),
            "enabled": cfg.get("enabled", True),
            "anchor_date": cfg["anchor_date"],
            "cycle_length": cfg.get("cycle_length", 28),
            "period_length": cfg.get("period_length", 5),
            "ovulation_day": cfg.get("ovulation_day", 14),
            "ovulation_window": cfg.get("ovulation_window", 3),
            "advance_days": cfg.get("advance_days", 0),
            "phase": info.phase,
            "phase_day": info.day,
            "total_day": info.total_day,
            "days_to_next": info.days_to_next,
            "phase_label": phase_labels.get(info.phase, info.phase),
        }

    async def _webapi_list_sessions(self):
        """GET /astrbot_plugin_period/sessions"""
        all_data = await self.store.get_all()
        sessions = []
        for umo, cfg in all_data.items():
            serialized = self._serialize_session(umo, cfg)
            if serialized:
                sessions.append(serialized)
        return jsonify(
            {"status": "ok", "data": {"sessions": sessions, "count": len(sessions)}}
        )

    async def _webapi_get_config(self):
        """GET /astrbot_plugin_period/config"""
        cycle_settings = self.config.get("cycle_settings", {})
        return jsonify(
            {
                "status": "ok",
                "data": {
                    "default_anchor_date": self.config.get("default_anchor_date", ""),
                    "default_enabled": self.config.get("default_enabled", False),
                    "default_cycle_length": self.config.get("default_cycle_length", 28),
                    "default_period_length": self.config.get("default_period_length", 5),
                    "cycle_settings": {
                        "ovulation_day": cycle_settings.get("ovulation_day", 14),
                        "ovulation_window": cycle_settings.get("ovulation_window", 3),
                    },
                },
            }
        )

    async def _webapi_toggle_session(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/toggle"""
        cfg = await self._get_session_config(umo)
        if not cfg or "anchor_date" not in cfg:
            return jsonify({"status": "error", "message": "会话未配置周期参数"}), 404

        if not await self.store.get(umo):
            await self.store.set(umo, cfg)

        await self.store.toggle(umo)
        cfg = await self.store.get(umo)
        return jsonify({"status": "ok", "data": self._serialize_session(umo, cfg)})

    async def _webapi_advance_session(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/advance"""
        body = await request.get_json() or {}
        days = body.get("days", 1)
        if isinstance(days, bool) or not isinstance(days, int):
            return jsonify({"status": "error", "message": "days 必须是整数"}), 400
        if not (-365 <= days <= 365):
            return jsonify({"status": "error", "message": "days 范围为 -365 ~ 365"}), 400

        cfg = await self._get_session_config(umo)
        if not cfg:
            return jsonify({"status": "error", "message": "会话未配置周期参数"}), 404

        if not await self.store.get(umo):
            await self.store.set(umo, cfg)

        cfg = await self.store.get(umo)
        cfg["advance_days"] = cfg.get("advance_days", 0) + days
        await self.store.set(umo, cfg)

        return jsonify({"status": "ok", "data": self._serialize_session(umo, cfg)})

    async def _webapi_set_anchor(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/anchor"""
        body = await request.get_json() or {}
        date_str = body.get("date", "")
        if not date_str:
            return jsonify({"status": "error", "message": "缺少 date 参数"}), 400

        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return (
                jsonify(
                    {"status": "error", "message": "日期格式错误，请使用 YYYY-MM-DD 格式"}
                ),
                400,
            )

        cfg = await self._get_session_config(umo)
        if not cfg:
            return jsonify({"status": "error", "message": "会话未配置周期参数"}), 404

        if not await self.store.get(umo):
            await self.store.set(umo, cfg)

        cfg = await self.store.get(umo)
        cfg["anchor_date"] = date_str
        cfg["advance_days"] = 0
        await self.store.set(umo, cfg)

        self._anchored_sessions.discard(umo)
        self._inject_counters.pop(umo, None)
        self._warmup_counters.pop(umo, None)

        return jsonify({"status": "ok", "data": self._serialize_session(umo, cfg)})

    async def _webapi_delete_session(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/delete"""
        if not await self.store.get(umo):
            return jsonify({"status": "error", "message": "会话不存在"}), 404
        await self.store.delete(umo)
        self._anchored_sessions.discard(umo)
        self._inject_counters.pop(umo, None)
        self._warmup_counters.pop(umo, None)
        return jsonify({"status": "ok", "data": {"umo": umo, "deleted": True}})

    async def _get_session_config(self, umo: str) -> dict | None:
        """Get session config, falling back to global defaults if available."""
        cfg = await self.store.get(umo)
        if cfg and "anchor_date" in cfg:
            return cfg

        # No session-specific config, check global defaults
        anchor = self.config.get("default_anchor_date", "")
        if not anchor:
            return None

        if not self.config.get("default_enabled", False):
            return None

        # Build config from global defaults
        cycle_settings = self.config.get("cycle_settings", {})
        return {
            "anchor_date": anchor,
            "cycle_length": self.config.get("default_cycle_length", 28),
            "period_length": self.config.get("default_period_length", 5),
            "ovulation_day": cycle_settings.get("ovulation_day", 14),
            "ovulation_window": cycle_settings.get("ovulation_window", 3),
            "enabled": True,
            "advance_days": 0,
        }

    async def _get_status_text(self, umo: str) -> str:
        """Generate human-readable status text for a session."""
        cfg = await self._get_session_config(umo)
        if not cfg or "anchor_date" not in cfg:
            return "当前会话未设置周期参数，且未配置全局默认值"

        enabled = cfg.get("enabled", True)
        if not enabled:
            return "当前会话的生理周期模拟已暂停，使用periodtoggle可恢复"

        info = self.engine.get_phase(
            cfg["anchor_date"],
            cfg.get("cycle_length", 28),
            cfg.get("period_length", 5),
            cfg.get("ovulation_day", 14),
            cfg.get("ovulation_window", 3),
            cfg.get("advance_days", 0),
        )

        phase_names = {
            "menstrual": "月经期",
            "follicular": "卵泡期",
            "ovulatory": "排卵期",
            "luteal": "黄体期",
        }

        lines = [
            f"当前生理状态{phase_names.get(info.phase, info.phase)}",
            f"阶段第{info.day}天周期第{info.total_day}天",
        ]
        if info.days_to_next > 0:
            lines.append(f"距离下次月经还有{info.days_to_next}天")
        else:
            lines.append("正处于月经期间")

        if cfg.get("advance_days", 0) != 0:
            lines.append(f"[调试]时间已快进{cfg['advance_days']}天")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Mood system helpers
    # ------------------------------------------------------------------ #

    def _get_phase_mood_base(self, phase: str) -> int:
        """Return the configured mood baseline for a cycle phase."""
        return {
            "menstrual": self.config.get("mood_base_menstrual", -3),
            "follicular": self.config.get("mood_base_follicular", 1),
            "ovulatory": self.config.get("mood_base_ovulatory", 2),
            "luteal": self.config.get("mood_base_luteal", -2),
        }.get(phase, 0)

    async def _auto_compress_prompts(self) -> None:
        """Background task to compress prompts on plugin init."""
        try:
            results = await self.prompt_compressor.compress_all()
            if results:
                logger.info(f"[PeriodPlugin] 后台提示词压缩完成，共 {len(results)} 条")
            else:
                logger.info("[PeriodPlugin] 后台提示词压缩完成，无新增压缩")
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 后台提示词压缩失败: {e}")

    def _extract_history(self, req: ProviderRequest) -> list[dict]:
        """Extract recent user/assistant exchanges from req.contexts."""
        contexts = getattr(req, "contexts", None) or []
        history: list[dict] = []
        max_len = self.config.get("mood_detector_context_length", 6)
        for entry in contexts[-max_len * 2 :]:
            if isinstance(entry, dict):
                role = entry.get("role", "")
                content = entry.get("content", "")
                if role in ("user", "assistant"):
                    history.append({"role": role, "content": str(content)})
        return history

    async def _apply_detection(
        self, mood_state: MoodState, detection: dict, user_message: str
    ) -> None:
        """Update mood state from detector output."""
        mood_state.mood_score = detection.get("new_mood_score", mood_state.mood_score)
        mood_state.energy = detection.get("new_energy", mood_state.energy)
        mood_state.intimacy = detection.get("new_intimacy", mood_state.intimacy)
        mood_state.dominant_emotion = detection.get(
            "new_dominant_emotion", mood_state.dominant_emotion
        )
        mood_state.clamp()

        # Handle lift_cold_violence
        if detection.get("lift_cold_violence"):
            mood_state.active_tools = [
                t for t in mood_state.active_tools if t["name"] != "cold_violence"
            ]
            mood_state.consecutive_unpleasant = 0

        # Handle new tool
        raw_tool = detection.get("tool", {})
        tool = raw_tool if isinstance(raw_tool, dict) else {}
        tool_name = tool.get("name", "none")
        if tool_name != "none" and self.config.get(f"enable_{tool_name}", True):
            params = self.mood_executor.validate_params(tool_name, tool.get("params"))
            self.mood_executor.execute(tool_name, params, mood_state)

        # Track consecutive unpleasant interactions
        attitude = detection.get("user_attitude", "neutral")
        if attitude in ("offensive", "boring"):
            mood_state.consecutive_unpleasant += 1
        elif attitude in ("caring", "concerned"):
            mood_state.consecutive_unpleasant = max(0, mood_state.consecutive_unpleasant - 1)
        else:
            # neutral / playful: slowly decay
            if mood_state.consecutive_unpleasant > 0:
                mood_state.consecutive_unpleasant -= 1

        # Record history
        max_hist = self.config.get("mood_history_length", 10)
        mood_state.add_history(
            event=f"detection:{attitude}",
            mood_change=detection.get("mood_change", 0),
            tool_used=tool_name if tool_name != "none" else None,
            user_message=user_message[:200],
            max_length=max_hist,
        )

        mood_state.last_interaction = datetime.datetime.now().isoformat()

    async def _handle_active_mood_tools(
        self,
        mood_state: MoodState,
        req: ProviderRequest,
        event: AstrMessageEvent,
    ) -> bool:
        """Process currently active mood tools. Returns True if LLM should be blocked."""
        for tool in list(mood_state.active_tools):
            name = tool["name"]

            if name == "cold_violence":
                logger.info(f"[PeriodPlugin] 执行冷暴力拦截，行为模式={self.config.get('cold_violence_behavior', 'angry_then_silent')}")
                behavior = self.config.get("cold_violence_behavior", "angry_then_silent")
                if behavior != "silent" and not tool.get("initiated"):
                    msg = self.mood_executor.get_initial_message(
                        behavior, mood_state.dominant_emotion
                    )
                    if msg:
                        from astrbot.core.message.components import Plain
                        from astrbot.core.message.message_event_result import MessageChain

                        await event.send(MessageChain([Plain(msg)]))
                        logger.info(f"[PeriodPlugin] 冷暴力初始消息已发送")
                    tool["initiated"] = True
                event.stop_event()
                event.should_call_llm(False)
                return True

            if name == "read_no_reply":
                tool["rounds_left"] = tool.get("rounds_left", 1) - 1
                if tool["rounds_left"] < 0:
                    logger.info("[PeriodPlugin] 已读不回轮数耗尽，解除拦截")
                    mood_state.active_tools.remove(tool)
                else:
                    logger.info(f"[PeriodPlugin] 执行已读不回拦截，剩余{tool['rounds_left']}轮")
                    event.stop_event()
                    event.should_call_llm(False)
                    return True

            if name in (
                "perfunctory_reply",
                "seek_comfort",
                "delayed_reply",
                "emotional_outburst",
                "topic_shift",
            ):
                logger.info(f"[PeriodPlugin] 注入情绪工具提示词: {name}")
                injection = self.mood_executor.get_prompt_injection(
                    name, tool.get("params", {})
                )
                if injection:
                    req.extra_user_content_parts.append(
                        TextPart(text=injection).mark_as_temp()
                    )
                # Single-use tools: remove after injection
                mood_state.active_tools.remove(tool)

        return False

    def _build_mood_prompt(self, mood_state: MoodState) -> str:
        """Build a prompt snippet describing current emotional state."""
        emotion_labels = {
            "happy": "开心",
            "calm": "平静",
            "irritable": "烦躁",
            "depressed": "低落",
            "angry": "生气",
            "playful": "撒娇",
        }
        label = emotion_labels.get(mood_state.dominant_emotion, mood_state.dominant_emotion)
        parts = [f"[当前情绪] 心情{mood_state.mood_score:.0f}/10，精力{mood_state.energy:.0f}/10，"
                 f"主导情绪：{label}。"]
        if mood_state.mood_score <= -5:
            parts.append("你现在心情很差，回复可能带刺或不耐烦。")
        elif mood_state.mood_score <= -2:
            parts.append("你现在心情不太好，回复比较冷淡。")
        elif mood_state.mood_score >= 5:
            parts.append("你现在心情很好，回复比较活泼热情。")
        return " ".join(parts)

    async def _get_mood_status_text(self, umo: str) -> str:
        """Generate human-readable mood status for a session."""
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        mood_state = await self.mood_store.get(mood_umo) or MoodState()

        emotion_labels = {
            "happy": "开心", "calm": "平静", "irritable": "烦躁",
            "depressed": "低落", "angry": "生气", "playful": "撒娇",
        }
        lines = [
            f"心情值：{mood_state.mood_score:.0f}/10",
            f"精力值：{mood_state.energy:.0f}/10",
            f"亲密度：{mood_state.intimacy:.0f}/10",
            f"主导情绪：{emotion_labels.get(mood_state.dominant_emotion, mood_state.dominant_emotion)}",
        ]
        if mood_state.active_tools:
            tools_str = ", ".join(t["name"] for t in mood_state.active_tools)
            lines.append(f"生效工具：{tools_str}")
        else:
            lines.append("生效工具：无")
        lines.append(f"连续不愉快：{mood_state.consecutive_unpleasant}次")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Commands
    # ------------------------------------------------------------------ #

    @filter.command_group("period")
    def period_group(self):
        """周期管理指令组"""
        pass

    @period_group.command("status")
    async def period_status(self, event: AstrMessageEvent):
        """查看当前周期状态 /period status"""
        allowed, msg = self._check_command_permission("status")
        if not allowed:
            yield event.plain_result(msg)
            return
        umo = event.unified_msg_origin
        logger.info(f"[PeriodPlugin] 用户 {umo} 执行 period status")
        text = await self._get_status_text(umo)
        yield event.plain_result(text)

    @period_group.command("set")
    async def period_set(
        self,
        event: AstrMessageEvent,
        date_str: str,
        cycle_len: int = 28,
        period_len: int = 5,
    ):
        """设置周期参数 /period set 2026-05-01 [28] [5]"""
        allowed, msg = self._check_command_permission("set")
        if not allowed:
            yield event.plain_result(msg)
            return
        umo = event.unified_msg_origin

        # Validate date format
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            yield event.plain_result("日期格式错误，请使用YYYYMMDD格式，例如2026-05-01")
            return

        # Validate parameters
        if not (21 <= cycle_len <= 35):
            yield event.plain_result("周期长度应在21至35天之间")
            return
        if not (2 <= period_len <= 10):
            yield event.plain_result("经期长度应在2至10天之间")
            return

        # Use cycle_settings defaults from config for ovulation
        cycle_settings = self.config.get("cycle_settings", {})
        ovulation_day = cycle_settings.get("ovulation_day", 14)
        ovulation_window = cycle_settings.get("ovulation_window", 3)

        data = {
            "anchor_date": date_str,
            "cycle_length": cycle_len,
            "period_length": period_len,
            "ovulation_day": ovulation_day,
            "ovulation_window": ovulation_window,
            "enabled": True,
            "advance_days": 0,
        }
        await self.store.set(umo, data)

        # Reset counters for this session
        self._anchored_sessions.discard(umo)
        self._inject_counters.pop(umo, None)
        self._warmup_counters.pop(umo, None)

        logger.info(
            f"[PeriodPlugin] 用户 {umo} 设置周期参数: "
            f"锚点={date_str}, 周期={cycle_len}天, 经期={period_len}天"
        )
        yield event.plain_result(
            f"周期参数已设置"
            f"经期首日{date_str}"
            f"周期长度{cycle_len}天"
            f"经期长度{period_len}天"
            f"排卵日第{ovulation_day}天"
            f"使用periodstatus查看当前状态"
        )

    @period_group.command("toggle")
    async def period_toggle(self, event: AstrMessageEvent):
        """切换模拟开关"""
        allowed, msg = self._check_command_permission("toggle")
        if not allowed:
            yield event.plain_result(msg)
            return
        umo = event.unified_msg_origin
        cfg = await self._get_session_config(umo)
        if not cfg or "anchor_date" not in cfg:
            yield event.plain_result("请先使用periodset设置周期参数，或在插件配置中填写全局默认值")
            return

        # If using global defaults (not yet persisted), write to store first
        if not await self.store.get(umo):
            await self.store.set(umo, cfg)

        new_state = await self.store.toggle(umo)
        state_text = "开启" if new_state else "暂停"
        logger.info(f"[PeriodPlugin] 用户 {umo} 切换周期模拟状态为: {state_text}")
        yield event.plain_result(f"生理周期模拟已{state_text}")

    @period_group.command("advance")
    async def period_advance(self, event: AstrMessageEvent, days: int = 1):
        """快进时间（调试）"""
        allowed, msg = self._check_command_permission("advance")
        if not allowed:
            yield event.plain_result(msg)
            return
        umo = event.unified_msg_origin
        cfg = await self._get_session_config(umo)
        if not cfg:
            yield event.plain_result("请先使用periodset设置周期参数，或在插件配置中填写全局默认值")
            return

        # If using global defaults (not yet persisted), write to store first
        if not await self.store.get(umo):
            await self.store.set(umo, cfg)

        cfg = await self.store.get(umo)
        cfg["advance_days"] = cfg.get("advance_days", 0) + days
        await self.store.set(umo, cfg)

        logger.info(f"[PeriodPlugin] 用户 {umo} 快进时间: {days}天, 累计={cfg['advance_days']}天")
        yield event.plain_result(
            f"时间已快进{days}天（累计快进{cfg['advance_days']}天）"
            f"使用periodstatus查看当前状态"
        )

    @period_group.command("reset")
    @permission_type(PermissionType.ADMIN)
    async def period_reset(self, event: AstrMessageEvent):
        """重置当前会话数据"""
        allowed, msg = self._check_command_permission("reset")
        if not allowed:
            yield event.plain_result(msg)
            return
        umo = event.unified_msg_origin
        await self.store.delete(umo)
        self._anchored_sessions.discard(umo)
        self._inject_counters.pop(umo, None)
        self._warmup_counters.pop(umo, None)
        logger.info(f"[PeriodPlugin] 用户 {umo} 重置周期数据")
        yield event.plain_result("当前会话的周期数据已重置")

    @period_group.command("mood")
    async def period_mood(self, event: AstrMessageEvent):
        """查看当前情绪状态 /period mood"""
        if not self.config.get("mood_system_enabled", True):
            yield event.plain_result("情绪管理系统已关闭")
            return
        umo = event.unified_msg_origin
        logger.info(f"[PeriodPlugin] 用户 {umo} 执行 period mood")
        text = await self._get_mood_status_text(umo)
        yield event.plain_result(text)

    @period_group.command("moodreset")
    @permission_type(PermissionType.ADMIN)
    async def period_mood_reset(self, event: AstrMessageEvent):
        """重置当前会话情绪状态 /period moodreset"""
        if not self.config.get("mood_system_enabled", True):
            yield event.plain_result("情绪管理系统已关闭")
            return
        umo = event.unified_msg_origin
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        await self.mood_store.delete(mood_umo)
        logger.info(f"[PeriodPlugin] 用户 {umo} 重置情绪状态")
        yield event.plain_result("当前会话的情绪状态已重置")

    @period_group.command("lift")
    async def period_lift(self, event: AstrMessageEvent):
        """手动解除冷暴力等活跃工具 /period lift"""
        if not self.config.get("mood_system_enabled", True):
            yield event.plain_result("情绪管理系统已关闭")
            return
        umo = event.unified_msg_origin
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        mood_state = await self.mood_store.get(mood_umo) or MoodState()
        if not mood_state.active_tools:
            yield event.plain_result("当前没有生效的情绪工具")
            return
        mood_state.active_tools.clear()
        mood_state.consecutive_unpleasant = 0
        await self.mood_store.set(mood_umo, mood_state)
        logger.info(f"[PeriodPlugin] 用户 {umo} 手动解除情绪工具")
        yield event.plain_result("已解除所有情绪工具限制")

    @period_group.command("compress")
    @permission_type(PermissionType.ADMIN)
    async def period_compress(self, event: AstrMessageEvent):
        """手动压缩提示词 /period compress"""
        if not self.config.get("prompt_compression_enabled", False):
            yield event.plain_result("提示词压缩功能未开启，请先在插件配置中启用")
            return
        logger.info(f"[PeriodPlugin] 用户 {event.unified_msg_origin} 手动触发提示词压缩")
        yield event.plain_result("正在压缩提示词，请稍候...")
        try:
            results = await self.prompt_compressor.compress_all()
            if results:
                lines = [f"压缩完成，共 {len(results)} 条提示词:"]
                for key, text in results.items():
                    original_len = len(self.prompt_compressor._get_original_text(key))
                    lines.append(f"  {key}: {original_len}字 → {len(text)}字")
                yield event.plain_result("\n".join(lines))
            else:
                yield event.plain_result("无可压缩的提示词，或压缩失败")
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 手动压缩失败: {e}")
            yield event.plain_result(f"压缩失败: {e}")

    # ------------------------------------------------------------------ #
    #  LLM Hooks
    # ------------------------------------------------------------------ #

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Inject physiological state into LLM request."""
        umo = event.unified_msg_origin

        # Check global switch
        if not self.config.get("auto_inject", True):
            return

        # UMO list filter
        if not self.config.get("global_inject", False):
            umo_list = self.config.get("umo_list", [])
            umo_mode = self.config.get("umo_mode", "whitelist")
            if umo_mode == "whitelist":
                if umo not in umo_list:
                    return
            elif umo_mode == "blacklist":
                if umo in umo_list:
                    return

        cfg = await self._get_session_config(umo)
        if not cfg or not cfg.get("enabled") or "anchor_date" not in cfg:
            return

        # Auto-persist global defaults so they appear in WebUI
        if not await self.store.get(umo):
            await self.store.set(umo, cfg)

        # Warmup check
        warmup = self.config.get("warmup_rounds", 0)
        if warmup > 0:
            count = self._warmup_counters.get(umo, 0) + 1
            self._warmup_counters[umo] = count
            if count <= warmup:
                return

        # Injection frequency check
        mode = self.config.get("inject_mode", "every_request")
        if mode == "only_status":
            return
        elif mode == "interval_3":
            count = self._inject_counters.get(umo, 0) + 1
            self._inject_counters[umo] = count
            if count % 3 != 1:  # Inject on 1st, 4th, 7th... requests
                return
        elif mode == "on_trigger":
            msg = event.message_str or ""
            keywords = self.config.get("trigger_keywords", ["怎么了", "还好吗", "不舒服", "心情不好", "你没事吧"])
            if not any(kw in msg for kw in keywords):
                return

        # Calculate cycle phase
        info = self.engine.get_phase(
            cfg["anchor_date"],
            cfg.get("cycle_length", 28),
            cfg.get("period_length", 5),
            cfg.get("ovulation_day", 14),
            cfg.get("ovulation_window", 3),
            cfg.get("advance_days", 0),
        )

        # Save original system prompt before injecting anchor
        # so mood detector sees the bot's persona without our anchor
        original_system_prompt = req.system_prompt or ""

        # Static anchor: inject once per session
        if umo not in self._anchored_sessions:
            anchor = self.prompt_builder.get_anchor()
            req.system_prompt = original_system_prompt + ("\n\n" if original_system_prompt else "") + anchor
            self._anchored_sessions.add(umo)

        # Dynamic state: inject every request (when frequency allows)
        hour = datetime.datetime.now().hour
        dynamic = self.prompt_builder.build_dynamic(info.phase, info.day, hour)
        req.extra_user_content_parts.append(
            TextPart(text=dynamic).mark_as_temp()
        )

        # ============================================================== #
        #  Mood / Emotion System
        # ============================================================== #
        if self.config.get("mood_system_enabled", True):
            logger.info(f"[PeriodPlugin] 用户 {umo} 触发情绪检测")
            await self._run_mood_system(event, req, umo, info, original_system_prompt)

    async def _run_mood_system(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        umo: str,
        phase_info,
        original_system_prompt: str = "",
    ) -> None:
        """Execute the mood detection and tool logic."""
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo

        # UMO-level lock prevents race conditions when multiple messages
        # from the same session arrive concurrently
        lock = self._mood_locks.setdefault(mood_umo, asyncio.Lock())
        async with lock:
            mood_state = await self.mood_store.get(mood_umo) or MoodState()
            logger.info(
                f"[PeriodPlugin] 情绪状态: umo={umo}, 心情={mood_state.mood_score:.0f}, "
                f"情绪={mood_state.dominant_emotion}, 工具={len(mood_state.active_tools)}个"
            )

            # Expire old tools
            now_iso = datetime.datetime.now().isoformat()
            expired = mood_state.expire_tools(now_iso)
            if expired:
                logger.info(f"[PeriodPlugin] 到期工具清理: {[t['name'] for t in expired]}")

            # Check if any intercepting tool is already active
            blocked = await self._handle_active_mood_tools(mood_state, req, event)
            if blocked:
                await self.mood_store.set(mood_umo, mood_state)
                return

            # Determine whether to run detection
            detection_mode = self.config.get("mood_detection_mode", "always")
            should_detect = True
            if detection_mode == "sensitive_only" and phase_info.phase not in (
                "menstrual",
                "luteal",
            ):
                should_detect = False
                logger.info("[PeriodPlugin] 非敏感期，跳过情绪检测")
            elif detection_mode == "rule_first":
                should_detect = self._rule_based_prescreen(mood_state, event.message_str or "")
                if not should_detect:
                    logger.info("[PeriodPlugin] 规则初筛未触发，跳过情绪检测")

            detection: dict = {"tool": {"name": "none"}}
            if should_detect:
                history = self._extract_history(req)
                # Pass the original system prompt (before our anchor injection)
                # so the mood detector sees the bot's persona, not our plugin anchor
                system_prompt = ""
                if self.config.get("mood_detector_read_system_prompt", True):
                    system_prompt = original_system_prompt
                try:
                    detection = await self.mood_detector.detect(
                        umo,
                        phase_info,
                        mood_state,
                        history,
                        event.message_str or "",
                        system_prompt,
                    )
                    logger.info(
                        f"[PeriodPlugin] 情绪检测结果: 态度={detection.get('user_attitude', 'unknown')}, "
                        f"工具={detection.get('tool', {}).get('name', 'none')}, "
                        f"原因={detection.get('reasoning', '无')}"
                    )
                except Exception as e:
                    logger.warning(f"[MoodSystem] Detection failed: {e}")

                await self._apply_detection(mood_state, detection, event.message_str or "")

            # If detection triggered a new tool, handle it immediately
            raw_tool = detection.get("tool", {})
            tool = raw_tool if isinstance(raw_tool, dict) else {}
            tool_name = tool.get("name", "none")
            if tool_name != "none" and self.config.get(f"enable_{tool_name}", True):
                logger.info(f"[PeriodPlugin] 激活情绪工具: {tool_name}")
                blocked = await self._handle_active_mood_tools(mood_state, req, event)
                if blocked:
                    await self.mood_store.set(mood_umo, mood_state)
                    return

            # Inject current mood into main model prompt
            if self.config.get("inject_mood_to_prompt", True):
                mp = self._build_mood_prompt(mood_state)
                req.extra_user_content_parts.append(
                    TextPart(text=mp).mark_as_temp()
                )

            await self.mood_store.set(mood_umo, mood_state)

    def _rule_based_prescreen(self, mood_state: MoodState, message: str) -> bool:
        """Lightweight rule-based prescreen for 'rule_first' detection mode.

        Returns True if the message warrants full LLM detection.
        """
        msg = message.strip()
        if not msg:
            return False

        # Triggers that always warrant detection
        negative_keywords = ["滚", "烦", "闭嘴", "垃圾", "废物", "傻", "蠢", "死", "他妈"]
        if any(kw in msg for kw in negative_keywords):
            return True

        comfort_keywords = ["对不起", "抱歉", "我错了", "别生气", "抱抱", "安慰", "还好吗"]
        if any(kw in msg for kw in comfort_keywords):
            return True

        # If mood is already negative, always detect
        if mood_state.mood_score <= -3:
            return True

        # If there are active tools, always detect
        if mood_state.active_tools:
            return True

        # Otherwise skip (reduce token cost)
        return False

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """Detect and optionally replace OOC words in LLM response."""
        umo = event.unified_msg_origin

        cfg = await self._get_session_config(umo)
        if not cfg or not cfg.get("enabled"):
            return
        if not self.config.get("ooc_shield", True):
            return

        text = resp.completion_text or ""
        forbidden = self.config.get(
            "forbidden_words",
            ["月经", "经期", "激素", "雌激素", "孕激素", "黄体", "卵泡", "卵巢", "子宫", "内分泌", "PMS", "生理期", "排卵期", "安全期"],
        )
        hit = [w for w in forbidden if w in text]
        if hit:
            logger.warning(f"[PeriodPlugin] OOC检测命中: umo={umo}, 命中词={hit}")
            if self.config.get("ooc_replace", False):
                for w in hit:
                    text = text.replace(w, "*" * len(w))
                resp.completion_text = text  # Setter syncs to result_chain
                logger.info(f"[PeriodPlugin] OOC词汇已替换为星号")

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def terminate(self):
        """Clean up resources on plugin unload."""
        self._anchored_sessions.clear()
        self._inject_counters.clear()
        self._warmup_counters.clear()
        logger.info("[PeriodPlugin] 插件已卸载，内存缓存已清理")
