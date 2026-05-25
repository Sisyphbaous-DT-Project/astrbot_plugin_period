"""AstrBot plugin for physiological cycle simulation."""

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


@register("astrbot_plugin_period", "C₂₂H₂₅NO₆", "生理周期模拟插件", "1.5.0",
          "https://github.com/Sisyphbaous-DT-Project/astrbot_plugin_period")
class PeriodPlugin(Star):
    """Plugin that simulates physiological cycles for female-persona bots."""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self.plugin_data_dir = StarTools.get_data_dir()
        self.engine = CycleEngine()
        self.store = CycleStore(self.plugin_data_dir)
        self.prompt_builder = PromptBuilder(self.config)
        self._anchored_sessions: set[str] = set()  # Sessions with anchor injected
        self._inject_counters: dict[str, int] = {}  # Interval injection counters
        self._warmup_counters: dict[str, int] = {}  # Warmup round counters

        self._register_web_apis()

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
        text = await self._get_status_text(event.unified_msg_origin)
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
        yield event.plain_result("当前会话的周期数据已重置")

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

        # Static anchor: inject once per session
        if umo not in self._anchored_sessions:
            anchor = self.prompt_builder.get_anchor()
            existing = req.system_prompt or ""
            req.system_prompt = existing + ("\n\n" if existing else "") + anchor
            self._anchored_sessions.add(umo)

        # Dynamic state: inject every request (when frequency allows)
        hour = datetime.datetime.now().hour
        dynamic = self.prompt_builder.build_dynamic(info.phase, info.day, hour)
        req.extra_user_content_parts.append(
            TextPart(text=dynamic).mark_as_temp()
        )

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
            logger.warning(f"[Period OOC] umo={umo}, words={hit}")
            if self.config.get("ooc_replace", False):
                for w in hit:
                    text = text.replace(w, "*" * len(w))
                resp.completion_text = text  # Setter syncs to result_chain

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def terminate(self):
        """Clean up resources on plugin unload."""
        self._anchored_sessions.clear()
        self._inject_counters.clear()
        self._warmup_counters.clear()
        # CycleStore writes are atomic, no pending data to flush
        logger.info("[PeriodPlugin] Terminated.")
