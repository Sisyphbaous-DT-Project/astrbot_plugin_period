"""AstrBot plugin for physiological cycle simulation."""

import asyncio
import datetime
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from quart import jsonify, request

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api import logger
import astrbot.api.message_components as Comp
from astrbot.api.event import MessageChain
from astrbot.core.agent.message import TextPart

from .core.engine import CycleEngine
from .core.store import CycleStore
from .core.prompt import PromptBuilder
from .core.prompt_compressor import PromptCompressor
from .core.mood import MoodState
from .core.mood_store import MoodStore
from .core.mood_tools import MoodToolExecutor
from .core.mood_detector import MoodDetector


class PeriodPlugin(Star):
    """Plugin that simulates physiological cycles for female-persona bots."""

    name = "astrbot_plugin_period"

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self.plugin_data_dir = StarTools.get_data_dir("astrbot_plugin_period")
        self.engine = CycleEngine()
        self.store = CycleStore(self.plugin_data_dir)

        # Prompt compression
        self.prompt_compressor = PromptCompressor(self.context, self.config, self.plugin_data_dir)
        self.prompt_builder = PromptBuilder(self.config, self.prompt_compressor)

        self._anchored_sessions: set[str] = set()  # Sessions visited (for WebUI tracking)
        self._inject_counters: dict[str, int] = {}  # Interval injection counters
        self._warmup_counters: dict[str, int] = {}  # Warmup round counters

        # Mood / emotion system
        self.mood_store = MoodStore(self.plugin_data_dir)
        self.mood_detector = MoodDetector(self.context, self.config)
        self.mood_executor = MoodToolExecutor()
        self._mood_locks: dict[str, asyncio.Lock] = {}
        self._mood_locks_lock = asyncio.Lock()  # Protects _mood_locks dict (WR-2)

        self._compression_task: asyncio.Task | None = None  # CR-3: tracked task reference

        self._register_web_apis()

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

    def _load_config_schema(self) -> dict:
        """读取插件配置 schema，供 WebUI 自动渲染和后端校验使用。"""
        schema_path = Path(__file__).with_name("_conf_schema.json")
        try:
            with schema_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[PeriodPlugin] 读取配置 schema 失败: {e}")
            return {}

    def _extract_schema_defaults(self, schema: dict) -> dict:
        """从 schema 中递归提取默认值。"""
        defaults = {}
        for key, meta in schema.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("type") == "object":
                defaults[key] = self._extract_schema_defaults(meta.get("items", {}))
            elif "default" in meta:
                defaults[key] = deepcopy(meta["default"])
        return defaults

    def _deep_merge_dicts(self, base: dict, override: dict) -> dict:
        """递归合并配置，保留未出现在 override 中的旧字段。"""
        merged = deepcopy(base)
        for key, value in override.items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = self._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def _build_full_config(self) -> tuple[dict, dict, dict]:
        """返回当前完整配置、schema 和默认值。"""
        schema = self._load_config_schema()
        defaults = self._extract_schema_defaults(schema)
        current = self._deep_merge_dicts(defaults, dict(self.config))
        return current, schema, defaults

    def _get_provider_options(self) -> list[dict[str, str]]:
        """列出可供 WebUI 下拉选择的聊天模型。"""
        try:
            providers = self.context.get_all_providers()
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 读取 provider 列表失败: {e}")
            return []

        options = []
        for provider in providers or []:
            try:
                meta = provider.meta()
                provider_id = str(getattr(meta, "id", "") or "")
                if not provider_id:
                    continue
                model = str(getattr(meta, "model", "") or "")
                provider_type = str(getattr(meta, "type", "") or "")
                label_parts = [provider_id]
                if model:
                    label_parts.append(model)
                if provider_type:
                    label_parts.append(provider_type)
                options.append(
                    {
                        "id": provider_id,
                        "label": " / ".join(label_parts),
                        "model": model,
                        "type": provider_type,
                    }
                )
            except Exception as e:
                logger.warning(f"[PeriodPlugin] 解析 provider 信息失败: {e}")
        return options

    def _coerce_config_value(
        self,
        key_path: str,
        value: Any,
        meta: dict,
        old_value: Any = None,
    ) -> tuple[Any, list[str]]:
        """按 schema 清洗单个配置值，返回清洗结果和错误列表。"""
        errors: list[str] = []
        field_type = meta.get("type", "string")

        if field_type == "object":
            if value is None:
                value = {}
            if not isinstance(value, dict):
                return old_value if isinstance(old_value, dict) else {}, [f"{key_path} 必须是对象"]
            old_obj = old_value if isinstance(old_value, dict) else {}
            cleaned, child_errors = self._sanitize_config(
                value,
                meta.get("items", {}),
                old_obj,
                key_path,
            )
            return cleaned, child_errors

        if field_type == "bool":
            if not isinstance(value, bool):
                return old_value, [f"{key_path} 必须是布尔值"]
            return value, []

        if field_type == "int":
            if isinstance(value, bool):
                return old_value, [f"{key_path} 必须是整数"]
            if isinstance(value, float) and not value.is_integer():
                return old_value, [f"{key_path} 必须是整数"]
            try:
                coerced = int(value)
            except (TypeError, ValueError):
                return old_value, [f"{key_path} 必须是整数"]
            min_value = meta.get("min")
            max_value = meta.get("max")
            if min_value is not None and coerced < min_value:
                errors.append(f"{key_path} 不能小于 {min_value}")
            if max_value is not None and coerced > max_value:
                errors.append(f"{key_path} 不能大于 {max_value}")
            return coerced, errors

        if field_type == "list":
            if not isinstance(value, list):
                return old_value if isinstance(old_value, list) else [], [f"{key_path} 必须是列表"]
            cleaned_list = []
            for item in value:
                if not isinstance(item, str):
                    errors.append(f"{key_path} 中的每一项都必须是字符串")
                    continue
                cleaned_item = item.strip()
                if cleaned_item:
                    cleaned_list.append(cleaned_item)
            return cleaned_list, errors

        if field_type in ("string", "text"):
            if value is None:
                value = ""
            if not isinstance(value, str):
                return old_value if isinstance(old_value, str) else "", [f"{key_path} 必须是字符串"]
            if key_path == "default_anchor_date" and value:
                try:
                    datetime.datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    return old_value, ["default_anchor_date 日期格式错误，请使用 YYYY-MM-DD"]
            options = meta.get("options")
            if options and value not in options:
                return old_value, [f"{key_path} 必须是以下值之一: {', '.join(options)}"]
            return value, []

        return deepcopy(value), []

    def _sanitize_config(
        self,
        incoming: dict,
        schema: dict,
        current: dict,
        prefix: str = "",
    ) -> tuple[dict, list[str]]:
        """递归清洗提交的配置，只处理 schema 中定义的字段并保留旧字段。"""
        cleaned = deepcopy(current)
        errors: list[str] = []
        for key, meta in schema.items():
            if not isinstance(meta, dict) or key not in incoming:
                continue
            key_path = f"{prefix}.{key}" if prefix else key
            value, value_errors = self._coerce_config_value(
                key_path,
                incoming[key],
                meta,
                cleaned.get(key),
            )
            if value_errors:
                errors.extend(value_errors)
                continue
            cleaned[key] = value
        return cleaned, errors

    def _save_live_config(self, next_config: dict) -> tuple[bool, str]:
        """保存配置文件并同步当前运行态。"""
        save_config = getattr(self.config, "save_config", None)
        persisted = callable(save_config)
        if persisted:
            save_config(next_config)
        else:
            self.config.clear()
            self.config.update(next_config)
            return False, "当前配置对象不支持 save_config，已仅更新运行态配置"

        # AstrBotConfig.save_config 会 update 自身；这里再同步一次，兼容不同实现。
        self.config.clear()
        self.config.update(next_config)
        return True, "配置已保存并即时生效"

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
            f"{base}/config",
            self._webapi_save_config,
            ["POST"],
            "Save plugin config",
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
        if cfg.get("source") == "global_default":
            return "global_default"
        return "manual"

    def _normalize_web_umo(self, umo: str) -> str:
        """还原 WebUI 路径里被编码过的 UMO，避免写出重复会话。"""
        decoded = unquote(umo)
        if decoded != umo and ":" in decoded and "/" not in decoded:
            return decoded
        return umo

    async def _migrate_encoded_session_alias(self, umo: str) -> str:
        """将旧版误写入的 percent-encoded UMO 合并回原始 UMO。"""
        normalized = self._normalize_web_umo(umo)
        if normalized == umo:
            return normalized

        alias_cfg = await self.store.get(umo)
        if alias_cfg:
            existing = await self.store.get(normalized)
            next_cfg = self._deep_merge_dicts(existing or {}, alias_cfg)
            await self.store.set(normalized, next_cfg)
            await self.store.delete(umo)
            logger.info(
                f"[PeriodPlugin] 迁移编码后的 WebUI 会话记录: {umo} -> {normalized}"
            )

        if umo in self._anchored_sessions:
            self._anchored_sessions.discard(umo)
            self._anchored_sessions.add(normalized)
        if umo in self._inject_counters:
            self._inject_counters[normalized] = self._inject_counters.pop(umo)
        if umo in self._warmup_counters:
            self._warmup_counters[normalized] = self._warmup_counters.pop(umo)

        return normalized

    async def _migrate_encoded_session_aliases(self) -> None:
        """清理 WebUI 旧版本留下的编码 UMO 会话。"""
        all_data = await self.store.get_all()
        for umo in list(all_data.keys()):
            await self._migrate_encoded_session_alias(umo)

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

    def _build_global_default_config(self, stored: dict | None = None) -> dict | None:
        """构建跟随全局默认值的会话配置。"""
        stored = stored or {}
        anchor_overridden = stored.get("anchor_overridden", False)
        if anchor_overridden:
            anchor = stored.get("anchor_date", "")
        else:
            anchor = self.config.get("default_anchor_date", "")
        if not anchor:
            return None

        cycle_settings = self.config.get("cycle_settings", {})
        return {
            "source": "global_default",
            "anchor_overridden": anchor_overridden,
            "anchor_date": anchor,
            "cycle_length": self.config.get("default_cycle_length", 28),
            "period_length": self.config.get("default_period_length", 5),
            "ovulation_day": cycle_settings.get("ovulation_day", 14),
            "ovulation_window": cycle_settings.get("ovulation_window", 3),
            "enabled": stored.get("enabled", self.config.get("default_enabled", False)),
            "advance_days": stored.get("advance_days", 0),
        }

    def _is_legacy_global_default_config(self, cfg: dict) -> bool:
        """判断旧版未标记来源的记录是否可安全视为全局默认会话。"""
        if cfg.get("source") or not cfg.get("anchor_date"):
            return False

        # 旧版本没有 source 字段，且最常见的遗留值是内置经期长度 5 天。
        # 若用户手动设置过其他经期长度，则继续按手动会话处理，避免误伤。
        # 完全同形的旧手动记录无法可靠区分；这里仅兼容默认经期已改动的旧默认会话。
        historical_period_length = 5
        if cfg.get("period_length") != historical_period_length:
            return False
        if self.config.get("default_period_length", 5) == historical_period_length:
            return False

        default_anchor = self.config.get("default_anchor_date", "")
        if not default_anchor or cfg.get("anchor_date") != default_anchor:
            return False

        cycle_settings = self.config.get("cycle_settings", {})
        expected = {
            "cycle_length": self.config.get("default_cycle_length", 28),
            "ovulation_day": cycle_settings.get("ovulation_day", 14),
            "ovulation_window": cycle_settings.get("ovulation_window", 3),
        }
        return all(cfg.get(key) == value for key, value in expected.items())

    async def _migrate_legacy_global_default_config(
        self,
        umo: str,
        cfg: dict,
    ) -> dict | None:
        """将可识别的旧版全局默认记录迁移为显式来源记录。"""
        migrated = self._build_global_default_config(cfg)
        if migrated:
            await self.store.set(umo, migrated)
            logger.info(f"[PeriodPlugin] 迁移旧版全局默认会话记录: {umo}")
        return migrated

    async def _webapi_list_sessions(self):
        """GET /astrbot_plugin_period/sessions"""
        await self._migrate_encoded_session_aliases()
        all_data = await self.store.get_all()
        seen: set[str] = set()
        sessions = []
        # 1) Explicitly configured sessions (from persistent store)
        for umo, cfg in all_data.items():
            if cfg.get("source") == "global_default" or (
                self._is_legacy_global_default_config(cfg)
            ):
                cfg = await self._get_session_config(umo)
            serialized = self._serialize_session(umo, cfg)
            if serialized:
                sessions.append(serialized)
                seen.add(umo)
        # 2) Sessions that use global defaults but have not been persisted yet
        for umo in self._anchored_sessions:
            if umo in seen:
                continue
            cfg = await self._get_session_config(umo)
            if not cfg or "anchor_date" not in cfg:
                continue
            serialized = self._serialize_session(umo, cfg)
            if serialized:
                sessions.append(serialized)
                seen.add(umo)
        return jsonify(
            {"status": "ok", "data": {"sessions": sessions, "count": len(sessions)}}
        )

    async def _webapi_get_config(self):
        """GET /astrbot_plugin_period/config"""
        current, schema, defaults = self._build_full_config()
        return jsonify(
            {
                "status": "ok",
                "data": {
                    **current,
                    "config": current,
                    "schema": schema,
                    "defaults": defaults,
                    "provider_options": self._get_provider_options(),
                },
            }
        )

    async def _webapi_save_config(self):
        """POST /astrbot_plugin_period/config"""
        body = await request.get_json()
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify({"status": "error", "message": "配置数据必须是对象"}), 400

        incoming = body.get("config", body)
        if not isinstance(incoming, dict):
            return jsonify({"status": "error", "message": "config 必须是对象"}), 400

        current, schema, defaults = self._build_full_config()
        base_config = self._deep_merge_dicts(defaults, current)
        next_config, errors = self._sanitize_config(incoming, schema, base_config)
        if errors:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "配置校验失败",
                        "errors": errors,
                    }
                ),
                400,
            )

        try:
            persisted, message = self._save_live_config(next_config)
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 保存 WebUI 配置失败: {e}")
            return jsonify({"status": "error", "message": f"保存配置失败: {e}"}), 500

        current, schema, defaults = self._build_full_config()
        return jsonify(
            {
                "status": "ok",
                "message": message,
                "data": {
                    **current,
                    "config": current,
                    "schema": schema,
                    "defaults": defaults,
                    "provider_options": self._get_provider_options(),
                    "persisted": persisted,
                },
            }
        )

    async def _webapi_toggle_session(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/toggle"""
        umo = await self._migrate_encoded_session_alias(umo)
        cfg = await self._get_session_config(umo)
        if not cfg or "anchor_date" not in cfg:
            return jsonify({"status": "error", "message": "会话未配置周期参数"}), 404

        if not await self.store.get(umo):
            await self.store.set(umo, cfg)

        await self.store.toggle(umo)
        cfg = await self._get_session_config(umo)
        return jsonify({"status": "ok", "data": self._serialize_session(umo, cfg)})

    async def _webapi_advance_session(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/advance"""
        umo = await self._migrate_encoded_session_alias(umo)
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
        cfg = await self._get_session_config(umo)

        return jsonify({"status": "ok", "data": self._serialize_session(umo, cfg)})

    async def _webapi_set_anchor(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/anchor"""
        umo = await self._migrate_encoded_session_alias(umo)
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
        if cfg.get("source") == "global_default":
            cfg["anchor_overridden"] = True
        await self.store.set(umo, cfg)
        cfg = await self._get_session_config(umo)

        self._anchored_sessions.discard(umo)
        self._inject_counters.pop(umo, None)
        self._warmup_counters.pop(umo, None)

        return jsonify({"status": "ok", "data": self._serialize_session(umo, cfg)})

    async def _webapi_delete_session(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/delete"""
        umo = await self._migrate_encoded_session_alias(umo)
        if not await self.store.get(umo):
            return jsonify({"status": "error", "message": "会话不存在"}), 404
        await self.store.delete(umo)
        self._anchored_sessions.discard(umo)
        self._inject_counters.pop(umo, None)
        self._warmup_counters.pop(umo, None)
        # Also clean up mood state for this session
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        await self.mood_store.delete(mood_umo)
        self._mood_locks.pop(mood_umo, None)
        return jsonify({"status": "ok", "data": {"umo": umo, "deleted": True}})

    async def _get_session_config(self, umo: str) -> dict | None:
        """Get session config, falling back to global defaults if available."""
        cfg = await self.store.get(umo)
        if cfg and "anchor_date" in cfg:
            if cfg.get("source") == "global_default":
                return self._build_global_default_config(cfg)
            if self._is_legacy_global_default_config(cfg):
                return await self._migrate_legacy_global_default_config(umo, cfg)
            return cfg

        # enabled reflects default_enabled so that on_llm_request won't inject
        # unless the admin explicitly opted in, but commands (status/toggle)
        # can still see and manipulate the session config.
        return self._build_global_default_config()

    async def _get_status_text(self, umo: str) -> str:
        """Generate human-readable status text for a session."""
        cfg = await self._get_session_config(umo)
        if not cfg or "anchor_date" not in cfg:
            return "当前会话未设置周期参数，且未配置全局默认值"

        enabled = cfg.get("enabled", True)
        if not enabled:
            return "当前会话的生理周期模拟已暂停，使用 period toggle 可恢复"

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

    async def _auto_compress_prompts(self) -> None:
        """Background task to compress prompts on plugin init."""
        try:
            results = await self.prompt_compressor.compress_all()
            if results:
                logger.info("[PeriodPlugin] 后台提示词压缩完成，共 %s 条", len(results))
            else:
                logger.info("[PeriodPlugin] 后台提示词压缩完成，无新增压缩")
        except Exception as e:
            logger.warning("[PeriodPlugin] 后台提示词压缩失败: %s", e)

    def _extract_history(self, req: ProviderRequest) -> list[dict]:
        """Extract recent user/assistant exchanges from req.contexts.

        Handles both dict entries (OpenAI-compatible) and astrbot Message objects.
        """
        contexts = getattr(req, "contexts", None) or []
        history: list[dict] = []
        # Fixed context length: last 6 rounds (12 messages) of user/assistant
        for entry in contexts[-12:]:
            role = ""
            content = ""
            if isinstance(entry, dict):
                role = entry.get("role", "")
                content = entry.get("content", "")
            elif hasattr(entry, "role") and hasattr(entry, "content"):
                # astrbot.core.agent.message.Message or similar Pydantic model
                role = entry.role
                raw_content = entry.content
                if isinstance(raw_content, str):
                    content = raw_content
                elif raw_content is not None:
                    content = str(raw_content)
            if role in ("user", "assistant") and content:
                history.append({"role": role, "content": content})
        return history

    def _inject_mood_prompts(self, mood_state: MoodState, req: ProviderRequest) -> bool:
        """Inject prompt snippets for active non-intercepting tools.

        Single-use tools are removed after injection.
        Returns True if any prompt was injected.
        """
        injected = False
        for tool in list(mood_state.active_tools):
            name = tool["name"]
            if name in (
                "perfunctory_reply",
                "seek_comfort",
                "delayed_reply",
                "emotional_outburst",
                "topic_shift",
            ):
                logger.info("[PeriodPlugin] 注入情绪工具提示词: %s", name)
                injection = self.mood_executor.get_prompt_injection(
                    name, tool.get("params", {})
                )
                if injection:
                    req.extra_user_content_parts.append(
                        TextPart(text=injection).mark_as_temp()
                    )
                    injected = True
                mood_state.active_tools.remove(tool)
        return injected

    async def _get_mood_status_text(self, umo: str) -> str:
        """Generate human-readable mood status for a session."""
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        mood_state = await self.mood_store.get(mood_umo) or MoodState()

        lines = []
        if mood_state.active_tools:
            tools_info = []
            for t in mood_state.active_tools:
                name = t["name"]
                extra = ""
                if t.get("rounds_left") is not None:
                    extra = f"(剩余{t['rounds_left']}轮)"
                elif t.get("expires_at"):
                    extra = f"(限时)"
                tools_info.append(f"{name}{extra}")
            lines.append(f"生效工具：{', '.join(tools_info)}")
        else:
            lines.append("生效工具：无")

        if mood_state.history:
            last = mood_state.history[-1]
            lines.append(f"最近事件：{last.get('event', '无')}")
            lines.append(f"原因：{last.get('reasoning', '无')}")

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
        cycle_len: int = None,
        period_len: int = None,
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

        try:
            cycle_len = (
                self.config.get("default_cycle_length", 28)
                if cycle_len is None
                else int(cycle_len)
            )
        except (TypeError, ValueError):
            yield event.plain_result("周期长度应在21至35天之间")
            return
        try:
            period_len = (
                self.config.get("default_period_length", 5)
                if period_len is None
                else int(period_len)
            )
        except (TypeError, ValueError):
            yield event.plain_result("经期长度应在2至10天之间")
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
            "source": "manual",
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
        if not self.config.get("mood_system_enabled", False):
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
        if not self.config.get("mood_system_enabled", False):
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
        if not self.config.get("mood_system_enabled", False):
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
        if not cfg or not cfg.get("enabled", True) or "anchor_date" not in cfg:
            return

        # NOTE: Do NOT auto-persist global defaults here.
        # Persisting would freeze the current default values for this session,
        # making it immune to future global default changes (BUG #1).
        # Sessions using global defaults appear in WebUI via _webapi_list_sessions.
        pass

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

        # Save original system prompt before injecting our content
        # so mood detector sees the bot's persona without our additions
        original_system_prompt = req.system_prompt or ""

        # Anchor is static content — inject into system_prompt on every request.
        # (Previously only once via _anchored_sessions; but req.system_prompt
        # is a fresh object each round, so the anchor was lost after round 1.)
        anchor = self.prompt_builder.get_anchor()
        req.system_prompt = original_system_prompt + ("\n\n" if original_system_prompt else "") + anchor
        logger.debug(
            "[PeriodPlugin][umo=%s] 锚点已注入 system_prompt, 长度=%d",
            umo, len(anchor),
        )

        # Track session for WebUI listing (sessions using global defaults)
        if umo not in self._anchored_sessions:
            self._anchored_sessions.add(umo)

        # Dynamic state: choose injection location based on config
        hour = datetime.datetime.now().hour
        dynamic = self.prompt_builder.build_dynamic(info.phase, info.day, hour)
        location = self.config.get("inject_location", "extra_user_content_parts")
        logger.info(
            "[PeriodPlugin][umo=%s] 动态状态注入位置: %s",
            umo, location,
        )

        if location == "system_prompt_append":
            req.system_prompt += "\n\n" + dynamic
            logger.debug(
                "[PeriodPlugin][umo=%s] 动态状态追加到 system_prompt, 长度=%d",
                umo, len(dynamic),
            )
        elif location == "user_message_before":
            req.prompt = dynamic + "\n\n" + (req.prompt or "")
            logger.debug(
                "[PeriodPlugin][umo=%s] 动态状态前置到用户消息, 长度=%d",
                umo, len(dynamic),
            )
        elif location == "fake_tool_call":
            provider = self.context.get_using_provider(umo)
            provider_type = ""
            if provider and hasattr(provider, "provider_config"):
                cfg = provider.provider_config
                provider_type = cfg.get("type", "") if isinstance(cfg, dict) else ""
            if provider_type == "googlegenai_chat_completion":
                logger.info(
                    "[PeriodPlugin][umo=%s] fake_tool_call 降级为 user_message_before (Gemini)",
                    umo,
                )
                req.prompt = dynamic + "\n\n" + (req.prompt or "")
            else:
                import uuid
                call_id = f"period_query_{uuid.uuid4().hex[:8]}"
                logger.debug(
                    "[PeriodPlugin][umo=%s] 伪造工具调用注入, call_id=%s",
                    umo, call_id,
                )
                req.contexts.extend([
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "query_period_status", "arguments": "{}"}
                        }]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": dynamic
                    }
                ])
        else:  # extra_user_content_parts
            req.extra_user_content_parts.append(
                TextPart(text=dynamic).mark_as_temp()
            )
            logger.debug(
                "[PeriodPlugin][umo=%s] 动态状态追加到 extra_user_content_parts, 长度=%d",
                umo, len(dynamic),
            )

        # ============================================================== #
        #  Mood / Emotion System
        # ============================================================== #
        if self.config.get("mood_system_enabled", False):
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
        """Execute the three-call mood detection architecture.

        Call 1 (screen):  Small model decides if intervention is needed.
        Call 2 (consult): Small model DMs the main model for a decision.
        Call 3 (interpret): Small model parses the main model's reply into tool calls.
        """
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo

        lock = await self._get_mood_lock(mood_umo)
        async with lock:
            mood_state = await self.mood_store.get(mood_umo) or MoodState()
            logger.info(
                "[PeriodPlugin][umo=%s] 情绪状态: 活跃工具=%s个",
                mood_umo, len(mood_state.active_tools),
            )

            # Expire old tools
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            expired = mood_state.expire_tools(now_iso)
            if expired:
                logger.info(
                    "[PeriodPlugin][umo=%s] 到期工具清理: %s",
                    mood_umo, [t["name"] for t in expired],
                )

            # Inject non-intercepting tool prompts (single-use)
            has_injection = self._inject_mood_prompts(mood_state, req)
            if has_injection:
                logger.debug("[PeriodPlugin][umo=%s] 已注入情绪工具提示词", mood_umo)

            history = self._extract_history(req)

            # Respect user preference: don't pass system prompt to mood detector if disabled
            system_prompt = (
                original_system_prompt
                if self.config.get("mood_detector_read_system_prompt", True)
                else ""
            )

            # ---------- Call 1: Screen ----------
            logger.info("[PeriodPlugin][umo=%s] 调用① 小模型筛选...", mood_umo)
            try:
                screen_result = await self.mood_detector.screen(
                    umo,
                    phase_info,
                    mood_state,
                    history,
                    event.message_str or "",
                    system_prompt,
                )
                need = screen_result.get("need_intervention", False)
                logger.info(
                    "[PeriodPlugin][umo=%s] 筛选结果: need=%s, reason=%s",
                    mood_umo, need, screen_result.get("reasoning", ""),
                )
            except Exception as e:
                logger.warning(
                    "[PeriodPlugin][umo=%s] 筛选调用失败: %s", mood_umo, e, exc_info=True,
                )
                await self.mood_store.set(mood_umo, mood_state)
                return

            if not need:
                await self.mood_store.set(mood_umo, mood_state)
                return

            # ---------- Call 2: Consult main model ----------
            logger.info("[PeriodPlugin][umo=%s] 调用② 主模型决策...", mood_umo)
            try:
                main_reply = await self.mood_detector.consult_main_model(
                    umo,
                    phase_info,
                    mood_state,
                    history,
                    event.message_str or "",
                    system_prompt,
                )
                logger.info(
                    "[PeriodPlugin][umo=%s] 主模型回复: %s",
                    mood_umo, main_reply[:200],
                )
                logger.debug(
                    "[PeriodPlugin][umo=%s] 主模型完整回复: %s",
                    mood_umo, main_reply,
                )
            except Exception as e:
                logger.warning(
                    "[PeriodPlugin][umo=%s] 主模型决策调用失败: %s", mood_umo, e, exc_info=True,
                )
                await self.mood_store.set(mood_umo, mood_state)
                return

            # ---------- Call 3: Interpret ----------
            logger.info("[PeriodPlugin][umo=%s] 调用③ 小模型理解...", mood_umo)
            try:
                interpret_result = await self.mood_detector.interpret(
                    umo, main_reply, mood_state.active_tools,
                )
                tool_name = interpret_result.get("tool_name")
                lift_tools = interpret_result.get("lift_tools", [])
                logger.info(
                    "[PeriodPlugin][umo=%s] 理解结果: tool=%s, lift=%s, reason=%s",
                    mood_umo,
                    tool_name,
                    lift_tools,
                    interpret_result.get("reasoning", ""),
                )
            except Exception as e:
                logger.warning(
                    "[PeriodPlugin][umo=%s] 理解调用失败: %s", mood_umo, e, exc_info=True,
                )
                await self.mood_store.set(mood_umo, mood_state)
                return

            # ---------- Execute ----------
            for lt in lift_tools:
                removed = mood_state.remove_tool(lt)
                if removed:
                    logger.info("[PeriodPlugin][umo=%s] 解除工具: %s", mood_umo, lt)

            if tool_name and self.config.get(f"enable_{tool_name}", True):
                params = self.mood_executor.validate_params(
                    tool_name, interpret_result.get("tool_params", {}),
                )
                expires_at = None
                rounds_left = None
                if tool_name == "cold_violence":
                    duration = params.get("duration", 30)
                    expires_at = (
                        datetime.datetime.now(datetime.timezone.utc)
                        + datetime.timedelta(minutes=duration)
                    ).isoformat()
                elif tool_name == "read_no_reply":
                    rounds_left = params.get("rounds", 3)

                mood_state.add_tool(
                    tool_name,
                    params,
                    expires_at=expires_at,
                    rounds_left=rounds_left,
                    initiated=False,
                )
                logger.info(
                    "[PeriodPlugin][umo=%s] 激活工具: %s, params=%s",
                    mood_umo, tool_name, params,
                )

            mood_state.add_history(
                event=f"intervention:yes,tool:{tool_name or 'none'}",
                reasoning=interpret_result.get("reasoning", ""),
                user_message=(event.message_str or "")[:200],
                max_length=self.config.get("mood_history_length", 20),
            )
            mood_state.last_interaction = now_iso
            await self.mood_store.set(mood_umo, mood_state)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """Handle OOC shield and intercepting mood tools."""
        umo = event.unified_msg_origin

        # ---- Mood system interception (cold_violence / read_no_reply) ----
        if self.config.get("mood_system_enabled", False):
            scope = self.config.get("mood_scope", "per_umo")
            mood_umo = "__global__" if scope == "global" else umo
            lock = await self._get_mood_lock(mood_umo)
            async with lock:
                mood_state = await self.mood_store.get(mood_umo)
                if mood_state and mood_state.active_tools:
                    for tool in list(mood_state.active_tools):
                        name = tool["name"]

                        if name == "cold_violence":
                            behavior = self.config.get("cold_violence_behavior", "angry_then_silent")
                            if behavior != "silent" and not tool.get("initiated"):
                                msg = self.mood_executor.get_initial_message(behavior, "")
                                if msg:
                                    try:
                                        await event.send(MessageChain([Comp.Plain(msg)]))
                                    except Exception as e:
                                        logger.warning(
                                            "[PeriodPlugin] 冷暴力初始消息发送失败: %s", e,
                                        )
                                tool["initiated"] = True
                                await self.mood_store.set(mood_umo, mood_state)

                            resp.completion_text = ""
                            if resp.result_chain:
                                resp.result_chain.chain.clear()
                            logger.info(
                                "[PeriodPlugin][umo=%s] cold_violence 拦截生效，丢弃回复",
                                mood_umo,
                            )
                            return

                        if name == "read_no_reply":
                            remaining = tool.get("rounds_left", 0)
                            if remaining <= 0:
                                logger.info(
                                    "[PeriodPlugin][umo=%s] 已读不回轮数耗尽，解除", mood_umo,
                                )
                                mood_state.remove_tool("read_no_reply")
                                await self.mood_store.set(mood_umo, mood_state)
                            else:
                                tool["rounds_left"] = remaining - 1
                                resp.completion_text = ""
                                if resp.result_chain:
                                    resp.result_chain.chain.clear()
                                logger.info(
                                    "[PeriodPlugin][umo=%s] read_no_reply 拦截生效，剩余%d轮",
                                    mood_umo, tool["rounds_left"],
                                )
                                await self.mood_store.set(mood_umo, mood_state)
                            return

        # ---- OOC Shield ----
        cfg = await self._get_session_config(umo)
        if not cfg or not cfg.get("enabled", True):
            return
        if not self.config.get("ooc_shield", True):
            return

        forbidden = self.config.get(
            "forbidden_words",
            ["月经", "经期", "激素", "雌激素", "孕激素", "黄体", "卵泡", "卵巢", "子宫", "内分泌", "PMS", "生理期", "排卵期", "安全期"],
        )

        if resp.result_chain:
            # WR-7 fix: Only replace in Plain components, preserve others (Image, At, etc.)
            hit_any = False
            for comp in resp.result_chain.chain:
                if isinstance(comp, Comp.Plain):
                    text = comp.text or ""
                    hit = [w for w in forbidden if w in text]
                    if hit:
                        hit_any = True
                        logger.warning(
                            "[PeriodPlugin] OOC检测命中: umo=%s, 命中词=%s", umo, hit,
                        )
                        if self.config.get("ooc_replace", False):
                            for w in hit:
                                text = text.replace(w, "*" * len(w))
                            comp.text = text
            if hit_any and self.config.get("ooc_replace", False):
                logger.info("[PeriodPlugin] OOC词汇已在 Plain 组件中替换为星号")
        else:
            text = resp.completion_text or ""
            hit = [w for w in forbidden if w in text]
            if hit:
                logger.warning(
                    "[PeriodPlugin] OOC检测命中: umo=%s, 命中词=%s", umo, hit,
                )
                if self.config.get("ooc_replace", False):
                    for w in hit:
                        text = text.replace(w, "*" * len(w))
                    resp.completion_text = text
                    logger.info("[PeriodPlugin] OOC词汇已替换为星号")

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        """AstrBot lifecycle hook: called when the bot is fully loaded."""
        if self.config.get("prompt_compression_enabled", False):
            if self.config.get("prompt_compression_auto_trigger", True):
                logger.info("[PeriodPlugin] 提示词压缩已启用，将在后台自动压缩...")
                self._compression_task = asyncio.create_task(self._auto_compress_prompts())

    async def _get_mood_lock(self, mood_umo: str) -> asyncio.Lock:
        """Get or create an asyncio.Lock for the given mood UMO (WR-2 fix).

        Uses _mood_locks_lock to prevent race conditions when multiple
        coroutines request a lock for the same UMO simultaneously.
        """
        async with self._mood_locks_lock:
            lock = self._mood_locks.get(mood_umo)
            if lock is None:
                lock = asyncio.Lock()
                self._mood_locks[mood_umo] = lock
            return lock

    async def terminate(self):
        """Clean up resources on plugin unload."""
        if self._compression_task and not self._compression_task.done():
            self._compression_task.cancel()
            try:
                await self._compression_task
            except asyncio.CancelledError:
                pass
        self._anchored_sessions.clear()
        self._inject_counters.clear()
        self._warmup_counters.clear()
        self._mood_locks.clear()
        logger.info("[PeriodPlugin] 插件已卸载，内存缓存已清理")
