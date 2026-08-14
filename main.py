"""AstrBot plugin for physiological cycle simulation."""

import asyncio
import datetime
import hashlib
import json
import uuid
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
from .core.mood import (
    RECOVERY_RETENTION_MESSAGES,
    MoodState,
    PersistentAction,
    RequestMoodDecision,
    utc_now_iso,
)
from .core.mood_store import MoodStore
from .core.mood_tools import MoodToolExecutor
from .core.mood_detector import MoodDetector
from .core.mood_journal import DiaryJournal
from .core.mood_lookup import add_diary_lookup_tool, build_diary_lookup_tool
from .core.mood_context import (
    apply_injection,
    history_to_contexts,
    is_umo_allowed,
    normalize_inject_location,
    parse_history,
    should_show_body_hint,
)
from .core.diagnostics import DiagnosticsStore


class PeriodPlugin(Star):
    """Plugin that simulates physiological cycles for female-persona bots."""

    name = "astrbot_plugin_period"

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self.plugin_data_dir = StarTools.get_data_dir("astrbot_plugin_period")
        self.engine = CycleEngine()
        self.store = CycleStore(self.plugin_data_dir)
        self.diagnostics = DiagnosticsStore(self.config, self.plugin_data_dir)

        # Prompt compression
        self.prompt_compressor = PromptCompressor(self.context, self.config, self.plugin_data_dir)
        self.prompt_builder = PromptBuilder(self.config, self.prompt_compressor)

        self._anchored_sessions: set[str] = set()  # Sessions visited (for WebUI tracking)
        self._inject_counters: dict[str, int] = {}  # Interval injection counters
        self._warmup_counters: dict[str, int] = {}  # Warmup round counters

        # Mood / emotion system
        self.mood_store = MoodStore(self.plugin_data_dir)
        self.mood_store.on_migration = self._on_mood_migration
        self.mood_detector = MoodDetector(self.context, self.config)
        self.mood_executor = MoodToolExecutor()
        self._mood_locks: dict[str, asyncio.Lock] = {}
        self._mood_locks_lock = asyncio.Lock()  # Protects _mood_locks dict (WR-2)
        # fake_tool_call 等废弃注入位置的降级诊断（每位置只记一次）
        self._location_downgrade_notified: set[str] = set()
        # 第三方 Runner / 无会话环境的跳过诊断限频（umo -> 上次记录的 monotonic 时间）
        self._runner_skip_notified: dict[str, float] = {}
        # 日记身份字段缺失的限频诊断（key -> monotonic 时间）
        self._diary_identity_notified: dict[str, float] = {}

        # 情绪日记系统（异步 outbox，不阻塞主链）
        self.diary_journal = DiaryJournal(
            self.plugin_data_dir,
            self._resolve_diary_provider,
            max_chars=self.config.get("diary_max_chars", 4000),
            config_getter=lambda key, default: self.config.get(key, default),
            enabled_getter=self._diary_active,
            umo_active_getter=self._umo_cycle_active,
        )

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
            if cmd_name in ("status", "mood"):
                return True, ""
            return False, "当前仅允许查看状态，设置类指令已被关闭，如需调整请前往插件配置修改指令权限控制"
        return True, ""

    async def _record_diagnostic_warning(
        self,
        title: str,
        message: str,
        *,
        source: str = "runtime",
        context: dict[str, Any] | None = None,
    ) -> None:
        """安全记录 warning 诊断，不影响主流程。"""
        try:
            await self.diagnostics.record_warning(
                title,
                message,
                source=source,
                context=context,
            )
        except Exception as e:
            logger.debug(f"[PeriodPlugin] 记录 warning 诊断失败: {e}")

    async def _record_diagnostic_error(
        self,
        title: str,
        error: BaseException | str,
        *,
        source: str = "runtime",
        context: dict[str, Any] | None = None,
    ) -> None:
        """安全记录 error 诊断，不影响主流程。"""
        try:
            await self.diagnostics.record_error(
                title,
                error,
                source=source,
                context=context,
            )
        except Exception as e:
            logger.debug(f"[PeriodPlugin] 记录 error 诊断失败: {e}")

    def _record_diagnostic_warning_background(
        self,
        title: str,
        message: str,
        *,
        source: str = "runtime",
        context: dict[str, Any] | None = None,
    ) -> None:
        """在同步路径中后台记录 warning 诊断。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._record_diagnostic_warning(
                title,
                message,
                source=source,
                context=context,
            )
        )

    def _record_diagnostic_error_background(
        self,
        title: str,
        error: BaseException | str,
        *,
        source: str = "runtime",
        context: dict[str, Any] | None = None,
    ) -> None:
        """在同步路径中后台记录 error 诊断。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._record_diagnostic_error(
                title,
                error,
                source=source,
                context=context,
            )
        )

    def _sync_diagnostics_config(self) -> None:
        """同步诊断运行态配置。"""
        try:
            self.diagnostics.max_entries = max(
                20,
                int(self.config.get("diagnostics_max_entries", 200)),
            )
        except (TypeError, ValueError):
            self.diagnostics.max_entries = 200

    def _safe_umo_hash(self, umo: str) -> str:
        """生成稳定短哈希，避免诊断日志保存原始 UMO。"""
        return hashlib.sha256(str(umo).encode("utf-8")).hexdigest()[:12]

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
            self._record_diagnostic_warning_background(
                "读取配置 schema 失败",
                e,
                source="config.schema",
            )
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
            self._record_diagnostic_warning_background(
                "读取 provider 列表失败",
                e,
                source="config.providers",
            )
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
                self._record_diagnostic_warning_background(
                    "解析 provider 信息失败",
                    e,
                    source="config.providers",
                    context={"provider_class": provider.__class__.__name__},
                )
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
            self._sync_diagnostics_config()
            return False, "当前配置对象不支持 save_config，已仅更新运行态配置"

        # AstrBotConfig.save_config 会 update 自身；这里再同步一次，兼容不同实现。
        self.config.clear()
        self.config.update(next_config)
        self._sync_diagnostics_config()
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
            f"{base}/diagnostics/summary",
            self._webapi_get_diagnostics_summary,
            ["GET"],
            "Get plugin diagnostics summary",
        )
        self.context.register_web_api(
            f"{base}/diagnostics",
            self._webapi_get_diagnostics,
            ["GET"],
            "List plugin diagnostics events",
        )
        self.context.register_web_api(
            f"{base}/diagnostics/read",
            self._webapi_mark_diagnostics_read,
            ["POST"],
            "Mark plugin diagnostics as read",
        )
        self.context.register_web_api(
            f"{base}/diagnostics/clear",
            self._webapi_clear_diagnostics,
            ["POST"],
            "Clear plugin diagnostics events",
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

    def _should_hide_session_in_dashboard(self, umo: str) -> bool:
        """判断某个会话是否只应从 WebUI 会话列表隐藏。"""
        umo = str(umo)
        return umo == "webchat" or umo.startswith(("webchat!", "webchat:"))

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
            # 规范记录优先：别名是旧数据，只能补齐规范记录缺失的字段，
            # 不得把新锚点/开关/快进天数覆盖回旧值；重试迁移时同样安全
            next_cfg = self._deep_merge_dicts(alias_cfg, existing or {})
            # 先成功写入目标再删别名、搬迁运行态：写入失败保留原 UMO
            # 返回（目标未持久化，返回规范 UMO 会造成假 404/重复记录）
            if await self.store.set(normalized, next_cfg):
                if await self.store.delete(umo):
                    logger.info(
                        f"[PeriodPlugin] 迁移编码后的 WebUI 会话记录: {umo} -> {normalized}"
                    )
                else:
                    logger.warning(
                        f"[PeriodPlugin] 编码会话别名删除落盘失败（下次将重试迁移）: {umo}"
                    )
            else:
                logger.warning(
                    f"[PeriodPlugin] 编码会话迁移落盘失败（下次将重试）: {umo} -> {normalized}"
                )
                return umo

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
            try:
                await self._migrate_encoded_session_alias(umo)
            except Exception as e:
                logger.warning(f"[PeriodPlugin] 迁移编码 UMO 会话失败: {e}")
                await self._record_diagnostic_error(
                    "迁移编码 UMO 会话失败",
                    e,
                    source="sessions.migrate_encoded_alias",
                    context={"encoded": "%" in umo},
                )

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
            self._record_diagnostic_warning_background(
                "序列化会话周期失败",
                e,
                source="sessions.serialize",
                context={
                    "source": cfg.get("source", "manual"),
                    "cycle_length": cfg.get("cycle_length", 28),
                    "period_length": cfg.get("period_length", 5),
                },
            )
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
            if await self.store.set(umo, migrated):
                logger.info(f"[PeriodPlugin] 迁移旧版全局默认会话记录: {umo}")
            else:
                # 运行态照常使用迁移结果；落盘失败时记录仍是旧版，
                # 下次加载会重新识别并迁移（幂等）
                logger.warning(
                    f"[PeriodPlugin] 旧版全局默认会话迁移落盘失败（运行态生效，将重试）: {umo}"
                )
        return migrated

    async def _webapi_list_sessions(self):
        """GET /astrbot_plugin_period/sessions"""
        try:
            await self._migrate_encoded_session_aliases()
            all_data = await self.store.get_all()
            seen: set[str] = set()
            sessions = []
            # 1) Explicitly configured sessions (from persistent store)
            for umo, cfg in all_data.items():
                if self._should_hide_session_in_dashboard(umo):
                    continue
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
                if self._should_hide_session_in_dashboard(umo):
                    continue
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
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 查询 WebUI 会话列表失败: {e}")
            await self._record_diagnostic_error(
                "查询 WebUI 会话列表失败",
                e,
                source="dashboard.sessions",
            )
            return jsonify({"status": "error", "message": f"查询会话失败: {e}"}), 500

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
            await self._record_diagnostic_error(
                "保存 WebUI 配置失败",
                e,
                source="dashboard.config.save",
            )
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

    async def _webapi_get_diagnostics_summary(self):
        """GET /astrbot_plugin_period/diagnostics/summary"""
        try:
            summary = await self.diagnostics.get_summary()
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 获取诊断摘要失败: {e}")
            return jsonify({"status": "error", "message": f"获取诊断摘要失败: {e}"}), 500
        return jsonify({"status": "ok", "data": summary})

    async def _webapi_get_diagnostics(self):
        """GET /astrbot_plugin_period/diagnostics"""
        args = getattr(request, "args", {}) or {}
        try:
            limit = int(args.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        unread_raw = str(args.get("unread_only", "")).lower()
        unread_only = unread_raw in ("1", "true", "yes")
        try:
            events = await self.diagnostics.list_events(
                limit=limit,
                unread_only=unread_only,
            )
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 获取诊断记录失败: {e}")
            return jsonify({"status": "error", "message": f"获取诊断记录失败: {e}"}), 500
        return jsonify({"status": "ok", "data": {"events": events}})

    async def _webapi_mark_diagnostics_read(self):
        """POST /astrbot_plugin_period/diagnostics/read"""
        body = await request.get_json()
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify({"status": "error", "message": "请求体必须是对象"}), 400
        ids = body.get("ids")
        if ids is not None and not isinstance(ids, list):
            return jsonify({"status": "error", "message": "ids 必须是列表"}), 400
        if isinstance(ids, list) and not all(isinstance(item, str) for item in ids):
            return jsonify({"status": "error", "message": "ids 必须是字符串列表"}), 400
        try:
            count, saved = await self.diagnostics.mark_read(ids=ids)
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 标记诊断已读失败: {e}")
            return jsonify({"status": "error", "message": f"标记诊断已读失败: {e}"}), 500
        if not saved:
            return jsonify({"status": "error", "message": "诊断已读状态保存失败"}), 500
        return jsonify({"status": "ok", "message": "已标记为已读", "data": {"marked": count}})

    async def _webapi_clear_diagnostics(self):
        """POST /astrbot_plugin_period/diagnostics/clear"""
        try:
            count, saved = await self.diagnostics.clear()
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 清空诊断记录失败: {e}")
            return jsonify({"status": "error", "message": f"清空诊断记录失败: {e}"}), 500
        if not saved:
            return jsonify({"status": "error", "message": "诊断记录清空失败"}), 500
        return jsonify({"status": "ok", "message": "诊断记录已清空", "data": {"cleared": count}})

    async def _webapi_toggle_session(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/toggle"""
        umo = await self._migrate_encoded_session_alias(umo)
        cfg = await self._get_session_config(umo)
        if not cfg or "anchor_date" not in cfg:
            return jsonify({"status": "error", "message": "会话未配置周期参数"}), 404

        if not await self.store.get(umo):
            if not await self.store.set(umo, cfg):
                return jsonify({
                    "status": "error",
                    "message": "会话配置保存失败（磁盘写入异常），请重试",
                }), 500

        new_state, persisted = await self.store.toggle(umo)
        if not persisted:
            return jsonify({
                "status": "error",
                "message": "状态切换保存失败（磁盘写入异常），请重试",
            }), 500
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
            if not await self.store.set(umo, cfg):
                return jsonify({
                    "status": "error",
                    "message": "会话配置保存失败（磁盘写入异常），请重试",
                }), 500

        cfg = await self.store.get(umo)
        cfg["advance_days"] = cfg.get("advance_days", 0) + days
        if not await self.store.set(umo, cfg):
            return jsonify({
                "status": "error",
                "message": "快进保存失败（磁盘写入异常），请重试",
            }), 500
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
            if not await self.store.set(umo, cfg):
                return jsonify({
                    "status": "error",
                    "message": "会话配置保存失败（磁盘写入异常），请重试",
                }), 500

        cfg = await self.store.get(umo)
        cfg["anchor_date"] = date_str
        cfg["advance_days"] = 0
        if cfg.get("source") == "global_default":
            cfg["anchor_overridden"] = True
        if not await self.store.set(umo, cfg):
            return jsonify({
                "status": "error",
                "message": "锚点保存失败（磁盘写入异常），请重试",
            }), 500
        cfg = await self._get_session_config(umo)

        self._anchored_sessions.discard(umo)
        self._inject_counters.pop(umo, None)
        self._warmup_counters.pop(umo, None)

        return jsonify({"status": "ok", "data": self._serialize_session(umo, cfg)})

    async def _webapi_delete_session(self, umo: str):
        """POST /astrbot_plugin_period/sessions/<umo>/delete

        周期记录、情绪状态、待处理日记事件是三个独立的持久化动作，
        不能假设一起成功：逐项执行并返回组合结果，任一项失败返回 500
        与明细。清理不依赖周期记录存在——周期已删但情绪/日记清理失败
        时，重试本接口仍能对残留数据继续清理（不会因 404 卡死）。

        作用域语义：global 模式下全局情绪是跨会话共享状态，既不是
        "该 UMO 存在"的证据，也不随单个会话删除而清除（清全局情绪
        用 /period moodreset）；per_umo 模式下情绪属于该会话自身，
        残留时重试必须可达。

        顺序语义：日记清理（含丢弃水位线）最先执行——一旦发起删除，
        该会话的旧日记事件立即失效，即使周期记录随后删除失败也不
        恢复；水位线落盘失败时周期记录仍在，重试本接口可继续。
        """
        umo = await self._migrate_encoded_session_alias(umo)
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        cycle_exists = await self.store.get(umo) is not None
        pending_exists = any(
            e.get("umo") == umo
            for e in await self.diary_journal.store.pending_events()
        )
        # global 模式下全局情绪与"该 UMO 是否存在"无关，不参与判据
        mood_exists = (
            scope != "global"
            and await self.mood_store.get(mood_umo) is not None
        )
        if not cycle_exists and not mood_exists and not pending_exists:
            return jsonify({"status": "error", "message": "会话不存在"}), 404

        # True 表示"已不存在或已成功清理/无需清理"
        results = {
            "umo": umo,
            "cycle_deleted": not cycle_exists,
            "mood_deleted": not mood_exists,
            "diary_pending_cleared": not pending_exists,
        }
        failures: list[str] = []

        # 情绪锁覆盖整个删除流程（per_umo 与 global 都取；global 取
        # __global__ 锁，不删情绪但必须在锁内完成清理）：在途请求的日记
        # 事件提交发生在同一把锁内（_run_mood_locked 末尾 flush）——要么
        # 在锁前完成、随即被下面的 discard 按 UMO 清掉，要么在锁后被
        # 源头复查挡掉，消除"水位线后、周期删除前提交"的窗口。锁序：
        # 情绪锁 → owner 提交锁 → 存储锁，与请求路径同序，无反转。
        # 不得 pop 锁对象：被弹出的锁若正被请求持有，下一个请求会新建
        # 一把锁，同一 mood_umo 出现两把锁并行，互斥失效。
        lock = await self._get_mood_lock(mood_umo)
        async with lock:
            # 1) 日记清理（含持久化丢弃水位线）先行：与 /period reset
            # 同款，丢弃该会话来源的待处理日记事件；已提交日记保留
            # （owner 跨 UMO 共享，可能还有其他有效会话来源）。
            # discard 返回 -1 表示落盘失败，不得假报成功。
            try:
                discarded = await self.diary_journal.discard_pending_for_umo(umo)
                if discarded < 0:
                    failures.append("待处理日记事件清理失败（磁盘写入异常）")
                else:
                    results["diary_pending_cleared"] = True
            except Exception as e:
                logger.warning(
                    "[PeriodPlugin] 删除会话后清理待处理日记事件失败: %s",
                    type(e).__name__,
                )
                failures.append("待处理日记事件清理失败")

            # 水位线落盘失败时立即中止：周期/情绪记录都还在，重试本接口
            # 可从头完成（若继续删除周期，重试会因记录不存在而无法再补
            # 写水位线，晚到的旧周期事件失去拦截）
            if failures:
                return jsonify({
                    "status": "error",
                    "message": "；".join(failures) + "。尚未执行其他删除，请重试本接口",
                    "data": results,
                }), 500

            # 2) 周期记录删除（事务化 Store：失败不污染缓存，可重试）。
            # 运行态计数器只在删除成功（或本就不存在）后清理——失败时
            # 周期记录仍在，预热/间隔状态不得被提前改变
            if cycle_exists:
                if await self.store.delete(umo):
                    results["cycle_deleted"] = True
                else:
                    failures.append("周期记录删除失败（磁盘写入异常）")
            if results["cycle_deleted"]:
                self._anchored_sessions.discard(umo)
                self._inject_counters.pop(umo, None)
                self._warmup_counters.pop(umo, None)

            # 3) 情绪状态清理（仅 per_umo；global 共享情绪由 moodreset
            # 管理）。已在情绪锁内，直接重新检查：进入本流程后可能有
            # 进行中的请求在锁前首次写入情绪状态，按流程入口的旧快照
            # 跳过清理会谎报"无需清理"。
            if scope != "global":
                if await self.mood_store.get(mood_umo) is None:
                    results["mood_deleted"] = True
                else:
                    deleted = await self.mood_store.delete(mood_umo)
                    if deleted:
                        results["mood_deleted"] = True
                    else:
                        failures.append("情绪状态清理失败（磁盘写入异常）")

        if failures:
            return jsonify({
                "status": "error",
                "message": "；".join(failures) + "。已成功的部分不会回滚，请重试本接口完成剩余清理",
                "data": results,
            }), 500
        data = {**results, "deleted": True}
        if scope == "global":
            data["note"] = "全局情绪状态为跨会话共享，已保留；如需清除请使用 /period moodreset"
        return jsonify({"status": "ok", "data": data})

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
            await self._record_diagnostic_error(
                "后台提示词压缩失败",
                e,
                source="prompt_compression.auto",
            )

    def _snapshot_user_turn(self, req: ProviderRequest) -> str:
        """在进入本插件注入流程前保存原始用户轮次快照。

        供硬沉默路径单独写入会话历史：剥离所有 _no_save 临时内容。
        """
        parts: list[str] = []
        if req.prompt:
            parts.append(req.prompt)
        for part in getattr(req, "extra_user_content_parts", None) or []:
            if getattr(part, "_no_save", False):
                continue
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                if part.get("_no_save"):
                    continue
                text = part.get("text")
            if text:
                parts.append(str(text))
        return "\n".join(parts)

    def _safe_inject_location(self, config_key: str) -> str:
        """读取注入位置配置并规整；废弃位置降级并记一次诊断。"""
        raw = self.config.get(config_key, "extra_user_content_parts")
        location, downgraded = normalize_inject_location(raw)
        if downgraded and config_key not in self._location_downgrade_notified:
            self._location_downgrade_notified.add(config_key)
            logger.warning(
                "[PeriodPlugin] 注入位置 %s=%r 已废弃或未知，降级为 %s",
                config_key, raw, location,
            )
            self._record_diagnostic_warning_background(
                "注入位置已降级",
                f"配置项 {config_key} 的值 {raw!r} 不再支持（fake_tool_call 已移除），"
                f"已自动降级为 {location}",
                source="mood.inject_location",
                context={"config_key": config_key, "fallback": location},
            )
        return location

    async def _get_effective_session(self, umo: str) -> dict | None:
        """周期有效门禁：总开关 + UMO 范围 + 会话启用 + 锚点存在。

        情绪系统与身体提示共用此门禁；warmup/inject_mode/关键词不在这里。
        """
        if not self.config.get("auto_inject", True):
            return None
        if not is_umo_allowed(self.config, umo):
            return None
        cfg = await self._get_session_config(umo)
        if not cfg or not cfg.get("enabled", True) or "anchor_date" not in cfg:
            return None
        return cfg

    async def _on_mood_migration(self, umo: str, notes: list[str]) -> None:
        """moods.json 旧数据迁移诊断回调。"""
        await self._record_diagnostic_warning(
            "旧情绪数据已迁移",
            f"迁移说明: {', '.join(notes)}",
            source="mood.migration",
            context={"mood_umo_hash": self._safe_umo_hash(umo), "notes": list(notes)},
        )

    async def _notify_runner_skip(self, umo: str) -> None:
        """第三方 Runner / 无会话环境的跳过诊断（每 UMO 每小时至多一条）。"""
        now = asyncio.get_running_loop().time()
        last = self._runner_skip_notified.get(umo, 0.0)
        if now - last < 3600:
            return
        self._runner_skip_notified[umo] = now
        await self._record_diagnostic_warning(
            "情绪系统已跳过：无正式会话",
            "当前请求缺少会话对象（可能是第三方 Agent Runner 或历史缺失），"
            "情绪与日记功能已按设计跳过",
            source="mood.runner_skip",
            context={"umo_hash": self._safe_umo_hash(umo)},
        )

    # ------------------------------------------------------------------ #
    #  日记系统
    # ------------------------------------------------------------------ #

    def _resolve_diary_provider(self, captured_provider_id: str):
        """日记模型回退链：diary_provider_id → mood_detector_provider_id → 入队时捕获的主 Provider。"""
        get_by_id = getattr(self.context, "get_provider_by_id", None)
        if get_by_id is None:
            return None
        for pid in (
            self.config.get("diary_provider_id", ""),
            self.config.get("mood_detector_provider_id", ""),
            captured_provider_id,
        ):
            if pid:
                provider = get_by_id(pid)
                if provider is not None:
                    return provider
        return None

    def _capture_main_provider_id(self, event: AstrMessageEvent) -> str:
        """事件入队时捕获本轮实际 Provider 的 ID（与②同源的 selected_provider/
        会话偏好解析），保证日记模型回退链末位与本轮正式模型一致。"""
        provider = self._resolve_round_provider(event, event.unified_msg_origin)
        cfg = getattr(provider, "provider_config", None) if provider else None
        if isinstance(cfg, dict):
            return str(cfg.get("id", "") or "")
        return ""

    async def _diary_owner_key(self, event: AstrMessageEvent) -> str | None:
        """platform_id + bot_self_id + sender_id；缺失时记限频诊断，禁止退化。"""
        try:
            platform_id = event.get_platform_id() or ""
            self_id = event.get_self_id() or ""
            sender_id = event.get_sender_id() or ""
        except Exception:
            platform_id = self_id = sender_id = ""
        key = DiaryJournal.make_owner_key(platform_id, self_id, sender_id)
        if key is None:
            notify_key = f"{platform_id}:{self_id}:{sender_id}"
            now = asyncio.get_running_loop().time()
            if now - self._diary_identity_notified.get(notify_key, 0.0) >= 3600:
                self._diary_identity_notified[notify_key] = now
                await self._record_diagnostic_warning(
                    "日记已跳过：身份字段缺失",
                    "platform_id、bot_self_id 或 sender_id 缺失，无法确定日记归属",
                    source="diary.identity",
                    context={"umo_hash": self._safe_umo_hash(event.unified_msg_origin)},
                )
        return key

    def _diary_active(self) -> bool:
        """日记只有在周期总开关、情绪总开关与 diary_enabled 同时满足时才工作。

        情绪是周期附属功能：auto_inject 关闭时情绪与日记同步停止
        （outbox 事件延后处理，重开自动恢复）。
        """
        return bool(
            self.config.get("auto_inject", True)
            and self.config.get("mood_system_enabled", False)
            and self.config.get("diary_enabled", True)
        )

    async def _umo_cycle_active(self, umo: str) -> bool:
        """日记 worker 的会话级周期门控：与请求链路的周期有效门禁同源。

        来源会话的周期失效（toggle、删除会话、白名单变更等）时，该会话
        滞留的日记事件延后处理，重新有效后自动恢复。与请求链路一样
        要求周期可正常计算（engine.get_phase 不抛异常）；任何判定异常
        都按失效处理（保守方向：不多写一条来源已失效的日记）。
        """
        try:
            cfg = await self._get_effective_session(umo)
            if cfg is None:
                return False
            # 与请求门禁同严：损坏的锚点/周期参数视为周期失效
            self.engine.get_phase(
                cfg["anchor_date"],
                cfg.get("cycle_length", 28),
                cfg.get("period_length", 5),
                cfg.get("ovulation_day", 14),
                cfg.get("ovulation_window", 3),
                cfg.get("advance_days", 0),
            )
            return True
        except Exception:
            return False

    async def _get_diary_text(self, event: AstrMessageEvent) -> str:
        """读取当前用户已提交日记的注入文本；无日记时返回空串。"""
        if not self._diary_active():
            return ""
        owner_key = await self._diary_owner_key(event)
        if owner_key is None:
            return ""
        diary = await self.diary_journal.store.get_diary(owner_key)
        # 读取侧实时上限：用户调低 diary_max_chars 后、后台 worker 裁剪
        # 完成前，超限旧日记不得完整进入模型上下文
        return self.diary_journal.build_injection_text(
            diary, max_chars=self.config.get("diary_max_chars", 4000),
        )

    async def _flush_diary_events(
        self, event: AstrMessageEvent, pending: list[tuple[str, dict]],
    ) -> None:
        """状态落盘成功后统一下发本轮收集的日记事件。"""
        for kind, payload in pending:
            await self._emit_diary_event(event, kind, payload)

    async def _emit_diary_event(self, event: AstrMessageEvent, kind: str, payload: dict) -> None:
        """产生脱敏日记事件（持久化 outbox 后异步处理，不阻塞主链）。"""
        if not self._diary_active():
            return
        # 事件发生时间在校验前捕获：水位线按事件实际发生时间判定过期，
        # 覆盖"复查通过 → reset → submit"的竞态窗口（reset 的持久化
        # 水位线早于该 occurred_at 时，enqueue 会拒绝入队）
        occurred_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # 源头闭窗：周期可能在请求/指令处理中途被 reset/toggle，入队前
        # 复查来源会话有效性——失效周期产生的事件不进 outbox
        if not await self._umo_cycle_active(event.unified_msg_origin):
            return
        owner_key = await self._diary_owner_key(event)
        if owner_key is None:
            return
        summary = self._format_diary_summary(kind, payload)
        if not summary:
            return
        try:
            display_name = event.get_sender_name() or ""
        except Exception:
            display_name = ""
        try:
            submitted = await self.diary_journal.submit(
                owner_key,
                kind,
                summary,
                display_name=display_name,
                provider_id=self._capture_main_provider_id(event),
                umo=event.unified_msg_origin,
                occurred_at=occurred_at,
            )
        except Exception as e:
            # 日记是附属功能：任何入队/落盘故障都不得影响主请求链；
            # 日志与诊断只记异常类型——自定义 Provider/存储异常可能
            # 在消息里回显摘要、提示词等请求信息
            logger.warning(
                "[PeriodPlugin] 日记事件入队失败: %s", type(e).__name__,
            )
            await self._record_diagnostic_error(
                "日记事件入队失败", type(e).__name__,
                source="diary.submit",
                context={"umo_hash": self._safe_umo_hash(event.unified_msg_origin)},
            )
            return
        if not submitted:
            # outbox 落盘失败或事件重复：事件静默丢失是不可接受的，
            # 如实记录诊断（不抛异常，不影响主链）
            logger.warning("[PeriodPlugin] 日记事件未能持久化到 outbox，已丢弃")
            await self._record_diagnostic_error(
                "日记事件入队失败", "outbox 落盘失败或事件重复",
                source="diary.submit",
                context={"umo_hash": self._safe_umo_hash(event.unified_msg_origin)},
            )

    @staticmethod
    def _format_diary_summary(kind: str, payload: dict) -> str:
        """把事件载荷格式化为脱敏摘要（不含聊天原文/人格/②原回答）。"""
        if kind == "mood_changed":
            text = (
                f"心境变为「{payload.get('summary', '')}」"
                f"（类别 {payload.get('cause_category', 'neutral')}，状态 {payload.get('status', '')}）"
            )
            if payload.get("latest_reason"):
                text += f"；原因：{payload['latest_reason']}"
            if payload.get("status") in ("active", "recovering"):
                text += f"；是否好转：{'是' if payload.get('improved') else '否'}"
            return text
        if kind == "action_activated":
            return (
                f"决定执行动作 {payload.get('action', '')}"
                f"（参数 {json.dumps(payload.get('params', {}), ensure_ascii=False)}，"
                f"方式 {payload.get('silence_mode', '')}）：{payload.get('reasoning', '')}"
            )
        if kind == "action_lifted":
            return f"解除动作 {payload.get('action', '')}：{payload.get('reasoning', '')}"
        if kind == "action_expired":
            return f"动作 {payload.get('action', '')} 到期自动解除（{payload.get('reason', '')}）"
        if kind == "fully_recovered":
            return f"完全恢复：{payload.get('recovery_reason', '')}"
        if kind == "manual_lift":
            actions = payload.get("actions") or []
            if actions:
                return f"用户手动解除了动作：{', '.join(actions)}"
            return "用户手动解除了当前情绪状态"
        return ""

    def _maybe_inject_diary_lookup_tool(
        self, event: AstrMessageEvent, req: ProviderRequest,
    ) -> None:
        """跨人只读检索：仅显式开启、日记可用且为内部 Agent 时注入。"""
        if not self.config.get("diary_cross_user_lookup_enabled", False):
            return
        if not self._diary_active():
            return
        # 第三方 Runner（无会话）不开放：内部 Agent 总是设置 req.conversation，
        # 第三方 Runner 不设置；不能用全局 conversation_manager 做判据。
        if getattr(req, "conversation", None) is None:
            return
        try:
            platform_id = event.get_platform_id() or ""
            self_id = event.get_self_id() or ""
        except Exception:
            return
        if not platform_id or not self_id:
            return
        try:
            max_chars = int(self.config.get("diary_cross_user_max_chars", 800))
        except (TypeError, ValueError):
            max_chars = 800
        tool = build_diary_lookup_tool(
            self.diary_journal.store, platform_id, self_id, max_chars,
        )
        if tool is None:
            if "diary_lookup" not in self._location_downgrade_notified:
                self._location_downgrade_notified.add("diary_lookup")
                self._record_diagnostic_warning_background(
                    "跨人日记检索不可用",
                    "当前 AstrBot 版本不支持自定义 FunctionTool，diary_cross_user_lookup 未生效",
                    source="diary.lookup",
                    context={"umo_hash": self._safe_umo_hash(event.unified_msg_origin)},
                )
            return
        add_diary_lookup_tool(req, tool)

    def _resolve_round_provider(self, event: AstrMessageEvent, umo: str):
        """解析本轮实际使用的 Provider：优先事件级 selected_provider
        （用户临时选择的模型），回退会话偏好 Provider。

        注意：图片能力回退发生在 OnLLMRequestEvent 之后，插件无法预知，
        这里是钩子时刻的最佳近似。
        """
        try:
            sel = event.get_extra("selected_provider")
        except Exception:
            sel = None
        if sel and isinstance(sel, str):
            get_by_id = getattr(self.context, "get_provider_by_id", None)
            provider = get_by_id(sel) if get_by_id else None
            if provider is not None:
                return provider
        try:
            return self.context.get_using_provider(umo)
        except Exception:
            return None

    def _sanitize_decision_text(
        self,
        decision: RequestMoodDecision,
        user_message: str,
        protected_texts: list[str] | None = None,
    ) -> None:
        """隐私兜底：脱敏字段若照抄用户原话（滑动窗口包含检测）则清空该字段。

        检测语料 = 当前消息 + 传入②的全部历史消息（小模型可能照抄历史原文）。
        脱敏主要依赖③的提示词约束，这里是代码侧的最后防线；
        命中只清空对应字段并记日志，不作废整份决策。
        """
        corpus = [t for t in (protected_texts or []) if t and t.strip()]
        if user_message and user_message.strip():
            corpus.append(user_message.strip())

        def leaked(fragment: str) -> bool:
            fragment = (fragment or "").strip()
            if not fragment or not corpus:
                return False
            window = 8
            for msg in corpus:
                if len(msg) <= window:
                    if msg in fragment:
                        return True
                    continue
                for i in range(len(msg) - window + 1):
                    if msg[i:i + window] in fragment:
                        return True
            return False

        if decision.mood_update:
            for key in ("summary", "latest_reason", "recovery_reason"):
                if leaked(decision.mood_update.get(key, "")):
                    decision.mood_update[key] = ""
                    logger.info("[PeriodPlugin] 脱敏字段 %s 含用户原话，已清空", key)
        if leaked(decision.reasoning_summary):
            decision.reasoning_summary = ""
            logger.info("[PeriodPlugin] reasoning_summary 含用户原话，已清空")

    async def _get_mood_status_text(self, umo: str) -> str:
        """Generate human-readable mood status for a session (v3)."""
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        mood_state = await self.mood_store.get(mood_umo) or MoodState()

        status_labels = {
            "stable": "平稳",
            "active": "波动中",
            "recovering": "好转中",
            "recovered": "已恢复",
        }
        lines = [f"情绪状态：{status_labels.get(mood_state.status, mood_state.status)}"]
        if mood_state.summary:
            lines.append(f"心境：{mood_state.summary}")
        if mood_state.latest_reason:
            lines.append(f"最近原因：{mood_state.latest_reason}")
        if mood_state.status in ("active", "recovering"):
            lines.append(f"是否好转：{'是' if mood_state.improved else '否'}")
        if mood_state.fully_recovered and mood_state.recovery_reason:
            lines.append(f"恢复原因：{mood_state.recovery_reason}")

        if mood_state.persistent_actions:
            actions_info = []
            for a in mood_state.persistent_actions:
                if a.name == "cold_violence":
                    actions_info.append(f"冷暴力(至{a.expires_at})")
                elif a.name == "read_no_reply":
                    actions_info.append(f"已读不回(剩余{a.remaining_replies}条)")
            lines.append(f"生效动作：{', '.join(actions_info)}")
        else:
            lines.append("生效动作：无")

        if mood_state.history:
            last = mood_state.history[-1]
            lines.append(f"最近事件：{last.get('event', '无')}")
            lines.append(f"摘要：{last.get('reasoning', '无')}")

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
        if not await self.store.set(umo, data):
            yield event.plain_result("周期参数保存失败（磁盘写入异常），请检查磁盘后重试")
            return

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
            if not await self.store.set(umo, cfg):
                yield event.plain_result("会话配置保存失败（磁盘写入异常），请检查磁盘后重试")
                return

        new_state, persisted = await self.store.toggle(umo)
        if not persisted:
            yield event.plain_result("状态切换保存失败（磁盘写入异常），请检查磁盘后重试")
            return
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
            if not await self.store.set(umo, cfg):
                yield event.plain_result("会话配置保存失败（磁盘写入异常），请检查磁盘后重试")
                return

        cfg = await self.store.get(umo)
        cfg["advance_days"] = cfg.get("advance_days", 0) + days
        if not await self.store.set(umo, cfg):
            yield event.plain_result("快进保存失败（磁盘写入异常），请检查磁盘后重试")
            return

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
        # 删除失败时中止：周期记录仍有效，不得继续丢弃日记事件或报成功
        if not await self.store.delete(umo):
            yield event.plain_result("周期数据删除失败（磁盘写入异常），请检查磁盘后重试")
            return
        self._anchored_sessions.discard(umo)
        self._inject_counters.pop(umo, None)
        self._warmup_counters.pop(umo, None)
        # 周期已失效：该会话失效前滞留的日记事件不再处理（已提交日记保留）。
        # 事件按来源 UMO 丢弃，其他会话产生的同 owner 事件不受影响。
        diary_cleanup_failed = False
        if self.diary_journal is not None:
            try:
                diary_cleanup_failed = (
                    await self.diary_journal.discard_pending_for_umo(umo) < 0
                )
            except Exception as e:
                diary_cleanup_failed = True
                logger.warning(
                    "[PeriodPlugin] 重置后清理待处理日记事件失败: %s",
                    type(e).__name__,
                )
        logger.info(f"[PeriodPlugin] 用户 {umo} 重置周期数据")
        if diary_cleanup_failed:
            yield event.plain_result(
                "当前会话的周期数据已重置（待处理日记事件清理失败，请检查磁盘后重试）"
            )
        else:
            yield event.plain_result("当前会话的周期数据已重置")

    @period_group.command("mood")
    async def period_mood(self, event: AstrMessageEvent):
        """查看当前情绪状态 /period mood（开关关闭时也可查看保存的状态）"""
        allowed, msg = self._check_command_permission("mood")
        if not allowed:
            yield event.plain_result(msg)
            return
        umo = event.unified_msg_origin
        logger.info(f"[PeriodPlugin] 用户 {umo} 执行 period mood")
        text = await self._get_mood_status_text(umo)
        yield event.plain_result(text)

    @period_group.command("moodreset")
    @permission_type(PermissionType.ADMIN)
    async def period_mood_reset(self, event: AstrMessageEvent):
        """重置当前会话情绪状态 /period moodreset（仅管理员；只删情绪，不删日记）"""
        allowed, msg = self._check_command_permission("moodreset")
        if not allowed:
            yield event.plain_result(msg)
            return
        umo = event.unified_msg_origin
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        # 必须在情绪锁内删除：请求链持锁跑完三段后会 set 回状态，
        # 不拿锁的 delete 会被进行中的请求复活
        lock = await self._get_mood_lock(mood_umo)
        async with lock:
            deleted = await self.mood_store.delete(mood_umo)
        if not deleted:
            logger.warning(f"[PeriodPlugin] 用户 {umo} 重置情绪状态落盘失败")
            yield event.plain_result("情绪状态重置失败（磁盘写入异常），请重试 /period moodreset")
            return
        logger.info(f"[PeriodPlugin] 用户 {umo} 重置情绪状态")
        yield event.plain_result("当前会话的情绪状态已重置（日记不受影响）")

    @period_group.command("lift")
    async def period_lift(self, event: AstrMessageEvent):
        """手动解除冷暴力等动作 /period lift

        安全出口：绕过情绪开关与 commands_enabled，任何用户任何时候都可执行；
        global 模式下同样允许当前用户解除全局状态。
        """
        umo = event.unified_msg_origin
        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        lock = await self._get_mood_lock(mood_umo)
        async with lock:
            mood_state = await self.mood_store.get(mood_umo) or MoodState()
            lifted = [a.name for a in mood_state.persistent_actions]
            had_inner = mood_state.status != "stable" or bool(mood_state.summary)
            if not lifted and not had_inner:
                yield event.plain_result("当前没有需要解除的情绪状态")
                return
            if lifted:
                mood_state.persistent_actions.clear()
                mood_state.revision += 1
            # 安全出口同时退出内在情绪：无硬动作时也能把 active/recovering
            # 心境标记为手动恢复（恢复事件按保留计数继续注入后自动收尾）。
            if had_inner:
                now = utc_now_iso()
                mood_state.status = "recovered"
                mood_state.improved = True
                mood_state.fully_recovered = True
                mood_state.recovery_reason = "用户手动解除"
                mood_state.recovered_at = now
                mood_state.recovered_messages_left = RECOVERY_RETENTION_MESSAGES
                mood_state.changed_at = now
                mood_state.revision += 1
            mood_state.add_history(
                "lift:manual", "用户手动解除", self.config.get("mood_history_length", 20),
            )
            mood_state.last_interaction_at = utc_now_iso()
            persisted = await self.mood_store.set(mood_umo, mood_state)
        if not persisted:
            # 安全出口不得假成功：落盘失败时旧状态（含硬动作）仍在，
            # 如实告知用户重试，且不发 manual_lift 日记事件（那是谎话）
            logger.warning(
                "[PeriodPlugin] 用户 %s 手动解除落盘失败，状态未生效", umo,
            )
            yield event.plain_result("情绪状态保存失败（磁盘写入异常），请重试 /period lift")
            return
        await self._emit_diary_event(event, "manual_lift", {"actions": lifted})
        logger.info(
            "[PeriodPlugin] 用户 %s 手动解除情绪状态: 动作=%s, 内在情绪=%s",
            umo, lifted, had_inner,
        )
        if lifted and had_inner:
            yield event.plain_result("已解除所有情绪动作，情绪状态已标记为手动恢复")
        elif lifted:
            yield event.plain_result("已解除所有情绪动作")
        else:
            yield event.plain_result("当前情绪已标记为手动恢复")

    @period_group.command("diary")
    @permission_type(PermissionType.ADMIN)
    async def period_diary(self, event: AstrMessageEvent):
        """查看指定 QQ 号的情绪日记 /period diary <QQ号>（仅管理员）"""
        allowed, msg = self._check_command_permission("diary")
        if not allowed:
            yield event.plain_result(msg)
            return
        target = (event.message_str or "").split("diary")[-1].strip()
        if not target:
            yield event.plain_result("用法：/period diary <QQ号>")
            return
        owner_key = DiaryJournal.make_owner_key(
            event.get_platform_id() or "", event.get_self_id() or "", target,
        )
        diary = await self.diary_journal.store.get_diary(owner_key) if owner_key else None
        if not diary or not diary.get("entries"):
            yield event.plain_result(f"没有找到 {target} 的情绪日记")
            return
        lines = [f"{diary.get('display_name') or target} 的情绪日记（{len(diary['entries'])} 条）："]
        for e in diary["entries"]:
            lines.append(f"- {e.get('text', '')}")
        yield event.plain_result("\n".join(lines))

    @period_group.command("diaryclear")
    @permission_type(PermissionType.ADMIN)
    async def period_diary_clear(self, event: AstrMessageEvent):
        """清除指定 QQ 号的情绪日记 /period diaryclear <QQ号>（仅管理员）"""
        allowed, msg = self._check_command_permission("diaryclear")
        if not allowed:
            yield event.plain_result(msg)
            return
        target = (event.message_str or "").split("diaryclear")[-1].strip()
        if not target:
            yield event.plain_result("用法：/period diaryclear <QQ号>")
            return
        owner_key = DiaryJournal.make_owner_key(
            event.get_platform_id() or "", event.get_self_id() or "", target,
        )
        if not owner_key:
            yield event.plain_result("无法确定日记归属（平台或机器人身份缺失）")
            return
        removed_diary, removed_events, persisted = await self.diary_journal.clear_owner(owner_key)
        if not persisted:
            # 落盘失败时磁盘与缓存均未变，不得谎报已清除
            yield event.plain_result("清除失败：日记数据写入磁盘异常，请检查磁盘后重试")
            return
        if removed_diary:
            logger.info(f"[PeriodPlugin] 管理员清除了 {target} 的情绪日记")
            yield event.plain_result(
                f"已清除 {target} 的情绪日记（含 {removed_events} 条待处理事件）"
            )
        elif removed_events:
            logger.info(f"[PeriodPlugin] 管理员清除了 {target} 的 {removed_events} 条待处理日记事件")
            yield event.plain_result(
                f"{target} 没有已保存的日记，已清理 {removed_events} 条待处理事件"
            )
        else:
            yield event.plain_result(f"没有找到 {target} 的情绪日记")

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
            await self._record_diagnostic_error(
                "手动提示词压缩失败",
                e,
                source="prompt_compression.manual",
                context={"umo_hash": self._safe_umo_hash(event.unified_msg_origin)},
            )
            yield event.plain_result(f"压缩失败: {e}")

    # ------------------------------------------------------------------ #
    #  LLM Hooks
    # ------------------------------------------------------------------ #

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Inject physiological state and run the mood system."""
        umo = event.unified_msg_origin

        # 0. 原始用户轮次快照（未经本插件注入，供硬沉默路径写历史）
        snapshot = self._snapshot_user_turn(req)

        # 1. 周期有效门禁（情绪系统与身体提示共用：总开关/UMO/会话/锚点）
        cfg = await self._get_effective_session(umo)
        if cfg is None:
            return

        # 2. 周期计算（不可正常计算视为门禁不过）
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
            logger.warning("[PeriodPlugin][umo=%s] 周期计算失败: %s", umo, e)
            await self._record_diagnostic_error(
                "周期计算失败",
                e,
                source="engine.get_phase",
                context={"umo_hash": self._safe_umo_hash(umo)},
            )
            return

        # NOTE: Do NOT auto-persist global defaults here.
        # Persisting would freeze the current default values for this session,
        # making it immune to future global default changes (BUG #1).
        # Sessions using global defaults appear in WebUI via _webapi_list_sessions.

        # Save original system prompt before injecting our content
        # so the mood consult call sees the bot's persona without our additions
        original_system_prompt = req.system_prompt or ""

        # 3. 情绪系统：只依赖周期有效门禁，与 warmup/inject_mode/关键词解耦
        if self.config.get("mood_system_enabled", False):
            handled = await self._run_mood_system(
                event, req, umo, info, original_system_prompt, snapshot,
            )
            if handled:
                return  # 硬沉默已接管（不产生正式回复），或等锁期间周期已失效：本轮不再注入

        # 3.5 跨人只读日记检索工具（显式开启 + 仅内部 Agent）
        self._maybe_inject_diary_lookup_tool(event, req)

        # 4. 身体状态提示的展示门禁（warmup/频率/关键词只控制展示）
        if not should_show_body_hint(
            self.config,
            umo,
            event.message_str or "",
            self._warmup_counters,
            self._inject_counters,
        ):
            return

        # 5. Anchor is static content — inject into system_prompt on every request.
        # (req.system_prompt is a fresh object each round, so the anchor would
        # be lost after round 1 if only injected once.)
        # 注意：必须基于当前 req.system_prompt 追加，不能用 original_system_prompt
        # 重建——情绪/日记若选择 system_prompt_append 位置，重建会把它们整体抹掉。
        anchor = self.prompt_builder.get_anchor()
        current_system_prompt = req.system_prompt or ""
        req.system_prompt = current_system_prompt + ("\n\n" if current_system_prompt else "") + anchor
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
        location = self._safe_inject_location("inject_location")
        logger.info(
            "[PeriodPlugin][umo=%s] 动态状态注入位置: %s",
            umo, location,
        )
        apply_injection(req, dynamic, location)
        logger.debug(
            "[PeriodPlugin][umo=%s] 动态状态已注入, 长度=%d",
            umo, len(dynamic),
        )

    async def _run_mood_system(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        umo: str,
        phase_info,
        original_system_prompt: str,
        snapshot: str,
    ) -> bool:
        """三段式情绪流程（请求级决策，本轮决定本轮生效）。

        返回 True 表示本轮不再继续插件处理：硬沉默已接管（事件已停止），
        或等待情绪锁期间周期已失效（锁内二次门禁不过）——两种情况下
        on_llm_request 都直接返回，不得再注入身体提示。
        """
        # 第三方 Runner / 无会话环境：跳过情绪与日记并限频诊断。
        # AstrBot 内部 Agent 自 v3.4 起总是设置 req.conversation；第三方 Runner
        # 用裸 ProviderRequest 触发钩子（conversation 为 None），而全局 Context
        # 恒有 conversation_manager，不能用后者做判据。
        conversation = getattr(req, "conversation", None)
        if conversation is None:
            await self._notify_runner_skip(umo)
            return False

        scope = self.config.get("mood_scope", "per_umo")
        mood_umo = "__global__" if scope == "global" else umo
        request_id = uuid.uuid4().hex[:12]

        lock = await self._get_mood_lock(mood_umo)
        async with lock:
            # 锁内先查情绪总开关：同一 mood_umo 的请求在锁上串行，排队
            # 期间管理员可能已关闭情绪系统（配置即时生效）。关闭即跳过
            # 情绪流程——返回 False 让身体周期提示照常注入（关闭的是
            # 情绪系统，不是周期系统）；已有硬动作与心境状态保留但
            # 立即停止拦截，lift 仍可解除。
            if not self.config.get("mood_system_enabled", False):
                logger.info(
                    "[PeriodPlugin][umo=%s] 等待情绪锁期间情绪系统已关闭，跳过情绪流程",
                    umo,
                )
                return False
            # 锁内二次门禁：排队等锁期间周期可能已被 reset/删除/toggle
            # （删除会话流程在同一把锁内执行清理）。若失效，不得用过期
            # phase_info 写情绪/日记，并告知外层直接结束本轮请求——
            # 身体提示也不得再注入（周期已不存在）
            if not await self._umo_cycle_active(umo):
                logger.info(
                    "[PeriodPlugin][umo=%s] 等待情绪锁期间周期已失效，本轮不再处理",
                    umo,
                )
                return True
            return await self._run_mood_locked(
                event, req, umo, mood_umo, phase_info,
                original_system_prompt, snapshot, request_id,
            )

    async def _run_mood_locked(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        umo: str,
        mood_umo: str,
        phase_info,
        original_system_prompt: str,
        snapshot: str,
        request_id: str,
    ) -> bool:
        mood_state = await self.mood_store.get(mood_umo) or MoodState()
        now_iso = utc_now_iso()
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        max_history = self.config.get("mood_history_length", 20)
        # 本轮产生的日记事件先收集，状态 set 成功后才统一下发：
        # 落盘失败时不得让日记记录一笔实际没保存的心境/动作变化
        pending_diary_events: list[tuple[str, dict]] = []

        # ---- 1. 到期/耗尽清理（产生日记事件）----
        for action in mood_state.expire_actions(now_iso):
            logger.info("[PeriodPlugin][umo=%s] 动作到期解除: %s", mood_umo, action.name)
            mood_state.add_history(f"expire:{action.name}", "时间到期自动解除", max_history)
            pending_diary_events.append(("action_expired", {
                "action": action.name, "reason": "timeout",
            }))
        stale = mood_state.get_action("read_no_reply")
        if stale is not None and (stale.remaining_replies or 0) <= 0:
            mood_state.remove_action("read_no_reply")
            logger.info("[PeriodPlugin][umo=%s] 已读不回轮数耗尽解除", mood_umo)
            mood_state.add_history("expire:read_no_reply", "轮数耗尽自动解除", max_history)
            pending_diary_events.append(("action_expired", {
                "action": "read_no_reply", "reason": "rounds_exhausted",
            }))

        hard = mood_state.persistent_actions[0] if mood_state.persistent_actions else None

        # ---- 2. 规范化历史与已提交日记 ----
        history = parse_history(
            req.contexts,
            self.config.get("mood_consult_history_messages", 30),
        )
        diary_text = await self._get_diary_text(event)
        # 当前消息以 req.prompt 为准（AstrBot 装配后的本轮用户文本）
        user_message = req.prompt or event.message_str or ""

        # ---- 3. 调用①：三段架构不可跳过。已有硬动作时照常筛选，
        # 结果为否也由宿主强制进入②③（每轮都要问主模型是否解除）。
        persona_summary = (
            original_system_prompt
            if self.config.get("mood_detector_read_system_prompt", True)
            else ""
        )
        screen = await self.mood_detector.screen(
            umo, phase_info, mood_state, history, user_message, persona_summary,
        )
        # reasoning 是小模型生成文本，可能照抄用户原话：只记录结果标志，不记内容
        logger.info(
            "[PeriodPlugin][umo=%s] 筛选结果: need=%s, failed=%s",
            mood_umo, screen.get("need_intervention"), bool(screen.get("failed")),
        )

        # ①失败走保守策略：有硬动作按原规则沉默（不进②③，不给解除机会）；
        # 无硬动作正常放行（不激活任何新动作）
        if screen.get("failed"):
            await self._record_diagnostic_warning(
                "情绪筛选调用失败",
                "调用①失败，本轮按保守策略处理",
                source="mood.screen",
                context={"mood_umo_hash": self._safe_umo_hash(mood_umo)},
            )
            if hard is not None:
                return await self._apply_silence(
                    event, req, umo, mood_umo, mood_state, hard, snapshot, max_history,
                    pending_diary_events,
                )
            await self._finalize_normal_round(
                event, req, mood_state, mood_umo, diary_text,
                pending_diary_events=pending_diary_events,
            )
            return False

        if not screen.get("need_intervention", False):
            if hard is None:
                await self._finalize_normal_round(
                    event, req, mood_state, mood_umo, diary_text,
                    pending_diary_events=pending_diary_events,
                )
                return False
            logger.info(
                "[PeriodPlugin][umo=%s] 已有硬动作 %s，筛选为否仍继续决策",
                mood_umo, hard.name,
            )

        # ---- 4. 调用②：当前人格主模型自然语言决策 ----
        # Provider 绑定本轮实际选择（selected_provider/会话偏好），
        # 避免用户临时选模型时情绪决策与正式回复由不同模型完成。
        round_provider = self._resolve_round_provider(event, umo)
        main_reply = await self.mood_detector.consult_main_model(
            umo,
            phase_info,
            mood_state,
            history_to_contexts(history),
            user_message,
            original_system_prompt,
            diary_text,
            model=getattr(req, "model", None),
            provider=round_provider,
        )
        if not main_reply.strip():
            # 保守：不更新心境、不激活新动作；已有硬状态按原规则处理本轮
            logger.info("[PeriodPlugin][umo=%s] 主模型未给出决策，保持原状态", mood_umo)
            await self._record_diagnostic_warning(
                "情绪系统主模型未给出决策",
                "调用②返回为空或失败，本轮保持原状态",
                source="mood.consult",
                context={"mood_umo_hash": self._safe_umo_hash(mood_umo)},
            )
            if hard is not None:
                return await self._apply_silence(
                    event, req, umo, mood_umo, mood_state, hard, snapshot, max_history,
                    pending_diary_events,
                )
            await self._finalize_normal_round(
                event, req, mood_state, mood_umo, diary_text,
                pending_diary_events=pending_diary_events,
            )
            return False

        # ---- 5. 调用③：翻译为严格 JSON 并校验 ----
        decision = await self.mood_detector.interpret(umo, main_reply, mood_state)
        if not decision.valid:
            await self._record_diagnostic_warning(
                "情绪决策校验失败",
                f"原因: {decision.reject_reason}",
                source="mood.interpret",
                context={"mood_umo_hash": self._safe_umo_hash(mood_umo)},
            )
            if hard is not None:
                return await self._apply_silence(
                    event, req, umo, mood_umo, mood_state, hard, snapshot, max_history,
                    pending_diary_events,
                )
            await self._finalize_normal_round(
                event, req, mood_state, mood_umo, diary_text,
                pending_diary_events=pending_diary_events,
            )
            return False

        # ---- 6. 原子提交：先解除，再心境，后激活 ----
        # 提交前做隐私兜底：脱敏字段不得照抄用户原话（当前消息+历史）
        self._sanitize_decision_text(
            decision, user_message, [h["content"] for h in history],
        )
        for name in decision.lift_actions:
            removed = mood_state.remove_action(name)
            if removed is not None:
                logger.info("[PeriodPlugin][umo=%s] 解除动作: %s", mood_umo, name)
                mood_state.add_history(f"lift:{name}", decision.reasoning_summary, max_history)
                pending_diary_events.append(("action_lifted", {
                    "action": name, "reasoning": decision.reasoning_summary,
                }))
        if hard is not None and mood_state.get_action(hard.name) is None:
            hard = None  # 本轮已解除

        newly_recovered = False
        if decision.mood_update:
            was_fully_recovered = mood_state.fully_recovered
            if mood_state.apply_mood_update(decision.mood_update):
                pending_diary_events.append(("mood_changed", {
                    "status": mood_state.status,
                    "summary": mood_state.summary,
                    "cause_category": mood_state.cause_category,
                    "latest_reason": mood_state.latest_reason,
                    "improved": mood_state.improved,
                }))
                if mood_state.fully_recovered and not was_fully_recovered:
                    newly_recovered = True
                    pending_diary_events.append(("fully_recovered", {
                        "recovery_reason": mood_state.recovery_reason,
                    }))

        if decision.actions_rejected:
            logger.info(
                "[PeriodPlugin][umo=%s] 动作组被拒绝: %s",
                mood_umo, decision.reject_reason,
            )

        new_hard: PersistentAction | None = None
        if not decision.actions_rejected and decision.new_hard_actions:
            spec = decision.new_hard_actions[0]
            if spec["name"] == "cold_violence":
                expires_at = (
                    now_dt + datetime.timedelta(minutes=spec["params"].get("duration", 30))
                ).isoformat()
                candidate = PersistentAction.create(
                    "cold_violence", spec["params"],
                    expires_at=expires_at, request_id=request_id,
                )
            else:
                candidate = PersistentAction.create(
                    "read_no_reply", spec["params"],
                    remaining_replies=spec["params"].get("rounds", 3),
                    request_id=request_id,
                )
            # 硬动作冲突检查：
            # - 同名旧动作仍在生效 → 拒绝续期，保持原到期时间/剩余轮数
            #   （防止模型每轮重新激活把已读不回/冷暴力无限“续杯”）；
            # - 不同名旧动作未解除 → 互斥，拒绝激活。
            existing_same = mood_state.get_action(candidate.name)
            others = [a for a in mood_state.persistent_actions if a.name != candidate.name]
            if existing_same is not None or others:
                if existing_same is not None:
                    detail = (
                        f"同名动作 {candidate.name} 仍在生效，拒绝续期，"
                        f"保持原到期时间/剩余轮数"
                    )
                else:
                    detail = f"新动作 {candidate.name} 与未解除的 {others[0].name} 互斥"
                logger.info(
                    "[PeriodPlugin][umo=%s] 硬动作被拒绝: %s", mood_umo, detail,
                )
                await self._record_diagnostic_warning(
                    "硬动作冲突，已拒绝激活",
                    detail,
                    source="mood.interpret",
                    context={"mood_umo_hash": self._safe_umo_hash(mood_umo)},
                )
            else:
                new_hard = candidate
                mood_state.add_action(new_hard)
                mood_state.add_history(
                    f"action:{new_hard.name}", decision.reasoning_summary, max_history,
                )
                pending_diary_events.append(("action_activated", {
                    "action": new_hard.name,
                    "params": dict(new_hard.params),
                    "silence_mode": decision.silence_mode,
                    "reasoning": decision.reasoning_summary,
                }))

        soft_actions = [] if decision.actions_rejected else decision.new_soft_actions
        mood_state.last_interaction_at = now_iso

        # ---- 7. 分支 ----
        # 7a. 新激活硬动作 immediate → 本轮即沉默
        if new_hard is not None and decision.silence_mode == "immediate":
            return await self._apply_silence(
                event, req, umo, mood_umo, mood_state, new_hard, snapshot, max_history,
                pending_diary_events,
            )

        # 7b. 已有硬动作未解除 → 本轮继续沉默
        if new_hard is None and hard is not None:
            return await self._apply_silence(
                event, req, umo, mood_umo, mood_state, hard, snapshot, max_history,
                pending_diary_events,
            )

        # 7c. after_expression / 软动作 / 正常：注入倾向 + 状态 + 日记，放行
        tendency_lines: list[str] = []
        if new_hard is not None and decision.silence_mode == "after_expression":
            tendency_lines.append(
                "[情绪倾向] 你决定暂时不再回应对方。这一轮用你自己的人格方式，"
                "自然地说出最后一句（比如表达你的边界或此刻的感受），随后结束这次交流。"
            )
        for a in soft_actions:
            text = self.mood_executor.get_prompt_injection(a["name"], a["params"])
            if text:
                tendency_lines.append(text)
        await self._finalize_normal_round(
            event, req, mood_state, mood_umo, diary_text, tendency_lines,
            skip_recovered_tick=newly_recovered,
            pending_diary_events=pending_diary_events,
        )
        return False

    async def _apply_silence(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        umo: str,
        mood_umo: str,
        mood_state: MoodState,
        action: PersistentAction,
        snapshot: str,
        max_history: int,
        pending_diary_events: list[tuple[str, dict]] | None = None,
    ) -> bool:
        """硬沉默：只把用户消息写入会话历史一次，不产生助手回复，随后停止事件。

        已读不回只在实际拦截成功的有效请求上递减；历史写入失败仍保持沉默。
        本轮收集的日记事件只在状态落盘成功后下发。
        """
        if action.name == "read_no_reply" and action.remaining_replies is not None:
            action.remaining_replies = max(0, action.remaining_replies - 1)
            mood_state.revision += 1
        mood_state.add_history(f"silence:{action.name}", "本轮未回应", max_history)
        mood_state.last_interaction_at = utc_now_iso()
        try:
            persisted = await self.mood_store.set(mood_umo, mood_state)
        except Exception as e:
            # 状态落盘失败不得打断沉默：记诊断后继续 stop_event，
            # 否则异常被框架捕获后正式回复会意外发出
            persisted = False
            logger.warning(
                "[PeriodPlugin][umo=%s] 沉默轮状态落盘失败: %s",
                mood_umo, type(e).__name__,
            )
        if not persisted:
            # 动作激活/已读不回轮数递减未保存：当轮沉默照常（已 stop_event），
            # 但必须如实记录——下轮读回的仍是旧状态，轮数可能多于名义值；
            # 本轮收集的日记事件一并丢弃（状态没保存，事件就是谎话）
            logger.warning(
                "[PeriodPlugin][umo=%s] 沉默轮状态未持久化，动作计数/激活可能未生效",
                mood_umo,
            )
            await self._record_diagnostic_error(
                "沉默轮情绪状态落盘失败",
                "动作激活或轮数递减未保存，当轮沉默仍生效",
                source="mood.silence.persist_state",
                context={"umo_hash": self._safe_umo_hash(umo)},
            )
        elif pending_diary_events:
            await self._flush_diary_events(event, pending_diary_events)

        await self._persist_silenced_user_turn(event, req, umo, snapshot)
        logger.info(
            "[PeriodPlugin][umo=%s] %s 沉默生效，本轮不回应",
            mood_umo, action.name,
        )
        event.stop_event()
        return True

    async def _persist_silenced_user_turn(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        umo: str,
        snapshot: str,
    ) -> None:
        """沉默轮：仅把剥离临时内容后的原始用户消息追加到会话历史一次。"""
        if not snapshot.strip():
            return
        try:
            conv_mgr = getattr(self.context, "conversation_manager", None)
            conversation = getattr(req, "conversation", None)
            cid = getattr(conversation, "cid", None) if conversation else None

            history: list = []
            if conversation is not None and getattr(conversation, "history", None):
                history = json.loads(conversation.history or "[]")
            elif conv_mgr is not None:
                cid = cid or await conv_mgr.get_curr_conversation_id(umo)
                conv = await conv_mgr.get_conversation(umo, cid) if cid else None
                if conv is not None and getattr(conv, "history", None):
                    history = json.loads(conv.history or "[]")
            if not isinstance(history, list):
                history = []

            history.append({"role": "user", "content": snapshot})
            if conv_mgr is not None:
                await conv_mgr.update_conversation(
                    umo, conversation_id=cid, history=history,
                )
            elif conversation is not None:
                # 无管理器时只能更新内存对象（无法落库），记诊断
                conversation.history = json.dumps(history, ensure_ascii=False)
                raise RuntimeError("缺少 conversation_manager，历史仅写入内存对象")
        except Exception as e:
            # 不记录异常消息内容（数据库层异常理论上可能回显写入内容）
            logger.warning(
                "[PeriodPlugin][umo=%s] 沉默轮用户历史写入失败: %s",
                umo, type(e).__name__,
            )
            await self._record_diagnostic_error(
                "沉默轮用户历史写入失败",
                e,
                source="mood.silence.persist",
                context={"umo_hash": self._safe_umo_hash(umo)},
            )

    async def _finalize_normal_round(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        mood_state: MoodState,
        mood_umo: str,
        diary_text: str,
        tendency_lines: list[str] | None = None,
        skip_recovered_tick: bool = False,
        pending_diary_events: list[tuple[str, dict]] | None = None,
    ) -> None:
        """注入本轮软动作倾向 + 即时状态 + 日记，持久化状态后放行正常回复。

        skip_recovered_tick：完全恢复提交当轮不计恢复保留条数
        （恢复事件从下一条有效消息开始计 10 条）。
        本轮收集的日记事件只在状态落盘成功后下发。
        """
        if tendency_lines:
            apply_injection(
                req, "\n".join(tendency_lines), "extra_user_content_parts",
            )
        # 状态空白（stable 且无动作）也注入极简占位：每个有效请求都要
        # 让模型知道当前情绪状态
        state_text = mood_state.build_snapshot_text() or (
            "[内在情绪状态（即时快照，仅供你感知，不要直接向用户复述机制）]\n"
            "- 当前心境：平稳（无特殊情绪状态）"
        )
        apply_injection(
            req, state_text,
            self._safe_inject_location("mood_state_inject_location"),
        )
        if diary_text:
            apply_injection(
                req, diary_text,
                self._safe_inject_location("diary_inject_location"),
            )
        # 恢复保留计数：先注入（含恢复事件）再递减，归零后清理回 stable
        if not skip_recovered_tick:
            mood_state.tick_recovered()
        if not await self.mood_store.set(mood_umo, mood_state):
            # 落盘失败：本轮注入与回复照常，但状态变更（恢复计数递减等）
            # 未保存，如实记录而不是假装成功；本轮收集的日记事件一并丢弃
            # （状态没保存，事件就是谎话）
            logger.warning(
                "[PeriodPlugin][umo=%s] 情绪状态落盘失败，本轮状态变更未保存",
                mood_umo,
            )
            await self._record_diagnostic_error(
                "情绪状态落盘失败",
                "本轮状态变更未保存（恢复计数/心境更新可能未生效）",
                source="mood.finalize.persist_state",
                context={"umo_hash": self._safe_umo_hash(mood_umo)},
            )
        elif pending_diary_events:
            await self._flush_diary_events(event, pending_diary_events)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """Handle OOC shield.

        硬动作（冷暴力/已读不回）的拦截已移至请求侧（沉默不产生正式 LLM 调用，
        也不会把用户看不到的幽灵回复写入 AstrBot 历史），本钩子不再清空回复。
        """
        umo = event.unified_msg_origin

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
                        await self._record_diagnostic_warning(
                            "OOC 检测命中",
                            "回复中出现禁用词",
                            source="ooc.shield",
                            context={
                                "umo_hash": self._safe_umo_hash(umo),
                                "hit_count": len(hit),
                                "replace": self.config.get("ooc_replace", False),
                            },
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
                await self._record_diagnostic_warning(
                    "OOC 检测命中",
                    "回复中出现禁用词",
                    source="ooc.shield",
                    context={
                        "umo_hash": self._safe_umo_hash(umo),
                        "hit_count": len(hit),
                        "replace": self.config.get("ooc_replace", False),
                    },
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
        try:
            self._sync_diagnostics_config()
            await self.diagnostics.load()
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 加载诊断日志失败: {e}")
        try:
            await self.diary_journal.start()
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 恢复日记事件队列失败: {e}")
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
        try:
            await self.diary_journal.shutdown()  # outbox 已逐次落盘，仅取消 worker
        except Exception as e:
            logger.warning(f"[PeriodPlugin] 日记系统关闭异常: {e}")
        self._anchored_sessions.clear()
        self._inject_counters.clear()
        self._warmup_counters.clear()
        self._mood_locks.clear()
        logger.info("[PeriodPlugin] 插件已卸载，内存缓存已清理")
