"""Mood state model v3 - natural-language inner state and persistent hard actions.

v3 与旧版（无 schema_version）的差异：
- 保存自然语言内在心境（summary/cause_category/latest_reason/恢复字段），不再只有工具列表。
- 软动作不再持久化，只存在于当前请求的 RequestMoodDecision。
- 硬动作（冷暴力/已读不回）收进 persistent_actions，含到期时间或剩余轮数。
- history 不再保存用户消息原文，只保留脱敏事件摘要。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 3

STATUS_VALUES = ("stable", "active", "recovering", "recovered")
CAUSE_CATEGORIES = (
    "neutral", "neglect", "dismissive", "conflict",
    "insult", "apology", "care", "boundary", "other",
)

HARD_ACTIONS = ("cold_violence", "read_no_reply")
SOFT_ACTIONS = (
    "perfunctory_reply", "seek_comfort", "delayed_reply",
    "emotional_outburst", "topic_shift",
)
ALL_ACTIONS = HARD_ACTIONS + SOFT_ACTIONS

SILENCE_MODES = ("none", "immediate", "after_expression")

# 完全恢复后，恢复事件继续注入的有效消息条数
RECOVERY_RETENTION_MESSAGES = 10

# 脱敏摘要级字段长度上限，防止原文级内容落盘
MAX_SUMMARY_CHARS = 200
MAX_REASON_CHARS = 200
MAX_REASONING_CHARS = 200


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_utc(value: Any) -> datetime | None:
    """把 ISO-8601 字符串解析为 UTC aware datetime。

    无时区信息或格式非法一律返回 None——字符串比较对不同时区偏移不可靠，
    所有时间判断必须先过这里。
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def clamp_str(value: Any, max_len: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value[:max_len]


@dataclass
class PersistentAction:
    """跨请求硬动作。expires_at 与 remaining_replies 按动作类型二选一。"""

    id: str
    name: str
    params: dict
    created_at: str
    expires_at: str | None = None          # cold_violence: UTC ISO 到期时间
    remaining_replies: int | None = None   # read_no_reply: 剩余可拦截条数
    activation_request_id: str = ""

    @classmethod
    def create(
        cls,
        name: str,
        params: dict,
        *,
        expires_at: str | None = None,
        remaining_replies: int | None = None,
        request_id: str = "",
    ) -> "PersistentAction":
        return cls(
            id=uuid.uuid4().hex[:12],
            name=name,
            params=dict(params),
            created_at=utc_now_iso(),
            expires_at=expires_at,
            remaining_replies=remaining_replies,
            activation_request_id=request_id,
        )

    def is_expired(self, now_iso: str) -> bool:
        """仅时间型动作按 UTC 到期；轮数型不随时间过期。

        非法到期时间按已到期处理：宁可提前解除，也不让损坏数据导致
        冷暴力永不结束。
        """
        if not self.expires_at:
            return False
        exp = parse_iso_utc(self.expires_at)
        if exp is None:
            return True
        now = parse_iso_utc(now_iso)
        if now is None:
            return False
        return exp <= now

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "params": dict(self.params),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "remaining_replies": self.remaining_replies,
            "activation_request_id": self.activation_request_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersistentAction":
        # 损坏文件防御：任何字段类型异常都回退默认值，不得抛异常
        raw_params = data.get("params")
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            name=str(data.get("name", "")),
            params=dict(raw_params) if isinstance(raw_params, dict) else {},
            created_at=str(data.get("created_at") or utc_now_iso()),
            expires_at=data.get("expires_at") if isinstance(data.get("expires_at"), str) else None,
            remaining_replies=(
                data.get("remaining_replies")
                # bool 是 int 子类，True 不得被当成合法轮数
                if isinstance(data.get("remaining_replies"), int)
                and not isinstance(data.get("remaining_replies"), bool) else None
            ),
            activation_request_id=str(data.get("activation_request_id") or ""),
        )


@dataclass
class MoodState:
    """情绪内在状态 v3（自然语言快照 + 持久硬动作 + 脱敏事件历史）。"""

    schema_version: int = SCHEMA_VERSION
    revision: int = 1
    status: str = "stable"
    summary: str = ""
    cause_category: str = "neutral"
    latest_reason: str = ""
    improved: bool = False
    fully_recovered: bool = False
    recovery_reason: str = ""
    changed_at: str = ""
    recovered_at: str | None = None
    recovered_messages_left: int = 0
    persistent_actions: list[PersistentAction] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    # history item: {"timestamp": str, "event": str, "reasoning": str}（无用户消息原文）
    last_interaction_at: str = ""

    # ------------------------------------------------------------------ #
    #  序列化
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": self.revision,
            "status": self.status,
            "summary": self.summary,
            "cause_category": self.cause_category,
            "latest_reason": self.latest_reason,
            "improved": self.improved,
            "fully_recovered": self.fully_recovered,
            "recovery_reason": self.recovery_reason,
            "changed_at": self.changed_at,
            "recovered_at": self.recovered_at,
            "recovered_messages_left": self.recovered_messages_left,
            "persistent_actions": [a.to_dict() for a in self.persistent_actions],
            "history": list(self.history),
            "last_interaction_at": self.last_interaction_at,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "MoodState":
        """按 v3 结构反序列化；缺字段/类型异常回退默认值，永不抛异常。

        旧格式请走 migrate()。损坏的 moods.json（如 params 是字符串、
        history 是 null、记录本身不是对象）不得在请求钩子上抛异常。
        """
        if not isinstance(data, dict) or not data:
            return cls()
        status = data.get("status", "stable")
        if status not in STATUS_VALUES:
            status = "stable"
        cause = data.get("cause_category", "neutral")
        if cause not in CAUSE_CATEGORIES:
            cause = "neutral"
        left = data.get("recovered_messages_left", 0)
        # bool 是 int 子类，True/False 不得被当成合法计数
        if not isinstance(left, int) or isinstance(left, bool) or left < 0:
            left = 0
        revision = data.get("revision", 1)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            revision = 1
        raw_history = data.get("history")
        # 只保留白名单键：v3 history 由 add_history 写入，本不含用户原话；
        # 外部损坏/手改的数据不得把 user_message 等字段重新带回来
        history = (
            [
                {k: h[k] for k in ("timestamp", "event", "reasoning") if k in h}
                for h in raw_history if isinstance(h, dict)
            ]
            if isinstance(raw_history, list) else []
        )
        improved = data.get("improved")
        fully_recovered = data.get("fully_recovered")
        return cls(
            schema_version=SCHEMA_VERSION,
            revision=revision,
            status=status,
            summary=clamp_str(data.get("summary", ""), MAX_SUMMARY_CHARS),
            cause_category=cause,
            latest_reason=clamp_str(data.get("latest_reason", ""), MAX_REASON_CHARS),
            # 严格要求真 bool：bool("false") 会变成 True
            improved=improved if isinstance(improved, bool) else False,
            fully_recovered=fully_recovered if isinstance(fully_recovered, bool) else False,
            recovery_reason=clamp_str(data.get("recovery_reason", ""), MAX_REASON_CHARS),
            changed_at=str(data.get("changed_at") or ""),
            recovered_at=data.get("recovered_at") if isinstance(data.get("recovered_at"), str) else None,
            recovered_messages_left=left,
            persistent_actions=cls._sanitize_actions(data.get("persistent_actions")),
            history=history,
            last_interaction_at=str(data.get("last_interaction_at") or ""),
        )

    # ------------------------------------------------------------------ #
    #  旧版迁移（无 schema_version 的 v1 数据）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sanitize_actions(raw: Any) -> list["PersistentAction"]:
        """清洗持久硬动作：同名去重（保留第一个）、非法到期时间丢弃、
        两种硬动作并存时全部清除（不猜测用户意图）。"""
        if not isinstance(raw, list):
            return []
        actions = [
            PersistentAction.from_dict(a)
            for a in raw
            if isinstance(a, dict) and a.get("name") in HARD_ACTIONS
        ]
        seen: set[str] = set()
        cleaned: list[PersistentAction] = []
        for a in actions:
            if a.name in seen:
                continue
            if a.name == "cold_violence" and parse_iso_utc(a.expires_at) is None:
                continue
            seen.add(a.name)
            cleaned.append(a)
        if len({a.name for a in cleaned}) > 1:
            return []
        return cleaned

    @classmethod
    def migrate(cls, data: dict | None) -> tuple["MoodState", list[str]]:
        """把旧格式字典迁移到 v3。幂等：v3 数据原样返回，notes 为空。

        返回 (state, notes)；notes 供调用方记录诊断。
        规则（按方案）：删除 history.user_message；软工具不恢复；合法硬动作
        保留并补齐字段；非法时间丢弃；两个硬动作并存时两者都清除。
        """
        if not data:
            return cls(), []
        if data.get("schema_version") == SCHEMA_VERSION:
            state = cls.from_dict(data)
            notes: list[str] = []
            # 损坏文件防御：标量/字符串等非法容器不得让迭代抛 TypeError
            raw_pa = data.get("persistent_actions")
            raw_actions = [
                a for a in (raw_pa if isinstance(raw_pa, list) else [])
                if isinstance(a, dict) and a.get("name") in HARD_ACTIONS
            ]
            if not isinstance(raw_pa, list) and raw_pa is not None:
                notes.append("v3_invalid_actions_sanitized")
            if len(raw_actions) != len(state.persistent_actions):
                notes.append("v3_invalid_actions_sanitized")
            if len({a.get("name") for a in raw_actions}) > 1:
                notes.append("v3_dual_hard_actions_cleared")
            raw_hist = data.get("history")
            # 字段缺失（键不存在）不动；显式 null、非列表、含非 dict 条目、
            # 含额外键都视为脏数据，触发落盘重写（磁盘原文一并清除）
            history_dirty = "history" in data and (
                raw_hist is None
                or not isinstance(raw_hist, list)
                or any(
                    not isinstance(h, dict)
                    or set(h) - {"timestamp", "event", "reasoning"}
                    for h in raw_hist
                )
            )
            if history_dirty:
                # from_dict 已按白名单剥离（含 user_message、非对象条目等）；
                # 补 note 触发落盘重写：磁盘原文不得只因内存脱敏而保留
                notes.append("v3_history_extra_keys_removed")
            return state, notes

        notes: list[str] = []
        state = cls()

        raw_tools = data.get("active_tools")
        legacy_tools = [
            t for t in (raw_tools if isinstance(raw_tools, list) else [])
            if isinstance(t, dict) and t.get("name")
        ]
        hard = [t for t in legacy_tools if t["name"] in HARD_ACTIONS]
        if len({t["name"] for t in hard}) > 1:
            state.persistent_actions = []
            notes.append("legacy_dual_hard_actions_cleared")
        else:
            for t in hard:
                name = t["name"]
                params = t.get("params")
                if not isinstance(params, dict):
                    # 损坏文件防御：dict("bad") 会 ValueError 打穿全量加载
                    if params:
                        notes.append("legacy_action_params_sanitized")
                    params = {}
                if name == "cold_violence":
                    expires_at = t.get("expires_at")
                    if parse_iso_utc(expires_at) is None:
                        notes.append("legacy_cold_violence_invalid_time_dropped")
                        continue
                    state.persistent_actions.append(PersistentAction.create(
                        name, params, expires_at=expires_at,
                    ))
                elif name == "read_no_reply":
                    rounds_left = t.get("rounds_left")
                    # bool 是 int 子类，True 不得被当成合法轮数
                    if (not isinstance(rounds_left, int)
                            or isinstance(rounds_left, bool) or rounds_left < 0):
                        notes.append("legacy_read_no_reply_invalid_rounds_dropped")
                        continue
                    state.persistent_actions.append(PersistentAction.create(
                        name, params, remaining_replies=rounds_left,
                    ))

        raw_hist = data.get("history")
        legacy_history = raw_hist if isinstance(raw_hist, list) else []
        for h in legacy_history:
            if not isinstance(h, dict):
                continue
            state.history.append({
                "timestamp": str(h.get("timestamp") or ""),
                "event": clamp_str(h.get("event", ""), MAX_REASONING_CHARS),
                "reasoning": clamp_str(h.get("reasoning", ""), MAX_REASONING_CHARS),
            })
        if any("user_message" in h for h in legacy_history if isinstance(h, dict)):
            notes.append("legacy_history_user_message_removed")

        state.last_interaction_at = str(data.get("last_interaction") or "")
        state.status = "active" if state.persistent_actions else "stable"
        state.changed_at = utc_now_iso()
        notes.append("migrated_v1_to_v3")
        return state, notes

    # ------------------------------------------------------------------ #
    #  持久硬动作
    # ------------------------------------------------------------------ #

    def get_action(self, name: str) -> PersistentAction | None:
        for a in self.persistent_actions:
            if a.name == name:
                return a
        return None

    def has_hard_action(self) -> bool:
        return bool(self.persistent_actions)

    def add_action(self, action: PersistentAction) -> None:
        self.remove_action(action.name)
        self.persistent_actions.append(action)
        self.revision += 1

    def remove_action(self, name: str) -> PersistentAction | None:
        for i, a in enumerate(self.persistent_actions):
            if a.name == name:
                removed = self.persistent_actions.pop(i)
                self.revision += 1
                return removed
        return None

    def expire_actions(self, now_iso: str) -> list[PersistentAction]:
        """移除 UTC 已到期动作，返回被移除项。"""
        expired = [a for a in self.persistent_actions if a.is_expired(now_iso)]
        if expired:
            self.persistent_actions = [a for a in self.persistent_actions if not a.is_expired(now_iso)]
            self.revision += 1
        return expired

    # ------------------------------------------------------------------ #
    #  心境快照与恢复保留
    # ------------------------------------------------------------------ #

    def apply_mood_update(self, mu: dict) -> bool:
        """应用经校验的心境快照。返回是否有实质变化（供日记事件判定）。

        mu 必须已通过 mood_tools.validate_decision 校验。
        """
        was_fully_recovered = self.fully_recovered
        changed = (
            mu["status"] != self.status
            or mu.get("summary", "") != self.summary
            or mu["cause_category"] != self.cause_category
            or mu.get("latest_reason", "") != self.latest_reason
            or bool(mu.get("improved", False)) != self.improved
            or bool(mu.get("fully_recovered", False)) != self.fully_recovered
            or mu.get("recovery_reason", "") != self.recovery_reason
        )
        if not changed:
            return False

        now = utc_now_iso()
        self.status = mu["status"]
        self.summary = clamp_str(mu.get("summary", ""), MAX_SUMMARY_CHARS)
        self.cause_category = mu["cause_category"]
        self.latest_reason = clamp_str(mu.get("latest_reason", ""), MAX_REASON_CHARS)
        self.improved = bool(mu.get("improved", False))
        self.fully_recovered = bool(mu.get("fully_recovered", False))
        self.recovery_reason = clamp_str(mu.get("recovery_reason", ""), MAX_REASON_CHARS)
        self.changed_at = now
        self.revision += 1

        if self.fully_recovered:
            if not was_fully_recovered:
                # 首次完全恢复：从下一条有效消息起继续注入恢复事件
                self.recovered_at = now
                self.recovered_messages_left = RECOVERY_RETENTION_MESSAGES
        else:
            self.recovered_at = None
            self.recovered_messages_left = 0
        return True

    def tick_recovered(self) -> bool:
        """每条有效消息调用一次（恢复提交当轮不计）。

        返回 True 表示本次调用完成收尾：清除最近原因与恢复说明并回到 stable。
        """
        if self.recovered_messages_left <= 0:
            return False
        self.recovered_messages_left -= 1
        self.revision += 1
        if self.recovered_messages_left > 0:
            return False
        if self.status == "recovered":
            self.status = "stable"
            self.latest_reason = ""
            self.recovery_reason = ""
            self.improved = False
            self.fully_recovered = False
            self.recovered_at = None
            self.changed_at = utc_now_iso()
        else:
            self.recovered_messages_left = 0
        return True

    # ------------------------------------------------------------------ #
    #  事件历史（脱敏）
    # ------------------------------------------------------------------ #

    def add_history(self, event: str, reasoning: str, max_length: int = 20) -> None:
        self.history.append({
            "timestamp": utc_now_iso(),
            "event": clamp_str(event, MAX_REASONING_CHARS),
            "reasoning": clamp_str(reasoning, MAX_REASONING_CHARS),
        })
        if len(self.history) > max_length:
            self.history = self.history[-max_length:]

    # ------------------------------------------------------------------ #
    #  注入文本（即时快照，给正式主模型与调用②使用）
    # ------------------------------------------------------------------ #

    def build_snapshot_text(self) -> str:
        """生成注入用的状态快照文本；状态为空时返回空串。"""
        if (
            self.status == "stable"
            and not self.summary
            and not self.persistent_actions
            and self.recovered_messages_left <= 0
        ):
            return ""

        lines = ["[内在情绪状态（即时快照，仅供你感知，不要直接向用户复述机制）]"]
        if self.summary:
            lines.append(f"- 当前心境：{self.summary}")
        if self.latest_reason:
            lines.append(f"- 最近原因：{self.latest_reason}")
        if self.status in ("active", "recovering"):
            lines.append(f"- 是否好转：{'是' if self.improved else '否'}")
        if self.status == "recovered" or self.fully_recovered:
            lines.append("- 是否完全恢复：是")
            if self.recovery_reason:
                lines.append(f"- 恢复原因：{self.recovery_reason}")
        for a in self.persistent_actions:
            if a.name == "cold_violence":
                lines.append(f"- 当前动作：冷暴力（至 {a.expires_at} 为止不想回应）")
            elif a.name == "read_no_reply":
                lines.append(f"- 当前动作：已读不回（还剩 {a.remaining_replies} 条不想回应）")
        return "\n".join(lines)


@dataclass
class RequestMoodDecision:
    """单请求内的情绪决策结果。软动作只活在这里，绝不持久化。

    valid=False          → 决策本身非法（JSON/心境/解除字段损坏），全部作废。
    actions_rejected=True → 解除仍生效，但新动作组整组拒绝（reject_reason 说明）。
    """

    valid: bool
    reject_reason: str = ""
    actions_rejected: bool = False
    mood_update: dict | None = None
    actions: list[dict] = field(default_factory=list)  # [{"name": str, "params": dict}]
    lift_actions: list[str] = field(default_factory=list)
    silence_mode: str = "none"
    reasoning_summary: str = ""

    @property
    def new_hard_actions(self) -> list[dict]:
        return [a for a in self.actions if a["name"] in HARD_ACTIONS]

    @property
    def new_soft_actions(self) -> list[dict]:
        return [a for a in self.actions if a["name"] in SOFT_ACTIONS]
