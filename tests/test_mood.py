"""Tests for core/mood.py — MoodState v3 (natural-language inner state)."""

import pytest

from core.mood import (
    MoodState,
    PersistentAction,
    RequestMoodDecision,
    RECOVERY_RETENTION_MESSAGES,
    SCHEMA_VERSION,
)


class TestMoodStateSerialization:
    """v3 round-trip serialization."""

    def test_to_dict_and_back(self):
        action = PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2030-05-25T18:00:00+00:00", request_id="req-1",
        )
        state = MoodState(
            status="active",
            summary="有些介意被敷衍",
            cause_category="dismissive",
            latest_reason="回应不够被重视",
            persistent_actions=[action],
        )
        state.add_history(event="action:cold_violence", reasoning="决定暂时不理会")

        d = state.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        restored = MoodState.from_dict(d)
        assert restored.status == "active"
        assert restored.summary == "有些介意被敷衍"
        assert restored.cause_category == "dismissive"
        assert len(restored.persistent_actions) == 1
        assert restored.persistent_actions[0].name == "cold_violence"
        assert restored.persistent_actions[0].expires_at == "2030-05-25T18:00:00+00:00"
        assert restored.persistent_actions[0].activation_request_id == "req-1"
        assert len(restored.history) == 1
        assert "user_message" not in restored.history[0]

    def test_from_dict_none_returns_default(self):
        state = MoodState.from_dict(None)
        assert state.status == "stable"
        assert state.persistent_actions == []
        assert state.history == []

    def test_from_dict_bad_enums_fall_back(self):
        state = MoodState.from_dict({
            "schema_version": 3, "status": "bogus", "cause_category": "xxx",
        })
        assert state.status == "stable"
        assert state.cause_category == "neutral"

    def test_summary_fields_clamped(self):
        long_text = "长" * 500
        state = MoodState.from_dict({
            "schema_version": 3, "status": "active", "summary": long_text,
            "latest_reason": long_text,
        })
        assert len(state.summary) == 200
        assert len(state.latest_reason) == 200

    def test_from_dict_corrupted_fields_never_raise(self):
        """P1 回归：损坏的 v3 记录（坏 params/null history/非对象）不得抛异常。"""
        # params 是字符串 → 回退为空 dict
        state = MoodState.from_dict({
            "schema_version": 3,
            "persistent_actions": [{
                "name": "cold_violence", "params": "oops",
                "expires_at": "2999-01-01T00:00:00+00:00",
            }],
        })
        assert state.persistent_actions[0].params == {}
        # history 是 null → 回退为空列表
        state = MoodState.from_dict({"schema_version": 3, "history": None})
        assert state.history == []
        # persistent_actions 不是列表 → 回退为空
        state = MoodState.from_dict({"schema_version": 3, "persistent_actions": 5})
        assert state.persistent_actions == []
        # 记录本身不是 dict → 返回默认状态
        assert MoodState.from_dict("garbage").status == "stable"
        assert MoodState.from_dict(["x"]).status == "stable"


class TestPersistentActions:
    def test_expire_time_based_only(self):
        state = MoodState(persistent_actions=[
            PersistentAction.create("cold_violence", {}, expires_at="2030-01-01T00:00:00+00:00"),
            PersistentAction.create("read_no_reply", {}, remaining_replies=2),
        ])
        expired = state.expire_actions("2031-01-01T00:00:00+00:00")
        # 两个硬动作并存只可能由直接构造产生；时间型到期、轮数型不受时间影响
        assert len(expired) == 1
        assert expired[0].name == "cold_violence"
        assert state.get_action("read_no_reply") is not None

    def test_add_replaces_same_name(self):
        state = MoodState()
        state.add_action(PersistentAction.create("cold_violence", {"duration": 10}))
        state.add_action(PersistentAction.create("cold_violence", {"duration": 20}))
        assert len(state.persistent_actions) == 1
        assert state.persistent_actions[0].params["duration"] == 20

    def test_remove_action(self):
        state = MoodState(persistent_actions=[
            PersistentAction.create("cold_violence", {}),
        ])
        assert state.remove_action("cold_violence") is not None
        assert state.remove_action("cold_violence") is None
        assert state.has_hard_action() is False


class TestMoodUpdateAndRecovery:
    def _mu(self, **kw):
        base = {
            "status": "active", "summary": "介意", "cause_category": "conflict",
            "latest_reason": "吵了一架", "improved": False,
            "fully_recovered": False, "recovery_reason": "",
        }
        base.update(kw)
        return base

    def test_apply_mood_update_changes_state(self):
        state = MoodState()
        changed = state.apply_mood_update(self._mu())
        assert changed is True
        assert state.status == "active"
        assert state.summary == "介意"
        assert state.changed_at != ""

    def test_apply_same_snapshot_returns_false(self):
        state = MoodState()
        state.apply_mood_update(self._mu())
        rev = state.revision
        assert state.apply_mood_update(self._mu()) is False
        assert state.revision == rev

    def test_fully_recovered_starts_retention(self):
        state = MoodState()
        state.apply_mood_update(self._mu(
            status="recovered", improved=True, fully_recovered=True,
            recovery_reason="对方认真道歉",
        ))
        assert state.recovered_at is not None
        assert state.recovered_messages_left == RECOVERY_RETENTION_MESSAGES

    def test_tick_recovered_counts_down_and_cleans(self):
        state = MoodState()
        state.apply_mood_update(self._mu(
            status="recovered", improved=True, fully_recovered=True,
            recovery_reason="对方认真道歉",
        ))
        for i in range(RECOVERY_RETENTION_MESSAGES - 1):
            assert state.tick_recovered() is False
            assert state.status == "recovered"
        # 最后一次 tick：清理原因并回到 stable
        assert state.tick_recovered() is True
        assert state.status == "stable"
        assert state.latest_reason == ""
        assert state.recovery_reason == ""
        assert state.fully_recovered is False
        # 再次 tick 无效果
        assert state.tick_recovered() is False

    def test_worsening_cancels_retention(self):
        state = MoodState()
        state.apply_mood_update(self._mu(
            status="recovered", fully_recovered=True, recovery_reason="道歉",
        ))
        state.apply_mood_update(self._mu(status="active", summary="又被激怒"))
        assert state.recovered_at is None
        assert state.recovered_messages_left == 0


class TestHistory:
    def test_add_history_trims_and_has_no_user_message(self):
        state = MoodState()
        for i in range(15):
            state.add_history(event=f"evt{i}", reasoning="r", max_length=5)
        assert len(state.history) == 5
        assert state.history[-1]["event"] == "evt14"
        assert set(state.history[0].keys()) == {"timestamp", "event", "reasoning"}


class TestSnapshotText:
    def test_stable_empty_returns_empty(self):
        assert MoodState().build_snapshot_text() == ""

    def test_snapshot_contains_key_fields(self):
        state = MoodState(
            status="recovering", summary="有所缓和", cause_category="dismissive",
            latest_reason="感到被敷衍", improved=True,
        )
        text = state.build_snapshot_text()
        assert "有所缓和" in text
        assert "感到被敷衍" in text
        assert "是否好转：是" in text

    def test_snapshot_shows_actions(self):
        state = MoodState(persistent_actions=[
            PersistentAction.create("read_no_reply", {}, remaining_replies=2),
        ])
        text = state.build_snapshot_text()
        assert "已读不回" in text and "2" in text


class TestLegacyMigration:
    """无 schema_version 的 v1 数据迁移。"""

    def test_migrate_keeps_valid_cold_violence(self):
        legacy = {
            "active_tools": [{
                "name": "cold_violence", "params": {"duration": 20},
                "expires_at": "2030-01-01T00:00:00+00:00",
                "rounds_left": None, "initiated": True,
            }],
            "history": [{
                "timestamp": "t", "event": "e", "reasoning": "r",
                "user_message": "用户原文不应保留",
            }],
            "last_interaction": "2025-01-01T00:00:00",
        }
        state, notes = MoodState.migrate(legacy)
        assert "migrated_v1_to_v3" in notes
        assert "legacy_history_user_message_removed" in notes
        assert len(state.persistent_actions) == 1
        action = state.persistent_actions[0]
        assert action.name == "cold_violence"
        assert action.expires_at == "2030-01-01T00:00:00+00:00"
        assert action.id and action.created_at
        assert state.history[0].get("user_message") is None
        assert state.last_interaction_at == "2025-01-01T00:00:00"
        assert state.status == "active"

    def test_migrate_drops_soft_tools(self):
        legacy = {"active_tools": [{"name": "seek_comfort", "params": {}}]}
        state, _ = MoodState.migrate(legacy)
        assert state.persistent_actions == []
        assert state.status == "stable"

    def test_migrate_drops_invalid_time(self):
        legacy = {"active_tools": [{"name": "cold_violence", "params": {}, "expires_at": None}]}
        state, notes = MoodState.migrate(legacy)
        assert state.persistent_actions == []
        assert "legacy_cold_violence_invalid_time_dropped" in notes

    def test_migrate_clears_dual_hard_actions(self):
        legacy = {
            "active_tools": [
                {"name": "cold_violence", "params": {}, "expires_at": "2030-01-01T00:00:00"},
                {"name": "read_no_reply", "params": {}, "rounds_left": 2},
            ],
        }
        state, notes = MoodState.migrate(legacy)
        assert state.persistent_actions == []
        assert "legacy_dual_hard_actions_cleared" in notes

    def test_migrate_read_no_reply_rounds(self):
        legacy = {"active_tools": [{"name": "read_no_reply", "params": {}, "rounds_left": 2}]}
        state, _ = MoodState.migrate(legacy)
        assert state.persistent_actions[0].remaining_replies == 2

    def test_migrate_is_idempotent_on_v3(self):
        state = MoodState(status="active", summary="介意")
        again, notes = MoodState.migrate(state.to_dict())
        assert notes == []
        assert again.summary == "介意"


class TestRequestMoodDecision:
    def test_defaults(self):
        d = RequestMoodDecision(valid=True)
        assert d.silence_mode == "none"
        assert d.new_hard_actions == []
        assert d.new_soft_actions == []

    def test_hard_soft_partition(self):
        d = RequestMoodDecision(
            valid=True,
            actions=[
                {"name": "cold_violence", "params": {}},
            ],
        )
        assert len(d.new_hard_actions) == 1


class TestTimeParsing:
    """P1-⑨ 回归：时间必须解析为 UTC datetime 比较，非法值不得残留。"""

    def test_parse_iso_utc_aware(self):
        from core.mood import parse_iso_utc
        dt = parse_iso_utc("2030-01-01T00:00:00+08:00")
        assert dt is not None
        assert dt.utcoffset().total_seconds() == 0  # 已归一到 UTC

    def test_parse_iso_utc_rejects_naive_and_garbage(self):
        from core.mood import parse_iso_utc
        assert parse_iso_utc("2030-01-01T00:00:00") is None  # 无时区
        assert parse_iso_utc("not-a-time") is None
        assert parse_iso_utc(None) is None
        assert parse_iso_utc("") is None

    def test_is_expired_compares_instants_not_strings(self):
        from core.mood import PersistentAction
        # +08:00 的 08:00 == UTC 00:00，字符串比较会判错
        a = PersistentAction.create(
            "cold_violence", {}, expires_at="2030-01-01T08:00:00+08:00",
        )
        assert a.is_expired("2030-01-01T00:00:01+00:00") is True
        assert a.is_expired("2029-12-31T23:59:59+00:00") is False

    def test_invalid_expires_at_treated_as_expired(self):
        from core.mood import PersistentAction
        a = PersistentAction.create("cold_violence", {}, expires_at="not-a-time")
        assert a.is_expired("2024-01-01T00:00:00+00:00") is True

    def test_migrate_drops_naive_legacy_time(self):
        legacy = {"active_tools": [
            {"name": "cold_violence", "params": {}, "expires_at": "2030-01-01T00:00:00"},
        ]}
        state, notes = MoodState.migrate(legacy)
        assert state.persistent_actions == []
        assert "legacy_cold_violence_invalid_time_dropped" in notes

    def test_migrate_v3_sanitizes_dual_hard_and_notes(self):
        data = MoodState().to_dict()
        data["persistent_actions"] = [
            {"id": "a", "name": "cold_violence", "params": {},
             "created_at": "2024-01-01T00:00:00+00:00",
             "expires_at": "2999-01-01T00:00:00+00:00",
             "remaining_replies": None, "activation_request_id": ""},
            {"id": "b", "name": "read_no_reply", "params": {},
             "created_at": "2024-01-01T00:00:00+00:00",
             "expires_at": None, "remaining_replies": 2,
             "activation_request_id": ""},
        ]
        state, notes = MoodState.migrate(data)
        assert state.persistent_actions == []
        assert "v3_dual_hard_actions_cleared" in notes

    def test_migrate_v3_drops_invalid_time_with_notes(self):
        data = MoodState().to_dict()
        data["persistent_actions"] = [
            {"id": "a", "name": "cold_violence", "params": {},
             "created_at": "2024-01-01T00:00:00+00:00",
             "expires_at": "not-a-time",
             "remaining_replies": None, "activation_request_id": ""},
        ]
        state, notes = MoodState.migrate(data)
        assert state.persistent_actions == []
        assert "v3_invalid_actions_sanitized" in notes


class TestCorruptedDataDefense:
    """P1 回归：损坏的 moods.json 不得在请求钩子上抛异常。

    migrate() 自己的原始数据遍历（v3 的 persistent_actions、旧版的
    active_tools/history）也必须对非标量容器免疫——异常会让已有
    冷暴力/已读不回无法载入，AstrBot 吞掉钩子异常后继续正式请求。
    """

    def test_migrate_v3_scalar_persistent_actions(self):
        state, notes = MoodState.migrate({"schema_version": 3, "persistent_actions": 5})
        assert state.persistent_actions == []
        assert "v3_invalid_actions_sanitized" in notes

    def test_migrate_legacy_scalar_containers(self):
        state, notes = MoodState.migrate({"active_tools": 5, "history": 5})
        assert state.persistent_actions == []
        assert state.history == []
        assert "migrated_v1_to_v3" in notes

    def test_migrate_legacy_bool_rounds_dropped(self):
        # bool 是 int 子类，True 不得被当成合法轮数
        state, notes = MoodState.migrate({
            "active_tools": [{"name": "read_no_reply", "rounds_left": True}],
        })
        assert state.persistent_actions == []
        assert "legacy_read_no_reply_invalid_rounds_dropped" in notes

    def test_from_dict_strict_bool_scalars(self):
        # bool("false") 会变成 True：持久化标量必须严格要求真 bool/int
        state = MoodState.from_dict({
            "schema_version": 3,
            "improved": "false",
            "fully_recovered": "yes",
            "recovered_messages_left": True,
            "revision": True,
        })
        assert state.improved is False
        assert state.fully_recovered is False
        assert state.recovered_messages_left == 0
        assert state.revision == 1

    def test_from_dict_history_whitelist_strips_user_message(self):
        # 脱敏不变量在反序列化边界强制执行：外部数据不得带回用户原话
        state = MoodState.from_dict({
            "schema_version": 3,
            "history": [{
                "timestamp": "t", "event": "e", "reasoning": "r",
                "user_message": "用户原话", "extra": "其他字段",
            }],
        })
        assert state.history == [{"timestamp": "t", "event": "e", "reasoning": "r"}]

    def test_action_from_dict_bool_remaining_replies(self):
        action = PersistentAction.from_dict({
            "name": "read_no_reply", "remaining_replies": True,
        })
        assert action.remaining_replies is None

    def test_migrate_legacy_bad_params_sanitized(self):
        # dict("bad") 会 ValueError 打穿全量加载：params 非 dict 一律置空
        state, notes = MoodState.migrate({
            "active_tools": [{
                "name": "cold_violence", "params": "bad",
                "expires_at": "2999-01-01T00:00:00+00:00",
            }],
        })
        assert len(state.persistent_actions) == 1
        assert state.persistent_actions[0].params == {}
        assert "legacy_action_params_sanitized" in notes
