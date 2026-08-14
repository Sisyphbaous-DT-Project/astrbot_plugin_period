"""Tests for core/mood_store.py — MoodStore persistence and v1->v3 migration."""

import json

import pytest

from core.mood import MoodState, PersistentAction
from core.mood_store import MoodStore


class TestMoodStoreCrud:
    """Basic create/read/update/delete."""

    @pytest.fixture
    def store(self, temp_data_dir):
        return MoodStore(temp_data_dir)

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, store):
        assert await store.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, store):
        state = MoodState(persistent_actions=[
            PersistentAction.create("cold_violence", {"duration": 30},
                                    expires_at="2030-01-01T00:00:00+00:00"),
        ])
        await store.set("test:umo", state)
        retrieved = await store.get("test:umo")
        assert retrieved is not None
        assert retrieved.get_action("cold_violence") is not None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.set("del:umo", MoodState())
        assert await store.get("del:umo") is not None
        await store.delete("del:umo")
        assert await store.get("del:umo") is None

    @pytest.mark.asyncio
    async def test_get_all(self, store):
        await store.set("a", MoodState(summary="甲"))
        await store.set("b", MoodState(summary="乙"))
        all_data = await store.get_all()
        assert len(all_data) == 2
        assert all_data["a"].summary == "甲"
        assert all_data["b"].summary == "乙"

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, temp_data_dir):
        store1 = MoodStore(temp_data_dir)
        await store1.set("persist", MoodState(summary="留存"))

        store2 = MoodStore(temp_data_dir)
        state = await store2.get("persist")
        assert state is not None
        assert state.summary == "留存"

    @pytest.mark.asyncio
    async def test_corrupted_file_returns_empty(self, temp_data_dir):
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_data_dir / "moods.json"
        file_path.write_text("not json", encoding="utf-8")
        store = MoodStore(temp_data_dir)
        assert await store.get("any") is None
        assert await store.get_all() == {}


class TestMoodStoreTransactional:
    """P1 回归：缓存镜像磁盘——落盘失败不污染内存、不抛出穿透请求钩子。"""

    @pytest.fixture
    def store(self, temp_data_dir):
        return MoodStore(temp_data_dir)

    @staticmethod
    def _break_disk(monkeypatch):
        from pathlib import Path

        def always_fail(*args, **kwargs):
            raise OSError("磁盘满")

        monkeypatch.setattr(Path, "write_text", always_fail)

    @pytest.mark.asyncio
    async def test_set_failure_keeps_old_state_and_does_not_raise(
        self, store, monkeypatch, temp_data_dir,
    ):
        await store.set("umo", MoodState(summary="旧状态"))
        self._break_disk(monkeypatch)
        ok = await store.set("umo", MoodState(summary="新状态"))  # 不得抛出 OSError
        assert ok is False
        # 缓存镜像磁盘：当前进程仍读旧状态（不会读到未持久化的新状态）
        assert (await store.get("umo")).summary == "旧状态"
        # 重启（新实例）也是旧状态
        assert (await MoodStore(temp_data_dir).get("umo")).summary == "旧状态"

    @pytest.mark.asyncio
    async def test_delete_failure_keeps_record(self, store, monkeypatch):
        await store.set("umo", MoodState())
        self._break_disk(monkeypatch)
        assert await store.delete("umo") is False
        assert await store.get("umo") is not None

    @pytest.mark.asyncio
    async def test_migration_save_failure_not_cached(self, temp_data_dir, monkeypatch):
        """迁移落盘失败不缓存：下次加载重新读取并迁移（幂等）。"""
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "moods.json").write_text(json.dumps({
            # 缺 expires_at 的旧冷暴力会被清洗 → 产生迁移说明 → 触发落盘
            "umo": {"active_tools": [{"name": "cold_violence", "params": {}}]},
        }), encoding="utf-8")
        store = MoodStore(temp_data_dir)
        self._break_disk(monkeypatch)
        state = await store.get("umo")  # 迁移结果照常在内存返回，不得抛出
        assert state is not None
        assert state.persistent_actions == []  # 非法动作已清洗
        assert store._cache is None  # 未落盘不缓存


class TestMoodStoreMigration:
    """Legacy moods.json (no schema_version) migration on load."""

    def _write_legacy(self, temp_data_dir, payload):
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "moods.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8",
        )

    @pytest.mark.asyncio
    async def test_legacy_file_migrated_and_persisted(self, temp_data_dir):
        self._write_legacy(temp_data_dir, {
            "umo:1": {
                "active_tools": [{
                    "name": "cold_violence", "params": {"duration": 15},
                    "expires_at": "2030-06-01T00:00:00+00:00",
                    "rounds_left": None, "initiated": False,
                }],
                "history": [{
                    "timestamp": "t", "event": "e",
                    "reasoning": "r", "user_message": "原文",
                }],
                "last_interaction": "2025-06-01T00:00:00",
            },
        })
        notes_seen = []

        async def on_migration(umo, notes):
            notes_seen.append((umo, notes))

        store = MoodStore(temp_data_dir)
        store.on_migration = on_migration
        state = await store.get("umo:1")
        assert state is not None
        assert state.get_action("cold_violence") is not None
        assert state.history[0].get("user_message") is None
        assert notes_seen and notes_seen[0][0] == "umo:1"

        # 迁移结果已原子落盘：文件内是 v3 结构
        on_disk = json.loads((temp_data_dir / "moods.json").read_text(encoding="utf-8"))
        assert on_disk["umo:1"]["schema_version"] == 3
        assert "user_message" not in json.dumps(on_disk, ensure_ascii=False)

        # 幂等：新实例再次加载不再触发迁移
        notes_seen.clear()
        store2 = MoodStore(temp_data_dir)
        store2.on_migration = on_migration
        again = await store2.get("umo:1")
        assert again is not None
        assert notes_seen == []

    @pytest.mark.asyncio
    async def test_legacy_dual_hard_actions_cleared_with_note(self, temp_data_dir):
        self._write_legacy(temp_data_dir, {
            "umo:2": {
                "active_tools": [
                    {"name": "cold_violence", "params": {}, "expires_at": "2030-01-01T00:00:00"},
                    {"name": "read_no_reply", "params": {}, "rounds_left": 1},
                ],
                "history": [],
            },
        })
        notes_seen = []

        async def on_migration(umo, notes):
            notes_seen.extend(notes)

        store = MoodStore(temp_data_dir)
        store.on_migration = on_migration
        state = await store.get("umo:2")
        assert state is not None
        assert state.persistent_actions == []
        assert "legacy_dual_hard_actions_cleared" in notes_seen

    @pytest.mark.asyncio
    async def test_v3_file_not_rewritten(self, temp_data_dir):
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        store = MoodStore(temp_data_dir)
        await store.set("umo:3", MoodState(summary="介意"))

        raw_before = (temp_data_dir / "moods.json").read_text(encoding="utf-8")
        notes_seen = []

        async def on_migration(umo, notes):
            notes_seen.append(notes)

        store2 = MoodStore(temp_data_dir)
        store2.on_migration = on_migration
        state = await store2.get("umo:3")
        assert state.summary == "介意"
        assert notes_seen == []
        raw_after = (temp_data_dir / "moods.json").read_text(encoding="utf-8")
        assert raw_before == raw_after


class TestV3HistoryScrubbedOnDisk:
    """P2 回归：v3 history 的非白名单键（如 user_message）——内存脱敏
    且通过迁移 note 触发落盘重写，磁盘原文不得保留。"""

    @pytest.mark.asyncio
    async def test_user_message_removed_and_rewritten(self, temp_data_dir):
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "moods.json").write_text(json.dumps({
            "umo": {
                "schema_version": 3,
                "history": [{
                    "timestamp": "t", "event": "e", "reasoning": "r",
                    "user_message": "用户原话",
                }],
            },
        }), encoding="utf-8")
        store = MoodStore(temp_data_dir)
        state = await store.get("umo")
        assert state.history == [{"timestamp": "t", "event": "e", "reasoning": "r"}]
        raw = json.loads((temp_data_dir / "moods.json").read_text(encoding="utf-8"))
        assert "user_message" not in raw["umo"]["history"][0]
        assert raw["umo"]["history"][0] == {
            "timestamp": "t", "event": "e", "reasoning": "r",
        }

    @pytest.mark.asyncio
    async def test_clean_history_not_rewritten(self, temp_data_dir):
        """白名单内的正常 v3 记录不产生 note、不重写文件。"""
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "moods.json").write_text(json.dumps({
            "umo": {
                "schema_version": 3,
                "history": [{"timestamp": "t", "event": "e", "reasoning": "r"}],
            },
        }), encoding="utf-8")
        raw_before = (temp_data_dir / "moods.json").read_text(encoding="utf-8")
        store = MoodStore(temp_data_dir)
        state = await store.get("umo")
        assert len(state.history) == 1
        raw_after = (temp_data_dir / "moods.json").read_text(encoding="utf-8")
        assert raw_before == raw_after

    @pytest.mark.asyncio
    async def test_non_dict_history_entries_scrubbed_on_disk(self, temp_data_dir):
        """history 含非对象条目（或整体非列表）同样触发落盘重写。"""
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "moods.json").write_text(json.dumps({
            "umo": {"schema_version": 3, "history": ["敏感原文"]},
        }), encoding="utf-8")
        store = MoodStore(temp_data_dir)
        state = await store.get("umo")
        assert state.history == []
        raw = json.loads((temp_data_dir / "moods.json").read_text(encoding="utf-8"))
        assert raw["umo"]["history"] == []

    @pytest.mark.asyncio
    async def test_explicit_null_history_rewritten(self, temp_data_dir):
        """显式 history: null 区分于字段缺失：同样触发落盘重写。"""
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "moods.json").write_text(json.dumps({
            "umo": {"schema_version": 3, "history": None},
        }), encoding="utf-8")
        store = MoodStore(temp_data_dir)
        state = await store.get("umo")
        assert state.history == []
        raw = json.loads((temp_data_dir / "moods.json").read_text(encoding="utf-8"))
        assert raw["umo"]["history"] == []

    @pytest.mark.asyncio
    async def test_missing_history_key_not_rewritten(self, temp_data_dir):
        """history 键缺失不是脏数据：不产生 note、不重写文件。"""
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "moods.json").write_text(json.dumps({
            "umo": {"schema_version": 3, "summary": "介意"},
        }), encoding="utf-8")
        raw_before = (temp_data_dir / "moods.json").read_text(encoding="utf-8")
        store = MoodStore(temp_data_dir)
        state = await store.get("umo")
        assert state.summary == "介意"
        raw_after = (temp_data_dir / "moods.json").read_text(encoding="utf-8")
        assert raw_before == raw_after
