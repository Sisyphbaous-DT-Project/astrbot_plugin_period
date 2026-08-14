"""Tests for core/mood_journal.py — DiaryStore envelope, outbox, identity keys."""

import json

import pytest

from core.mood_journal import DiaryJournal, DiaryStore, count_diary_chars


OWNER_A = "qq_1:10000:12345"
OWNER_B = "qq_1:10000:67890"       # 同机器人不同用户
OWNER_OTHER_BOT = "qq_1:20000:12345"  # 同用户不同机器人
OWNER_OTHER_PLATFORM = "wx_1:10000:12345"  # 同用户不同平台实例


def _entry(eid, text, event_id="evt"):
    return {"id": eid, "event_id": event_id, "occurred_at": "2026-08-12T00:00:00+00:00", "text": text}


class TestOwnerKey:
    def test_make_owner_key(self):
        assert DiaryJournal.make_owner_key("qq_1", "10000", "12345") == OWNER_A

    @pytest.mark.parametrize("platform_id,self_id,sender_id", [
        ("", "10000", "12345"),
        ("qq_1", "", "12345"),
        ("qq_1", "10000", ""),
        ("", "", ""),
    ])
    def test_missing_identity_returns_none(self, platform_id, self_id, sender_id):
        assert DiaryJournal.make_owner_key(platform_id, self_id, sender_id) is None

    def test_namespace(self):
        assert DiaryJournal.namespace_of(OWNER_A) == "qq_1:10000"


class TestCharCount:
    def test_empty(self):
        assert count_diary_chars([]) == 0

    def test_includes_newlines_between_entries(self):
        entries = [_entry("a", "一二三四"), _entry("b", "五六")]
        assert count_diary_chars(entries) == 4 + 1 + 2


class TestDiaryStoreCrud:
    @pytest.fixture
    def store(self, temp_data_dir):
        return DiaryStore(temp_data_dir)

    @pytest.mark.asyncio
    async def test_upsert_and_get(self, store):
        await store.upsert_diary(OWNER_A, [_entry("e1", "条目一")], display_name="小明")
        diary = await store.get_diary(OWNER_A)
        assert diary["display_name"] == "小明"
        assert len(diary["entries"]) == 1
        assert diary["updated_at"]

    @pytest.mark.asyncio
    async def test_aliases_accumulate(self, store):
        await store.upsert_diary(OWNER_A, [], display_name="小明", aliases=["小明"])
        await store.upsert_diary(OWNER_A, [], display_name="明哥", aliases=["明哥"])
        diary = await store.get_diary(OWNER_A)
        assert diary["display_name"] == "明哥"
        assert set(diary["aliases"]) == {"小明", "明哥"}

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.upsert_diary(OWNER_A, [_entry("e1", "x")])
        removed, events, persisted = await store.delete_diary(OWNER_A)
        assert removed is True and events == 0 and persisted is True
        assert await store.get_diary(OWNER_A) is None
        removed, events, persisted = await store.delete_diary(OWNER_A)
        assert removed is False and events == 0 and persisted is True

    @pytest.mark.asyncio
    async def test_delete_also_removes_pending_events(self, store):
        """删除必须连同该 owner 的 outbox 事件一起清理。"""
        await store.enqueue({
            "id": "evt-1", "owner_key": OWNER_A, "kind": "mood_changed",
            "summary": "s", "display_name": "", "provider_id": "",
            "occurred_at": "2026-08-12T00:00:00+00:00",
        })
        removed, events, persisted = await store.delete_diary(OWNER_A)
        assert removed is False and events == 1 and persisted is True
        assert await store.pending_events() == []

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, store):
        await store.upsert_diary(OWNER_A, [_entry("e1", "甲")])
        await store.upsert_diary(OWNER_B, [_entry("e2", "乙")])
        await store.upsert_diary(OWNER_OTHER_BOT, [_entry("e3", "丙")])
        await store.upsert_diary(OWNER_OTHER_PLATFORM, [_entry("e4", "丁")])

        same_bot = await store.list_namespace("qq_1", "10000")
        keys = {d["owner_key"] for d in same_bot}
        assert keys == {OWNER_A, OWNER_B}  # 同机器人跨用户可见；其他机器人/平台严格隔离

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, temp_data_dir):
        store1 = DiaryStore(temp_data_dir)
        await store1.upsert_diary(OWNER_A, [_entry("e1", "留存")])
        store2 = DiaryStore(temp_data_dir)
        diary = await store2.get_diary(OWNER_A)
        assert diary["entries"][0]["text"] == "留存"


class TestOutbox:
    @pytest.fixture
    def store(self, temp_data_dir):
        return DiaryStore(temp_data_dir)

    def _event(self, eid="ev1", owner=OWNER_A):
        return {"id": eid, "owner_key": owner, "kind": "mood_changed", "summary": "摘要"}

    @pytest.mark.asyncio
    async def test_enqueue_and_pending(self, store):
        assert await store.enqueue(self._event()) is True
        pending = await store.pending_events()
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_enqueue_dedupes_same_id(self, store):
        await store.enqueue(self._event())
        assert await store.enqueue(self._event()) is False
        assert len(await store.pending_events()) == 1

    @pytest.mark.asyncio
    async def test_ack_moves_to_processed_ring(self, store):
        await store.enqueue(self._event())
        await store.ack("ev1")
        assert await store.pending_events() == []
        assert await store.is_processed("ev1") is True
        # 已处理事件不再入队
        assert await store.enqueue(self._event()) is False

    @pytest.mark.asyncio
    async def test_outbox_survives_restart(self, temp_data_dir):
        store1 = DiaryStore(temp_data_dir)
        await store1.enqueue(self._event())
        store2 = DiaryStore(temp_data_dir)
        assert len(await store2.pending_events()) == 1

    @pytest.mark.asyncio
    async def test_envelope_shape_on_disk(self, store, temp_data_dir):
        await store.enqueue(self._event())
        raw = json.loads((temp_data_dir / "emotion_diaries.json").read_text(encoding="utf-8"))
        assert raw["schema_version"] == 1
        assert set(raw.keys()) == {
            "schema_version", "diaries", "pending_events",
            "processed_event_ids", "umo_watermarks",
        }


class TestTransactionalCache:
    """P1 回归：缓存必须镜像磁盘——落盘失败不得污染内存、删除不得假成功。"""

    @pytest.fixture
    def store(self, temp_data_dir):
        return DiaryStore(temp_data_dir)

    @staticmethod
    def _break_disk(monkeypatch):
        from pathlib import Path

        def always_fail(*args, **kwargs):
            raise OSError("磁盘满")

        monkeypatch.setattr(Path, "write_text", always_fail)

    @pytest.mark.asyncio
    async def test_delete_save_failure_reports_and_preserves(
        self, store, monkeypatch, temp_data_dir,
    ):
        await store.upsert_diary(OWNER_A, [_entry("e1", "x")])
        self._break_disk(monkeypatch)
        removed, events, persisted = await store.delete_diary(OWNER_A)
        # 如实上报失败，不得返回"已删除"
        assert persisted is False and removed is False and events == 0
        # 缓存镜像磁盘：当前进程仍能看到日记（不会假装已删）
        assert await store.get_diary(OWNER_A) is not None
        # 重启（新实例）后数据也还在——清除确实未生效
        store2 = DiaryStore(temp_data_dir)
        assert await store2.get_diary(OWNER_A) is not None

    @pytest.mark.asyncio
    async def test_upsert_save_failure_not_visible_in_process(self, store, monkeypatch):
        self._break_disk(monkeypatch)
        ok = await store.upsert_diary(OWNER_A, [_entry("e1", "x")])
        assert ok is False
        # 未落盘的日记不得被当前进程读取（否则会注入模型且重启后消失）
        assert await store.get_diary(OWNER_A) is None

    @pytest.mark.asyncio
    async def test_ack_failure_keeps_event_pending(self, store, monkeypatch):
        await store.enqueue({"id": "ev1", "owner_key": OWNER_A})
        self._break_disk(monkeypatch)
        assert await store.ack("ev1") is False
        # 缓存镜像磁盘：事件仍 pending、未进已处理环
        assert len(await store.pending_events()) == 1
        assert await store.is_processed("ev1") is False


class TestDiscardPendingEvents:
    """P2 回归：/period reset 按来源 UMO 丢弃滞留事件。"""

    @pytest.fixture
    def store(self, temp_data_dir):
        return DiaryStore(temp_data_dir)

    @staticmethod
    def _event(eid, umo, owner=OWNER_A):
        return {
            "id": eid, "owner_key": owner, "kind": "mood_changed",
            "summary": "s", "umo": umo,
            "occurred_at": "2026-08-12T00:00:00+00:00",
        }

    @pytest.mark.asyncio
    async def test_discard_only_matching_umo(self, store):
        await store.enqueue(self._event("e1", "umo_a"))
        await store.enqueue(self._event("e2", "umo_b"))
        await store.enqueue(self._event("e3", "umo_a", owner=OWNER_B))
        removed = await store.discard_pending_events("umo_a")
        assert removed == 2  # 同 UMO 的都被丢弃，与 owner 无关
        pending = await store.pending_events()
        assert [e["id"] for e in pending] == ["e2"]

    @pytest.mark.asyncio
    async def test_discard_keeps_committed_diary(self, store):
        await store.upsert_diary(OWNER_A, [_entry("e1", "已提交条目")])
        await store.enqueue(self._event("e1", "umo_a"))
        await store.discard_pending_events("umo_a")
        # reset 只丢弃滞留事件，已提交日记保留
        diary = await store.get_diary(OWNER_A)
        assert diary is not None and diary["entries"][0]["text"] == "已提交条目"

    @pytest.mark.asyncio
    async def test_discard_save_failure_returns_minus_one(self, store, monkeypatch):
        """落盘失败返回 -1，与"没有事件可丢"（0）严格区分。"""
        from pathlib import Path

        await store.enqueue(self._event("e1", "umo_a"))

        def always_fail(*args, **kwargs):
            raise OSError("磁盘满")

        monkeypatch.setattr(Path, "write_text", always_fail)
        assert await store.discard_pending_events("umo_a") == -1
        assert len(await store.pending_events()) == 1  # 未生效，事件仍在

    @pytest.mark.asyncio
    async def test_discard_nothing_returns_zero(self, store):
        assert await store.discard_pending_events("umo_x") == 0


class TestCorruptedFileLoad:
    """P1 回归：损坏的 emotion_diaries.json 逐本清洗，不得打穿请求链。

    日记读取发生在调用①之前：损坏数据若抛异常，已有硬动作会进不了
    沉默分支（AstrBot 吞掉钩子异常后继续正式请求）。
    """

    @pytest.mark.asyncio
    async def test_per_diary_and_event_cleaning(self, temp_data_dir):
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "emotion_diaries.json").write_text(json.dumps({
            "schema_version": 1,
            "diaries": {
                OWNER_A: {"entries": 5, "aliases": "x", "display_name": 3},
                "broken": 42,
                OWNER_B: {"entries": [{"text": "正常条目"}, "坏条目", {"no_text": 1}]},
            },
            "pending_events": [
                {"id": "e1"},  # 缺 owner_key：恢复时下标访问会崩，丢弃
                {"id": "", "owner_key": OWNER_A},  # 空 id：丢弃
                {"id": "e2", "owner_key": OWNER_A, "kind": 3, "summary": None},
                "not-a-dict",
            ],
            "processed_event_ids": [],
        }), encoding="utf-8")
        store = DiaryStore(temp_data_dir)

        diary = await store.get_diary(OWNER_A)
        assert diary["entries"] == []
        assert diary["aliases"] == []
        assert diary["display_name"] == ""
        assert diary["owner_key"] == OWNER_A
        assert await store.get_diary("broken") is None

        diary_b = await store.get_diary(OWNER_B)
        assert [e["text"] for e in diary_b["entries"]] == ["正常条目"]

        pending = await store.pending_events()
        assert [e["id"] for e in pending] == ["e2"]
        assert pending[0]["kind"] == "" and pending[0]["summary"] == ""

        # 注入文本构建（请求链路径）不得抛异常
        journal = DiaryJournal(temp_data_dir, lambda pid: None)
        assert journal.build_injection_text(diary) == ""
        assert "正常条目" in journal.build_injection_text(diary_b)
    @pytest.mark.asyncio
    async def test_entry_missing_id_backfilled(self, temp_data_dir):
        """缺 id 的条目补生成而非丢弃：工具循环 earliest_id 不再 KeyError。"""
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "emotion_diaries.json").write_text(json.dumps({
            "schema_version": 1,
            "diaries": {OWNER_A: {"entries": [{"text": "没id的条目"}]}},
            "pending_events": [],
        }), encoding="utf-8")
        store = DiaryStore(temp_data_dir)
        diary = await store.get_diary(OWNER_A)
        assert len(diary["entries"]) == 1
        entry = diary["entries"][0]
        assert entry["text"] == "没id的条目"
        assert entry["id"] and isinstance(entry["id"], str)
        assert entry["event_id"] == "" and entry["occurred_at"] == ""


class TestUmoWatermark:
    """P1 回归：reset/删除会话的丢弃水位线。

    水位线持久化（重启后仍生效）；早于水位线的事件在 enqueue 存储锁内
    被拒绝，覆盖"源头复查通过 → reset → submit"的竞态窗口。
    """

    @pytest.fixture
    def store(self, temp_data_dir):
        return DiaryStore(temp_data_dir)

    @staticmethod
    def _event(eid, umo, occurred_at):
        return {
            "id": eid, "owner_key": OWNER_A, "kind": "mood_changed",
            "summary": "摘要", "display_name": "", "provider_id": "",
            "umo": umo, "occurred_at": occurred_at,
        }

    @pytest.mark.asyncio
    async def test_stale_event_refused_at_enqueue(self, store):
        assert await store.mark_umo_watermark("umo_a") is True
        stale = self._event("e1", "umo_a", "2020-01-01T00:00:00+00:00")
        assert await store.enqueue(stale) is False
        assert await store.pending_events() == []

    @pytest.mark.asyncio
    async def test_fresh_event_after_watermark_accepted(self, store):
        await store.mark_umo_watermark("umo_a")
        fresh = self._event("e2", "umo_a", "2999-01-01T00:00:00+00:00")
        assert await store.enqueue(fresh) is True
        assert len(await store.pending_events()) == 1

    @pytest.mark.asyncio
    async def test_other_umo_unaffected(self, store):
        await store.mark_umo_watermark("umo_a")
        ev = self._event("e3", "umo_b", "2020-01-01T00:00:00+00:00")
        assert await store.enqueue(ev) is True

    @pytest.mark.asyncio
    async def test_event_without_umo_unaffected(self, store):
        # 无 umo 的旧事件保持向后兼容，不参与水位线判定
        await store.mark_umo_watermark("umo_a")
        ev = self._event("e4", "", "2020-01-01T00:00:00+00:00")
        assert await store.enqueue(ev) is True

    @pytest.mark.asyncio
    async def test_watermark_survives_restart(self, store, temp_data_dir):
        await store.mark_umo_watermark("umo_a")
        store2 = DiaryStore(temp_data_dir)  # 模拟插件重启
        stale = self._event("e5", "umo_a", "2020-01-01T00:00:00+00:00")
        assert await store2.enqueue(stale) is False
