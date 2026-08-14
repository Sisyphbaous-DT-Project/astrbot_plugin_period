"""Tests for core/store.py — CycleStore persistence."""

import json
import pytest

from core.store import CycleStore


@pytest.fixture
def store(temp_data_dir):
    return CycleStore(temp_data_dir)


class TestCycleStoreBasic:
    """Core CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Getting unknown umo returns None."""
        result = await store.get("unknown:umo")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, store):
        """Round-trip write then read."""
        await store.set("test:umo", {"anchor_date": "2026-05-01", "enabled": True})
        result = await store.get("test:umo")
        assert result["anchor_date"] == "2026-05-01"
        assert result["enabled"] is True

    @pytest.mark.asyncio
    async def test_set_overwrites(self, store):
        """Second set replaces first."""
        await store.set("test:umo", {"anchor_date": "2026-05-01"})
        await store.set("test:umo", {"anchor_date": "2026-06-01"})
        result = await store.get("test:umo")
        assert result["anchor_date"] == "2026-06-01"

    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Delete removes record."""
        await store.set("test:umo", {"anchor_date": "2026-05-01"})
        await store.delete("test:umo")
        result = await store.get("test:umo")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_no_crash(self, store):
        """Deleting unknown umo is a no-op."""
        await store.delete("never:existed")


class TestCycleStoreToggle:
    """Toggle enabled state."""

    @pytest.mark.asyncio
    async def test_toggle_creates_default(self, store):
        """Toggle on nonexistent record creates one with enabled=True."""
        state, persisted = await store.toggle("new:umo")
        assert state is True and persisted is True
        data = await store.get("new:umo")
        assert data["enabled"] is True

    @pytest.mark.asyncio
    async def test_toggle_flips(self, store):
        """Toggle flips enabled boolean."""
        await store.set("test:umo", {"enabled": True})
        state, persisted = await store.toggle("test:umo")
        assert state is False and persisted is True
        state, persisted = await store.toggle("test:umo")
        assert state is True and persisted is True


class TestCycleStorePersistence:
    """Data survives store re-instantiation."""

    @pytest.mark.asyncio
    async def test_data_survives_reinit(self, temp_data_dir):
        """Simulate plugin reload: new Store instance sees old data."""
        store1 = CycleStore(temp_data_dir)
        await store1.set("persist:umo", {"anchor_date": "2026-04-01"})

        store2 = CycleStore(temp_data_dir)
        result = await store2.get("persist:umo")
        assert result["anchor_date"] == "2026-04-01"

    @pytest.mark.asyncio
    async def test_json_file_created(self, store, temp_data_dir):
        """ cycles.json should exist after first write."""
        await store.set("test:umo", {"anchor_date": "2026-05-01"})
        cycles_path = temp_data_dir / "cycles.json"
        assert cycles_path.exists()

    @pytest.mark.asyncio
    async def test_json_valid_format(self, store, temp_data_dir):
        """Written JSON must be valid and top-level is a dict."""
        await store.set("test:umo", {"anchor_date": "2026-05-01"})
        cycles_path = temp_data_dir / "cycles.json"
        data = json.loads(cycles_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "test:umo" in data

    @pytest.mark.asyncio
    async def test_concurrent_writes_safe(self, store):
        """Multiple rapid sets to same key should not corrupt file."""
        import asyncio
        async def write(i):
            await store.set("concurrent:umo", {"version": i})

        await asyncio.gather(*(write(i) for i in range(10)))
        result = await store.get("concurrent:umo")
        assert isinstance(result["version"], int)


class TestCycleStoreMultiSession:
    """Isolation between sessions."""

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self, store):
        """Different umos do not interfere."""
        await store.set("guild1:user1", {"anchor_date": "2026-05-01"})
        await store.set("guild2:user2", {"anchor_date": "2026-06-01"})

        r1 = await store.get("guild1:user1")
        r2 = await store.get("guild2:user2")
        assert r1["anchor_date"] == "2026-05-01"
        assert r2["anchor_date"] == "2026-06-01"

    @pytest.mark.asyncio
    async def test_partial_delete(self, store):
        """Deleting one session leaves others intact."""
        await store.set("a:umo", {"anchor_date": "2026-05-01"})
        await store.set("b:umo", {"anchor_date": "2026-06-01"})
        await store.delete("a:umo")

        assert await store.get("a:umo") is None
        assert (await store.get("b:umo"))["anchor_date"] == "2026-06-01"


class TestCycleStoreTransactional:
    """P1 回归：CycleStore 事务化——落盘失败不污染缓存、如实返回 False。

    旧实现先更新缓存再写盘、delete 直接改缓存本体：双写失败时缓存已删、
    磁盘还在，Dashboard 重试 404 而重启后记录复活。
    """

    @staticmethod
    def _break_disk(monkeypatch):
        from pathlib import Path

        original_write = Path.write_text

        def always_fail(*args, **kwargs):
            raise OSError("磁盘满")

        monkeypatch.setattr(Path, "write_text", always_fail)
        return original_write

    @pytest.mark.asyncio
    async def test_delete_failure_keeps_cache_and_disk(self, store, monkeypatch):
        await store.set("umo", {"enabled": True, "anchor_date": "2024-01-15"})
        self._break_disk(monkeypatch)
        assert await store.delete("umo") is False
        # 缓存不被污染：当前进程仍读得到，重试不会 404
        assert await store.get("umo") is not None

    @pytest.mark.asyncio
    async def test_delete_failure_retry_succeeds(self, store, monkeypatch):
        from pathlib import Path

        await store.set("umo", {"enabled": True})
        original_write = self._break_disk(monkeypatch)
        assert await store.delete("umo") is False
        monkeypatch.setattr(Path, "write_text", original_write)  # 恢复磁盘
        assert await store.delete("umo") is True
        assert await store.get("umo") is None

    @pytest.mark.asyncio
    async def test_set_failure_returns_false_and_not_cached(self, store, monkeypatch):
        self._break_disk(monkeypatch)
        assert await store.set("umo", {"enabled": True}) is False
        assert await store.get("umo") is None

    @pytest.mark.asyncio
    async def test_toggle_failure_reports_not_persisted(self, store, monkeypatch):
        await store.set("umo", {"enabled": True})
        self._break_disk(monkeypatch)
        state, persisted = await store.toggle("umo")
        assert persisted is False
        # 缓存与磁盘都保持原值
        assert (await store.get("umo"))["enabled"] is True

    @pytest.mark.asyncio
    async def test_get_returns_deep_copy(self, store):
        """get() 必须返回深拷贝：调用方改返回值不得污染缓存。"""
        await store.set("umo", {"enabled": True, "anchor_date": "2024-01-15"})
        cfg = await store.get("umo")
        cfg["anchor_date"] = "1999-01-01"
        cfg["enabled"] = False
        again = await store.get("umo")
        assert again["anchor_date"] == "2024-01-15"
        assert again["enabled"] is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_success(self, store):
        """记录本就不存在视为成功（无需写盘）。"""
        assert await store.delete("never:existed") is True


class TestCycleStoreDefense:
    """P2 回归：存储边界深拷贝与损坏单条记录防御。"""

    @pytest.mark.asyncio
    async def test_set_stores_deep_copy(self, store):
        """set 成功后调用方继续修改原字典不得污染缓存（缓存镜像磁盘）。"""
        cfg = {"enabled": True, "anchor_date": "2024-01-15"}
        await store.set("umo", cfg)
        cfg["anchor_date"] = "1999-01-01"
        assert (await store.get("umo"))["anchor_date"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_corrupted_record_filtered(self, temp_data_dir):
        """单条记录值不是对象时忽略，不影响其他记录读取与 toggle。"""
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "cycles.json").write_text(json.dumps({
            "bad": 7,
            "good": {"enabled": True, "anchor_date": "2024-01-15"},
        }), encoding="utf-8")
        store = CycleStore(temp_data_dir)
        assert await store.get("bad") is None
        assert (await store.get("good"))["anchor_date"] == "2024-01-15"
        # 不得在损坏记录上抛 AttributeError
        state, persisted = await store.toggle("bad")
        assert state is True and persisted is True
