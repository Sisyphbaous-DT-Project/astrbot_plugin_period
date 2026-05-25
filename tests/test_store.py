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
        state = await store.toggle("new:umo")
        assert state is True
        data = await store.get("new:umo")
        assert data["enabled"] is True

    @pytest.mark.asyncio
    async def test_toggle_flips(self, store):
        """Toggle flips enabled boolean."""
        await store.set("test:umo", {"enabled": True})
        state = await store.toggle("test:umo")
        assert state is False
        state = await store.toggle("test:umo")
        assert state is True


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
