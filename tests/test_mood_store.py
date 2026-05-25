"""Tests for core/mood_store.py — MoodStore persistence (v2.1)."""

import pytest

from core.mood import MoodState
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
        state = MoodState(active_tools=[{"name": "cold_violence"}])
        await store.set("test:umo", state)
        retrieved = await store.get("test:umo")
        assert retrieved is not None
        assert retrieved.is_tool_active("cold_violence")

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.set("del:umo", MoodState())
        assert await store.get("del:umo") is not None
        await store.delete("del:umo")
        assert await store.get("del:umo") is None

    @pytest.mark.asyncio
    async def test_get_all(self, store):
        await store.set("a", MoodState(active_tools=[{"name": "a"}]))
        await store.set("b", MoodState(active_tools=[{"name": "b"}]))
        all_data = await store.get_all()
        assert len(all_data) == 2
        assert all_data["a"].is_tool_active("a")
        assert all_data["b"].is_tool_active("b")

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, temp_data_dir):
        store1 = MoodStore(temp_data_dir)
        await store1.set("persist", MoodState(active_tools=[{"name": "test"}]))

        store2 = MoodStore(temp_data_dir)
        state = await store2.get("persist")
        assert state is not None
        assert state.is_tool_active("test")

    @pytest.mark.asyncio
    async def test_corrupted_file_returns_empty(self, temp_data_dir):
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_data_dir / "moods.json"
        file_path.write_text("not json", encoding="utf-8")
        store = MoodStore(temp_data_dir)
        assert await store.get("any") is None
        assert await store.get_all() == {}
