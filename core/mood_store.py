"""Session mood state persistence (JSON with atomic writes)."""

import copy
import json
import os
import asyncio
from pathlib import Path

from astrbot.api import logger
from .mood import MoodState


class MoodStore:
    """Persistent storage for per-session emotional state."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._data_dir / "moods.json"
        self._lock = asyncio.Lock()
        self._cache: dict[str, dict] | None = None

    async def _load(self) -> dict[str, dict]:
        """Load all mood data from disk."""
        if self._cache is not None:
            return self._cache

        if not self._file_path.exists():
            self._cache = {}
            logger.info(f"[MoodStore] 情绪数据文件不存在，创建空存储")
            return self._cache

        try:
            content = self._file_path.read_text(encoding="utf-8")
            data = json.loads(content) if content.strip() else {}
            if not isinstance(data, dict):
                data = {}
            logger.info(f"[MoodStore] 加载情绪数据成功，共 {len(data)} 条记录")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[MoodStore] 情绪数据文件读取失败: {e}，使用空存储")
            data = {}

        self._cache = data
        return self._cache

    async def _save(self, data: dict[str, dict]) -> None:
        """Save all mood data to disk atomically."""
        self._cache = data
        tmp_path = self._file_path.with_suffix(".tmp")

        try:
            content = json.dumps(data, ensure_ascii=False, indent=2)
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(self._file_path))
        except OSError:
            # If atomic write fails, try direct write as fallback
            self._file_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def get(self, umo: str) -> MoodState | None:
        """Get mood state by unified message origin."""
        async with self._lock:
            data = await self._load()
            raw = data.get(umo)
            if raw is None:
                return None
            return MoodState.from_dict(raw)

    async def set(self, umo: str, state: MoodState) -> None:
        """Set mood state for a session."""
        async with self._lock:
            all_data = await self._load()
            is_new = umo not in all_data
            all_data[umo] = state.to_dict()
            await self._save(all_data)
            tool_names = [t["name"] for t in state.active_tools]
            if is_new:
                logger.info("[MoodStore] 新增情绪记录: %s, 工具=%s", umo, tool_names)
            else:
                logger.info("[MoodStore] 更新情绪记录: %s, 工具=%s", umo, tool_names)

    async def delete(self, umo: str) -> None:
        """Delete mood state for a session."""
        async with self._lock:
            all_data = await self._load()
            if umo in all_data:
                del all_data[umo]
                await self._save(all_data)
                logger.info("[MoodStore] 删除情绪记录: %s", umo)

    async def get_all(self) -> dict[str, MoodState]:
        """Return all persisted mood states."""
        async with self._lock:
            result: dict[str, MoodState] = {}
            for umo, raw in (await self._load()).items():
                state = MoodState.from_dict(raw)
                result[umo] = state
            return result
