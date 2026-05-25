"""Session state persistence (JSON with atomic writes)."""

import copy
import json
import os
import asyncio
from pathlib import Path
from typing import Any

from astrbot.api import logger


class CycleStore:
    """Persistent storage for per-session cycle configuration."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._data_dir / "cycles.json"
        self._lock = asyncio.Lock()
        self._cache: dict[str, dict[str, Any]] | None = None

    async def _load(self) -> dict[str, dict[str, Any]]:
        """Load all session data from disk."""
        if self._cache is not None:
            return self._cache
        
        if not self._file_path.exists():
            self._cache = {}
            logger.info(f"[CycleStore] 数据文件不存在，创建空存储: {self._file_path}")
            return self._cache
        
        try:
            content = self._file_path.read_text(encoding="utf-8")
            data = json.loads(content) if content.strip() else {}
            if not isinstance(data, dict):
                data = {}
            logger.info(f"[CycleStore] 加载数据文件成功，共 {len(data)} 条会话记录")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[CycleStore] 数据文件读取失败: {e}，使用空存储")
            data = {}
        
        self._cache = data
        return self._cache

    async def _save(self, data: dict[str, dict[str, Any]]) -> None:
        """Save all session data to disk atomically."""
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

    async def get(self, umo: str) -> dict[str, Any] | None:
        """Get session configuration by unified message origin."""
        async with self._lock:
            data = await self._load()
            return data.get(umo)

    async def set(self, umo: str, data: dict[str, Any]) -> None:
        """Set session configuration."""
        async with self._lock:
            all_data = await self._load()
            is_new = umo not in all_data
            all_data[umo] = data
            await self._save(all_data)
            if is_new:
                logger.info(f"[CycleStore] 新增会话记录: {umo}")
            else:
                logger.info(f"[CycleStore] 更新会话记录: {umo}")

    async def delete(self, umo: str) -> None:
        """Delete session configuration."""
        async with self._lock:
            all_data = await self._load()
            if umo in all_data:
                del all_data[umo]
                await self._save(all_data)
                logger.info(f"[CycleStore] 删除会话记录: {umo}")

    async def get_all(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of all persisted session data."""
        async with self._lock:
            return copy.deepcopy(await self._load())

    async def toggle(self, umo: str) -> bool:
        """Toggle enabled state for a session. Returns new state."""
        async with self._lock:
            all_data = await self._load()
            if umo not in all_data:
                all_data[umo] = {"enabled": True}
            else:
                current = all_data[umo].get("enabled", False)
                all_data[umo]["enabled"] = not current
            await self._save(all_data)
            return all_data[umo]["enabled"]
