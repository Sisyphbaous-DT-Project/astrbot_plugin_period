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
            else:
                # 损坏文件防御：单条记录的值不是对象时，cfg.get()/in 运算
                # 会在请求链与 Dashboard 抛 TypeError/AttributeError
                bad = [k for k, v in data.items() if not isinstance(v, dict)]
                if bad:
                    logger.warning(
                        f"[CycleStore] 忽略 {len(bad)} 条损坏的会话记录（值不是对象）"
                    )
                    data = {k: v for k, v in data.items() if isinstance(v, dict)}
            logger.info(f"[CycleStore] 加载数据文件成功，共 {len(data)} 条会话记录")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[CycleStore] 数据文件读取失败: {e}，使用空存储")
            data = {}
        
        self._cache = data
        return self._cache

    async def _mutable(self) -> dict[str, dict[str, Any]]:
        """返回当前数据的深拷贝，供修改后交给 _save 原子替换。"""
        return copy.deepcopy(await self._load())

    async def _save(self, data: dict[str, dict[str, Any]]) -> bool:
        """落盘成功后才替换缓存并返回 True；失败时缓存保持原样。

        缓存必须镜像磁盘：若失败时缓存已更新，当前进程会读到未持久化
        的状态（删除假成功、重启后磁盘数据复活，重试又因缓存已删而
        404）。失败只记日志返回 False 不抛出，调用方依据返回值如实
        告知用户。传入的 data 应为 _mutable() 的拷贝。
        """
        tmp_path = self._file_path.with_suffix(".tmp")
        try:
            content = json.dumps(data, ensure_ascii=False, indent=2)
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(self._file_path))
        except OSError:
            try:
                self._file_path.write_text(content, encoding="utf-8")
            except OSError as e:
                logger.warning(f"[CycleStore] 数据落盘失败: {e}")
                return False
        self._cache = data
        return True

    async def get(self, umo: str) -> dict[str, Any] | None:
        """Get session configuration by unified message origin.

        返回深拷贝：调用方修改返回值（如更新 anchor_date 后重新 set）
        不得在 _save 成功前污染缓存。
        """
        async with self._lock:
            data = await self._load()
            cfg = data.get(umo)
            return copy.deepcopy(cfg) if cfg is not None else None

    async def set(self, umo: str, data: dict[str, Any]) -> bool:
        """Set session configuration. 返回是否已持久化。

        存储边界深拷贝：调用方在 set 成功后继续修改原字典不得让
        缓存与磁盘不一致（缓存必须始终镜像磁盘）。
        """
        async with self._lock:
            all_data = await self._mutable()
            is_new = umo not in all_data
            all_data[umo] = copy.deepcopy(data)
            if not await self._save(all_data):
                return False
            if is_new:
                logger.info(f"[CycleStore] 新增会话记录: {umo}")
            else:
                logger.info(f"[CycleStore] 更新会话记录: {umo}")
            return True

    async def delete(self, umo: str) -> bool:
        """Delete session configuration.

        返回是否已持久化；记录本就不存在视为成功（无需写盘）。
        """
        async with self._lock:
            all_data = await self._mutable()
            if umo not in all_data:
                return True
            del all_data[umo]
            if not await self._save(all_data):
                return False
            logger.info(f"[CycleStore] 删除会话记录: {umo}")
            return True

    async def get_all(self) -> dict[str, dict[str, Any]]:
        """Return a deep copy of all persisted session data."""
        async with self._lock:
            return copy.deepcopy(await self._load())

    async def toggle(self, umo: str) -> tuple[bool, bool]:
        """Toggle enabled state for a session. 返回 (新状态, 是否已持久化)。"""
        async with self._lock:
            all_data = await self._mutable()
            if umo not in all_data:
                all_data[umo] = {"enabled": True}
            else:
                current = all_data[umo].get("enabled", False)
                all_data[umo]["enabled"] = not current
            persisted = await self._save(all_data)
            return all_data[umo]["enabled"], persisted
