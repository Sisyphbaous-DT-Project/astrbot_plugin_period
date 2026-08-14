"""Session mood state persistence (JSON with atomic writes)."""

import json
import os
import asyncio
from pathlib import Path
from typing import Any

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
        # 可选回调：旧数据迁移产生诊断说明时由插件层记录
        self.on_migration: Any = None  # async (umo, notes: list[str]) -> None

    async def _load(self) -> dict[str, dict]:
        """Load all mood data from disk, migrating legacy records to v3."""
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

        # v1 -> v3 幂等迁移：migrate 对 v3 数据原样返回且 notes 为空
        migrated: dict[str, dict] = {}
        migration_notes: dict[str, list[str]] = {}
        for umo, raw in data.items():
            state, notes = MoodState.migrate(raw if isinstance(raw, dict) else {})
            migrated[umo] = state.to_dict()
            if notes:
                migration_notes[umo] = notes

        if migration_notes:
            saved = await self._save(migrated)  # 迁移结果原子落盘，保证幂等
            for umo, notes in migration_notes.items():
                logger.warning(
                    "[MoodStore] 旧情绪数据已迁移: umo=%s, 说明=%s", umo, notes,
                )
                if self.on_migration is not None:
                    try:
                        await self.on_migration(umo, notes)
                    except Exception as e:
                        logger.warning("[MoodStore] 迁移诊断回调失败: %s", e)
            if saved:
                return self._cache  # _save 已更新缓存
            # 落盘失败不缓存：下次 _load 重新读取并迁移（幂等）
            return migrated

        self._cache = migrated
        return self._cache

    async def _mutable(self) -> dict[str, dict]:
        """返回当前数据的深拷贝，供修改后交给 _save 原子替换。"""
        return json.loads(json.dumps(await self._load()))

    async def _save(self, data: dict[str, dict]) -> bool:
        """落盘成功后才替换缓存并返回 True；失败时缓存保持原样。

        缓存必须镜像磁盘：失败时若缓存已更新，当前进程会读到未持久化的
        状态（重启后凭空回退），删除也会假成功。失败只记日志返回 False，
        绝不抛出——情绪是附属功能，写盘故障不得穿透请求钩子，
        也不得让 /period lift 这类安全出口失效。
        """
        tmp_path = self._file_path.with_suffix(".tmp")
        try:
            content = json.dumps(data, ensure_ascii=False, indent=2)
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(self._file_path))
        except OSError:
            # If atomic write fails, try direct write as fallback
            try:
                self._file_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as e:
                logger.warning("[MoodStore] 情绪数据落盘失败: %s", e)
                return False
        self._cache = data
        return True

    async def get(self, umo: str) -> MoodState | None:
        """Get mood state by unified message origin."""
        async with self._lock:
            data = await self._load()
            raw = data.get(umo)
            if raw is None:
                return None
            return MoodState.from_dict(raw)

    async def set(self, umo: str, state: MoodState) -> bool:
        """Set mood state for a session. 返回是否已持久化（失败时缓存不变）。"""
        async with self._lock:
            all_data = await self._mutable()
            is_new = umo not in all_data
            all_data[umo] = state.to_dict()
            if not await self._save(all_data):
                logger.warning("[MoodStore] 情绪状态落盘失败（未生效）: %s", umo)
                return False
            action_names = [a.name for a in state.persistent_actions]
            if is_new:
                logger.info("[MoodStore] 新增情绪记录: %s, 动作=%s", umo, action_names)
            else:
                logger.info("[MoodStore] 更新情绪记录: %s, 动作=%s", umo, action_names)
            return True

    async def delete(self, umo: str) -> bool:
        """Delete mood state for a session. 返回是否已持久化（无记录也算成功）。"""
        async with self._lock:
            all_data = await self._mutable()
            if umo not in all_data:
                return True
            del all_data[umo]
            if not await self._save(all_data):
                logger.warning("[MoodStore] 情绪记录删除落盘失败（未生效）: %s", umo)
                return False
            logger.info("[MoodStore] 删除情绪记录: %s", umo)
            return True

    async def get_all(self) -> dict[str, MoodState]:
        """Return all persisted mood states."""
        async with self._lock:
            result: dict[str, MoodState] = {}
            for umo, raw in (await self._load()).items():
                state = MoodState.from_dict(raw)
                result[umo] = state
            return result
