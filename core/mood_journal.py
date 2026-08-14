"""Emotion diary system - per-person long-term memory maintained by a small model.

架构：
- 存储：emotion_diaries.json 原子信封（schema_version/diaries/pending_events outbox/
  processed_event_ids 有界环）；
- 键：platform_id + bot_self_id + sender_id（同平台实例同机器人下跨群/私聊共用）；
- 处理：事件先持久化到 outbox 再触发后台任务；同一日记 FIFO 单 worker，
  不同日记最多 max_parallel 个 worker 并行；插件重启后自动续跑；
- 工具循环：插件私有手工 JSON 循环（diary_write/diary_edit/diary_count），
  不注册全局 AstrBot 工具，不持有 event/req 对象；内存草稿操作，
  满足提交条件（事件已写 + 最后一次成功工具为 count + 不超限）才原子提交。

隐私：事件与日记内容均为脱敏摘要，本模块不接触聊天原文、完整人格或②的原回答。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from astrbot.api import logger

SCHEMA_VERSION = 1
PROCESSED_IDS_RING = 500
MAX_ALIASES = 10

# 工具循环默认约束
DEFAULT_MAX_STEPS = 12
DEFAULT_STEP_TIMEOUT = 30.0
DEFAULT_TOTAL_TIMEOUT = 120.0
DEFAULT_MAX_PARALLEL = 2
DEFAULT_MAX_CHARS = 4000

# 失败事件的运行期重试：阻塞同键队列以保住严格 FIFO，指数退避封顶。
# 确认不再需要的“毒事件”用 diaryclear 显式清除（会一并清出 outbox）。
DEFAULT_RETRY_DELAY = 60.0
MAX_RETRY_DELAY = 300.0

EVENT_KIND_LABELS = {
    "mood_changed": "心境变化",
    "action_activated": "决定动作",
    "action_lifted": "解除动作",
    "action_expired": "动作到期",
    "fully_recovered": "完全恢复",
    "manual_lift": "用户手动解除",
}


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def count_diary_chars(entries: list[dict]) -> int:
    """日记字符数：条目文本之和 + 条目间换行。"""
    if not entries:
        return 0
    return sum(len(e.get("text", "")) for e in entries) + (len(entries) - 1)


class DiaryStore:
    """emotion_diaries.json 的原子持久化（信封 + outbox）。"""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._data_dir / "emotion_diaries.json"
        self._lock = asyncio.Lock()
        self._cache: dict | None = None

    # ------------------------------------------------------------------ #
    #  基础读写
    # ------------------------------------------------------------------ #

    def _empty_envelope(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "diaries": {},
            "pending_events": [],
            "processed_event_ids": [],
            "umo_watermarks": {},
        }

    async def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if not self._file_path.exists():
            self._cache = self._empty_envelope()
            return self._cache
        try:
            content = self._file_path.read_text(encoding="utf-8")
            data = json.loads(content) if content.strip() else {}
            if not isinstance(data, dict):
                data = {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[DiaryStore] 日记文件读取失败: %s，使用空存储", e)
            data = {}
        envelope = self._empty_envelope()
        if isinstance(data.get("diaries"), dict):
            # 逐本清洗：损坏文件里 entries/aliases 等字段可能是标量，
            # 不清洗会在注入文本构建（请求链上、调用①之前）抛 TypeError
            for key, diary in data["diaries"].items():
                if not isinstance(diary, dict):
                    continue
                entries = diary.get("entries")
                cleaned_entries = []
                if isinstance(entries, list):
                    for e in entries:
                        if not isinstance(e, dict) or not isinstance(e.get("text"), str):
                            continue
                        # 工具循环按下标读取 id（diary_count 的 earliest_id），
                        # 缺 id 的条目会变成毒事件永久阻塞 FIFO：
                        # 补生成而非丢弃，保留用户数据
                        if not isinstance(e.get("id"), str) or not e["id"]:
                            e["id"] = uuid.uuid4().hex[:12]
                        if not isinstance(e.get("event_id"), str):
                            e["event_id"] = ""
                        if not isinstance(e.get("occurred_at"), str):
                            e["occurred_at"] = ""
                        cleaned_entries.append(e)
                diary["entries"] = cleaned_entries
                aliases = diary.get("aliases")
                diary["aliases"] = (
                    [a for a in aliases if isinstance(a, str) and a]
                    if isinstance(aliases, list) else []
                )
                if not isinstance(diary.get("display_name"), str):
                    diary["display_name"] = ""
                if not isinstance(diary.get("updated_at"), str):
                    diary["updated_at"] = ""
                diary["owner_key"] = key  # 以信封键为准
                envelope["diaries"][key] = diary
        if isinstance(data.get("pending_events"), list):
            # 坏事件会让恢复/worker 下标访问崩溃或永久阻塞队列：
            # id 与 owner_key 是路由必需字段，缺失即丢弃；其余字段规整类型
            for e in data["pending_events"]:
                if not isinstance(e, dict):
                    continue
                if not isinstance(e.get("id"), str) or not e["id"]:
                    continue
                if not isinstance(e.get("owner_key"), str) or not e["owner_key"]:
                    continue
                for field in ("kind", "summary", "display_name", "provider_id", "umo", "occurred_at"):
                    if not isinstance(e.get(field), str):
                        e[field] = ""
                envelope["pending_events"].append(e)
        if isinstance(data.get("processed_event_ids"), list):
            envelope["processed_event_ids"] = [
                str(i) for i in data["processed_event_ids"]
            ][-PROCESSED_IDS_RING:]
        if isinstance(data.get("umo_watermarks"), dict):
            # reset/删除会话的丢弃水位线（持久化，重启后仍然生效）
            envelope["umo_watermarks"] = {
                str(k): v for k, v in data["umo_watermarks"].items()
                if isinstance(v, str) and v
            }
        self._cache = envelope
        return self._cache

    async def _mutable(self) -> dict:
        """返回当前信封的深拷贝，供调用方修改后交给 _save 原子替换。"""
        return json.loads(json.dumps(await self._load()))

    async def _save(self, data: dict) -> bool:
        """落盘成功后才替换内存缓存并返回 True；失败时缓存保持原样。

        缓存必须始终镜像磁盘：若失败时缓存已更新，未落盘的数据会被当前
        进程读取并注入模型、重启后却凭空消失；删除操作也会"假成功"
        （指令报已清除，重启后复活）。失败只记日志不抛出——日记是附属
        功能，任何写入故障都不得打断主请求链；调用方依据返回值区分
        "已持久化"与"未生效"。传入的 data 应为 _mutable() 的拷贝。
        """
        tmp_path = self._file_path.with_suffix(".tmp")
        content = json.dumps(data, ensure_ascii=False, indent=2)
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(self._file_path))
        except OSError:
            try:
                self._file_path.write_text(content, encoding="utf-8")
            except OSError as e:
                logger.warning("[DiaryStore] 日记落盘失败: %s", e)
                return False
        self._cache = data
        return True

    # ------------------------------------------------------------------ #
    #  日记本体
    # ------------------------------------------------------------------ #

    async def get_diary(self, owner_key: str) -> dict | None:
        async with self._lock:
            data = await self._load()
            diary = data["diaries"].get(owner_key)
            return json.loads(json.dumps(diary)) if diary else None

    async def upsert_diary(
        self,
        owner_key: str,
        entries: list[dict],
        display_name: str = "",
        aliases: list[str] | None = None,
    ) -> bool:
        """写入日记本体，返回落盘是否成功（False 时调用方可选择重试）。"""
        async with self._lock:
            data = await self._mutable()
            diary = data["diaries"].get(owner_key) or {
                "owner_key": owner_key,
                "display_name": "",
                "aliases": [],
                "entries": [],
                "updated_at": "",
            }
            if display_name:
                diary["display_name"] = display_name
            for alias in aliases or []:
                if alias and alias not in diary["aliases"]:
                    diary["aliases"].append(alias)
            diary["aliases"] = diary["aliases"][-MAX_ALIASES:]
            diary["entries"] = entries
            diary["updated_at"] = _utc_now_iso()
            data["diaries"][owner_key] = diary
            return await self._save(data)

    async def delete_diary(self, owner_key: str) -> tuple[bool, int, bool]:
        """删除已提交日记与该 owner 的未处理事件。

        返回 (是否删掉了已提交日记, 清理的待处理事件数, 是否已持久化)。
        清除必须连同 pending 事件一起删除，否则后台 worker 会把已清除
        内容重新写回；落盘失败时磁盘与缓存都保持原样（第三个返回值
        为 False），调用方必须如实告知用户清除未生效，不得假成功。
        """
        async with self._lock:
            data = await self._mutable()
            removed = owner_key in data["diaries"]
            if removed:
                del data["diaries"][owner_key]
            before = len(data["pending_events"])
            data["pending_events"] = [
                e for e in data["pending_events"] if e.get("owner_key") != owner_key
            ]
            removed_events = before - len(data["pending_events"])
            if not removed and not removed_events:
                return False, 0, True
            if not await self._save(data):
                logger.warning(
                    "[DiaryStore] 删除日记落盘失败，清除未生效: %s", owner_key,
                )
                return False, 0, False
            return removed, removed_events, True

    async def discard_pending_events(
        self, umo: str, owner_key: str | None = None,
    ) -> int:
        """丢弃指定会话（UMO）来源的未处理事件，返回丢弃数；落盘失败返回 -1。

        供 /period reset 调用：周期锚点被清除后，该会话失效前滞留的
        日记事件不应再被处理（日记是周期附属功能）。已提交日记保留。
        指定 owner_key 时只丢弃该 owner 的匹配事件（调用方按 owner
        逐个在提交锁内调用，与提交临界区互斥）。
        """
        async with self._lock:
            data = await self._mutable()

            def _match(e: dict) -> bool:
                if e.get("umo") != umo:
                    return False
                return owner_key is None or e.get("owner_key") == owner_key

            before = len(data["pending_events"])
            data["pending_events"] = [
                e for e in data["pending_events"] if not _match(e)
            ]
            removed = before - len(data["pending_events"])
            if not removed:
                return 0
            if not await self._save(data):
                logger.warning(
                    "[DiaryStore] 丢弃会话待处理事件落盘失败: %s", umo,
                )
                return -1
            return removed

    async def mark_umo_watermark(self, umo: str) -> bool:
        """记录会话（UMO）的丢弃水位线，返回是否已持久化。

        /period reset 或删除会话时调用：水位线之前发生的事件属于已被
        重置的周期，即使因竞态在 reset 之后才提交到 outbox，也会被
        enqueue/worker 按过期事件拒绝。持久化是为了重启后仍然生效
        （outbox 本身是磁盘数据，内存水位线重启即失效）。
        """
        async with self._lock:
            data = await self._mutable()
            data["umo_watermarks"][umo] = _utc_now_iso()
            if not await self._save(data):
                logger.warning(
                    "[DiaryStore] 会话丢弃水位线落盘失败: %s", umo,
                )
                return False
            return True

    @staticmethod
    def is_stale_event(event: dict, envelope: dict) -> bool:
        """事件是否早于其来源会话的丢弃水位线（即属于已重置的旧周期）。

        occurred_at 与水位线均由 _utc_now_iso 生成（同一 UTC ISO 格式），
        字符串比较即时间先后。无 umo 的旧事件或无水位线时不过期。
        """
        umo = event.get("umo") or ""
        if not umo:
            return False
        watermark = envelope.get("umo_watermarks", {}).get(umo)
        if not watermark:
            return False
        occurred = event.get("occurred_at") or ""
        return bool(occurred) and occurred <= watermark

    async def is_event_stale(self, event: dict) -> bool:
        """带锁判定事件是否早于其来源会话的丢弃水位线。"""
        async with self._lock:
            return self.is_stale_event(event, await self._load())

    async def list_namespace(self, platform_id: str, bot_self_id: str) -> list[dict]:
        """列出同一命名空间（平台实例+机器人账号）下的所有日记。"""
        prefix = f"{platform_id}:{bot_self_id}:"
        async with self._lock:
            data = await self._load()
            return [
                json.loads(json.dumps(d))
                for k, d in data["diaries"].items() if k.startswith(prefix)
            ]

    # ------------------------------------------------------------------ #
    #  Outbox 与去重
    # ------------------------------------------------------------------ #

    async def enqueue(self, event: dict) -> bool:
        """事件先持久化到 outbox。重复事件 ID 直接忽略。

        落盘失败时缓存保持原样并返回 False——outbox 的意义就是崩溃
        可恢复，不能把仅入内存的事件谎报为成功。
        """
        async with self._lock:
            data = await self._mutable()
            if self.is_stale_event(event, data):
                # reset 与进行中请求的竞态：reset 已记录水位线，旧周期事件
                # 即使此刻才提交也不得进入 outbox（与 discard 在同一把
                # 存储锁内线性化）
                logger.info(
                    "[DiaryStore] 事件 %s 早于会话丢弃水位线（来源周期已重置），拒绝入队",
                    event["id"],
                )
                return False
            if event["id"] in data["processed_event_ids"]:
                return False
            if any(e["id"] == event["id"] for e in data["pending_events"]):
                return False
            data["pending_events"].append(event)
            if not await self._save(data):
                logger.warning(
                    "[DiaryStore] 事件 %s 落盘失败，未入队", event["id"],
                )
                return False
            return True

    async def pending_events(self) -> list[dict]:
        async with self._lock:
            data = await self._load()
            return list(data["pending_events"])

    async def ack(self, event_id: str) -> bool:
        """事件处理完成：移出 outbox 并记入已处理环。返回是否已持久化。

        落盘失败时事件仍留在 outbox（缓存镜像磁盘），调用方可据此
        退避重试而不是出队丢失。
        """
        async with self._lock:
            data = await self._mutable()
            data["pending_events"] = [
                e for e in data["pending_events"] if e["id"] != event_id
            ]
            if event_id not in data["processed_event_ids"]:
                data["processed_event_ids"].append(event_id)
                data["processed_event_ids"] = data["processed_event_ids"][-PROCESSED_IDS_RING:]
            if not await self._save(data):
                logger.warning(
                    "[DiaryStore] ack 落盘失败，事件 %s 仍在 outbox", event_id,
                )
                return False
            return True

    async def is_processed(self, event_id: str) -> bool:
        async with self._lock:
            data = await self._load()
            return event_id in data["processed_event_ids"]

    async def is_pending(self, event_id: str) -> bool:
        async with self._lock:
            data = await self._load()
            return any(e.get("id") == event_id for e in data["pending_events"])


# --------------------------------------------------------------------------- #
#  日记维护器（异步 worker + 私有手工工具循环）
# --------------------------------------------------------------------------- #

DIARY_SYSTEM_PROMPT = """你是 Bot 的长期情绪日记维护助手。你的任务：根据给定事件，把日记维护成简洁、真实、有连续性的自然语言记录。

【工作方式】
你每轮只能输出一个 JSON 工具调用（不要输出其他内容，不要 markdown）：
- {"tool": "diary_write", "args": {"text": "为当前事件写的完整日记条目"}}
- {"tool": "diary_edit", "args": {"entry_id": "条目ID", "operation": "replace", "text": "替换后的完整条目"}}
- {"tool": "diary_edit", "args": {"entry_id": "条目ID", "operation": "delete"}}
- {"tool": "diary_count", "args": {}}

【硬性规则】
1. 每个事件必须用 diary_write 写一条新条目；同一事件只能写一次。
2. 日记总字符数不得超过上限；超限时用 diary_edit(delete) 删除最早条目，再用 diary_count 确认。
3. diary_edit(delete) 只能删除当前最早的完整条目。
4. 结束前最后一次工具调用必须是 diary_count，且确认不超限。
5. 条目用 Bot 第一人称的自然语言，简洁（每条 100 字以内），不抄录用户原话。"""


class DiaryJournal:
    """情绪日记的队列调度与工具循环处理。"""

    def __init__(
        self,
        data_dir: Path,
        resolve_provider: Callable[[str], Any],
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_steps: int = DEFAULT_MAX_STEPS,
        step_timeout: float = DEFAULT_STEP_TIMEOUT,
        total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        max_attempts: int = 2,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        config_getter: Callable[[str, Any], Any] | None = None,
        enabled_getter: Callable[[], bool] | None = None,
        umo_active_getter: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self.store = DiaryStore(data_dir)
        self._resolve_provider = resolve_provider
        self.max_chars = max_chars
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.total_timeout = total_timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self._config_getter = config_getter
        # 情绪/日记总开关回调：关闭时事件延后处理（不调用模型、不写日记），
        # 留在 outbox，重开或重启后自然恢复
        self._enabled_getter = enabled_getter
        # 会话级周期门控回调：事件来源 UMO 的周期失效（toggle/删会话/白名单
        # 变更等）时延后处理；判定异常按失效处理（保守）
        self._umo_active_getter = umo_active_getter
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: dict[str, asyncio.Task] = {}
        # 每个 owner 的清除纪元：diaryclear 时递增，处理中的事件在提交前
        # 比对纪元，发现已清除则放弃提交，防止已删日记被 in-flight 草稿复活
        self._epochs: dict[str, int] = {}
        # 每个 owner 的提交锁：clear_owner 与提交/裁剪在同一临界区比对纪元，
        # 消除“检查通过后被清除”的竞态
        self._commit_locks: dict[str, asyncio.Lock] = {}
        self._started = False

    def _current_max_chars(self) -> int:
        """处理时实时读取上限，保证配置修改即时生效。"""
        if self._config_getter is not None:
            try:
                value = int(self._config_getter("diary_max_chars", self.max_chars))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        return self.max_chars

    # ------------------------------------------------------------------ #
    #  键与注入文本
    # ------------------------------------------------------------------ #

    @staticmethod
    def make_owner_key(platform_id: str, bot_self_id: str, sender_id: str) -> str | None:
        """任一身份字段缺失时返回 None（调用方记诊断，禁止退化键）。"""
        if not platform_id or not bot_self_id or not sender_id:
            return None
        return f"{platform_id}:{bot_self_id}:{sender_id}"

    @staticmethod
    def namespace_of(owner_key: str) -> str:
        parts = owner_key.split(":")
        return ":".join(parts[:2]) if len(parts) >= 3 else owner_key

    def build_injection_text(self, diary: dict | None, max_chars: int | None = None) -> str:
        """生成注入正式主模型与调用②的日记文本（长期记忆）。

        max_chars 为读取侧实时上限：用户调低 diary_max_chars 后、后台
        worker 完成裁剪前，旧日记仍可能超限，注入时按当前配置截断，
        避免超限内容持续进入模型上下文。从尾部保留最新条目（长期记忆
        越新越相关）。
        """
        if not diary or not diary.get("entries"):
            return ""
        header = "[情绪日记（长期记忆，按时间从旧到新，仅供你感知）]"
        lines = [header]
        for e in diary["entries"]:
            lines.append(f"- {e.get('text', '')}")
        text = "\n".join(lines)
        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            marker = "\n…（较早内容已省略）\n"
            budget = max_chars - len(header) - len(marker)
            if budget <= 0:
                return text[-max_chars:]
            text = header + marker + text[-budget:]
        return text

    # ------------------------------------------------------------------ #
    #  事件提交与生命周期
    # ------------------------------------------------------------------ #

    async def submit(
        self,
        owner_key: str,
        kind: str,
        summary: str,
        *,
        display_name: str = "",
        provider_id: str = "",
        umo: str = "",
        occurred_at: str = "",
    ) -> bool:
        """事件持久化到 outbox 后触发后台处理。不阻塞调用方。

        umo 记录事件来源会话：/period reset 后按会话丢弃滞留事件。
        occurred_at 由调用方在产生事件时捕获（而非入队时刻）：reset 的
        丢弃水位线按事件实际发生时间判定过期，覆盖"检查通过→reset→
        入队"的竞态窗口。enqueue 与入队在 owner 提交锁内完成：与
        clear_owner/discard/提交临界区互斥——先于清除的事件必被清除
        （不会清除后才落盘复活），后于清除的事件才算新事件；也不会出现
        "磁盘 pending 有、队列没有"的失联事件。
        """
        event = {
            "id": uuid.uuid4().hex[:12],
            "owner_key": owner_key,
            "kind": kind,
            "summary": summary[:200],
            "display_name": display_name,
            "provider_id": provider_id,
            "umo": umo,
            "occurred_at": occurred_at or _utc_now_iso(),
        }
        async with self._commit_lock(owner_key):
            queued = await self.store.enqueue(event)
            if not queued:
                return False
            self._ensure_worker(owner_key).put_nowait(event["id"])
        return True

    def _ensure_worker(self, owner_key: str) -> asyncio.Queue:
        queue = self._queues.get(owner_key)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[owner_key] = queue
        worker = self._workers.get(owner_key)
        if worker is None or worker.done():
            self._workers[owner_key] = asyncio.create_task(self._worker(owner_key, queue))
        return queue

    async def start(self) -> None:
        """插件加载后调用：恢复 outbox 中未处理的事件。"""
        if self._started:
            return
        self._started = True
        for event in await self.store.pending_events():
            self._ensure_worker(event["owner_key"]).put_nowait(event["id"])
        pending = len(await self.store.pending_events())
        if pending:
            logger.info("[DiaryJournal] 恢复 %d 条未处理日记事件", pending)

    async def shutdown(self) -> None:
        """插件卸载：outbox 已随每次操作落盘，这里只取消 worker。"""
        for worker in self._workers.values():
            if not worker.done():
                worker.cancel()
        for worker in self._workers.values():
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass
        self._workers.clear()
        self._queues.clear()
        self._started = False

    async def wait_idle(self) -> None:
        """等待当前所有日记队列清空（测试与关停前使用）。"""
        for queue in list(self._queues.values()):
            await queue.join()

    def _commit_lock(self, owner_key: str) -> asyncio.Lock:
        lock = self._commit_locks.get(owner_key)
        if lock is None:
            lock = asyncio.Lock()
            self._commit_locks[owner_key] = lock
        return lock

    async def clear_owner(self, owner_key: str) -> tuple[bool, int, bool]:
        """彻底清除某人的日记：删除已提交内容与未处理事件、取消队列中
        尚未处理的事件，并递增纪元使 in-flight 处理放弃提交。

        与提交路径共用每 owner 提交锁，纪元比对和删除在同一临界区完成，
        消除“检查通过后才被清除”的竞态。
        返回 (是否删掉了已提交日记, 清理的待处理事件数, 是否已持久化)；
        第三个值为 False 时清除未生效，指令层必须如实告知用户。
        """
        async with self._commit_lock(owner_key):
            # 先删除：落盘失败时不动纪元、不抽队列——清除未生效，
            # in-flight 提交照常，磁盘 outbox 中的事件仍由 worker 继续
            # 处理，FIFO 不受影响
            removed_diary, removed_events, persisted = await self.store.delete_diary(
                owner_key,
            )
            if not persisted:
                return False, 0, False
            # 全程持有提交锁：bump 移到删除后不会重开竞态——提交方要么已在
            # 本临界区之前完成（其内容刚被删掉），要么在锁外等待（随后发现
            # 纪元不匹配/事件不在 outbox 而放弃）
            self._epochs[owner_key] = self._epochs.get(owner_key, 0) + 1
            queue = self._queues.get(owner_key)
            if queue is not None:
                drained = 0
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    queue.task_done()
                    drained += 1
                if drained:
                    logger.info(
                        "[DiaryJournal] 已丢弃 %s 队列中 %d 条未处理事件", owner_key, drained,
                    )
            return removed_diary, removed_events, True

    async def discard_pending_for_umo(self, umo: str) -> int:
        """/period reset 后调用：丢弃该会话来源的未处理事件。

        返回丢弃数；任何 owner 落盘失败返回 -1（调用方应如实提示用户，
        不得与"没有事件"混淆）。

        已提交的日记保留（reset 只重置周期，不是清除日记）。按受影响
        owner 逐个在其提交锁内删除：提交路径的"is_pending 检查 + upsert"
        在同一临界区，两者互斥，焊死"检查通过后才被 reset"的竞态。
        不递增纪元：其他会话来源的同 owner 事件不受影响。

        无论当前 outbox 是否有匹配事件，都先记录该 UMO 的持久化丢弃
        水位线：进行中请求可能在 reset 之后才提交旧周期事件（快照竞态），
        enqueue 会在存储锁内比对水位线拒绝入队，重启后仍然生效。
        """
        if not await self.store.mark_umo_watermark(umo):
            logger.warning(
                "[DiaryJournal] 会话周期已重置，但丢弃水位线落盘失败",
            )
            return -1
        owners = sorted({
            e.get("owner_key")
            for e in await self.store.pending_events()
            if e.get("umo") == umo and e.get("owner_key")
        })
        total = 0
        for owner_key in owners:
            async with self._commit_lock(owner_key):
                removed = await self.store.discard_pending_events(
                    umo, owner_key=owner_key,
                )
            if removed < 0:
                logger.warning(
                    "[DiaryJournal] 会话周期已重置，但待处理日记事件清理落盘失败",
                )
                return -1
            total += removed
        if total:
            logger.info(
                "[DiaryJournal] 会话周期已重置，丢弃 %d 条待处理日记事件", total,
            )
        return total

    def _epoch(self, owner_key: str) -> int:
        return self._epochs.get(owner_key, 0)

    async def _worker(self, owner_key: str, queue: asyncio.Queue) -> None:
        while True:
            event_id = await queue.get()
            try:
                async with self._semaphore:
                    done = await self._process_by_id(event_id)
                cycle = 0
                while not done:
                    # 严格 FIFO：失败/被暂停的事件持续退避重试，后续事件不得
                    # 超车；确认不再需要的事件用 diaryclear 显式清除
                    cycle += 1
                    await asyncio.sleep(self._retry_delay_for(cycle))
                    async with self._semaphore:
                        done = await self._process_by_id(event_id)
                    if not done and cycle == 1:
                        logger.warning(
                            "[DiaryJournal] 事件 %s 处理失败/暂停，进入退避重试",
                            event_id,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[DiaryJournal] 事件处理异常: %s", type(e).__name__)
            finally:
                queue.task_done()

    def _retry_delay_for(self, cycle: int) -> float:
        """指数退避：retry_delay × 2^(cycle-1)，封顶 MAX_RETRY_DELAY。

        先限制指数再算幂：cycle 极大（约 3.5 天持续失败）时 2**(cycle-1)
        转 float 会抛 OverflowError，导致事件被 worker 吞异常后出队丢失。
        """
        exponent = min(max(cycle - 1, 0), 20)
        return min(self.retry_delay * (2 ** exponent), MAX_RETRY_DELAY)

    async def _source_umo_active(self, event: dict) -> bool:
        """事件来源会话的周期是否仍有效。

        无 umo 的旧事件视为有效（向后兼容）；判定异常按失效处理（保守）。
        """
        if self._umo_active_getter is None or not event.get("umo"):
            return True
        try:
            return bool(await self._umo_active_getter(event["umo"]))
        except Exception:
            return False

    async def _process_by_id(self, event_id: str) -> bool:
        """处理一条事件。返回 True 表示已完成（或无需再处理）。"""
        if self._enabled_getter is not None and not self._enabled_getter():
            return False  # 开关关闭：延后处理，退避循环周期性复查，事件留 outbox
        if await self.store.is_processed(event_id):
            await self.store.ack(event_id)  # 确保 outbox 清理
            return True
        pending = await self.store.pending_events()
        event = next((e for e in pending if e["id"] == event_id), None)
        if event is None:
            return True  # 已被清除或不存在，无需重试
        # 丢弃水位线：事件属于已被 reset/删除的旧周期，直接移出 outbox，
        # 不处理、不阻塞 FIFO（enqueue 侧已拦截，此处兜底存量事件）
        if await self.store.is_event_stale(event):
            logger.info(
                "[DiaryJournal] 事件 %s 早于会话丢弃水位线（来源周期已重置），丢弃",
                event_id,
            )
            await self.store.ack(event_id)
            return True
        # 会话级周期门控：日记是有效周期的附属功能，来源 UMO 的周期已失效
        # （toggle/删除会话/白名单变更等）时延后处理，重新有效后自动恢复
        if not await self._source_umo_active(event):
            return False
        return await self._process_with_retry(event)

    # ------------------------------------------------------------------ #
    #  工具循环
    # ------------------------------------------------------------------ #

    async def _process_with_retry(self, event: dict) -> bool:
        for attempt in range(1, self.max_attempts + 1):
            # 尝试之间快速检查：事件可能在上一次尝试的模型调用窗口里
            # 被 diaryclear 清出 outbox（纪元会在清除后被重新捕获，单靠
            # 纪元比对防不住这种复活）
            if not await self.store.is_pending(event["id"]):
                logger.info(
                    "[DiaryJournal] 事件 %s 已不在 outbox（可能被清除），放弃处理",
                    event["id"],
                )
                return True
            try:
                done = await self._process_event(event)
            except Exception as e:
                logger.warning(
                    "[DiaryJournal] 事件 %s 第 %d 次处理异常: %s",
                    event["id"], attempt, type(e).__name__,
                )
                done = False
            if done:
                if await self.store.ack(event["id"]):
                    return True
                # ack 落盘失败：事件仍在 outbox，退避重试而非出队丢失
                logger.warning(
                    "[DiaryJournal] 事件 %s ack 落盘失败，将延迟重试", event["id"],
                )
                return False
        logger.warning(
            "[DiaryJournal] 事件 %s 本轮处理失败，将延迟重试", event["id"],
        )
        return False

    async def _process_event(self, event: dict) -> bool:
        """处理单条事件。返回 True 表示已提交可 ack（或已被清除需放弃）。"""
        owner_key = event["owner_key"]
        epoch = self._epoch(owner_key)
        provider = self._resolve_provider(event.get("provider_id", ""))
        if provider is None:
            logger.warning("[DiaryJournal] 无可用日记模型，事件 %s 暂缓", event["id"])
            return False

        diary = await self.store.get_diary(owner_key) or {
            "owner_key": owner_key,
            "display_name": event.get("display_name", ""),
            "aliases": [],
            "entries": [],
            "updated_at": "",
        }
        max_chars = self._current_max_chars()

        # 宿主确定性硬裁剪：用户调低上限导致存量超限时，先删最早条目保硬上限
        entries = list(diary["entries"])
        trimmed = False
        while entries and count_diary_chars(entries) > max_chars:
            entries.pop(0)
            trimmed = True
        if trimmed:
            async with self._commit_lock(owner_key):
                if self._epoch(owner_key) != epoch or not await self.store.is_pending(event["id"]):
                    return True  # 处理期间已被 diaryclear，放弃提交
                await self.store.upsert_diary(
                    owner_key, entries,
                    display_name=diary.get("display_name", ""),
                    aliases=diary.get("aliases", []),
                )
            logger.info("[DiaryJournal] 日记超出上限，宿主已裁剪最早条目")

        draft = list(entries)  # 内存草稿，提交前不落盘
        event_written = any(e.get("event_id") == event["id"] for e in draft)
        last_successful_tool = ""
        started = time.monotonic()

        kind_label = EVENT_KIND_LABELS.get(event["kind"], event["kind"])
        diary_view = self._format_diary_for_model(draft)
        sender_id = owner_key.split(":")[-1]
        display_name = event.get("display_name", "")
        owner_line = (
            f"【日记主人】{display_name}（QQ号 {sender_id}）\n"
            if display_name else f"【日记主人】QQ号 {sender_id}\n"
        )
        messages = [
            (
                f"{owner_line}"
                f"【当前事件】{kind_label}：{event['summary']}"
                f"（发生于 {event['occurred_at']}）\n\n"
                f"【当前日记】（字符上限 {max_chars}）\n{diary_view}\n\n"
                f"请维护日记：为事件写条目，确保不超限，最后用 diary_count 确认。"
            ),
        ]

        for step in range(self.max_steps):
            # 处理中关闭开关必须即时生效：每步模型调用前复查；
            # 来源会话周期失效同理（工具循环可能长达数分钟）
            if self._enabled_getter is not None and not self._enabled_getter():
                logger.info(
                    "[DiaryJournal] 处理期间开关关闭，事件 %s 延后", event["id"],
                )
                return False
            if not await self._source_umo_active(event):
                logger.info(
                    "[DiaryJournal] 处理期间来源会话周期失效，事件 %s 延后",
                    event["id"],
                )
                return False
            # 总超时是硬上限：单步只给剩余时间，不允许再跑满一个 step_timeout
            remaining = self.total_timeout - (time.monotonic() - started)
            if remaining <= 0:
                logger.warning("[DiaryJournal] 事件 %s 处理总超时", event["id"])
                return False
            try:
                resp = await asyncio.wait_for(
                    provider.text_chat(
                        prompt=messages[-1],
                        system_prompt=DIARY_SYSTEM_PROMPT,
                        contexts=[{"role": "user", "content": m} for m in messages[:-1]],
                    ),
                    timeout=min(self.step_timeout, remaining),
                )
                text = resp.completion_text or ""
            except Exception as e:
                # 只记异常类型：Provider 异常消息可能回显请求体（含日记内容）
                logger.warning(
                    "[DiaryJournal] 日记模型调用失败: %s", type(e).__name__,
                )
                return False

            call = self._parse_tool_call(text)
            if call is None:
                messages.append("无法解析你的工具调用，请只输出一个 JSON 工具调用。")
                continue

            tool, args = call
            result, success = self._execute_tool(
                tool, args, draft, event, event_written, max_chars,
            )
            if success:
                last_successful_tool = tool
                if tool in ("diary_write", "diary_edit"):
                    # 删除工具可能删掉的正是当前事件的条目，必须重新计算
                    event_written = any(
                        e.get("event_id") == event["id"] for e in draft
                    )
            messages.append(f"工具结果：{json.dumps(result, ensure_ascii=False)}")

            # 提交条件：事件已写 + 最后一次成功工具为 count + 不超限
            if (
                event_written
                and last_successful_tool == "diary_count"
                and count_diary_chars(draft) <= max_chars
            ):
                async with self._commit_lock(owner_key):
                    # 纪元比对 + outbox 成员资格双重校验：清除可能发生在
                    # 上一次尝试的窗口里（纪元会被重新捕获），只有"事件仍在
                    # outbox"才是未清除的可靠证据
                    if (
                        self._epoch(owner_key) != epoch
                        or not await self.store.is_pending(event["id"])
                    ):
                        logger.info(
                            "[DiaryJournal] 事件 %s 处理期间日记已被清除，放弃提交",
                            event["id"],
                        )
                        return True
                    # 提交前最后复查开关与来源会话周期：失效即停，
                    # 事件留 outbox 延后处理
                    if self._enabled_getter is not None and not self._enabled_getter():
                        logger.info(
                            "[DiaryJournal] 提交前开关已关闭，事件 %s 延后",
                            event["id"],
                        )
                        return False
                    if not await self._source_umo_active(event):
                        logger.info(
                            "[DiaryJournal] 提交前来源会话周期已失效，事件 %s 延后",
                            event["id"],
                        )
                        return False
                    committed = await self.store.upsert_diary(
                        owner_key, draft,
                        display_name=event.get("display_name", "") or diary.get("display_name", ""),
                        aliases=([event.get("display_name", "")] if event.get("display_name") else []),
                    )
                if not committed:
                    # 落盘失败不 ack：事件留 outbox，退避后重试
                    logger.warning(
                        "[DiaryJournal] 事件 %s 日记提交落盘失败，稍后重试",
                        event["id"],
                    )
                    return False
                logger.info(
                    "[DiaryJournal] 事件 %s 已写入日记（%d 条，%d 字）",
                    event["id"], len(draft), count_diary_chars(draft),
                )
                return True

        logger.warning("[DiaryJournal] 事件 %s 达到最大工具步数仍未完成", event["id"])
        return False

    # ------------------------------------------------------------------ #
    #  工具解析与执行
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_tool_call(text: str) -> tuple[str, dict] | None:
        text = text.strip()
        candidates = [text]
        first, last = text.find("{"), text.rfind("}")
        if 0 <= first < last:
            candidates.append(text[first : last + 1])
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                args = obj.get("args")
                return obj["tool"], args if isinstance(args, dict) else {}
        return None

    def _execute_tool(
        self,
        tool: str,
        args: dict,
        draft: list[dict],
        event: dict,
        event_written: bool,
        max_chars: int,
    ) -> tuple[dict, bool]:
        """在内存草稿上执行工具，返回 (结果, 是否成功)。"""
        if tool == "diary_write":
            if event_written:
                return {"error": "当前事件已写过条目，同一事件不能重复写"}, False
            text = str(args.get("text", "")).strip()
            if not text:
                return {"error": "条目内容为空"}, False
            draft.append({
                "id": uuid.uuid4().hex[:10],
                "event_id": event["id"],
                "occurred_at": event["occurred_at"],
                "text": text[:500],
            })
            return {"ok": True, "entry_id": draft[-1]["id"]}, True

        if tool == "diary_edit":
            entry_id = str(args.get("entry_id", ""))
            operation = str(args.get("operation", ""))
            target = next((e for e in draft if e.get("id") == entry_id), None)
            if target is None:
                return {"error": "条目不存在"}, False
            if operation == "replace":
                text = str(args.get("text", "")).strip()
                if not text:
                    return {"error": "替换内容为空"}, False
                target["text"] = text[:500]
                return {"ok": True}, True
            if operation == "delete":
                earliest = draft[0] if draft else None
                if earliest is None or earliest.get("id") != entry_id:
                    return {"error": "只能删除当前最早的完整条目"}, False
                draft.pop(0)
                return {"ok": True}, True
            return {"error": "未知操作"}, False

        if tool == "diary_count":
            chars = count_diary_chars(draft)
            return {
                "chars": chars,
                "limit": max_chars,
                "overflow": max(0, chars - max_chars),
                "entries": len(draft),
                "earliest_id": draft[0]["id"] if draft else None,
            }, True

        return {"error": "未知工具"}, False

    @staticmethod
    def _format_diary_for_model(entries: list[dict]) -> str:
        if not entries:
            return "  （日记为空）"
        return "\n".join(
            f"  [{e.get('id')}] {e.get('occurred_at', '')} {e.get('text', '')}"
            for e in entries
        )
