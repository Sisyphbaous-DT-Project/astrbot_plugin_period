"""Tests for core/mood_journal.py — async workers and the private tool loop."""

import asyncio

import pytest

from core.mood_journal import DiaryJournal, DiaryStore, count_diary_chars

from conftest import ProgrammableProvider

OWNER = "qq_1:10000:12345"
OWNER2 = "qq_1:10000:67890"

WRITE = '{"tool": "diary_write", "args": {"text": "今天有点介意被敷衍。"}}'
COUNT = '{"tool": "diary_count", "args": {}}'


@pytest.fixture
def provider():
    return ProgrammableProvider()


@pytest.fixture
def journal(temp_data_dir, provider):
    return DiaryJournal(
        temp_data_dir,
        lambda pid: provider,
        max_chars=4000,
        step_timeout=5,
        total_timeout=30,
        retry_delay=0.01,  # 测试不等真实退避
    )


class RuleProvider(ProgrammableProvider):
    """按提示内容给出工具调用：首轮 write，之后 count。"""

    async def text_chat(self, prompt: str = "", **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if "工具结果" in prompt or "无法解析" in prompt:
            from astrbot.api.provider import LLMResponse
            return LLMResponse(COUNT)
        from astrbot.api.provider import LLMResponse
        return LLMResponse(WRITE)


class TestToolLoop:
    @pytest.mark.asyncio
    async def test_write_then_count_commits(self, journal, provider):
        provider.queue(WRITE, COUNT)
        await journal.submit(OWNER, "mood_changed", "心境变为「有点介意」", display_name="小明")
        await journal.wait_idle()

        diary = await journal.store.get_diary(OWNER)
        assert diary is not None
        assert len(diary["entries"]) == 1
        assert "介意被敷衍" in diary["entries"][0]["text"]
        assert diary["display_name"] == "小明"
        # outbox 已清理且记入已处理
        assert await journal.store.pending_events() == []

    @pytest.mark.asyncio
    async def test_duplicate_write_same_event_rejected(self, journal, provider):
        provider.queue(WRITE, WRITE, COUNT)  # 第二次 write 应被拒绝
        await journal.submit(OWNER, "mood_changed", "事件")
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        assert len(diary["entries"]) == 1
        # 第二次 write 的错误结果被喂回给模型
        assert any("同一事件不能重复写" in c["prompt"] or
                   any("同一事件不能重复写" in m.get("content", "") for m in c.get("contexts", []))
                   for c in provider.calls)

    @pytest.mark.asyncio
    async def test_delete_only_earliest_allowed(self, journal, provider, temp_data_dir):
        # 预置两条日记
        await journal.store.upsert_diary(OWNER, [
            {"id": "e1", "event_id": "old1", "occurred_at": "t", "text": "最早的条目"},
            {"id": "e2", "event_id": "old2", "occurred_at": "t", "text": "第二条"},
        ])
        provider.queue(
            WRITE,
            '{"tool": "diary_edit", "args": {"entry_id": "e2", "operation": "delete"}}',  # 非最早 → 拒绝
            '{"tool": "diary_edit", "args": {"entry_id": "e1", "operation": "delete"}}',  # 最早 → 允许
            COUNT,
        )
        await journal.submit(OWNER, "action_activated", "决定冷暴力")
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        texts = [e["text"] for e in diary["entries"]]
        assert "最早的条目" not in texts  # e1 被删
        assert "第二条" in texts          # e2 保留
        assert len(texts) == 2            # + 新写入的一条

    @pytest.mark.asyncio
    async def test_model_driven_overflow_cleanup(self, journal, provider):
        journal.max_chars = 40
        await journal.store.upsert_diary(OWNER, [
            {"id": "e1", "event_id": "old1", "occurred_at": "t", "text": "一" * 20},
        ])
        # 新条目写入后超限 → 模型反复 count、删最早、再 count
        provider.queue(
            '{"tool": "diary_write", "args": {"text": "' + "新" * 30 + '"}}',
            COUNT,  # overflow>0, earliest_id=e1
            '{"tool": "diary_edit", "args": {"entry_id": "e1", "operation": "delete"}}',
            COUNT,  # 现在不超限 → 提交
        )
        await journal.submit(OWNER, "mood_changed", "事件")
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        texts = [e["text"] for e in diary["entries"]]
        assert "一" * 20 not in texts
        assert count_diary_chars(diary["entries"]) <= 40

    @pytest.mark.asyncio
    async def test_host_hard_trim_when_limit_lowered(self, journal, provider):
        """用户调低上限：宿主先确定性删最早条目保硬上限，事件继续补写。"""
        await journal.store.upsert_diary(OWNER, [
            {"id": "e1", "event_id": "o1", "occurred_at": "t", "text": "甲" * 30},
            {"id": "e2", "event_id": "o2", "occurred_at": "t", "text": "乙" * 30},
        ])
        journal.max_chars = 25  # 存量 61 字超限
        provider.queue(WRITE, COUNT)
        await journal.submit(OWNER, "mood_changed", "事件")
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        # 宿主裁剪后只剩新写入的条目（旧条目均被裁剪，新条目“今天有点介意被敷衍。”12字 ≤ 25）
        assert all(not e["text"].startswith("甲") and not e["text"].startswith("乙")
                   for e in diary["entries"])
        assert count_diary_chars(diary["entries"]) <= 25

    @pytest.mark.asyncio
    async def test_provider_failure_rolls_back_and_keeps_outbox(self, journal, provider):
        provider.queue(RuntimeError("模型挂了"), RuntimeError("又挂了"))
        await journal.submit(OWNER, "mood_changed", "事件")
        await asyncio.sleep(0.1)  # 让首轮尝试与若干退避重试跑完
        # 草稿回滚：无日记；事件留在 outbox 持续退避重试
        assert await journal.store.get_diary(OWNER) is None
        pending = await journal.store.pending_events()
        assert len(pending) == 1
        await journal.shutdown()  # 取消仍在退避的 worker

    @pytest.mark.asyncio
    async def test_max_steps_gives_up(self, journal, provider):
        journal.max_steps = 3
        provider.default_response = "不是工具调用"
        await journal.submit(OWNER, "mood_changed", "事件")
        await asyncio.sleep(0.1)
        assert await journal.store.get_diary(OWNER) is None
        assert len(await journal.store.pending_events()) == 1
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_unparseable_output_feeds_back_error(self, journal, provider):
        provider.queue("随便说说", WRITE, COUNT)
        await journal.submit(OWNER, "mood_changed", "事件")
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        assert diary is not None and len(diary["entries"]) == 1


class TestAsyncSemantics:
    @pytest.mark.asyncio
    async def test_fifo_order_per_owner(self, temp_data_dir):
        provider = RuleProvider()
        journal = DiaryJournal(temp_data_dir, lambda pid: provider)
        await journal.submit(OWNER, "mood_changed", "事件一")
        await journal.submit(OWNER, "mood_changed", "事件二")
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        assert len(diary["entries"]) == 2  # 同键串行，两事件按序写入

    @pytest.mark.asyncio
    async def test_cross_owner_parallel_limited_to_two(self, temp_data_dir):
        class SlowRuleProvider(RuleProvider):
            def __init__(self):
                super().__init__()
                self.current = 0
                self.max_seen = 0

            async def text_chat(self, prompt: str = "", **kwargs):
                self.current += 1
                self.max_seen = max(self.max_seen, self.current)
                await asyncio.sleep(0.05)
                self.current -= 1
                return await super().text_chat(prompt, **kwargs)

        provider = SlowRuleProvider()
        journal = DiaryJournal(temp_data_dir, lambda pid: provider, max_parallel=2)
        owners = [f"qq_1:10000:{i}" for i in range(4)]
        for owner in owners:
            await journal.submit(owner, "mood_changed", "事件")
        await journal.wait_idle()
        for owner in owners:
            assert await journal.store.get_diary(owner) is not None
        assert provider.max_seen <= 2  # 跨键最多 2 并发

    @pytest.mark.asyncio
    async def test_submit_does_not_block_caller(self, temp_data_dir):
        class HangProvider(ProgrammableProvider):
            async def text_chat(self, prompt: str = "", **kwargs):
                await asyncio.sleep(60)

        journal = DiaryJournal(temp_data_dir, lambda pid: HangProvider(), step_timeout=60)
        await asyncio.wait_for(
            journal.submit(OWNER, "mood_changed", "事件"), timeout=1.0,
        )  # submit 本身必须快速返回
        await journal.shutdown()  # 取消卡住的 worker

    @pytest.mark.asyncio
    async def test_restart_resumes_pending_events(self, temp_data_dir):
        # 第一次运行：事件入队但不处理（直接写 outbox，模拟处理前崩溃）
        store = DiaryStore(temp_data_dir)
        await store.enqueue({
            "id": "evt-restart", "owner_key": OWNER, "kind": "mood_changed",
            "summary": "重启前的事件", "display_name": "", "provider_id": "",
            "occurred_at": "2026-08-12T00:00:00+00:00",
        })
        # “重启”：新实例 + start() 恢复
        provider = RuleProvider()
        journal = DiaryJournal(temp_data_dir, lambda pid: provider)
        await journal.start()
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        assert diary is not None and len(diary["entries"]) == 1
        assert await journal.store.is_processed("evt-restart") is True


class TestInjectionText:
    def test_empty_diary(self):
        journal = DiaryJournal.__new__(DiaryJournal)  # 纯函数行为无需构造
        assert DiaryJournal.build_injection_text(journal, None) == ""
        assert DiaryJournal.build_injection_text(journal, {"entries": []}) == ""

    def test_text_contains_entries(self):
        journal = DiaryJournal.__new__(DiaryJournal)
        diary = {"entries": [{"text": "条目甲"}, {"text": "条目乙"}]}
        text = DiaryJournal.build_injection_text(journal, diary)
        assert "情绪日记" in text and "条目甲" in text and "条目乙" in text

    def test_live_max_chars_cap(self):
        """P2 回归：读取侧实时上限——调低上限后、worker 裁剪完成前，
        注入文本总长不得超过当前配置。"""
        journal = DiaryJournal.__new__(DiaryJournal)
        diary = {"entries": [{"text": "甲" * 300}, {"text": "乙" * 300}]}
        text = DiaryJournal.build_injection_text(journal, diary, max_chars=200)
        assert len(text) <= 200
        assert "情绪日记" in text  # 头部保留，从尾部保留最新内容
        assert "乙" in text
        # 未传上限时保持原样（向后兼容）
        full = DiaryJournal.build_injection_text(journal, diary)
        assert len(full) > 200

    def test_tiny_max_chars_falls_back_to_tail(self):
        journal = DiaryJournal.__new__(DiaryJournal)
        diary = {"entries": [{"text": "甲" * 100}]}
        text = DiaryJournal.build_injection_text(journal, diary, max_chars=10)
        assert len(text) <= 10


class TestFailureFIFO:
    """P1-⑦ 回归：失败事件阻塞同键队列，运行期延迟重试，不被后续事件超车。"""

    @pytest.mark.asyncio
    async def test_failed_event_blocks_later_events_until_success(self, temp_data_dir):
        provider = ProgrammableProvider()
        journal = DiaryJournal(temp_data_dir, lambda pid: provider, retry_delay=0.01)
        # 事件一：本轮两次尝试都失败；延迟重试后成功
        provider.queue(
            RuntimeError("挂1"), RuntimeError("挂2"),
            WRITE, COUNT,
        )
        await journal.submit(OWNER, "mood_changed", "事件一")
        # 事件二必须在事件一之后写入
        provider.queue(
            '{"tool": "diary_write", "args": {"text": "第二条内容"}}', COUNT,
        )
        await journal.submit(OWNER, "mood_changed", "事件二")
        await journal.wait_idle()

        diary = await journal.store.get_diary(OWNER)
        texts = [e["text"] for e in diary["entries"]]
        assert texts == ["今天有点介意被敷衍。", "第二条内容"]  # FIFO 顺序保持
        assert await journal.store.pending_events() == []


class TestClearOwner:
    """P1-⑧ 回归：diaryclear 必须连同未处理/在处理事件一起清除。"""

    @pytest.mark.asyncio
    async def test_clear_removes_pending_events(self, journal, provider):
        await journal.submit(OWNER, "mood_changed", "事件")
        removed_diary, removed_events, persisted = await journal.clear_owner(OWNER)
        assert removed_diary is False  # 还没有已提交日记
        assert removed_events == 1     # 但清理了 1 条待处理事件
        assert persisted is True
        assert await journal.store.pending_events() == []
        assert await journal.store.get_diary(OWNER) is None
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_clear_removes_committed_diary_and_events(self, journal, provider):
        provider.queue(WRITE, COUNT)
        await journal.submit(OWNER, "mood_changed", "事件")
        await journal.wait_idle()
        assert await journal.store.get_diary(OWNER) is not None

        await journal.submit(OWNER, "action_activated", "事件二")
        removed_diary, removed_events, persisted = await journal.clear_owner(OWNER)
        assert removed_diary is True
        assert removed_events == 1
        assert persisted is True
        assert await journal.store.get_diary(OWNER) is None
        assert await journal.store.pending_events() == []
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_inflight_commit_aborted_after_clear(self, temp_data_dir):
        """worker 处理期间执行 diaryclear：in-flight 草稿不得复活已清除的日记。"""
        gate = asyncio.Event()

        class GateProvider(ProgrammableProvider):
            async def text_chat(self, prompt: str = "", **kwargs):
                if not self.calls:
                    await gate.wait()  # 第一次模型调用被卡住，模拟处理中
                return await super().text_chat(prompt=prompt, **kwargs)

        provider = GateProvider()
        provider.queue(WRITE, COUNT)
        journal = DiaryJournal(temp_data_dir, lambda pid: provider, retry_delay=0.01)
        await journal.store.upsert_diary(OWNER, [
            {"id": "e1", "event_id": "old", "occurred_at": "t", "text": "旧条目"},
        ])
        await journal.submit(OWNER, "mood_changed", "事件")
        await asyncio.sleep(0.05)  # 让 worker 进入被卡住的模型调用

        await journal.clear_owner(OWNER)  # 处理期间清除
        gate.set()
        await journal.wait_idle()

        assert await journal.store.get_diary(OWNER) is None  # 没有复活
        assert await journal.store.pending_events() == []
        await journal.shutdown()


class TestSwitchGating:
    """P1-⑤ 回归：情绪/日记开关关闭时，outbox 事件不得调用模型。"""

    @pytest.mark.asyncio
    async def test_disabled_defers_processing_without_model_calls(self, temp_data_dir):
        provider = ProgrammableProvider()
        state = {"enabled": False}
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: state["enabled"],
        )
        await journal.submit(OWNER, "mood_changed", "事件")
        await asyncio.sleep(0.1)
        assert provider.calls == []  # 关闭时不调用模型、不创建日记
        assert len(await journal.store.pending_events()) == 1  # 留在 outbox
        assert await journal.store.get_diary(OWNER) is None

        # 重新开启后退避循环周期性复查，自动恢复处理
        provider.queue(WRITE, COUNT)
        state["enabled"] = True
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        assert diary is not None and len(diary["entries"]) == 1

    @pytest.mark.asyncio
    async def test_start_resume_respects_switch(self, temp_data_dir):
        """重启恢复 outbox 同样受开关门控。"""
        store = DiaryStore(temp_data_dir)
        await store.enqueue({
            "id": "evt-off", "owner_key": OWNER, "kind": "mood_changed",
            "summary": "s", "display_name": "", "provider_id": "",
            "occurred_at": "2026-08-12T00:00:00+00:00",
        })
        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,
        )
        await journal.start()
        await asyncio.sleep(0.05)
        assert provider.calls == []
        assert len(await journal.store.pending_events()) == 1
        await journal.shutdown()


class TestClearDuringFailedAttempt:
    """P1-① 回归：清除发生在失败尝试的模型调用窗口里，重试不得复活日记。"""

    @pytest.mark.asyncio
    async def test_retry_after_clear_does_not_revive(self, temp_data_dir):
        gate = asyncio.Event()

        class FailOnceGateProvider(ProgrammableProvider):
            async def text_chat(self, prompt: str = "", **kwargs):
                if not self.calls:
                    self.calls.append({"prompt": prompt})
                    await gate.wait()  # 第一次模型调用被卡住，清除在此窗口发生
                    raise RuntimeError("第一次尝试失败")
                return await super().text_chat(prompt=prompt, **kwargs)

        provider = FailOnceGateProvider()
        provider.queue(WRITE, COUNT)  # 若重试继续，用这组响应可以成功提交
        journal = DiaryJournal(temp_data_dir, lambda pid: provider, retry_delay=0.01)
        await journal.store.upsert_diary(OWNER, [
            {"id": "e1", "event_id": "old", "occurred_at": "t", "text": "旧条目"},
        ])
        await journal.submit(OWNER, "mood_changed", "事件")
        await asyncio.sleep(0.05)  # worker 进入被卡住的第一次尝试

        await journal.clear_owner(OWNER)  # 失败尝试的窗口里清除
        gate.set()
        await journal.wait_idle()

        # 旧纪元会被重试重新捕获，单靠纪元防不住；outbox 成员资格必须拦住
        assert await journal.store.get_diary(OWNER) is None
        assert await journal.store.pending_events() == []
        await journal.shutdown()


class TestEnqueueDurability:
    """P1-② 回归：outbox 落盘失败不得谎报成功入队。"""

    @pytest.mark.asyncio
    async def test_save_failure_rolls_back_and_skips_worker(
        self, temp_data_dir, monkeypatch,
    ):
        from pathlib import Path

        def always_fail(*args, **kwargs):
            raise OSError("磁盘满")

        monkeypatch.setattr(Path, "write_text", always_fail)
        provider = ProgrammableProvider()
        journal = DiaryJournal(temp_data_dir, lambda pid: provider, retry_delay=0.01)
        ok = await journal.submit(OWNER, "mood_changed", "事件")
        assert ok is False  # 不谎报成功
        assert await journal.store.pending_events() == []  # 缓存回滚
        await asyncio.sleep(0.05)
        assert provider.calls == []  # 未触发 worker
        await journal.shutdown()


class TestRetryDelayOverflow:
    """P1 回归：无限退避不得因浮点溢出失效（2**cycle 转 float 溢出）。"""

    def test_huge_cycle_does_not_overflow(self, temp_data_dir):
        from core.mood_journal import MAX_RETRY_DELAY

        journal = DiaryJournal(temp_data_dir, lambda pid: None)
        assert journal._retry_delay_for(1) == journal.retry_delay
        # 第 2000 轮退避不得抛 OverflowError，直接封顶
        assert journal._retry_delay_for(2000) == MAX_RETRY_DELAY


class TestTotalTimeoutHardCap:
    """P2 回归：总超时是硬上限，单步只给剩余时间。"""

    @pytest.mark.asyncio
    async def test_step_limited_by_remaining_total(self, temp_data_dir):
        import time

        class SlowProvider(ProgrammableProvider):
            def __init__(self):
                super().__init__()
                self.completed = False

            async def text_chat(self, prompt: str = "", **kwargs):
                await asyncio.sleep(5)
                self.completed = True

        provider = SlowProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider,
            step_timeout=30, total_timeout=0.3, retry_delay=60,
        )
        await journal.store.enqueue({
            "id": "evt-slow", "owner_key": OWNER, "kind": "mood_changed",
            "summary": "s", "display_name": "", "provider_id": "",
            "occurred_at": "2026-08-12T00:00:00+00:00",
        })
        started = time.monotonic()
        done = await journal._process_by_id("evt-slow")
        elapsed = time.monotonic() - started
        assert done is False
        assert provider.completed is False
        # 硬上限 ~0.3s 即中止，不会先跑满一个 30s 单步
        assert elapsed < 2


class TestSwitchMidProcessing:
    """P1 回归：处理期间关闭开关必须即时生效（步骤间与提交前都复查）。"""

    @pytest.mark.asyncio
    async def test_switch_off_between_steps_defers(self, temp_data_dir):
        state = {"enabled": True}

        class FlipRuleProvider(RuleProvider):
            async def text_chat(self, prompt: str = "", **kwargs):
                if not self.calls:
                    state["enabled"] = False  # 第一次模型调用期间关闭开关
                return await super().text_chat(prompt=prompt, **kwargs)

        provider = FlipRuleProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: state["enabled"],
        )
        await journal.submit(OWNER, "mood_changed", "事件")
        await asyncio.sleep(0.2)
        # 只进行了第一次模型调用；第二步前检测到开关关闭，不提交、不写盘
        assert len(provider.calls) == 1
        assert await journal.store.get_diary(OWNER) is None
        assert len(await journal.store.pending_events()) == 1

        # 重新开启后自动恢复处理
        state["enabled"] = True
        await journal.wait_idle()
        diary = await journal.store.get_diary(OWNER)
        assert diary is not None and len(diary["entries"]) == 1
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_switch_off_before_commit_defers(self, temp_data_dir):
        """提交临界区内复查：count 之后、落盘之前关闭开关，不得提交。"""
        state = {"enabled": True, "flipped": False}

        class FlipOnCountProvider(RuleProvider):
            async def text_chat(self, prompt: str = "", **kwargs):
                resp = await super().text_chat(prompt=prompt, **kwargs)
                if "工具结果" in prompt and not state["flipped"]:
                    state["flipped"] = True
                    state["enabled"] = False  # count 响应返回时关闭
                return resp

        provider = FlipOnCountProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: state["enabled"],
        )
        await journal.submit(OWNER, "mood_changed", "事件")
        await asyncio.sleep(0.2)
        # write + count 都执行了，但提交被开关拦下
        assert len(provider.calls) == 2
        assert await journal.store.get_diary(OWNER) is None
        assert len(await journal.store.pending_events()) == 1

        state["enabled"] = True
        await journal.wait_idle()
        assert await journal.store.get_diary(OWNER) is not None
        await journal.shutdown()


class TestDiscardPendingForUmo:
    """P2 回归：/period reset 丢弃该会话来源的滞留事件。"""

    @pytest.mark.asyncio
    async def test_discard_only_matching_umo(self, temp_data_dir):
        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,  # 开关关闭让事件滞留在 outbox
        )
        await journal.submit(OWNER, "mood_changed", "事件一", umo="umo_a")
        await journal.submit(OWNER, "mood_changed", "事件二", umo="umo_b")
        removed = await journal.discard_pending_for_umo("umo_a")
        assert removed == 1
        pending = await journal.store.pending_events()
        assert len(pending) == 1 and pending[0]["umo"] == "umo_b"
        await journal.shutdown()


class TestDiscardCommitRace:
    """P1 回归：reset 与提交共用提交锁，提交窗口里的 reset 不得复活日记。"""

    @pytest.mark.asyncio
    async def test_discard_waits_for_commit_lock(self, temp_data_dir):
        """discard 必须获取受影响 owner 的提交锁（与提交临界区互斥）。"""
        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,
        )
        await journal.submit(OWNER, "mood_changed", "事件", umo="umo_a")
        lock = journal._commit_lock(OWNER)
        await lock.acquire()
        try:
            task = asyncio.create_task(journal.discard_pending_for_umo("umo_a"))
            await asyncio.sleep(0.05)
            assert not task.done()  # 被提交锁挡住，不能绕过临界区
        finally:
            lock.release()
        assert await task == 1
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_reset_during_commit_window_does_not_revive(self, temp_data_dir):
        """count 之后、提交落盘之前执行 reset：被丢弃事件不得写入日记。"""
        journal = None

        class ResetOnCountProvider(RuleProvider):
            async def text_chat(self, prompt: str = "", **kwargs):
                resp = await super().text_chat(prompt=prompt, **kwargs)
                if "工具结果" in prompt:
                    # count 返回后、提交临界区之前执行 reset
                    await journal.discard_pending_for_umo("umo_a")
                return resp

        provider = ResetOnCountProvider()
        journal = DiaryJournal(temp_data_dir, lambda pid: provider, retry_delay=0.01)
        await journal.store.upsert_diary(OWNER, [
            {"id": "e1", "event_id": "old", "occurred_at": "t", "text": "旧条目"},
        ])
        await journal.submit(OWNER, "mood_changed", "事件", umo="umo_a")
        await journal.wait_idle()

        diary = await journal.store.get_diary(OWNER)
        assert [e["text"] for e in diary["entries"]] == ["旧条目"]  # 新事件未提交
        assert await journal.store.pending_events() == []
        await journal.shutdown()


class TestClearFailureKeepsQueue:
    """P1 回归：diaryclear 落盘失败不得抽空队列、不得丢事件。"""

    @pytest.mark.asyncio
    async def test_clear_save_failure_preserves_state(
        self, temp_data_dir, monkeypatch,
    ):
        from pathlib import Path

        provider = RuleProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,  # 开关关闭：事件滞留，worker 退避
        )
        await journal.store.upsert_diary(OWNER, [
            {"id": "e1", "event_id": "old", "occurred_at": "t", "text": "旧条目"},
        ])
        await journal.submit(OWNER, "mood_changed", "事件一")
        await journal.submit(OWNER, "mood_changed", "事件二")
        await asyncio.sleep(0.05)  # worker 拿起事件一（退避中），事件二在队列

        def always_fail(*args, **kwargs):
            raise OSError("磁盘满")

        monkeypatch.setattr(Path, "write_text", always_fail)
        removed_diary, removed_events, persisted = await journal.clear_owner(OWNER)
        assert persisted is False and removed_diary is False and removed_events == 0
        # 磁盘与缓存原样：日记仍在、两条事件都在 outbox
        assert await journal.store.get_diary(OWNER) is not None
        assert len(await journal.store.pending_events()) == 2
        # 队列未被抽干：事件二仍在等待，FIFO 不被后续事件超车
        assert journal._queues[OWNER].qsize() == 1
        await journal.shutdown()


class TestSubmitClearRace:
    """P1 回归：submit 与 clear/discard 共用 owner 提交锁（线性化）。"""

    @pytest.mark.asyncio
    async def test_submit_takes_commit_lock(self, temp_data_dir):
        """submit 必须获取 owner 提交锁：清除持锁时入队会被挡住。"""
        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,
        )
        lock = journal._commit_lock(OWNER)
        await lock.acquire()
        try:
            task = asyncio.create_task(journal.submit(OWNER, "mood_changed", "事件"))
            await asyncio.sleep(0.05)
            assert not task.done()  # 被提交锁挡住，不能绕过临界区
        finally:
            lock.release()
        assert await task is True
        assert len(await journal.store.pending_events()) == 1
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_no_orphan_pending_after_clear_race(self, temp_data_dir):
        """并发 submit/clear 不得产生"磁盘 pending 有、队列没有"的失联事件。"""
        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,
        )
        for i in range(10):
            await asyncio.gather(
                journal.submit(OWNER, "mood_changed", f"事件{i}"),
                journal.clear_owner(OWNER),
            )
            pending = await journal.store.pending_events()
            queue = journal._queues.get(OWNER)
            queued = queue.qsize() if queue is not None else 0
            # 不变量：每条 pending 事件要么在队列里，要么正被 worker 持有
            assert len(pending) <= queued + 1
        await journal.shutdown()


class TestUmoCycleGating:
    """P2 回归：来源会话周期失效时，滞留事件延后处理（不调用模型）。"""

    @pytest.mark.asyncio
    async def test_inactive_umo_defers_then_recovers(self, temp_data_dir):
        active = {"umo_a": False}

        async def umo_active(umo: str) -> bool:
            return active.get(umo, True)

        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            umo_active_getter=umo_active,
        )
        await journal.submit(OWNER, "mood_changed", "事件", umo="umo_a")
        await asyncio.sleep(0.1)
        assert provider.calls == []  # 周期失效：不调用模型、不写日记
        assert len(await journal.store.pending_events()) == 1

        # 周期重新有效后自动恢复
        provider.queue(WRITE, COUNT)
        active["umo_a"] = True
        await journal.wait_idle()
        assert await journal.store.get_diary(OWNER) is not None
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_getter_exception_treated_inactive(self, temp_data_dir):
        """门控判定异常按失效处理（保守）。"""

        async def boom(umo: str) -> bool:
            raise RuntimeError("判定失败")

        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            umo_active_getter=boom,
        )
        await journal.submit(OWNER, "mood_changed", "事件", umo="umo_a")
        await asyncio.sleep(0.1)
        assert provider.calls == []
        assert len(await journal.store.pending_events()) == 1
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_event_without_umo_skips_gate(self, temp_data_dir):
        """无 umo 的旧事件不受会话级门控影响（向后兼容）。"""

        async def always_inactive(umo: str) -> bool:
            return False

        provider = ProgrammableProvider()
        provider.queue(WRITE, COUNT)
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            umo_active_getter=always_inactive,
        )
        await journal.submit(OWNER, "mood_changed", "事件")  # umo 为空
        await journal.wait_idle()
        assert await journal.store.get_diary(OWNER) is not None
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_umo_invalidated_between_steps_defers(self, temp_data_dir):
        """P1 回归：工具循环中途来源会话周期失效，不得继续调用模型。"""
        active = {"umo_a": True}

        async def umo_active(umo: str) -> bool:
            return active.get("umo_a", True)

        class FlipProvider(RuleProvider):
            async def text_chat(self, prompt: str = "", **kwargs):
                if not self.calls:
                    active["umo_a"] = False  # 第一次调用期间周期失效
                return await super().text_chat(prompt=prompt, **kwargs)

        provider = FlipProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            umo_active_getter=umo_active,
        )
        await journal.submit(OWNER, "mood_changed", "事件", umo="umo_a")
        await asyncio.sleep(0.2)
        assert len(provider.calls) == 1  # 第二步前检测到失效，不再调用
        assert await journal.store.get_diary(OWNER) is None
        assert len(await journal.store.pending_events()) == 1
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_umo_invalidated_before_commit_defers(self, temp_data_dir):
        """P1 回归：count 之后、落盘之前周期失效，不得提交。"""
        active = {"umo_a": True, "flipped": False}

        async def umo_active(umo: str) -> bool:
            return active["umo_a"]

        class FlipOnCount(RuleProvider):
            async def text_chat(self, prompt: str = "", **kwargs):
                resp = await super().text_chat(prompt=prompt, **kwargs)
                if "工具结果" in prompt and not active["flipped"]:
                    active["flipped"] = True
                    active["umo_a"] = False  # count 响应返回时周期失效
                return resp

        provider = FlipOnCount()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            umo_active_getter=umo_active,
        )
        await journal.submit(OWNER, "mood_changed", "事件", umo="umo_a")
        await asyncio.sleep(0.2)
        # write + count 都执行了，但提交被 umo 门控拦下
        assert len(provider.calls) == 2
        assert await journal.store.get_diary(OWNER) is None
        assert len(await journal.store.pending_events()) == 1
        await journal.shutdown()


class TestStaleEventWatermark:
    """P1 回归：reset 水位线焊死 submit 竞态。

    快照为空的 reset 也必须记录水位线；进行中请求晚到的旧周期事件
    （occurred_at 早于水位线）不得入队；已在 outbox 的存量过期事件
    由 worker 直接丢弃（不调用模型、不阻塞 FIFO）。
    """

    @pytest.mark.asyncio
    async def test_empty_snapshot_discard_still_marks_watermark(self, temp_data_dir):
        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,
        )
        # reset 时 outbox 没有该会话事件（快照为空），水位线仍要落盘
        removed = await journal.discard_pending_for_umo("umo_a")
        assert removed == 0
        # 进行中请求此刻才提交的旧周期事件：入队被拒
        ok = await journal.submit(
            OWNER, "mood_changed", "旧事件", umo="umo_a",
            occurred_at="2020-01-01T00:00:00+00:00",
        )
        assert ok is False
        assert await journal.store.pending_events() == []
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_new_cycle_event_after_watermark_accepted(self, temp_data_dir):
        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,
        )
        await journal.discard_pending_for_umo("umo_a")
        # 重新设置周期后产生的新事件：occurred_at 晚于水位线，正常入队
        ok = await journal.submit(OWNER, "mood_changed", "新事件", umo="umo_a")
        assert ok is True
        assert len(await journal.store.pending_events()) == 1
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_existing_stale_event_discarded_without_model_call(self, temp_data_dir):
        provider = RuleProvider()
        journal = DiaryJournal(temp_data_dir, lambda pid: provider, retry_delay=0.01)
        # 先入队（此时无水位线），再 reset：模拟存量过期事件
        await journal.store.enqueue({
            "id": "e1", "owner_key": OWNER, "kind": "mood_changed",
            "summary": "s", "display_name": "", "provider_id": "",
            "umo": "umo_a", "occurred_at": "2020-01-01T00:00:00+00:00",
        })
        await journal.store.mark_umo_watermark("umo_a")
        done = await journal._process_by_id("e1")
        assert done is True
        assert await journal.store.pending_events() == []
        assert provider.calls == []  # 不调用模型、不阻塞 FIFO
        await journal.shutdown()

    @pytest.mark.asyncio
    async def test_watermark_write_failure_reports_minus_one(self, temp_data_dir, monkeypatch):
        provider = ProgrammableProvider()
        journal = DiaryJournal(
            temp_data_dir, lambda pid: provider, retry_delay=0.01,
            enabled_getter=lambda: False,
        )

        async def fail_save(data):
            return False

        monkeypatch.setattr(journal.store, "_save", fail_save)
        assert await journal.discard_pending_for_umo("umo_a") == -1
        await journal.shutdown()
