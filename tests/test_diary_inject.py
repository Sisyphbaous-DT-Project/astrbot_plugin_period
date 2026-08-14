"""Pipeline-level diary tests: injection into formal request and call ②, identity rules."""

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import asyncio

_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import pytest

from astrbot_plugin_period.main import PeriodPlugin

from conftest import MockConversationManager, ProgrammableProvider

GROUP_UMO = "aiocqhttp:group:1001_12345"
PRIVATE_UMO = "aiocqhttp:private:12345"
OWNER = "qq_1:10000:12345"

SCREEN_YES = '{"need_intervention": true, "reasoning": "试探"}'
SCREEN_NO = '{"need_intervention": false}'


@pytest.fixture
def pipeline(tmp_path, sample_config, monkeypatch):
    from astrbot.api.star import Context, StarTools

    config = deepcopy(sample_config)
    config.update({
        "default_anchor_date": "2024-01-15",
        "default_enabled": True,
        "global_inject": True,
        "mood_system_enabled": True,
        "diary_enabled": True,
        "mood_scope": "per_umo",
    })
    ctx = Context()
    provider = ProgrammableProvider()
    ctx.get_using_provider = lambda umo=None: provider
    ctx.get_provider_by_id = lambda pid: None
    conv_mgr = MockConversationManager()
    conv_mgr.seed(GROUP_UMO, [])
    conv_mgr.seed(PRIVATE_UMO, [])
    ctx.conversation_manager = conv_mgr
    monkeypatch.setattr(StarTools, "get_data_dir", lambda _name=None: tmp_path)
    plugin = PeriodPlugin(ctx, config)
    return plugin, provider


def _make_req():
    from astrbot.api.provider import ProviderRequest
    req = ProviderRequest()
    req.prompt = "用户消息"
    req.system_prompt = "人格"
    req.contexts = []
    req.conversation = SimpleNamespace(cid="cid-1", history=None)
    return req


async def _seed_diary(plugin, owner_key=OWNER, text="上次他敷衍我，我介意过。"):
    await plugin.diary_journal.store.upsert_diary(
        owner_key,
        [{"id": "e1", "event_id": "old", "occurred_at": "t", "text": text}],
        display_name="测试用户",
    )


class TestDiaryInjection:
    @pytest.mark.asyncio
    async def test_diary_injected_into_formal_request(self, pipeline, event_factory):
        plugin, provider = pipeline
        await _seed_diary(plugin)
        provider.queue(SCREEN_NO)
        req = _make_req()
        await plugin.on_llm_request(event_factory(umo=GROUP_UMO), req)
        texts = [p.text for p in req.extra_user_content_parts]
        assert any("情绪日记" in t and "介意过" in t for t in texts)
        # 默认临时注入，不落历史
        diary_parts = [p for p in req.extra_user_content_parts if "情绪日记" in p.text]
        assert all(p._no_save for p in diary_parts)

    @pytest.mark.asyncio
    async def test_diary_reaches_consult_call(self, pipeline, event_factory):
        plugin, provider = pipeline
        await _seed_diary(plugin)
        provider.queue(SCREEN_YES, "自然语言决策",
                       '{"mood_update": {"status": "stable", "summary": "", '
                       '"cause_category": "neutral", "latest_reason": "", '
                       '"improved": false, "fully_recovered": false, "recovery_reason": ""}, '
                       '"actions": [], "lift_actions": [], "silence_mode": "none"}')
        req = _make_req()
        await plugin.on_llm_request(event_factory(umo=GROUP_UMO), req)
        consult = provider.calls[1]
        assert "介意过" in consult["prompt"]  # ②看到已提交日记

    @pytest.mark.asyncio
    async def test_same_qq_shared_across_group_and_private(self, pipeline, event_factory):
        """同平台实例同机器人：跨群聊/私聊共用一本日记。"""
        plugin, provider = pipeline
        await _seed_diary(plugin)
        provider.queue(SCREEN_NO)
        req = _make_req()
        await plugin.on_llm_request(event_factory(umo=PRIVATE_UMO), req)
        assert any("介意过" in p.text for p in req.extra_user_content_parts)

    @pytest.mark.asyncio
    async def test_different_bot_isolated(self, pipeline, event_factory):
        plugin, provider = pipeline
        await _seed_diary(plugin)
        provider.queue(SCREEN_NO)
        req = _make_req()
        ev = event_factory(umo=GROUP_UMO, self_id="20000")  # 另一机器人账号
        await plugin.on_llm_request(ev, req)
        assert not any("情绪日记" in p.text for p in req.extra_user_content_parts)

    @pytest.mark.asyncio
    async def test_missing_identity_no_diary_no_crash(self, pipeline, event_factory):
        plugin, provider = pipeline
        await _seed_diary(plugin)
        provider.queue(SCREEN_NO)
        req = _make_req()
        ev = event_factory(umo=GROUP_UMO, sender_id="")  # 身份缺失
        await plugin.on_llm_request(ev, req)
        assert not any("情绪日记" in p.text for p in req.extra_user_content_parts)

    @pytest.mark.asyncio
    async def test_diary_disabled_by_config(self, pipeline, event_factory):
        plugin, provider = pipeline
        plugin.config["diary_enabled"] = False
        await _seed_diary(plugin)
        provider.queue(SCREEN_NO)
        req = _make_req()
        await plugin.on_llm_request(event_factory(umo=GROUP_UMO), req)
        assert not any("情绪日记" in p.text for p in req.extra_user_content_parts)


class TestDiaryEventsFromPipeline:
    @pytest.mark.asyncio
    async def test_mood_events_enter_outbox(self, pipeline, event_factory):
        """动作激活产生脱敏事件进入 outbox；载荷不含用户原消息。"""
        plugin, provider = pipeline
        provider.queue(SCREEN_YES, "决定", (
            '{"mood_update": {"status": "active", "summary": "介意", '
            '"cause_category": "dismissive", "latest_reason": "被轻慢", '
            '"improved": false, "fully_recovered": false, "recovery_reason": ""}, '
            '"actions": [{"name": "cold_violence", "params": {"duration": 10}}], '
            '"lift_actions": [], "silence_mode": "immediate", "reasoning_summary": "想静静"}'
        ))
        ev = event_factory(umo=GROUP_UMO, message_str="用户的原始消息不应出现在事件里")
        await plugin.on_llm_request(ev, _make_req())

        pending = await plugin.diary_journal.store.pending_events()
        kinds = {e["kind"] for e in pending}
        assert "action_activated" in kinds
        assert "mood_changed" in kinds
        for e in pending:
            assert "用户的原始消息不应出现在事件里" not in str(e)
            assert e["owner_key"] == OWNER
        await plugin.diary_journal.shutdown()

    @pytest.mark.asyncio
    async def test_manual_lift_event(self, pipeline, event_factory):
        plugin, provider = pipeline
        from astrbot_plugin_period.core.mood import MoodState, PersistentAction
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(GROUP_UMO, state)

        ev = event_factory(umo=GROUP_UMO)
        async for _ in plugin.period_lift(ev):
            pass
        pending = await plugin.diary_journal.store.pending_events()
        assert any(e["kind"] == "manual_lift" for e in pending)
        await plugin.diary_journal.shutdown()


class TestCycleMasterSwitchGating:
    """P1-③ 回归：周期总开关关闭时，日记 outbox 事件不得调用模型。"""

    @pytest.mark.asyncio
    async def test_auto_inject_off_defers_outbox(self, pipeline, event_factory):
        plugin, provider = pipeline
        plugin.config["auto_inject"] = False  # 周期总开关关闭
        await plugin.diary_journal.submit(OWNER, "mood_changed", "事件")
        await asyncio.sleep(0.1)
        assert provider.calls == []  # 不调用模型、不创建日记
        assert len(await plugin.diary_journal.store.pending_events()) == 1
        await plugin.diary_journal.shutdown()


class TestResetDiscardsPendingEvents:
    """P2 回归：/period reset 后该会话来源的滞留日记事件被丢弃。"""

    @staticmethod
    def _event(eid, umo):
        return {
            "id": eid, "owner_key": OWNER, "kind": "mood_changed",
            "summary": "s", "display_name": "", "provider_id": "",
            "umo": umo, "occurred_at": "2026-08-12T00:00:00+00:00",
        }

    @pytest.mark.asyncio
    async def test_reset_discards_only_this_umo(self, pipeline, event_factory):
        plugin, provider = pipeline
        await plugin.diary_journal.store.enqueue(self._event("evt-group", GROUP_UMO))
        await plugin.diary_journal.store.enqueue(self._event("evt-private", PRIVATE_UMO))
        await _seed_diary(plugin)  # 已提交日记必须保留

        ev = event_factory(umo=GROUP_UMO)
        async for _ in plugin.period_reset(ev):
            pass

        pending = await plugin.diary_journal.store.pending_events()
        assert [e["id"] for e in pending] == ["evt-private"]  # 只丢群聊来源
        assert await plugin.diary_journal.store.get_diary(OWNER) is not None
        await plugin.diary_journal.shutdown()

    @pytest.mark.asyncio
    async def test_reset_discard_failure_does_not_break_command(
        self, pipeline, event_factory, monkeypatch,
    ):
        """清理滞留事件失败不得打断 reset 指令，但须如实提示清理失败。"""
        plugin, provider = pipeline
        await plugin.diary_journal.store.enqueue(self._event("evt-group", GROUP_UMO))

        async def boom(umo):
            raise OSError("磁盘满")

        monkeypatch.setattr(
            plugin.diary_journal, "discard_pending_for_umo", boom,
        )
        ev = event_factory(umo=GROUP_UMO)
        results = [r async for r in plugin.period_reset(ev)]
        assert any("已重置" in r for r in results)
        assert any("清理失败" in r for r in results)  # 不得谎报完全成功
        await plugin.diary_journal.shutdown()

    @pytest.mark.asyncio
    async def test_reset_reports_cleanup_failure_on_minus_one(
        self, pipeline, event_factory, monkeypatch,
    ):
        """discard 返回 -1（落盘失败）时回复须提示清理失败。"""
        plugin, provider = pipeline
        await plugin.diary_journal.store.enqueue(self._event("evt-group", GROUP_UMO))

        async def minus_one(umo):
            return -1

        monkeypatch.setattr(
            plugin.diary_journal, "discard_pending_for_umo", minus_one,
        )
        ev = event_factory(umo=GROUP_UMO)
        results = [r async for r in plugin.period_reset(ev)]
        assert any("清理失败" in r for r in results)
        # 事件未被丢弃（清理未生效）
        assert len(await plugin.diary_journal.store.pending_events()) == 1
        await plugin.diary_journal.shutdown()


class TestDiarySubmitErrorPrivacy:
    """P1 回归：日记入队异常只记异常类型，不回显异常正文。"""

    @pytest.mark.asyncio
    async def test_submit_error_records_type_only(
        self, pipeline, event_factory, monkeypatch,
    ):
        plugin, provider = pipeline
        recorded = {}

        async def fake_record(title, error, **kwargs):
            recorded["error"] = error

        monkeypatch.setattr(plugin, "_record_diagnostic_error", fake_record)

        async def boom(*args, **kwargs):
            raise RuntimeError("敏感摘要:用户原话内容")

        monkeypatch.setattr(plugin.diary_journal, "submit", boom)
        await plugin._emit_diary_event(
            event_factory(umo=GROUP_UMO), "manual_lift",
            {"actions": ["cold_violence"]},
        )
        assert recorded["error"] == "RuntimeError"
        assert "用户原话" not in str(recorded)

    @pytest.mark.asyncio
    async def test_submit_false_records_diagnostic(
        self, pipeline, event_factory, monkeypatch,
    ):
        """P2 回归：submit 返回 False（outbox 落盘失败）不得静默丢事件。"""
        plugin, provider = pipeline
        recorded = {}

        async def fake_record(title, error, **kwargs):
            recorded["error"] = error

        monkeypatch.setattr(plugin, "_record_diagnostic_error", fake_record)

        async def ret_false(*args, **kwargs):
            return False

        monkeypatch.setattr(plugin.diary_journal, "submit", ret_false)
        await plugin._emit_diary_event(
            event_factory(umo=GROUP_UMO), "manual_lift",
            {"actions": ["cold_violence"]},
        )
        assert recorded.get("error") == "outbox 落盘失败或事件重复"


class TestEmitSourceWindowClosed:
    """P2 回归：周期失效后不再产生新日记事件（源头闭窗）。"""

    @pytest.mark.asyncio
    async def test_emit_dropped_when_cycle_inactive(self, pipeline, event_factory):
        plugin, provider = pipeline
        # 会话周期失效（等效 toggle 关闭 / reset 后）
        await plugin.store.set(GROUP_UMO, {
            "enabled": False, "anchor_date": "2024-01-15",
        })
        await plugin._emit_diary_event(
            event_factory(umo=GROUP_UMO), "manual_lift",
            {"actions": ["cold_violence"]},
        )
        assert await plugin.diary_journal.store.pending_events() == []


class TestUmoCycleGatingPipeline:
    """P2 回归：会话级周期失效（toggle 关闭）时滞留事件延后处理。"""

    @pytest.mark.asyncio
    async def test_session_disabled_defers_diary_processing(
        self, pipeline, event_factory, monkeypatch,
    ):
        plugin, provider = pipeline
        monkeypatch.setattr(
            plugin.diary_journal, "_resolve_provider", lambda pid: provider,
        )
        plugin.diary_journal.retry_delay = 0.01  # 测试不等真实退避
        # 会话级关闭周期（等效 /period toggle 关闭）
        await plugin.store.set(GROUP_UMO, {
            "enabled": False, "anchor_date": "2024-01-15",
        })
        await plugin.diary_journal.submit(
            OWNER, "mood_changed", "事件", umo=GROUP_UMO,
        )
        await asyncio.sleep(0.1)
        assert provider.calls == []  # 不调用模型、不创建日记
        assert len(await plugin.diary_journal.store.pending_events()) == 1

        # 重新启用后自动恢复
        provider.queue(
            '{"tool": "diary_write", "args": {"text": "恢复后写入。"}}',
            '{"tool": "diary_count", "args": {}}',
        )
        await plugin.store.set(GROUP_UMO, {
            "enabled": True, "anchor_date": "2024-01-15",
        })
        await plugin.diary_journal.wait_idle()
        assert await plugin.diary_journal.store.get_diary(OWNER) is not None
        await plugin.diary_journal.shutdown()

    @pytest.mark.asyncio
    async def test_uncomputable_cycle_defers_diary_processing(
        self, pipeline, event_factory, monkeypatch,
    ):
        """P2 回归：锚点/周期参数损坏（周期不可计算）同样视为失效。"""
        plugin, provider = pipeline
        monkeypatch.setattr(
            plugin.diary_journal, "_resolve_provider", lambda pid: provider,
        )
        plugin.diary_journal.retry_delay = 0.01
        await plugin.store.set(GROUP_UMO, {
            "enabled": True, "anchor_date": "2024-13-99",  # 损坏日期
        })
        await plugin.diary_journal.submit(
            OWNER, "mood_changed", "事件", umo=GROUP_UMO,
        )
        await asyncio.sleep(0.1)
        assert provider.calls == []  # 与请求链路同严：不可计算即失效
        assert len(await plugin.diary_journal.store.pending_events()) == 1
        await plugin.diary_journal.shutdown()

    @pytest.mark.asyncio
    async def test_dashboard_delete_session_discards_pending_events(
        self, pipeline, event_factory, monkeypatch,
    ):
        """P2 回归：Dashboard 永久删除会话，按 reset 同款丢弃滞留事件。"""
        plugin, provider = pipeline
        await plugin.store.set(GROUP_UMO, {
            "enabled": True, "anchor_date": "2024-01-15",
        })
        await plugin.diary_journal.store.enqueue({
            "id": "evt-del", "owner_key": OWNER, "kind": "mood_changed",
            "summary": "s", "display_name": "", "provider_id": "",
            "umo": GROUP_UMO, "occurred_at": "2026-08-12T00:00:00+00:00",
        })
        await _seed_diary(plugin)  # 已提交日记必须保留

        await plugin._webapi_delete_session(GROUP_UMO)

        assert await plugin.diary_journal.store.pending_events() == []
        assert await plugin.diary_journal.store.get_diary(OWNER) is not None
        await plugin.diary_journal.shutdown()
