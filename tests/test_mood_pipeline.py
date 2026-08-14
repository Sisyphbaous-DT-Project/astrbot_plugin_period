"""Integration tests for the vNext mood pipeline (on_llm_request chain).

驱动真实 PeriodPlugin.on_llm_request，使用可编程 Provider 脚本化三段调用，
验证：调用次数、门禁解耦、本轮生效、硬沉默、精确轮数、失败保守、安全出口。
"""

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

# Add parent dir so astrbot_plugin_period is importable as a package
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import pytest

from astrbot_plugin_period.main import PeriodPlugin
from astrbot_plugin_period.core.mood import MoodState, PersistentAction

from conftest import MockConversationManager, ProgrammableProvider


UMO = "aiocqhttp:group:1001_12345"

SCREEN_NO = '{"need_intervention": false, "reasoning": "正常"}'
SCREEN_YES = '{"need_intervention": true, "reasoning": "用户敷衍"}'

MOOD_ACTIVE_UPDATE = {
    "status": "active", "summary": "有点介意", "cause_category": "dismissive",
    "latest_reason": "回应很轻", "improved": False,
    "fully_recovered": False, "recovery_reason": "",
}


def _decision_json(**kw):
    import json
    base = {
        "mood_update": MOOD_ACTIVE_UPDATE,
        "actions": [],
        "lift_actions": [],
        "silence_mode": "none",
        "reasoning_summary": "摘要",
    }
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


@pytest.fixture
def pipeline(tmp_path, sample_config, monkeypatch):
    from astrbot.api.star import Context, StarTools

    config = deepcopy(sample_config)
    config.update({
        "default_anchor_date": "2024-01-15",
        "default_enabled": True,
        "global_inject": True,
        "mood_system_enabled": True,
        "mood_scope": "per_umo",
        "mood_consult_history_messages": 30,
    })
    ctx = Context()
    provider = ProgrammableProvider()
    ctx.get_using_provider = lambda umo=None: provider
    ctx.get_provider_by_id = lambda pid: None
    conv_mgr = MockConversationManager()
    conv_mgr.seed(UMO, [{"role": "user", "content": "旧消息"}])
    ctx.conversation_manager = conv_mgr
    monkeypatch.setattr(StarTools, "get_data_dir", lambda _name=None: tmp_path)
    plugin = PeriodPlugin(ctx, config)
    return plugin, provider, conv_mgr


def _make_req(contexts=None):
    """内部 Agent 形态的请求：AstrBot 自 v3.4 起总是设置 req.conversation。"""
    from astrbot.api.provider import ProviderRequest
    req = ProviderRequest()
    req.prompt = "用户消息"
    req.system_prompt = "人格设定"
    req.contexts = contexts or []
    req.model = "model-x"
    req.conversation = SimpleNamespace(cid="cid-1", history=None)
    return req


def _extra_texts(req):
    return [p.text for p in req.extra_user_content_parts]


class TestCallCounts:
    @pytest.mark.asyncio
    async def test_screen_no_single_call(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        provider.queue(SCREEN_NO)
        ev = event_factory(umo=UMO)
        req = _make_req()
        await plugin.on_llm_request(ev, req)

        assert len(provider.calls) == 1
        assert ev.is_stopped() is False
        # 身体提示照常注入
        assert "[身体感知系统]" in req.system_prompt

    @pytest.mark.asyncio
    async def test_screen_yes_three_calls(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        provider.queue(SCREEN_YES, "我有点介意，想敷衍他一下", _decision_json(
            actions=[{"name": "perfunctory_reply", "params": {"level": 2}}],
        ))
        ev = event_factory(umo=UMO)
        req = _make_req(contexts=[{"role": "user", "content": "上一条"}])
        await plugin.on_llm_request(ev, req)

        assert len(provider.calls) == 3
        # ② 继承人格/模型/真实历史
        consult = provider.calls[1]
        assert consult["system_prompt"] == "人格设定"
        assert consult["model"] == "model-x"
        assert consult["contexts"] == [{"role": "user", "content": "上一条"}]
        # ③ 不接触隐私上下文
        interpret = provider.calls[2]
        assert "contexts" not in interpret
        assert "system_prompt" in interpret  # 只有翻译器系统提示

    @pytest.mark.asyncio
    async def test_soft_action_takes_effect_same_round(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        provider.queue(SCREEN_YES, "想求安慰", _decision_json(
            actions=[{"name": "seek_comfort", "params": {"type": "emotional"}}],
        ))
        ev = event_factory(umo=UMO)
        req = _make_req()
        await plugin.on_llm_request(ev, req)

        assert ev.is_stopped() is False
        texts = _extra_texts(req)
        assert any("倾向" in t and "安慰" in t for t in texts)
        # 即时状态快照注入
        assert any("当前心境：有点介意" in t for t in texts)
        # 软动作不持久化
        state = await plugin.mood_store.get(UMO)
        assert state.persistent_actions == []
        assert state.summary == "有点介意"


class TestGateDecoupling:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("inject_mode", ["only_status", "on_trigger", "interval_3"])
    async def test_display_gate_never_blocks_mood(self, pipeline, event_factory, inject_mode):
        plugin, provider, _ = pipeline
        plugin.config["inject_mode"] = inject_mode
        plugin.config["trigger_keywords"] = ["绝不会命中的词"]
        if inject_mode == "interval_3":
            # 先打两发不计入命中的请求，让第三次落在 %3!=1 分支
            provider.queue(SCREEN_NO, SCREEN_NO)
            for _ in range(2):
                await plugin.on_llm_request(event_factory(umo=UMO), _make_req())
        provider.queue(SCREEN_YES, "介意", _decision_json())
        ev = event_factory(umo=UMO, message_str="普通消息")
        req = _make_req()
        await plugin.on_llm_request(ev, req)

        # 情绪三段照常执行（only_status/on_trigger 以及 interval_3 的非展示轮）
        if inject_mode == "interval_3":
            assert len(provider.calls) == 2 + 3
            assert "[当前生理状态]" not in req.system_prompt
        else:
            assert len(provider.calls) == 3
            assert "[身体感知系统]" not in req.system_prompt

    @pytest.mark.asyncio
    async def test_warmup_does_not_block_mood(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        plugin.config["warmup_rounds"] = 5  # 前 5 轮不展示身体提示
        provider.queue(SCREEN_YES, "介意", _decision_json())
        ev = event_factory(umo=UMO)
        req = _make_req()
        await plugin.on_llm_request(ev, req)
        assert len(provider.calls) == 3
        assert "[身体感知系统]" not in req.system_prompt

    @pytest.mark.asyncio
    async def test_cycle_gate_blocks_everything(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        plugin.config["auto_inject"] = False  # 周期总开关关闭
        ev = event_factory(umo=UMO)
        req = _make_req()
        await plugin.on_llm_request(ev, req)
        assert len(provider.calls) == 0
        assert "[身体感知系统]" not in req.system_prompt


class TestColdViolence:
    @pytest.mark.asyncio
    async def test_immediate_silence_stops_and_persists_user_only(
        self, pipeline, event_factory,
    ):
        plugin, provider, conv_mgr = pipeline
        provider.queue(SCREEN_YES, "不想理他了", _decision_json(
            actions=[{"name": "cold_violence", "params": {"duration": 30}}],
            silence_mode="immediate",
        ))
        ev = event_factory(umo=UMO)
        req = _make_req()
        await plugin.on_llm_request(ev, req)

        assert ev.is_stopped() is True
        # 只写一次用户消息，无助手幽灵回复
        history = conv_mgr.update_calls[-1]["history"]
        assert len(history) == 2
        assert history[-1] == {"role": "user", "content": "用户消息"}
        assert not any(h["role"] == "assistant" for h in history)
        # 沉默轮不注入任何提示
        assert req.extra_user_content_parts == []

    @pytest.mark.asyncio
    async def test_after_expression_replies_once_then_silences(
        self, pipeline, event_factory,
    ):
        plugin, provider, conv_mgr = pipeline
        # 第一轮：after_expression，本轮正常表达
        provider.queue(SCREEN_YES, "我说完这句就不想理他了", _decision_json(
            actions=[{"name": "cold_violence", "params": {"duration": 30}}],
            silence_mode="after_expression",
        ))
        ev1 = event_factory(umo=UMO)
        req1 = _make_req()
        await plugin.on_llm_request(ev1, req1)
        assert ev1.is_stopped() is False
        assert any("最后一句" in t for t in _extra_texts(req1))
        # 状态已落库
        state = await plugin.mood_store.get(UMO)
        assert state.get_action("cold_violence") is not None

        # 第二轮：已有硬动作 → ①照常执行，筛选为否也强制进②；主模型不想解除 → 沉默
        provider.queue(SCREEN_NO, "还是不想理他", _decision_json())
        ev2 = event_factory(umo=UMO)
        req2 = _make_req()
        await plugin.on_llm_request(ev2, req2)
        assert ev2.is_stopped() is True
        # 三段架构不可跳过：第二轮同样是 screen+consult+interpret 三次调用
        assert len(provider.calls) == 3 + 3

    @pytest.mark.asyncio
    async def test_expired_cold_violence_auto_lifts(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 1},
            expires_at="2000-01-01T00:00:00+00:00",  # 早已过期
        ))
        await plugin.mood_store.set(UMO, state)

        provider.queue(SCREEN_NO)
        ev = event_factory(umo=UMO)
        req = _make_req()
        await plugin.on_llm_request(ev, req)
        assert ev.is_stopped() is False
        state = await plugin.mood_store.get(UMO)
        assert state.persistent_actions == []


class TestReadNoReply:
    @pytest.mark.asyncio
    async def test_exactly_three_interceptions(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        # 轮1：激活（rounds=3 含当轮）
        provider.queue(SCREEN_YES, "不想回", _decision_json(
            actions=[{"name": "read_no_reply", "params": {"rounds": 3}}],
            silence_mode="immediate",
        ))
        # 轮2、3：①照常执行（筛选否也强制进②），主模型不解除
        provider.queue(SCREEN_NO, "还是不想回", _decision_json())
        provider.queue(SCREEN_NO, "不想回", _decision_json())
        # 轮4：轮数耗尽，走①且筛选否
        provider.queue(SCREEN_NO)

        events = [event_factory(umo=UMO) for _ in range(4)]
        for ev in events:
            await plugin.on_llm_request(ev, _make_req())

        assert [ev.is_stopped() for ev in events] == [True, True, True, False]
        state = await plugin.mood_store.get(UMO)
        assert state.persistent_actions == []
        # 调用数：3 + 3 + 3 + 1
        assert len(provider.calls) == 10


class TestFailureConservative:
    @pytest.mark.asyncio
    async def test_consult_failure_keeps_hard_state(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        provider.queue(SCREEN_NO, RuntimeError("模型超时"))  # ①正常，②失败
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        assert ev.is_stopped() is True  # 硬状态仍按原规则沉默
        state = await plugin.mood_store.get(UMO)
        action = state.get_action("cold_violence")
        assert action is not None
        assert action.expires_at == "2999-01-01T00:00:00+00:00"  # 不延长

    @pytest.mark.asyncio
    async def test_interpret_invalid_keeps_state(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        provider.queue(SCREEN_YES, "模糊回答", "这不是JSON")
        ev = event_factory(umo=UMO)
        req = _make_req()
        await plugin.on_llm_request(ev, req)
        assert ev.is_stopped() is False
        state = await plugin.mood_store.get(UMO)
        assert state is None or state.persistent_actions == []


class TestSafetyExit:
    @pytest.mark.asyncio
    async def test_lift_bypasses_switches(self, pipeline, event_factory):
        plugin, _, _ = pipeline
        plugin.config["mood_system_enabled"] = False
        plugin.config["commands_enabled"] = "none"
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        ev = event_factory(umo=UMO)
        results = []
        async for r in plugin.period_lift(ev):
            results.append(r)
        state = await plugin.mood_store.get(UMO)
        assert state.persistent_actions == []
        # 该状态只有硬动作、无内在情绪：文案不声称手动恢复
        assert any("已解除所有情绪动作" in str(r) for r in results)

    @pytest.mark.asyncio
    async def test_lift_works_in_global_scope(self, pipeline, event_factory):
        plugin, _, _ = pipeline
        plugin.config["mood_scope"] = "global"
        state = MoodState()
        state.add_action(PersistentAction.create("read_no_reply", {}, remaining_replies=2))
        await plugin.mood_store.set("__global__", state)

        ev = event_factory(umo=UMO)
        async for _ in plugin.period_lift(ev):
            pass
        state = await plugin.mood_store.get("__global__")
        assert state.persistent_actions == []

    @pytest.mark.asyncio
    async def test_mood_viewable_when_disabled(self, pipeline, event_factory):
        plugin, _, _ = pipeline
        plugin.config["mood_system_enabled"] = False
        await plugin.mood_store.set(UMO, MoodState(summary="介意", status="active"))

        ev = event_factory(umo=UMO)
        texts = []
        async for r in plugin.period_mood(ev):
            texts.append(str(r))
        assert any("介意" in t for t in texts)


class TestRecoveryRetention:
    @pytest.mark.asyncio
    async def test_recovery_event_injected_then_cleaned(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        state = MoodState(
            status="recovered", summary="缓和了", latest_reason="之前吵了一架",
            fully_recovered=True, recovery_reason="对方认真道歉",
            recovered_messages_left=2,
        )
        await plugin.mood_store.set(UMO, state)

        # 两条有效消息都注入恢复事件
        for _ in range(2):
            provider.queue(SCREEN_NO)
            req = _make_req()
            await plugin.on_llm_request(event_factory(umo=UMO), req)
            assert any("恢复原因" in t for t in _extra_texts(req))

        # 计数归零：原因清空、回到 stable
        state = await plugin.mood_store.get(UMO)
        assert state.status == "stable"
        assert state.latest_reason == ""
        assert state.recovery_reason == ""

        # 第三条不再注入恢复事件
        provider.queue(SCREEN_NO)
        req = _make_req()
        await plugin.on_llm_request(event_factory(umo=UMO), req)
        assert not any("恢复原因" in t for t in _extra_texts(req))


class TestRunnerSkip:
    @pytest.mark.asyncio
    async def test_no_conversation_skips_even_with_manager(self, pipeline, event_factory):
        """第三方 Runner：req.conversation 为 None 即跳过——全局 Context 恒有
        conversation_manager，不能用后者做判据（4.27.2 third_party.py 实测形态）。"""
        plugin, provider, _ = pipeline
        # 注意：保留 conversation_manager，模拟真实第三方 Runner 环境
        ev = event_factory(umo=UMO)
        req = _make_req()
        req.conversation = None
        await plugin.on_llm_request(ev, req)
        assert len(provider.calls) == 0
        assert ev.is_stopped() is False


class TestHardActionNoRefill:
    @pytest.mark.asyncio
    async def test_same_name_hard_action_cannot_refill(self, pipeline, event_factory):
        """模型重复激活同名硬动作不得重置期限/轮数（防“续杯”）。"""
        plugin, provider, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "read_no_reply", {"rounds": 3}, remaining_replies=2,
        ))
        await plugin.mood_store.set(UMO, state)

        # ①筛选是 → ②想再来一轮 read_no_reply(rounds=3) → ③翻译
        provider.queue(SCREEN_YES, "还是不想回他", _decision_json(
            actions=[{"name": "read_no_reply", "params": {"rounds": 3}}],
            silence_mode="immediate",
        ))
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        assert ev.is_stopped() is True  # 旧动作继续沉默
        state = await plugin.mood_store.get(UMO)
        action = state.get_action("read_no_reply")
        assert action is not None
        # 旧动作本轮递减为 1，而不是被重置为 3 后再减为 2
        assert action.remaining_replies == 1

    @pytest.mark.asyncio
    async def test_cold_violence_cannot_extend_expiry(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        provider.queue(SCREEN_YES, "继续冷暴力", _decision_json(
            actions=[{"name": "cold_violence", "params": {"duration": 1440}}],
            silence_mode="immediate",
        ))
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        state = await plugin.mood_store.get(UMO)
        action = state.get_action("cold_violence")
        assert action is not None
        assert action.expires_at == "2999-01-01T00:00:00+00:00"  # 原到期时间不变
        assert action.params.get("duration") == 30  # 参数也不被新值覆盖


class TestRoundProviderBinding:
    @pytest.mark.asyncio
    async def test_consult_uses_selected_provider(self, pipeline, event_factory):
        """②必须绑定本轮 selected_provider，而不是重新取会话默认 Provider。"""
        plugin, provider, _ = pipeline
        alt = ProgrammableProvider()
        plugin.context.get_provider_by_id = lambda pid: alt if pid == "alt-p" else None

        provider.queue(SCREEN_YES, _decision_json())
        alt.queue("临时模型的人格回答")

        ev = event_factory(umo=UMO)
        ev.get_extra.side_effect = lambda key, *a: (
            "alt-p" if key == "selected_provider" else None
        )
        await plugin.on_llm_request(ev, _make_req())

        # ②绑定到 selected Provider；①③仍走小模型/默认链
        assert len(provider.calls) == 2
        assert len(alt.calls) == 1
        assert alt.calls[0]["system_prompt"] == "人格设定"


class TestLiftWithoutHardAction:
    @pytest.mark.asyncio
    async def test_lift_marks_manual_recovery_without_actions(self, pipeline, event_factory):
        """无硬动作时 lift 也能退出内在情绪（标记手动恢复）。"""
        plugin, _, _ = pipeline
        await plugin.mood_store.set(UMO, MoodState(
            status="active", summary="很介意", latest_reason="被敷衍",
        ))

        ev = event_factory(umo=UMO)
        texts = []
        async for r in plugin.period_lift(ev):
            texts.append(str(r))
        assert any("手动恢复" in t for t in texts)

        state = await plugin.mood_store.get(UMO)
        assert state.status == "recovered"
        assert state.fully_recovered is True
        assert state.recovery_reason == "用户手动解除"
        assert state.recovered_messages_left == 10

    @pytest.mark.asyncio
    async def test_lift_actions_only_message(self, pipeline, event_factory):
        """只有硬动作、无内在情绪时，回复不得谎称已标记手动恢复。"""
        plugin, _, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create("read_no_reply", {}, remaining_replies=2))
        await plugin.mood_store.set(UMO, state)
        ev = event_factory(umo=UMO)
        texts = [str(r) async for r in plugin.period_lift(ev)]
        assert any("已解除所有情绪动作" in t for t in texts)
        assert not any("手动恢复" in t for t in texts)

    @pytest.mark.asyncio
    async def test_lift_noop_when_nothing_to_lift(self, pipeline, event_factory):
        plugin, _, _ = pipeline
        ev = event_factory(umo=UMO)
        texts = []
        async for r in plugin.period_lift(ev):
            texts.append(str(r))
        assert any("没有需要解除" in t for t in texts)

    @pytest.mark.asyncio
    async def test_lift_survives_mood_store_disk_failure(
        self, pipeline, event_factory, monkeypatch,
    ):
        """安全出口不得被落盘故障击穿，也不得假成功：失败如实告知且状态不变。"""
        from pathlib import Path

        plugin, _, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        def always_fail(*args, **kwargs):
            raise OSError("磁盘满")

        monkeypatch.setattr(Path, "write_text", always_fail)
        ev = event_factory(umo=UMO)
        texts = [str(r) async for r in plugin.period_lift(ev)]  # 不得抛出
        assert any("保存失败" in t for t in texts)  # 不得谎称"已解除"
        assert not any("已解除" in t for t in texts)
        await plugin.diary_journal.shutdown()

        # 解除未生效：旧硬动作仍在（缓存镜像磁盘）
        monkeypatch.undo()
        state_after = await plugin.mood_store.get(UMO)
        assert state_after.get_action("cold_violence") is not None
        # 未发 manual_lift 日记事件（那会是谎话）
        pending = await plugin.diary_journal.store.pending_events()
        assert not any(e["kind"] == "manual_lift" for e in pending)


class TestScreenFailureConservative:
    """P1-① 回归：调用①失败必须走保守策略，与'筛选为否'严格区分。"""

    @pytest.mark.asyncio
    async def test_screen_failure_with_hard_action_keeps_silence(self, pipeline, event_factory):
        """①失败 + 已有硬动作：按原规则沉默，不进②③（不给解除机会）。"""
        plugin, provider, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        provider.queue(RuntimeError("筛选模型挂了"))
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        assert ev.is_stopped() is True
        assert len(provider.calls) == 1  # 只有①，没有②③
        state = await plugin.mood_store.get(UMO)
        action = state.get_action("cold_violence")
        assert action is not None  # 原状态保持
        assert action.expires_at == "2999-01-01T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_screen_failure_without_hard_action_passes(self, pipeline, event_factory):
        """①失败 + 无硬动作：正常放行，不激活任何新动作。"""
        plugin, provider, _ = pipeline
        provider.queue(RuntimeError("筛选模型挂了"))
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        assert ev.is_stopped() is False
        assert len(provider.calls) == 1
        state = await plugin.mood_store.get(UMO)
        assert state is None or state.persistent_actions == []


class TestDiarySubmitIsolation:
    """P1-② 回归：日记入队/落盘故障不得打断主请求链。"""

    @pytest.mark.asyncio
    async def test_submit_failure_does_not_break_pipeline(self, pipeline, event_factory):
        plugin, provider, _ = pipeline

        async def boom(*args, **kwargs):
            raise OSError("模拟磁盘写失败")

        plugin.diary_journal.submit = boom
        provider.queue(SCREEN_YES, "我很介意", _decision_json(
            actions=[{"name": "perfunctory_reply", "params": {"level": 1}}],
        ))
        ev = event_factory(umo=UMO)
        req = _make_req()
        await plugin.on_llm_request(ev, req)  # 不抛异常

        assert ev.is_stopped() is False
        assert len(provider.calls) == 3  # 三段照常
        state = await plugin.mood_store.get(UMO)
        assert state.summary == "有点介意"  # 心境提交不受日记故障影响


class TestScreenBadJsonConservative:
    """P1-④ 回归：①响应不可解析 = 失败，不是'无需介入'。"""

    @pytest.mark.asyncio
    async def test_bad_json_with_hard_action_keeps_silence(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        provider.queue("这不是JSON")  # ①返回坏 JSON
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        assert ev.is_stopped() is True   # 保守沉默
        assert len(provider.calls) == 1  # 不进②③
        state = await plugin.mood_store.get(UMO)
        assert state.get_action("cold_violence") is not None  # 旧动作不被解除

    @pytest.mark.asyncio
    async def test_bad_json_without_hard_action_passes(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        provider.queue("")  # 空响应同样算失败
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())
        assert ev.is_stopped() is False
        assert len(provider.calls) == 1


class TestCorruptedMoodFile:
    """P1 回归：损坏的 moods.json 不得在请求钩子上抛异常。"""

    @pytest.mark.asyncio
    async def test_corrupted_v3_record_survives(self, pipeline, event_factory):
        import json as _json

        plugin, provider, _ = pipeline
        # params 是字符串、history 是 null 的损坏 v3 记录
        plugin.mood_store._file_path.write_text(_json.dumps({
            UMO: {
                "schema_version": 3, "history": None,
                "persistent_actions": [{
                    "name": "cold_violence", "params": "oops",
                    "expires_at": "2999-01-01T00:00:00+00:00",
                }],
            },
        }), encoding="utf-8")
        provider.queue(SCREEN_NO)
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())  # 不得抛出
        state = await plugin.mood_store.get(UMO)
        assert state is not None
        assert state.persistent_actions[0].params == {}  # 坏 params 回退为空


class TestMoodResetSerialized:
    """P1 回归：moodreset/删除会话必须与请求链共用情绪锁，且检查删除结果。"""

    @pytest.mark.asyncio
    async def test_moodreset_waits_for_mood_lock(self, pipeline, event_factory):
        import asyncio

        plugin, _, _ = pipeline
        await plugin.mood_store.set(UMO, MoodState(summary="介意"))
        lock = await plugin._get_mood_lock(UMO)
        await lock.acquire()
        try:
            ev = event_factory(umo=UMO)

            async def run():
                return [str(r) async for r in plugin.period_mood_reset(ev)]

            task = asyncio.create_task(run())
            await asyncio.sleep(0.05)
            assert not task.done()  # 被情绪锁挡住，不能绕过请求链
        finally:
            lock.release()
        texts = await task
        assert any("已重置" in t for t in texts)
        assert await plugin.mood_store.get(UMO) is None

    @pytest.mark.asyncio
    async def test_moodreset_reports_delete_failure(
        self, pipeline, event_factory, monkeypatch,
    ):
        plugin, _, _ = pipeline
        await plugin.mood_store.set(UMO, MoodState(summary="介意"))

        async def fail(*args, **kwargs):
            return False

        monkeypatch.setattr(plugin.mood_store, "delete", fail)
        ev = event_factory(umo=UMO)
        texts = [str(r) async for r in plugin.period_mood_reset(ev)]
        assert any("失败" in t for t in texts)
        assert not any("已重置" in t for t in texts)

    @pytest.mark.asyncio
    async def test_webapi_delete_keeps_lock_and_checks_result(
        self, pipeline, event_factory, monkeypatch,
    ):
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        lock = await plugin._get_mood_lock(UMO)

        await plugin._webapi_delete_session(UMO)
        assert plugin._mood_locks.get(UMO) is lock  # 不得 pop（pop 会导致双锁并行）

        # 情绪删除落盘失败 → 返回错误而非假成功（组合结果包含各项明细）
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        await plugin.mood_store.set(UMO, MoodState(summary="介意"))

        async def fail(*args, **kwargs):
            return False

        original_delete = plugin.mood_store.delete
        monkeypatch.setattr(plugin.mood_store, "delete", fail)
        resp, status = await plugin._webapi_delete_session(UMO)
        assert status == 500

        # 周期已删、情绪残留：重试本接口不得 404，应继续清理残留
        monkeypatch.setattr(plugin.mood_store, "delete", original_delete)
        result2 = await plugin._webapi_delete_session(UMO)
        assert result2["status"] == "ok"


class TestDiaryEventsAfterPersist:
    """P2 回归：状态落盘失败时，本轮收集的日记事件不得下发。"""

    @pytest.mark.asyncio
    async def test_set_failure_drops_collected_events(
        self, pipeline, event_factory, monkeypatch,
    ):
        plugin, provider, _ = pipeline
        provider.queue(SCREEN_YES, "我很介意", _decision_json())  # 产生 mood_changed

        async def fail(*args, **kwargs):
            return False

        monkeypatch.setattr(plugin.mood_store, "set", fail)
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())
        # 状态没保存：事件不得入队（否则日记记了一笔没保存的心境）
        assert await plugin.diary_journal.store.pending_events() == []

    @pytest.mark.asyncio
    async def test_set_success_flushes_collected_events(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        provider.queue(SCREEN_YES, "我很介意", _decision_json())
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())
        pending = await plugin.diary_journal.store.pending_events()
        assert any(e["kind"] == "mood_changed" for e in pending)
        await plugin.diary_journal.shutdown()


class TestConsultNonStringResponse:
    """P1 回归：②返回非字符串（畸形 Provider）不得穿透请求钩子。"""

    @pytest.mark.asyncio
    async def test_non_string_with_hard_action_keeps_silence(
        self, pipeline, event_factory,
    ):
        """②返回 dict：按调用失败保守沉默，不得 AttributeError 穿透后意外回复。"""
        plugin, provider, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        provider.queue(SCREEN_YES, {"not": "a string"})  # ①是，②返回 dict
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())  # 不得抛出

        assert ev.is_stopped() is True   # 保守沉默
        assert len(provider.calls) == 2  # ②被调用但按失败处理，不进③
        state = await plugin.mood_store.get(UMO)
        assert state.get_action("cold_violence") is not None  # 旧动作不被解除

    @pytest.mark.asyncio
    async def test_non_string_without_hard_action_passes(
        self, pipeline, event_factory,
    ):
        plugin, provider, _ = pipeline
        provider.queue(SCREEN_YES, ["list", "response"])
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())
        assert ev.is_stopped() is False
        assert len(provider.calls) == 2  # 不进③，正常放行


class TestSilencePersistFailure:
    """P1-⑤ 回归：沉默轮状态落盘失败不得让正式回复意外发出。"""

    @pytest.mark.asyncio
    async def test_silence_survives_store_failure(self, pipeline, event_factory, monkeypatch):
        plugin, provider, _ = pipeline
        state = MoodState()
        state.add_action(PersistentAction.create(
            "read_no_reply", {"rounds": 3}, remaining_replies=2,
        ))
        await plugin.mood_store.set(UMO, state)

        async def boom(*args, **kwargs):
            raise OSError("磁盘写失败")

        monkeypatch.setattr(plugin.mood_store, "set", boom)
        provider.queue(SCREEN_NO, "不想回", _decision_json())  # ①②③正常，不解除
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        assert ev.is_stopped() is True  # 落盘失败，沉默仍然生效


class TestConsultTemplateConservative:
    """P1 回归：自定义②模板语法错误按调用失败处理，已有硬动作仍保守沉默。"""

    @pytest.mark.asyncio
    async def test_malformed_template_keeps_silence(self, pipeline, event_factory):
        plugin, provider, _ = pipeline
        plugin.config["mood_detector_consult_prompt"] = "当前状态 {oops"
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        provider.queue(SCREEN_YES)  # ①正常 → 进②；②模板构建失败 → 保守
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        assert ev.is_stopped() is True  # 硬沉默不失效
        state = await plugin.mood_store.get(UMO)
        action = state.get_action("cold_violence")
        assert action is not None
        assert action.expires_at == "2999-01-01T00:00:00+00:00"  # 不延长
        # ②未发起模型调用（模板构建失败），只有①一次调用
        assert len(provider.calls) == 1


class TestCorruptedDiaryFilePipeline:
    """P1 回归：损坏的 emotion_diaries.json 不得打穿请求链。

    日记读取在调用①之前，清洗后按无日记处理，硬动作照常沉默。
    """

    @pytest.mark.asyncio
    async def test_corrupted_diaries_keeps_silence(
        self, pipeline, event_factory, tmp_path,
    ):
        import json as _json

        plugin, provider, _ = pipeline
        (tmp_path / "emotion_diaries.json").write_text(_json.dumps({
            "schema_version": 1,
            "diaries": {"qq_1:10000:12345": {"entries": 5, "aliases": "x"}},
            "pending_events": [{"id": "bad"}],  # 缺 owner_key，应被清洗
        }), encoding="utf-8")

        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        provider.queue(SCREEN_YES, "不想理他", _decision_json())
        ev = event_factory(umo=UMO)
        await plugin.on_llm_request(ev, _make_req())

        assert ev.is_stopped() is True
        assert len(provider.calls) == 3  # 三段照常执行
        # 清洗后按无日记处理：②提示词里是占位而非异常
        assert "暂无日记" in provider.calls[1]["prompt"]


class TestWebapiDeleteGlobalScope:
    """P1 回归：global 模式下删除会话不得触碰跨会话共享的全局情绪。

    全局情绪既不是"该 UMO 存在"的证据（删除打错字的 UMO 不得清全局
    状态），也不随单个会话删除而清除（由 /period moodreset 管理）。
    """

    @pytest.mark.asyncio
    async def test_delete_nonexistent_umo_keeps_global_mood(self, pipeline):
        plugin, _, _ = pipeline
        plugin.config["mood_scope"] = "global"
        await plugin.mood_store.set(
            "__global__", MoodState(summary="介意", status="active"),
        )
        resp, status = await plugin._webapi_delete_session("never-existed-umo")
        assert status == 404
        state = await plugin.mood_store.get("__global__")
        assert state is not None and state.summary == "介意"

    @pytest.mark.asyncio
    async def test_delete_real_session_keeps_global_mood(self, pipeline):
        plugin, _, _ = pipeline
        plugin.config["mood_scope"] = "global"
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        await plugin.mood_store.set(
            "__global__", MoodState(summary="介意", status="active"),
        )
        result = await plugin._webapi_delete_session(UMO)
        assert result["status"] == "ok"
        assert "moodreset" in result["data"]["note"]
        assert await plugin.store.get(UMO) is None
        state = await plugin.mood_store.get("__global__")
        assert state is not None and state.summary == "介意"

    @pytest.mark.asyncio
    async def test_per_umo_residual_retry_still_reachable(self, pipeline, monkeypatch):
        """per_umo 模式：周期已删但情绪残留时，重试不得 404。"""
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        await plugin.mood_store.set(UMO, MoodState(summary="介意"))

        original_delete = plugin.mood_store.delete

        async def fail(*args, **kwargs):
            return False

        monkeypatch.setattr(plugin.mood_store, "delete", fail)
        resp, status = await plugin._webapi_delete_session(UMO)
        assert status == 500
        monkeypatch.setattr(plugin.mood_store, "delete", original_delete)
        result2 = await plugin._webapi_delete_session(UMO)
        assert result2["status"] == "ok"
        assert await plugin.mood_store.get(UMO) is None


class TestWebapiDeleteWatermarkFirst:
    """P2 回归：日记清理（含水位线）先于周期删除——水位线写失败时
    周期记录仍在，重试本接口可完成剩余清理。"""

    @pytest.mark.asyncio
    async def test_watermark_failure_keeps_cycle_retryable(self, pipeline, monkeypatch):
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})

        original_mark = plugin.diary_journal.store.mark_umo_watermark

        async def fail_mark(umo):
            return False

        monkeypatch.setattr(
            plugin.diary_journal.store, "mark_umo_watermark", fail_mark,
        )
        resp, status = await plugin._webapi_delete_session(UMO)
        assert status == 500
        assert "待处理日记事件" in resp["message"]
        # 周期记录未删：重试可达（不会因 404 卡死）
        assert await plugin.store.get(UMO) is not None

        monkeypatch.setattr(
            plugin.diary_journal.store, "mark_umo_watermark", original_mark,
        )
        result2 = await plugin._webapi_delete_session(UMO)
        assert result2["status"] == "ok"
        assert await plugin.store.get(UMO) is None


class TestCycleStoreWriteFailureHonest:
    """P1 回归：CycleStore 写路径故障——指令与 WebUI 如实报错、缓存不污染。"""

    @staticmethod
    def _break_disk(monkeypatch):
        from pathlib import Path

        def always_fail(*args, **kwargs):
            raise OSError("磁盘满")

        monkeypatch.setattr(Path, "write_text", always_fail)

    @pytest.mark.asyncio
    async def test_reset_command_delete_failure_reports(
        self, pipeline, event_factory, monkeypatch,
    ):
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        self._break_disk(monkeypatch)
        ev = event_factory(umo=UMO)
        texts = []
        async for r in plugin.period_reset(ev):
            texts.append(str(r))
        assert any("删除失败" in t for t in texts)
        # 缓存未污染：记录仍在，可重试
        assert await plugin.store.get(UMO) is not None

    @pytest.mark.asyncio
    async def test_toggle_command_failure_reports(
        self, pipeline, event_factory, monkeypatch,
    ):
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        self._break_disk(monkeypatch)
        ev = event_factory(umo=UMO)
        texts = []
        async for r in plugin.period_toggle(ev):
            texts.append(str(r))
        assert any("保存失败" in t for t in texts)
        assert (await plugin.store.get(UMO))["enabled"] is True

    @pytest.mark.asyncio
    async def test_webapi_toggle_failure_500(self, pipeline, monkeypatch):
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        self._break_disk(monkeypatch)
        resp, status = await plugin._webapi_toggle_session(UMO)
        assert status == 500
        assert (await plugin.store.get(UMO))["enabled"] is True


class TestWebapiDeleteMoodRace:
    """P1 回归：per_umo 删除无条件取锁并在锁内重新检查情绪状态。

    快照之后进行中请求首次写入情绪时，不得按旧快照跳过清理并谎报成功。
    """

    @pytest.mark.asyncio
    async def test_mood_created_after_snapshot_still_deleted(
        self, pipeline, monkeypatch,
    ):
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})

        real_get = plugin.mood_store.get
        real_set = plugin.mood_store.set
        calls = {"n": 0}

        async def get_then_create(umo):
            calls["n"] += 1
            if calls["n"] == 1:
                # 快照时刻尚无情绪；模拟进行中请求随即首次写入
                await real_set(UMO, MoodState(summary="介意", status="active"))
                return None
            return await real_get(umo)

        monkeypatch.setattr(plugin.mood_store, "get", get_then_create)
        result = await plugin._webapi_delete_session(UMO)

        assert result["status"] == "ok"
        assert result["data"]["mood_deleted"] is True
        # 锁内重检后已真实删除，不是谎报
        assert await real_get(UMO) is None


class TestDeleteLockCoversDiaryCleanup:
    """P1 回归：情绪锁覆盖整个删除流程。

    在途请求的日记事件提交发生在同一把锁内——锁前提交的事件随即被
    discard 按 UMO 清掉（与 occurred_at 无关），锁后提交被源头复查
    挡掉；"水位线后、周期删除前提交"不再能留下 outbox 残留。
    """

    @pytest.mark.asyncio
    async def test_event_submitted_before_delete_lock_is_discarded(self, pipeline):
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        # 在途请求在删除拿锁前完成了事件提交
        ok = await plugin.diary_journal.submit(
            "qq_1:10000:12345", "mood_changed", "在途事件", umo=UMO,
        )
        assert ok is True
        result = await plugin._webapi_delete_session(UMO)
        assert result["status"] == "ok"
        assert await plugin.diary_journal.store.pending_events() == []

    @pytest.mark.asyncio
    async def test_event_submitted_before_delete_lock_is_discarded_global(
        self, pipeline,
    ):
        plugin, _, _ = pipeline
        plugin.config["mood_scope"] = "global"
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        ok = await plugin.diary_journal.submit(
            "qq_1:10000:12345", "mood_changed", "在途事件", umo=UMO,
        )
        assert ok is True
        result = await plugin._webapi_delete_session(UMO)
        assert result["status"] == "ok"
        assert await plugin.diary_journal.store.pending_events() == []


class TestSecondGateInsideMoodLock:
    """P1 回归：锁内二次门禁——已通过入口门禁但排队等锁的请求，
    在删除完成后再拿到锁时不得写情绪/日记，外层也不得注入身体提示。"""

    @pytest.mark.asyncio
    async def test_request_waiting_for_lock_aborts_after_delete(
        self, pipeline, event_factory,
    ):
        import asyncio

        plugin, provider, _ = pipeline
        # 关闭全局默认兜底，确保删除后周期真正失效（否则回退到全局默认，
        # 二次门禁正确放行）
        plugin.config["default_anchor_date"] = ""
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        lock = await plugin._get_mood_lock(UMO)
        await lock.acquire()
        ev = event_factory(umo=UMO)
        req = _make_req()
        task = asyncio.create_task(plugin.on_llm_request(ev, req))
        await asyncio.sleep(0.05)
        assert not task.done()  # 已通过入口门禁，卡在情绪锁前

        # 删除（持锁方）先完成：清周期配置 + 写水位线
        await plugin.store.delete(UMO)
        await plugin.diary_journal.store.mark_umo_watermark(UMO)
        lock.release()
        await task

        assert provider.calls == []  # 三段未执行
        assert ev.is_stopped() is False  # 不是硬沉默，是无注入放行
        assert "[身体感知系统]" not in (req.system_prompt or "")  # 身体提示未注入
        assert req.extra_user_content_parts == []  # 情绪状态/日记未注入
        assert await plugin.mood_store.get(UMO) is None  # 情绪未复活
        assert await plugin.diary_journal.store.pending_events() == []  # 无日记事件


class TestCycleDeleteFailureKeepsRuntimeState:
    """P2 回归：周期删除写盘失败时运行态计数器保持原样。"""

    @pytest.mark.asyncio
    async def test_counters_kept_on_delete_failure(self, pipeline, monkeypatch):
        plugin, _, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        plugin._anchored_sessions.add(UMO)
        plugin._inject_counters[UMO] = 3
        plugin._warmup_counters[UMO] = 2

        async def fail_save(data):
            return False

        # 只让 CycleStore 写盘失败（水位线等日记清理不受影响）
        monkeypatch.setattr(plugin.store, "_save", fail_save)
        resp, status = await plugin._webapi_delete_session(UMO)
        assert status == 500
        assert "周期记录删除失败" in resp["message"]
        assert UMO in plugin._anchored_sessions
        assert plugin._inject_counters.get(UMO) == 3
        assert plugin._warmup_counters.get(UMO) == 2


class TestSwitchOffWhileWaitingForLock:
    """P2 回归：锁内先查情绪总开关（开关即时生效）。

    同一 mood_umo 的请求在情绪锁上串行，等锁期间管理员关闭
    mood_system_enabled 后，排队请求不得再执行三段调用或提交状态；
    跳过情绪但身体周期提示照常（关闭的是情绪系统，不是周期系统）。
    """

    @pytest.mark.asyncio
    async def test_switch_off_skips_mood_keeps_body_hint(
        self, pipeline, event_factory,
    ):
        import asyncio

        plugin, provider, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        lock = await plugin._get_mood_lock(UMO)
        await lock.acquire()
        ev = event_factory(umo=UMO)
        req = _make_req()
        task = asyncio.create_task(plugin.on_llm_request(ev, req))
        await asyncio.sleep(0.05)
        assert not task.done()  # 已通过入口检查，卡在情绪锁前

        # 等锁期间管理员关闭情绪总开关（配置即时生效）
        plugin.config["mood_system_enabled"] = False
        lock.release()
        await task

        assert provider.calls == []  # 三段未执行
        assert ev.is_stopped() is False
        assert await plugin.mood_store.get(UMO) is None  # 状态未写入
        # 身体周期提示照常注入（周期系统未关）
        assert "[身体感知系统]" in (req.system_prompt or "")

    @pytest.mark.asyncio
    async def test_switch_off_keeps_hard_state_without_silence(
        self, pipeline, event_factory,
    ):
        import asyncio

        plugin, provider, _ = pipeline
        await plugin.store.set(UMO, {"enabled": True, "anchor_date": "2024-01-15"})
        state = MoodState()
        state.add_action(PersistentAction.create(
            "cold_violence", {"duration": 30},
            expires_at="2999-01-01T00:00:00+00:00",
        ))
        await plugin.mood_store.set(UMO, state)

        lock = await plugin._get_mood_lock(UMO)
        await lock.acquire()
        ev = event_factory(umo=UMO)
        req = _make_req()
        task = asyncio.create_task(plugin.on_llm_request(ev, req))
        await asyncio.sleep(0.05)
        assert not task.done()

        plugin.config["mood_system_enabled"] = False
        lock.release()
        await task

        assert provider.calls == []
        assert ev.is_stopped() is False  # 立即停止拦截
        kept = await plugin.mood_store.get(UMO)
        # 硬状态保留（lift 仍可解除），不因关闭被清除也不被执行
        assert kept.get_action("cold_violence") is not None
