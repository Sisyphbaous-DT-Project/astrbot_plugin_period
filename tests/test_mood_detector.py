"""Tests for core/mood_detector.py — three-call architecture (vNext)."""

import pytest

from core.engine import PhaseInfo
from core.mood import MoodState, PersistentAction
from core.mood_detector import MoodDetector

from conftest import ProgrammableProvider


@pytest.fixture
def phase_info():
    return PhaseInfo(phase="luteal", day=3, days_to_next=4, total_day=25)


@pytest.fixture
def detector(programmable_provider):
    class FakeContext:
        def __init__(self, provider):
            self._provider = provider

        def get_using_provider(self, umo=None):
            return self._provider

        def get_provider_by_id(self, pid):
            return None

    return MoodDetector(FakeContext(programmable_provider), {})


class TestScreen:
    @pytest.mark.asyncio
    async def test_screen_yes(self, detector, programmable_provider, phase_info):
        programmable_provider.queue('{"need_intervention": true, "reasoning": "用户语气冲"}')
        result = await detector.screen(
            "umo", phase_info, MoodState(), [], "你很烦",
        )
        assert result["need_intervention"] is True
        assert result["reasoning"] == "用户语气冲"

    @pytest.mark.asyncio
    async def test_screen_no_on_garbage(self, detector, programmable_provider, phase_info):
        programmable_provider.queue("不是JSON")
        result = await detector.screen("umo", phase_info, MoodState(), [], "你好")
        assert result["need_intervention"] is False

    @pytest.mark.asyncio
    async def test_screen_failure_conservative(self, detector, programmable_provider, phase_info):
        programmable_provider.queue(RuntimeError("network down"))
        result = await detector.screen("umo", phase_info, MoodState(), [], "你好")
        assert result["need_intervention"] is False
        assert "调用失败" in result["reasoning"]

    @pytest.mark.asyncio
    async def test_screen_receives_compact_state_and_history(
        self, detector, programmable_provider, phase_info,
    ):
        state = MoodState(summary="有点介意")
        state.add_action(PersistentAction.create("read_no_reply", {}, remaining_replies=2))
        programmable_provider.queue('{"need_intervention": false}')
        history = [{"role": "user", "content": "上一条消息"}]
        await detector.screen("umo", phase_info, state, history, "在吗", persona_summary="人格" * 500)

        call = programmable_provider.calls[0]
        prompt = call["prompt"]
        assert "有点介意" in prompt
        assert "read_no_reply" in prompt
        assert "上一条消息" in prompt
        assert "在吗" in prompt
        # 人格摘要被截断
        assert "人格" in prompt
        assert len(prompt) < 500 * 2
        # 不携带 contexts（①不使用真实历史结构）
        assert "contexts" not in call

    @pytest.mark.asyncio
    async def test_screen_without_persona(self, detector, programmable_provider, phase_info):
        programmable_provider.queue('{"need_intervention": false}')
        await detector.screen("umo", phase_info, MoodState(), [], "你好", persona_summary="")
        assert "人格摘要" not in programmable_provider.calls[0]["prompt"]

    @pytest.mark.asyncio
    async def test_screen_string_false_is_failure_not_no(
        self, detector, programmable_provider, phase_info,
    ):
        """字符串 "false" 经 bool() 会变 True：必须按调用失败处理，不得当"否"。"""
        programmable_provider.queue('{"need_intervention": "false"}')
        result = await detector.screen("umo", phase_info, MoodState(), [], "你好")
        assert result["need_intervention"] is False
        assert result["failed"] is True

    @pytest.mark.asyncio
    async def test_screen_missing_field_is_failure(
        self, detector, programmable_provider, phase_info,
    ):
        """缺少 need_intervention 字段 = 输出不完整，按调用失败处理。"""
        programmable_provider.queue('{"reasoning": "只给了理由"}')
        result = await detector.screen("umo", phase_info, MoodState(), [], "你好")
        assert result["need_intervention"] is False
        assert result["failed"] is True

    @pytest.mark.asyncio
    async def test_screen_strict_false_passes(
        self, detector, programmable_provider, phase_info,
    ):
        programmable_provider.queue('{"need_intervention": false, "reasoning": "日常闲聊"}')
        result = await detector.screen("umo", phase_info, MoodState(), [], "你好")
        assert result["need_intervention"] is False
        assert result.get("failed") is not True

    @pytest.mark.asyncio
    async def test_screen_non_string_response_is_failure(
        self, detector, programmable_provider, phase_info,
    ):
        """畸形 Provider 返回非字符串 completion_text：按失败处理，不得抛异常。"""
        programmable_provider.queue({"need_intervention": True})  # dict 而非 str
        result = await detector.screen("umo", phase_info, MoodState(), [], "你好")
        assert result["need_intervention"] is False
        assert result["failed"] is True

    def test_parse_json_non_string_returns_empty(self, detector):
        assert detector._parse_json({"a": 1}) == {}
        assert detector._parse_json(None) == {}
        assert detector._parse_json(["x"]) == {}

    @pytest.mark.asyncio
    async def test_screen_none_response_is_failure(
        self, detector, programmable_provider, phase_info, monkeypatch,
    ):
        """Provider 返回 None 响应：按失败处理，不得 AttributeError 穿透。"""

        async def none_chat(*args, **kwargs):
            return None

        monkeypatch.setattr(detector, "_text_chat", none_chat)
        result = await detector.screen("umo", phase_info, MoodState(), [], "你好")
        assert result["need_intervention"] is False
        assert result["failed"] is True

    @pytest.mark.asyncio
    async def test_consult_none_response_returns_empty(
        self, detector, programmable_provider, phase_info, monkeypatch,
    ):
        async def none_chat(*args, **kwargs):
            return None

        monkeypatch.setattr(detector, "_text_chat", none_chat)
        reply = await detector.consult_main_model(
            "umo", phase_info, MoodState(), [], "你好",
            system_prompt="人格", diary_text="", model=None,
        )
        assert reply == ""

    @pytest.mark.asyncio
    async def test_interpret_none_response_invalid(
        self, detector, programmable_provider, monkeypatch,
    ):
        async def none_chat(*args, **kwargs):
            return None

        monkeypatch.setattr(detector, "_text_chat", none_chat)
        d = await detector.interpret("umo", "我很生气", MoodState())
        assert d.valid is False


class TestConsult:
    @pytest.mark.asyncio
    async def test_consult_non_string_response_returns_empty(
        self, detector, programmable_provider, phase_info,
    ):
        """畸形 Provider 返回非字符串：按调用失败返回空，不得抛异常。"""
        programmable_provider.queue({"not": "a string"})
        reply = await detector.consult_main_model(
            "umo", phase_info, MoodState(), [], "你好",
            system_prompt="人格", diary_text="", model=None,
        )
        assert reply == ""

    @pytest.mark.asyncio
    async def test_consult_uses_main_provider_persona_model_contexts(
        self, detector, programmable_provider, phase_info,
    ):
        programmable_provider.queue("我有点不高兴，想冷他几分钟")
        history_contexts = [
            {"role": "user", "content": "在干嘛"},
            {"role": "assistant", "content": "没干嘛"},
        ]
        reply = await detector.consult_main_model(
            "umo", phase_info, MoodState(summary="平静"), history_contexts,
            "哦", system_prompt="完整人格设定", diary_text="日记内容", model="gpt-x",
        )
        assert reply == "我有点不高兴，想冷他几分钟"

        call = programmable_provider.calls[0]
        assert call["system_prompt"] == "完整人格设定"
        assert call["contexts"] == history_contexts
        assert call["model"] == "gpt-x"
        assert "日记内容" in call["prompt"]
        assert "哦" in call["prompt"]

    @pytest.mark.asyncio
    async def test_consult_failure_returns_empty(
        self, detector, programmable_provider, phase_info,
    ):
        programmable_provider.queue(TimeoutError("slow"))
        reply = await detector.consult_main_model(
            "umo", phase_info, MoodState(), [], "你好",
        )
        assert reply == ""

    @pytest.mark.asyncio
    async def test_consult_prompt_contains_actions_and_state(
        self, detector, programmable_provider, phase_info,
    ):
        programmable_provider.queue("回答")
        state = MoodState(summary="介意", status="active")
        await detector.consult_main_model("umo", phase_info, state, [], "msg")
        prompt = programmable_provider.calls[0]["prompt"]
        assert "介意" in prompt
        assert "cold_violence" in prompt  # 动作描述
        assert "after_expression" in prompt


class TestInterpret:
    @pytest.mark.asyncio
    async def test_interpret_valid_decision(self, detector, programmable_provider):
        programmable_provider.queue("""{
            "mood_update": {
                "status": "active", "summary": "被敷衍得有点介意",
                "cause_category": "dismissive", "latest_reason": "回应很轻",
                "improved": false, "fully_recovered": false, "recovery_reason": ""
            },
            "actions": [{"name": "perfunctory_reply", "params": {"level": 2}}],
            "lift_actions": [],
            "silence_mode": "none",
            "reasoning_summary": "决定冷淡回应"
        }""")
        decision = await detector.interpret("umo", "我很介意，想冷淡点回他", MoodState())
        assert decision.valid is True
        assert decision.mood_update["cause_category"] == "dismissive"
        assert decision.actions[0]["name"] == "perfunctory_reply"
        assert decision.actions[0]["params"]["level"] == 2

    @pytest.mark.asyncio
    async def test_interpret_does_not_receive_private_context(
        self, detector, programmable_provider,
    ):
        """③ 不接触聊天历史、人格或日记：签名上就不接受这些参数。"""
        programmable_provider.queue(
            '{"mood_update": {"status": "stable", "summary": "", '
            '"cause_category": "neutral", "latest_reason": "", '
            '"improved": false, "fully_recovered": false, "recovery_reason": ""}, '
            '"actions": [], "lift_actions": [], "silence_mode": "none", '
            '"reasoning_summary": "正常"}'
        )
        decision = await detector.interpret("umo", "正常回复", MoodState())
        assert decision.valid is True
        call = programmable_provider.calls[0]
        assert "contexts" not in call
        # system 中不应出现人格/日记字样以外的隐私内容
        assert "人格" not in call["system_prompt"] or "人格" in ""

    @pytest.mark.asyncio
    async def test_interpret_empty_reply_invalid(self, detector, programmable_provider):
        decision = await detector.interpret("umo", "   ", MoodState())
        assert decision.valid is False
        assert programmable_provider.calls == []  # 不发起调用

    @pytest.mark.asyncio
    async def test_interpret_bad_json_invalid(self, detector, programmable_provider):
        programmable_provider.queue("这不是JSON")
        decision = await detector.interpret("umo", "随便", MoodState())
        assert decision.valid is False

    @pytest.mark.asyncio
    async def test_interpret_call_failure_invalid(self, detector, programmable_provider):
        programmable_provider.queue(RuntimeError("boom"))
        decision = await detector.interpret("umo", "想冷暴力", MoodState())
        assert decision.valid is False

    @pytest.mark.asyncio
    async def test_interpret_respects_disabled_actions(self, programmable_provider):
        class FakeContext:
            def get_using_provider(self, umo=None):
                return programmable_provider

            def get_provider_by_id(self, pid):
                return None

        detector = MoodDetector(FakeContext(), {"enable_cold_violence": False})
        programmable_provider.queue("""{
            "mood_update": {"status": "active", "summary": "介意", "cause_category": "conflict",
                            "latest_reason": "被顶撞", "improved": false,
                            "fully_recovered": false, "recovery_reason": ""},
            "actions": [{"name": "cold_violence", "params": {"duration": 10}}],
            "lift_actions": [], "silence_mode": "immediate", "reasoning_summary": "想静静"
        }""")
        decision = await detector.interpret("umo", "不想理他了", MoodState())
        assert decision.valid is True
        assert decision.actions_rejected is True
        assert decision.reject_reason == "action_disabled"


class TestProviderCompat:
    @pytest.mark.asyncio
    async def test_unsupported_kwargs_dropped(self, phase_info):
        """旧版 Provider（无 contexts/model 参数）自动降级。"""

        class LegacyProvider(ProgrammableProvider):
            async def text_chat(self, prompt: str = "", system_prompt: str = ""):
                self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
                from astrbot.api.provider import LLMResponse
                return LLMResponse("自然语言回答")

        provider = LegacyProvider()

        class FakeContext:
            def get_using_provider(self, umo=None):
                return provider

            def get_provider_by_id(self, pid):
                return None

        detector = MoodDetector(FakeContext(), {})
        reply = await detector.consult_main_model(
            "umo", phase_info, MoodState(),
            [{"role": "user", "content": "历史"}], "消息",
            system_prompt="人格", model="x",
        )
        assert reply == "自然语言回答"
        call = provider.calls[0]
        assert "contexts" not in call and "model" not in call


class TestFailureLogPrivacy:
    """P1-⑦ 回归：Provider 异常消息可能回显请求体，日志只记异常类型。"""

    @pytest.mark.asyncio
    async def test_consult_failure_log_has_no_request_echo(
        self, detector, programmable_provider, phase_info,
    ):
        from astrbot.api import logger
        secret = "人格设定与用户历史的隐私回显内容"
        programmable_provider.queue(RuntimeError(f"boom: {secret}"))
        logger.warning.reset_mock()
        result = await detector.consult_main_model(
            "umo", phase_info, MoodState(), [], "你好",
        )
        assert result == ""
        assert logger.warning.call_count >= 1
        for call in logger.warning.call_args_list:
            assert secret not in str(call)

    @pytest.mark.asyncio
    async def test_screen_failure_log_has_no_request_echo(
        self, detector, programmable_provider, phase_info,
    ):
        from astrbot.api import logger
        secret = "筛选请求体的隐私回显内容"
        programmable_provider.queue(RuntimeError(f"boom: {secret}"))
        logger.warning.reset_mock()
        result = await detector.screen("umo", phase_info, MoodState(), [], "你好")
        assert result["failed"] is True
        for call in logger.warning.call_args_list:
            assert secret not in str(call)


class TestConsultTemplateError:
    """P1 回归：自定义②模板语法错误按调用失败处理。

    format_map 的 ValueError 发生在模型调用之前——若穿透请求钩子，
    已有硬动作将失去保守沉默路径（AstrBot 吞异常后继续正式请求）。
    """

    @pytest.mark.asyncio
    async def test_malformed_custom_template_returns_empty(
        self, programmable_provider, phase_info,
    ):
        class FakeContext:
            def get_using_provider(self, umo=None):
                return programmable_provider

            def get_provider_by_id(self, pid):
                return None

        detector = MoodDetector(
            FakeContext(), {"mood_detector_consult_prompt": "当前状态 {oops"},
        )
        reply = await detector.consult_main_model(
            "umo", phase_info, MoodState(), [], "你好",
        )
        assert reply == ""
        assert programmable_provider.calls == []  # 未发起模型调用


class TestCompletionTextHardening:
    """P2 加固：畸形 Provider 对象的属性 getter 抛异常也不得穿透。"""

    def test_raising_property_returns_empty(self):
        class EvilResp:
            @property
            def completion_text(self):
                raise RuntimeError("boom")

        assert MoodDetector._completion_text(EvilResp()) == ""

    def test_none_and_non_string(self):
        assert MoodDetector._completion_text(None) == ""
        assert MoodDetector._completion_text(object()) == ""
