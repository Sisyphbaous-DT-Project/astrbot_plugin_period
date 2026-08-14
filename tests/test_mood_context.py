"""Tests for core/mood_context.py — real-history parser, gates, safe injector."""

import pytest

from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

from core.mood_context import (
    apply_injection,
    extract_text_content,
    history_to_contexts,
    is_umo_allowed,
    normalize_inject_location,
    parse_history,
    should_show_body_hint,
)
from conftest import make_realistic_history


class TestParseHistory:
    def test_realistic_full_shape(self):
        history = parse_history(make_realistic_history(), limit=30)
        contents = [h["content"] for h in history]
        # system/tool/_checkpoint 已跳过
        assert "你是某个人格" not in contents
        assert "工具结果" not in contents
        # 字符串形态保留
        assert "第一条用户消息" in contents
        # 多段只拼 text，think/图片忽略
        assert "多段用户消息" in contents
        assert not any("内部思考" in c for c in contents)
        # 多段 assistant 拼接
        assert "多段\n助手回复" in contents
        # _no_save 消息跳过
        assert not any("临时情绪指令" in c for c in contents)
        # 角色正确
        roles = {h["role"] for h in history}
        assert roles <= {"user", "assistant"}

    def test_limit_trims_from_tail(self):
        history = parse_history(make_realistic_history(), limit=2)
        assert len(history) == 2
        assert history[-1]["content"] == "最近一条助手回复"
        assert history[-2]["content"] == "最近一条用户消息"

    def test_limit_zero_returns_empty(self):
        assert parse_history(make_realistic_history(), limit=0) == []

    def test_limit_100_keeps_all_valid(self):
        history = parse_history(make_realistic_history(), limit=100)
        # 11 条目中有效 user/assistant 文本共 6 条
        assert len(history) == 6

    def test_message_object_shape(self):
        class FakeMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        class FakeTextPart:
            def __init__(self, text):
                self.text = text
                self._no_save = False

        FakeTextPart.__name__ = "TextPart"
        contexts = [
            FakeMessage("user", "对象形态用户消息"),
            FakeMessage("assistant", [FakeTextPart("对象形态助手回复")]),
        ]
        # TextPart 判定基于类名；这里类名伪造为 TextPart
        history = parse_history(contexts, limit=10)
        assert history[0]["content"] == "对象形态用户消息"
        assert history[1]["content"] == "对象形态助手回复"

    def test_empty_and_garbage(self):
        assert parse_history(None, 30) == []
        assert parse_history([], 30) == []
        assert parse_history([{"foo": "bar"}], 30) == []

    def test_extract_text_content_shapes(self):
        assert extract_text_content("纯文本") == "纯文本"
        assert extract_text_content([{"type": "text", "text": "甲"},
                                     {"type": "think", "think": "乙"},
                                     {"type": "text", "text": "丙"}]) == "甲\n丙"
        assert extract_text_content(None) == ""
        assert extract_text_content(123) == ""

    def test_part_level_no_save_skipped(self):
        contexts = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "可见部分"},
                {"type": "text", "text": "临时部分", "_no_save": True},
            ],
        }]
        history = parse_history(contexts, 10)
        assert history[0]["content"] == "可见部分"

    def test_history_to_contexts_round_trip(self):
        history = [{"role": "user", "content": "甲"}, {"role": "assistant", "content": "乙"}]
        assert history_to_contexts(history) == history


class TestGates:
    def test_umo_whitelist(self):
        cfg = {"global_inject": False, "umo_list": ["a:1"], "umo_mode": "whitelist"}
        assert is_umo_allowed(cfg, "a:1") is True
        assert is_umo_allowed(cfg, "b:2") is False

    def test_umo_blacklist(self):
        cfg = {"global_inject": False, "umo_list": ["a:1"], "umo_mode": "blacklist"}
        assert is_umo_allowed(cfg, "a:1") is False
        assert is_umo_allowed(cfg, "b:2") is True

    def test_umo_global(self):
        cfg = {"global_inject": True}
        assert is_umo_allowed(cfg, "any:thing") is True

    def test_warmup_blocks_first_rounds(self):
        cfg = {"warmup_rounds": 2, "inject_mode": "every_request"}
        wc, ic = {}, {}
        assert should_show_body_hint(cfg, "u", "你好", wc, ic) is False
        assert should_show_body_hint(cfg, "u", "你好", wc, ic) is False
        assert should_show_body_hint(cfg, "u", "你好", wc, ic) is True

    def test_interval_3(self):
        cfg = {"warmup_rounds": 0, "inject_mode": "interval_3"}
        wc, ic = {}, {}
        results = [should_show_body_hint(cfg, "u", "x", wc, ic) for _ in range(4)]
        assert results == [True, False, False, True]

    def test_only_status_never_shows(self):
        cfg = {"warmup_rounds": 0, "inject_mode": "only_status"}
        assert should_show_body_hint(cfg, "u", "怎么了", {}, {}) is False

    def test_on_trigger_keyword(self):
        cfg = {"warmup_rounds": 0, "inject_mode": "on_trigger",
               "trigger_keywords": ["怎么了"]}
        assert should_show_body_hint(cfg, "u", "你怎么了", {}, {}) is True
        assert should_show_body_hint(cfg, "u", "吃饭了吗", {}, {}) is False


class TestInjectLocation:
    def test_normalize_safe_locations(self):
        for loc in ("extra_user_content_parts", "system_prompt_append"):
            assert normalize_inject_location(loc) == (loc, False)

    def test_fake_tool_call_downgraded(self):
        loc, downgraded = normalize_inject_location("fake_tool_call")
        assert loc == "extra_user_content_parts"
        assert downgraded is True

    def test_user_message_before_downgraded(self):
        """user_message_before 会写入聊天历史，已废弃并降级。"""
        loc, downgraded = normalize_inject_location("user_message_before")
        assert loc == "extra_user_content_parts"
        assert downgraded is True

    def test_unknown_downgraded(self):
        loc, downgraded = normalize_inject_location("weird")
        assert loc == "extra_user_content_parts"
        assert downgraded is True

    def test_apply_extra_parts_is_temp(self):
        req = ProviderRequest()
        apply_injection(req, "状态文本", "extra_user_content_parts")
        assert len(req.extra_user_content_parts) == 1
        part = req.extra_user_content_parts[0]
        assert isinstance(part, TextPart)
        assert part.text == "状态文本"
        assert part._no_save is True

    def test_apply_system_prompt_append(self):
        req = ProviderRequest()
        req.system_prompt = "人格"
        apply_injection(req, "状态文本", "system_prompt_append")
        assert req.system_prompt == "人格\n\n状态文本"

    def test_apply_empty_text_noop(self):
        req = ProviderRequest()
        apply_injection(req, "", "extra_user_content_parts")
        assert req.extra_user_content_parts == []
