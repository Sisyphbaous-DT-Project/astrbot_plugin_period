"""Tests for cross-user diary lookup (period_diary_lookup) — phase 8.

必须证明的不变量（方案 §七）：
- 实际检索结果可供下一轮模型读取（进入 run_context.messages）；
- 顺序合法（追加为 user 消息）；
- 不落会话历史（消息级 + part 级 _no_save，模拟 internal.py 保存过滤）；
- 工具自身只返回通用确认文本；
- 命名空间隔离、昵称歧义、长度截断、无发现接口。
"""

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import pytest

from astrbot_plugin_period.main import PeriodPlugin
from astrbot_plugin_period.core.mood_lookup import (
    build_diary_lookup_tool,
    lookup_diary,
    render_diary_for_lookup,
)

from conftest import MockConversationManager, ProgrammableProvider

UMO = "aiocqhttp:group:1001_12345"
PLATFORM = "qq_1"
BOT = "10000"
TARGET_KEY = f"{PLATFORM}:{BOT}:67890"
OTHER_BOT_KEY = f"{PLATFORM}:20000:67890"


async def _seed(store):
    await store.upsert_diary(
        TARGET_KEY,
        [{"id": "e1", "event_id": "x", "occurred_at": "t", "text": "他上次忘记纪念日，我介意了好久。"}],
        display_name="小芳",
        aliases=["小芳"],
    )
    await store.upsert_diary(
        OTHER_BOT_KEY,
        [{"id": "e2", "event_id": "y", "occurred_at": "t", "text": "另一个机器人的日记。"}],
        display_name="小芳",  # 故意同名：跨机器人也不应可见
        aliases=["小芳"],
    )


def _wrapper():
    """模拟 AstrBot ContextWrapper：messages 由 runner 维护并发送给模型。"""
    return SimpleNamespace(messages=[], context=SimpleNamespace(event=None))


class TestLookupLogic:
    @pytest.mark.asyncio
    async def test_exact_user_id(self, temp_data_dir):
        from core.mood_journal import DiaryStore
        store = DiaryStore(temp_data_dir)
        await _seed(store)
        diary, err = await lookup_diary(store, PLATFORM, BOT, target_user_id="67890")
        assert err is None and diary["entries"][0]["text"].startswith("他上次")

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, temp_data_dir):
        from core.mood_journal import DiaryStore
        store = DiaryStore(temp_data_dir)
        await _seed(store)
        # 另一个机器人账号的命名空间里看不到
        diary, err = await lookup_diary(store, PLATFORM, "30000", target_user_id="67890")
        assert diary is None and err is not None

    @pytest.mark.asyncio
    async def test_nickname_unique_match(self, temp_data_dir):
        from core.mood_journal import DiaryStore
        store = DiaryStore(temp_data_dir)
        await _seed(store)
        diary, err = await lookup_diary(store, PLATFORM, BOT, nickname="小芳")
        assert err is None and diary is not None

    @pytest.mark.asyncio
    async def test_nickname_ambiguous_within_namespace(self, temp_data_dir):
        from core.mood_journal import DiaryStore
        store = DiaryStore(temp_data_dir)
        await store.upsert_diary(f"{PLATFORM}:{BOT}:11111", [
            {"id": "a", "event_id": "e", "occurred_at": "t", "text": "甲"},
        ], display_name="重名", aliases=["重名"])
        await store.upsert_diary(f"{PLATFORM}:{BOT}:22222", [
            {"id": "b", "event_id": "e", "occurred_at": "t", "text": "乙"},
        ], display_name="重名", aliases=["重名"])
        diary, err = await lookup_diary(store, PLATFORM, BOT, nickname="重名")
        assert diary is None
        assert "QQ 号" in err  # 要求改用 QQ 号，且不提供用户列表

    @pytest.mark.asyncio
    async def test_nickname_zero_match(self, temp_data_dir):
        from core.mood_journal import DiaryStore
        store = DiaryStore(temp_data_dir)
        diary, err = await lookup_diary(store, PLATFORM, BOT, nickname="不存在")
        assert diary is None and "QQ 号" in err

    @pytest.mark.asyncio
    async def test_no_args_returns_error(self, temp_data_dir):
        from core.mood_journal import DiaryStore
        store = DiaryStore(temp_data_dir)
        diary, err = await lookup_diary(store, PLATFORM, BOT)
        assert diary is None and err is not None

    def test_render_truncation(self):
        diary = {"display_name": "小芳", "entries": [
            {"text": "长" * 1000},
        ]}
        text = render_diary_for_lookup(diary, 100)
        assert len(text) <= 120  # 截断 + 截断标记
        assert "截断" in text


class TestToolCallInvariant:
    """高保真验证：临时注入可读、顺序合法、不落历史。"""

    @pytest.mark.asyncio
    async def test_call_injects_temp_user_message(self, temp_data_dir):
        from core.mood_journal import DiaryStore
        store = DiaryStore(temp_data_dir)
        await _seed(store)
        tool = build_diary_lookup_tool(store, PLATFORM, BOT, 800)
        wrapper = _wrapper()

        result = await tool.call(wrapper, target_user_id="67890")

        # 工具自身只返回通用确认文本，不含日记内容
        assert "介意" not in result
        assert "已检索到" in result

        # 实际内容进入运行上下文（下一轮模型可读），且为合法 user 消息
        assert len(wrapper.messages) == 1
        msg = wrapper.messages[0]
        assert msg.role == "user"
        assert "介意" in msg.content[0].text

        # 不落历史：消息级 + part 级双标记
        assert msg._no_save is True
        assert msg.content[0]._no_save is True

        # 模拟 internal.py:470 的保存过滤：该消息必须被排除
        persisted = [
            m for m in wrapper.messages
            if not (m.role in ("assistant", "user") and m._no_save)
        ]
        assert persisted == []

    @pytest.mark.asyncio
    async def test_error_path_injects_nothing(self, temp_data_dir):
        from core.mood_journal import DiaryStore
        store = DiaryStore(temp_data_dir)
        await _seed(store)
        tool = build_diary_lookup_tool(store, PLATFORM, BOT, 800)
        wrapper = _wrapper()
        result = await tool.call(wrapper, target_user_id="99999")
        assert "没有找到" in result
        assert wrapper.messages == []  # 错误路径不注入任何内容


class TestPipelineInjection:
    @pytest.fixture
    def pipeline(self, tmp_path, sample_config, monkeypatch):
        from astrbot.api.star import Context, StarTools

        config = deepcopy(sample_config)
        config.update({
            "default_anchor_date": "2024-01-15",
            "default_enabled": True,
            "global_inject": True,
            "mood_system_enabled": True,
            "diary_enabled": True,
        })
        ctx = Context()
        provider = ProgrammableProvider()
        ctx.get_using_provider = lambda umo=None: provider
        ctx.get_provider_by_id = lambda pid: None
        conv_mgr = MockConversationManager()
        conv_mgr.seed(UMO, [])
        ctx.conversation_manager = conv_mgr
        monkeypatch.setattr(StarTools, "get_data_dir", lambda _name=None: tmp_path)
        return PeriodPlugin(ctx, config), provider

    def _make_req(self):
        from astrbot.api.provider import ProviderRequest
        from types import SimpleNamespace
        req = ProviderRequest()
        req.prompt = "用户消息"
        req.system_prompt = "人格"
        req.contexts = []
        req.conversation = SimpleNamespace(cid="cid-1", history=None)
        return req

    @pytest.mark.asyncio
    async def test_tool_present_when_enabled(self, pipeline, event_factory):
        plugin, provider = pipeline
        plugin.config["diary_cross_user_lookup_enabled"] = True
        provider.queue('{"need_intervention": false}')
        req = self._make_req()
        await plugin.on_llm_request(event_factory(umo=UMO), req)
        assert req.func_tool is not None
        assert req.func_tool.get_tool("period_diary_lookup") is not None

    @pytest.mark.asyncio
    async def test_tool_absent_when_disabled(self, pipeline, event_factory):
        plugin, provider = pipeline
        plugin.config["diary_cross_user_lookup_enabled"] = False
        provider.queue('{"need_intervention": false}')
        req = self._make_req()
        await plugin.on_llm_request(event_factory(umo=UMO), req)
        assert req.func_tool is None or req.func_tool.get_tool("period_diary_lookup") is None

    @pytest.mark.asyncio
    async def test_tool_absent_for_third_party_runner(self, pipeline, event_factory):
        plugin, provider = pipeline
        plugin.config["diary_cross_user_lookup_enabled"] = True
        del plugin.context.conversation_manager
        provider.queue('{"need_intervention": false}')
        req = self._make_req()
        req.conversation = None
        await plugin.on_llm_request(event_factory(umo=UMO), req)
        assert req.func_tool is None or req.func_tool.get_tool("period_diary_lookup") is None

    @pytest.mark.asyncio
    async def test_existing_tools_preserved(self, pipeline, event_factory):
        plugin, provider = pipeline
        plugin.config["diary_cross_user_lookup_enabled"] = True
        provider.queue('{"need_intervention": false}')
        req = self._make_req()
        from astrbot.core.agent.tool import FunctionTool, ToolSet
        req.func_tool = ToolSet([FunctionTool(name="other_tool", description="x")])
        await plugin.on_llm_request(event_factory(umo=UMO), req)
        assert req.func_tool.get_tool("other_tool") is not None
        assert req.func_tool.get_tool("period_diary_lookup") is not None


class TestRenderTruncation:
    """P2 回归：截断后的总长度（含后缀）不得超过配置上限。"""

    def test_truncated_text_within_limit(self):
        from astrbot_plugin_period.core.mood_lookup import render_diary_for_lookup
        diary = {"display_name": "小明", "entries": [{"text": "长" * 200}]}
        text = render_diary_for_lookup(diary, 50)
        assert len(text) <= 50
        assert "截断" in text

    def test_short_text_untouched(self):
        from astrbot_plugin_period.core.mood_lookup import render_diary_for_lookup
        diary = {"display_name": "小明", "entries": [{"text": "短"}]}
        text = render_diary_for_lookup(diary, 800)
        assert "截断" not in text
