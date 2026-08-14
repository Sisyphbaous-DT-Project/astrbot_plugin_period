"""Pytest fixtures and mocks for astrbot_plugin_period tests."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import pytest

# Ensure project root is on sys.path so absolute imports like `from core.engine ...` work
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --------------------------------------------------------------------------- #
#  Mock AstrBot modules before any plugin code imports them
# --------------------------------------------------------------------------- #

def _make_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod

# Build mock module tree
astrbot = _make_module("astrbot")
astrbot.api = _make_module("astrbot.api")
astrbot.api.event = _make_module("astrbot.api.event")
astrbot.api.event.filter = _make_module("astrbot.api.event.filter")
astrbot.api.star = _make_module("astrbot.api.star")
astrbot.api.provider = _make_module("astrbot.api.provider")
astrbot.core = _make_module("astrbot.core")
astrbot.core.agent = _make_module("astrbot.core.agent")
astrbot.core.agent.message = _make_module("astrbot.core.agent.message")
astrbot.core.message = _make_module("astrbot.core.message")
astrbot.core.message.components = _make_module("astrbot.core.message.components")
astrbot.core.star = _make_module("astrbot.core.star")
astrbot.core.star.register = _make_module("astrbot.core.star.register")
astrbot.core.star.filter = _make_module("astrbot.core.star.filter")

# --- astrbot.api.message_components ---
msg_comp_mod = _make_module("astrbot.api.message_components")

class Plain:
    def __init__(self, text=""):
        self.text = text

msg_comp_mod.Plain = Plain

# --- astrbot.api.event MessageChain ---
class MessageChain:
    def __init__(self, chain=None):
        self.chain = chain or []

sys.modules["astrbot.api.event"].MessageChain = MessageChain

# --- quart (for web api tests) ---
quart_mod = _make_module("quart")

class _MockRequest:
    _json = {}
    @classmethod
    async def get_json(cls):
        return cls._json
    @classmethod
    def set_json(cls, data):
        cls._json = data

class _MockJsonify:
    def __init__(self, data):
        self.data = data
    def __call__(self, *args, **kwargs):
        return self.data
    def get_json(self):
        return self.data

def mock_jsonify(data, *args, **kwargs):
    return data

quart_mod.request = _MockRequest()
quart_mod.jsonify = mock_jsonify
sys.modules["quart"] = quart_mod

# --- astrbot.api.event.filter ---
filter_mod = sys.modules["astrbot.api.event.filter"]

class _MockFilter:
    @staticmethod
    def command(name: str | None = None, **kwargs):
        def decorator(func):
            func._cmd_name = name
            return func
        return decorator

    @staticmethod
    def command_group(name: str | None = None, **kwargs):
        def decorator(func):
            group = MagicMock()
            group.command = _MockFilter.command
            group._group_name = name
            func._group = group
            return group
        return decorator

    @staticmethod
    def on_llm_request(**kwargs):
        def decorator(func):
            return func
        return decorator

    @staticmethod
    def on_llm_response(**kwargs):
        def decorator(func):
            return func
        return decorator

    @staticmethod
    def permission_type(perm):
        def decorator(func):
            return func
        return decorator

    @staticmethod
    def on_astrbot_loaded(**kwargs):
        def decorator(func):
            return func
        return decorator

class PermissionType:
    ADMIN = 1
    MEMBER = 2

filter_mod.filter = _MockFilter()
filter_mod.PermissionType = PermissionType
filter_mod.permission_type = _MockFilter.permission_type

# --- astrbot.api.event ---
event_mod = sys.modules["astrbot.api.event"]
event_mod.filter = filter_mod.filter
event_mod.AstrMessageEvent = MagicMock
event_mod.MessageEventResult = MagicMock

# --- astrbot.api.star ---
star_mod = sys.modules["astrbot.api.star"]

class Star:
    def __init__(self, context):
        self.context = context

class Context:
    def __init__(self):
        self.registered_web_apis = []

    def register_web_api(self, route, view_handler, methods, desc):
        self.registered_web_apis.append((route, view_handler, methods, desc))

    def get_using_provider(self, umo=None):
        return None

class StarTools:
    @classmethod
    def get_data_dir(cls):
        return Path(__file__).parent / "_test_data"

class register:
    def __init__(self, name, author, desc, version, repo):
        self.name = name

    def __call__(self, cls):
        cls._plugin_name = self.name
        cls.name = self.name  # AstrBot injects this before instantiation
        return cls

star_mod.Star = Star
star_mod.Context = Context
star_mod.StarTools = StarTools
star_mod.register = register

# --- astrbot.api.provider ---
provider_mod = sys.modules["astrbot.api.provider"]

class Provider:
    """Mock Chat Completion provider."""
    pass

class ProviderRequest:
    def __init__(self):
        self.prompt = ""
        self.system_prompt = ""
        self.contexts = []
        self.extra_user_content_parts = []
        # 与 AstrBot 4.27.2 entities.ProviderRequest 对齐的其余字段
        self.model = None
        self.conversation = None
        self.session_id = ""
        self.func_tool = None
        self.tool_calls_result = None
        self.image_urls = []
        self.audio_urls = []

class LLMResponse:
    def __init__(self, text=""):
        self._text = text
        self.result_chain = None

    @property
    def completion_text(self):
        if self.result_chain:
            # Match real AstrBot behavior: return plain text from chain
            getter = getattr(self.result_chain, "get_plain_text", None)
            if getter:
                return getter()
            return str(self.result_chain)
        return self._text

    @completion_text.setter
    def completion_text(self, value):
        self._text = value

provider_mod.Provider = Provider
provider_mod.ProviderRequest = ProviderRequest
provider_mod.LLMResponse = LLMResponse

# --- astrbot.api ---
api_mod = sys.modules["astrbot.api"]
api_mod.logger = MagicMock()

# --- astrbot.core.agent.message ---
msg_mod = sys.modules["astrbot.core.agent.message"]

class TextPart:
    def __init__(self, text=""):
        self.text = text
        self._no_save = False

    def mark_as_temp(self):
        self._no_save = True
        return self

class Message:
    """对齐 astrbot.core.agent.message.Message 的最小形态。"""

    def __init__(self, role="", content=None, **kwargs):
        self.role = role
        self.content = content
        self._no_save = False
        for k, v in kwargs.items():
            setattr(self, k, v)

msg_mod.TextPart = TextPart
msg_mod.Message = Message

# --- astrbot.core.agent.tool（FunctionTool/ToolSet，跨人日记检索用）---
tool_mod = _make_module("astrbot.core.agent.tool")

class FunctionTool:
    """对齐 4.27.2 FunctionTool：子类覆盖 call(context, **kwargs) 被调度。"""

    def __init__(self, name="", description="", parameters=None, handler=None,
                 active=True, **kwargs):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.handler = handler
        self.active = active

    async def call(self, context, **kwargs):
        raise NotImplementedError

class ToolSet:
    def __init__(self, tools=None):
        self.tools = list(tools or [])

    def add_tool(self, tool):
        for i, existing in enumerate(self.tools):
            if existing.name == tool.name:
                self.tools[i] = tool
                return
        self.tools.append(tool)

    def remove_tool(self, name):
        self.tools = [t for t in self.tools if t.name != name]

    def get_tool(self, name):
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def empty(self):
        return len(self.tools) == 0

tool_mod.FunctionTool = FunctionTool
tool_mod.ToolSet = ToolSet

# --- astrbot.core.star.register ---
reg_mod = sys.modules["astrbot.core.star.register"]
reg_mod.register_star = register

# --- astrbot.core.star.filter ---
sc_mod = sys.modules["astrbot.core.star.filter"]
sc_mod.PermissionType = PermissionType


# --------------------------------------------------------------------------- #
#  pytest fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def temp_data_dir(tmp_path):
    """Provide a temporary directory for store tests."""
    return tmp_path / "plugin_data"


@pytest.fixture
def sample_config():
    """Minimal valid config dict matching _conf_schema.json defaults."""
    return {
        "anchor_prompt": (
            "[身体感知系统] 你拥有真实的女性身体。"
            "禁止在回复中使用以下词汇：{forbidden_words}。"
        ),
        "phases": {
            "menstrual": {
                "prompt": "下腹坠胀，疲倦。",
                "time_morning": "早晨绞痛。",
                "time_afternoon": "午后犯困。",
                "time_night": "深夜安静。",
            },
            "follicular": {
                "prompt": "精力充沛，心情好。",
                "time_morning": "早晨清爽。",
                "time_afternoon": "午后活跃。",
                "time_night": "深夜不想睡。",
            },
            "ovulatory": {
                "prompt": "身体轻盈，情绪积极。",
                "time_morning": "早晨发热。",
                "time_afternoon": "午后慵懒。",
                "time_night": "深夜敏锐。",
            },
            "luteal": {
                "prompt": "胸闷烦躁，想吃甜食。",
                "time_morning": "早晨多梦。",
                "time_afternoon": "午后血糖波动。",
                "time_night": "深夜委屈。",
            },
        },
        "forbidden_words": ["月经", "激素", "黄体"],
        "max_prompt_length": 120,
        "include_time_modifier": True,
        "include_day_number": True,
        "include_phase_name": False,
        "auto_inject": True,
        "inject_mode": "every_request",
        "trigger_keywords": ["怎么了", "还好吗"],
        "warmup_rounds": 0,
        "ooc_shield": True,
        "ooc_replace": False,
        "debug_mode": False,
        "commands_enabled": "all",
        "cycle_settings": {
            "period_length": 5,
            "ovulation_day": 14,
            "ovulation_window": 3,
        },
    }


@pytest.fixture(autouse=True)
def reset_mock_request():
    """Reset mock request body between tests."""
    yield
    from quart import request
    request.set_json({})


@pytest.fixture
def mock_event():
    """Minimal mock AstrMessageEvent."""
    ev = MagicMock()
    ev.unified_msg_origin = "test_platform:test_guild:test_user"
    ev.message_str = "你好"
    return ev


# --------------------------------------------------------------------------- #
#  vNext: 高保真测试替身（真实历史形态 / Provider / ConversationManager）
# --------------------------------------------------------------------------- #

import json


class ProgrammableProvider(Provider):
    """记录 text_chat 调用次数与入参、按队列返回脚本化响应的 Provider 替身。

    - responses 队列每项可以是 str（作为 completion_text 返回）或 Exception（抛出）。
    - 队列耗尽后返回 default_response。
    - 每次调用的完整 kwargs 追加到 calls，供断言历史/人格/日记是否传入。
    """

    def __init__(self, provider_id: str = "main"):
        self.provider_id = provider_id
        self.provider_config = {"type": "openai_chat_completion", "id": provider_id}
        self.calls: list[dict] = []
        self.responses: list = []
        self.default_response: str = "{}"

    def queue(self, *responses) -> None:
        self.responses.extend(responses)

    async def text_chat(self, prompt: str = "", **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return LLMResponse(item)
        return LLMResponse(self.default_response)


class MockConversation:
    """对齐 astrbot.core.db.po.Conversation 的最小形态：history 为 JSON 字符串。"""

    def __init__(self, cid: str = "conv-1", history: list | None = None, persona_id: str = ""):
        self.cid = cid
        self.history = json.dumps(history or [], ensure_ascii=False)
        self.persona_id = persona_id

    def history_list(self) -> list:
        return json.loads(self.history or "[]")


class MockConversationManager:
    """对齐 conversation_mgr.ConversationManager 的读-改-写语义（整体替换历史）。"""

    def __init__(self):
        self._convos: dict[str, MockConversation] = {}
        self.update_calls: list[dict] = []

    def seed(self, umo: str, history: list, cid: str = "conv-1", persona_id: str = "") -> MockConversation:
        conv = MockConversation(cid=cid, history=history, persona_id=persona_id)
        self._convos[umo] = conv
        return conv

    async def get_curr_conversation_id(self, umo: str):
        conv = self._convos.get(umo)
        return conv.cid if conv else None

    async def get_conversation(self, umo: str, cid=None, create_if_not_exists=False):
        return self._convos.get(umo)

    async def update_conversation(self, unified_msg_origin: str, conversation_id=None, history=None, **kwargs):
        self.update_calls.append({
            "umo": unified_msg_origin,
            "conversation_id": conversation_id,
            "history": history,
        })
        conv = self._convos.get(unified_msg_origin)
        if conv is not None and history is not None:
            conv.history = json.dumps(history, ensure_ascii=False)


def make_realistic_history():
    """覆盖 AstrBot 真实历史全部形态：字符串/多段 content、system/tool/_checkpoint、
    _no_save 临时消息、think/图片 part。"""
    return [
        {"role": "system", "content": "你是某个人格"},
        {"role": "user", "content": "第一条用户消息"},
        {"role": "assistant", "content": "第一条助手回复"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "some_tool", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "工具结果"},
        {"role": "_checkpoint", "content": [], "ckpt": {}},
        {
            "role": "user",
            "content": [
                {"type": "think", "think": "内部思考不应提取"},
                {"type": "text", "text": "多段用户消息"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "多段"},
                {"type": "text", "text": "助手回复"},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "临时情绪指令", "_no_save": True}],
            "_no_save": True,
        },
        {"role": "user", "content": "最近一条用户消息"},
        {"role": "assistant", "content": "最近一条助手回复"},
    ]


@pytest.fixture
def programmable_provider():
    return ProgrammableProvider()


@pytest.fixture
def conversation_manager():
    return MockConversationManager()


@pytest.fixture
def event_factory():
    """构造带完整身份 API 与 stop_event 语义的事件替身。"""
    def _make(
        umo: str = "aiocqhttp:group:1001_12345",
        message_str: str = "你好",
        platform_id: str = "qq_1",
        self_id: str = "10000",
        sender_id: str = "12345",
        sender_name: str = "测试用户",
        role: str = "member",
    ):
        ev = MagicMock()
        ev.unified_msg_origin = umo
        ev.message_str = message_str
        ev.get_platform_id.return_value = platform_id
        ev.get_self_id.return_value = self_id
        ev.get_sender_id.return_value = sender_id
        ev.get_sender_name.return_value = sender_name
        ev.get_role.return_value = role
        ev.plain_result.side_effect = lambda text, **kwargs: text
        ev._stopped_flag = False
        ev.stop_event.side_effect = lambda: setattr(ev, "_stopped_flag", True)
        ev.is_stopped.side_effect = lambda: ev._stopped_flag
        return ev
    return _make
