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
        self.system_prompt = ""
        self.extra_user_content_parts = []

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

msg_mod.TextPart = TextPart

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
