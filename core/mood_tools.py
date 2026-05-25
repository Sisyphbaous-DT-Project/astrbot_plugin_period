"""Mood tool definitions, parameter validation, and prompt injection."""

from __future__ import annotations

from typing import Any

from astrbot.api import logger
from .mood import MoodState


class MoodToolExecutor:
    """Defines available mood tools, validates params, and executes them."""

    TOOLS: dict[str, dict[str, Any]] = {
        "cold_violence": {
            "description": "冷暴力：截断消息，N分钟内不回复",
            "params": {
                "duration": {"type": "int", "min": 1, "max": 1440, "default": 30}
            },
            "intercept": True,
        },
        "read_no_reply": {
            "description": "已读不回：假装看到但不回，持续N轮",
            "params": {
                "rounds": {"type": "int", "min": 1, "max": 10, "default": 3}
            },
            "intercept": True,
        },
        "perfunctory_reply": {
            "description": "敷衍回复：允许回但语气冷淡、简短",
            "params": {
                "level": {"type": "int", "min": 1, "max": 3, "default": 1}
            },
            "intercept": False,
        },
        "seek_comfort": {
            "description": "求安慰：向用户撒娇/索求关怀",
            "params": {
                "type": {
                    "type": "str",
                    "options": ["emotional", "physical", "attention"],
                    "default": "emotional",
                }
            },
            "intercept": False,
        },
        "delayed_reply": {
            "description": "延迟回复风格：模拟\"刚看到消息\"或\"刚才在忙\"的回复风格，并非真正延迟发送",
            "params": {
                "minutes": {"type": "int", "min": 1, "max": 60, "default": 5}
            },
            "intercept": False,
        },
        "emotional_outburst": {
            "description": "情绪爆发：突然情绪化回复",
            "params": {
                "type": {
                    "type": "str",
                    "options": ["angry", "sad", "playful"],
                    "default": "angry",
                }
            },
            "intercept": False,
        },
        "topic_shift": {
            "description": "转移话题：忽略用户问题，聊自己感兴趣的",
            "params": {},
            "intercept": False,
        },
    }

    # ------------------------------------------------------------------ #
    #  Param validation
    # ------------------------------------------------------------------ #

    @classmethod
    def validate_params(cls, tool_name: str, params: dict | None) -> dict:
        """Validate and clamp tool parameters, filling defaults."""
        params = dict(params) if params else {}
        spec = cls.TOOLS.get(tool_name, {})
        result: dict[str, Any] = {}

        for key, rule in spec.get("params", {}).items():
            val = params.get(key)
            if val is None:
                val = rule["default"]

            if rule["type"] == "int":
                try:
                    val = int(val)
                except (TypeError, ValueError):
                    val = rule["default"]
                val = max(rule["min"], min(rule["max"], val))

            elif rule["type"] == "str":
                val = str(val) if val else rule["default"]
                if "options" in rule and val not in rule["options"]:
                    val = rule["default"]

            result[key] = val

        return result

    # ------------------------------------------------------------------ #
    #  Execution
    # ------------------------------------------------------------------ #

    @classmethod
    def execute(cls, tool_name: str, params: dict, mood_state: MoodState) -> None:
        """Apply a tool to the mood state (adds to active_tools)."""
        if tool_name not in cls.TOOLS:
            return

        # Remove existing instance of same tool to avoid stacking
        mood_state.active_tools = [
            t for t in mood_state.active_tools if t["name"] != tool_name
        ]

        from datetime import datetime, timedelta

        tool_entry: dict[str, Any] = {"name": tool_name, "params": params}

        if tool_name == "cold_violence":
            duration = params.get("duration", 30)
            expires = datetime.now() + timedelta(minutes=duration)
            tool_entry["expires_at"] = expires.isoformat()
            tool_entry["initiated"] = False
            logger.info(f"[MoodTool] 激活冷暴力工具，持续时间={duration}分钟")

        elif tool_name == "read_no_reply":
            rounds = params.get("rounds", 3)
            tool_entry["rounds_left"] = rounds
            logger.info(f"[MoodTool] 激活已读不回工具，轮数={rounds}")

        elif tool_name in (
            "perfunctory_reply",
            "seek_comfort",
            "delayed_reply",
            "emotional_outburst",
            "topic_shift",
        ):
            # These are single-use prompt injections; no expiry needed
            logger.info(f"[MoodTool] 激活注入型工具: {tool_name}")

        mood_state.active_tools.append(tool_entry)

    # ------------------------------------------------------------------ #
    #  Prompt injection
    # ------------------------------------------------------------------ #

    @classmethod
    def get_prompt_injection(cls, tool_name: str, params: dict) -> str:
        """Return the prompt injection string for non-intercept tools."""
        if tool_name == "perfunctory_reply":
            level = params.get("level", 1)
            intensity = {1: "简短冷淡", 2: "明显敷衍", 3: "非常敷衍，爱答不理"}
            return (
                f"[情绪指令] 你现在没什么心情，回复要{intensity.get(level, '冷淡')}，"
                f"不要热情，字数尽量控制在30字以内。"
            )

        if tool_name == "seek_comfort":
            ctype = params.get("type", "emotional")
            mapping = {
                "emotional": "心里有点难受，用撒娇的方式向用户索取情感安慰",
                "physical": "身体不太舒服，暗示用户关心一下你",
                "attention": "觉得被忽略了，用委屈的方式吸引用户注意",
            }
            return f"[情绪指令] 你现在{mapping.get(ctype, '心情低落')}，但不要直接说出原因。"

        if tool_name == "delayed_reply":
            return (
                '[情绪指令] 你刚刚才看到这条消息（或者刚忙完），'
                '回复时带一点"姗姗来迟"的感觉，不要太急切。'
            )

        if tool_name == "emotional_outburst":
            etype = params.get("type", "angry")
            mapping = {
                "angry": "突然很生气，语气强烈地回应用户，带一点质问的感觉",
                "sad": "突然很委屈，语气带着哭腔或低落，让用户感到内疚",
                "playful": '突然撒娇式地"发作"，假装生气其实想引起关注',
            }
            return f"[情绪指令] 你现在{mapping.get(etype, '情绪失控')}，让情绪自然流露在回复中。"

        if tool_name == "topic_shift":
            return (
                "[情绪指令] 你对用户当前的话题完全不感兴趣，"
                "巧妙地转移话题到自己关心或想聊的事情上，不要生硬。"
            )

        return ""

    # ------------------------------------------------------------------ #
    #  Intercept helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def get_initial_message(cls, behavior: str, emotion: str) -> str:
        """Return the initial message sent before entering silent interception."""
        if behavior == "angry_then_silent":
            angry = {
                "happy": "我现在不想说话。",
                "calm": "……让我静一静。",
                "irritable": "烦不烦啊，我现在不想理你。",
                "depressed": "别烦我了……",
                "angry": "滚。我现在不想看到你。",
                "playful": "哼！不理你了！",
            }
            return angry.get(emotion, "我现在不想理你。")

        if behavior == "outburst_then_silent":
            outburst = {
                "happy": "好吧好吧，我去忙了！",
                "calm": "我需要一个人待会儿。",
                "irritable": "你能不能别一直发消息？！让我安静一下不行吗！",
                "depressed": "……你为什么总是这样对我……（已读不回）",
                "angry": "够了！我不想再看到你的名字出现在屏幕上！",
                "playful": "啊啊啊你好烦！我要消失一会儿！",
            }
            return outburst.get(emotion, "我现在情绪不太好，别烦我。")

        return ""
