"""Mood tool definitions and execution."""

from __future__ import annotations

from typing import Any

from astrbot.api import logger


class MoodToolExecutor:
    """Defines mood tools, validates parameters, and generates prompt injections."""

    TOOLS: dict[str, dict] = {
        "cold_violence": {
            "description": "冷暴力：N分钟内不回复用户。期间每条消息都会询问主模型是否回复。",
            "params": {
                "duration": {"type": "int", "min": 1, "max": 1440, "default": 30}
            },
            "intercept": True,
        },
        "read_no_reply": {
            "description": "已读不回：假装看到但不回，持续N轮。每轮都询问主模型是否回复。",
            "params": {
                "rounds": {"type": "int", "min": 1, "max": 10, "default": 3}
            },
            "intercept": True,
        },
        "perfunctory_reply": {
            "description": "敷衍回复：允许回但语气冷淡、简短、没有感情。",
            "params": {
                "level": {"type": "int", "min": 1, "max": 3, "default": 1}
            },
            "intercept": False,
        },
        "seek_comfort": {
            "description": "求安慰：向用户撒娇/索求关怀。",
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
            "description": "延迟回复风格：回复带有一点姗姗来迟的感觉。",
            "params": {
                "minutes": {"type": "int", "min": 1, "max": 60, "default": 5}
            },
            "intercept": False,
        },
        "emotional_outburst": {
            "description": "情绪爆发：突然情绪化地回应用户。",
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
            "description": "转移话题：巧妙忽略用户的问题，把话题转移到自己感兴趣的。",
            "params": {},
            "intercept": False,
        },
    }

    # ------------------------------------------------------------------ #
    #  Parameter validation
    # ------------------------------------------------------------------ #

    @classmethod
    def validate_params(cls, tool_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
        """Validate and clamp tool parameters."""
        definition = cls.TOOLS.get(tool_name)
        if not definition:
            return {}

        param_specs = definition.get("params", {})
        result: dict[str, Any] = {}
        raw = params or {}

        for key, spec in param_specs.items():
            val = raw.get(key, spec.get("default"))
            ptype = spec.get("type", "str")

            if ptype == "int":
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = spec.get("default", 0)
                val = max(spec.get("min", val), min(spec.get("max", val), val))
            elif ptype == "str":
                val = str(val)
                if "options" in spec and val not in spec["options"]:
                    val = spec.get("default", "")

            result[key] = val

        return result

    # ------------------------------------------------------------------ #
    #  Prompt injections for non-intercepting tools
    # ------------------------------------------------------------------ #

    @classmethod
    def get_prompt_injection(cls, tool_name: str, params: dict[str, Any]) -> str:
        """Generate a prompt injection for the given tool."""
        injections = {
            "perfunctory_reply": cls._perfunctory_injection,
            "seek_comfort": cls._seek_comfort_injection,
            "delayed_reply": cls._delayed_reply_injection,
            "emotional_outburst": cls._emotional_outburst_injection,
            "topic_shift": cls._topic_shift_injection,
        }
        fn = injections.get(tool_name)
        if fn:
            return fn(params)
        return ""

    @staticmethod
    def _perfunctory_injection(params: dict[str, Any]) -> str:
        level = params.get("level", 1)
        intensity = ["稍冷淡", "比较冷淡", "非常敷衍"][min(level, 3) - 1]
        return f"[情绪指令] 你现在心情一般，回复要{intensity}，不要热情。"

    @staticmethod
    def _seek_comfort_injection(params: dict[str, Any]) -> str:
        ctype = params.get("type", "emotional")
        labels = {
            "emotional": "情感安慰",
            "physical": "身体关怀",
            "attention": "需要关注",
        }
        return f"[情绪指令] 你现在有点低落，用撒娇的方式向用户索取{labels.get(ctype, '关怀')}。"

    @staticmethod
    def _delayed_reply_injection(_params: dict[str, Any]) -> str:
        return "[情绪指令] 你刚忙完/刚看到消息，回复时要带点\"姗姗来迟\"的感觉。"

    @staticmethod
    def _emotional_outburst_injection(params: dict[str, Any]) -> str:
        etype = params.get("type", "angry")
        labels = {"angry": "生气发火", "sad": "委屈哭诉", "playful": "撒娇式发作"}
        return f"[情绪指令] 你现在情绪突然失控，{labels.get(etype, '情绪化')}地回应用户，语气要强烈。"

    @staticmethod
    def _topic_shift_injection(_params: dict[str, Any]) -> str:
        return "[情绪指令] 你对用户的话题不感兴趣，巧妙地把话题转移到自己关心的事情上。"

    # ------------------------------------------------------------------ #
    #  Cold violence initial messages
    # ------------------------------------------------------------------ #

    @classmethod
    def get_initial_message(cls, behavior: str, _emotion_desc: str = "") -> str | None:
        """Generate the initial message when cold violence starts.

        Args:
            behavior: "silent" | "angry_then_silent" | "outburst_then_silent"
            _emotion_desc: Natural language emotion description (unused, reserved)
        """
        if behavior == "silent":
            return None
        if behavior == "angry_then_silent":
            return "我现在不想说话。"
        if behavior == "outburst_then_silent":
            return "……算了，别理我。"
        return None
