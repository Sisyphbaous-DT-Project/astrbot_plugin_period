"""Mood tool definitions, parameter clamping and decision legality matrix.

固定台词已彻底移除：所有注入只描述情绪倾向，由主模型按人格自然组织语言。
"""

from __future__ import annotations

from typing import Any

from .mood import (
    HARD_ACTIONS,
    SILENCE_MODES,
    SOFT_ACTIONS,
    RequestMoodDecision,
    clamp_str,
    MAX_REASONING_CHARS,
    STATUS_VALUES,
    CAUSE_CATEGORIES,
)


class MoodToolExecutor:
    """Defines mood actions, validates parameters, and generates tendency prompts."""

    TOOLS: dict[str, dict] = {
        "cold_violence": {
            "description": "冷暴力：一段时间内完全不回应用户。参数：duration（分钟，1-1440）。",
            "params": {
                "duration": {"type": "int", "min": 1, "max": 1440, "default": 30}
            },
            "intercept": True,
        },
        "read_no_reply": {
            "description": "已读不回：假装看到但连续几条消息都不回应（含触发当轮）。参数：rounds（1-10）。",
            "params": {
                "rounds": {"type": "int", "min": 1, "max": 10, "default": 3}
            },
            "intercept": True,
        },
        "perfunctory_reply": {
            "description": "敷衍回复：允许回但语气冷淡、简短、没有感情。参数：level（1-3）。",
            "params": {
                "level": {"type": "int", "min": 1, "max": 3, "default": 1}
            },
            "intercept": False,
        },
        "seek_comfort": {
            "description": "求安慰：向用户撒娇/索求关怀。参数：type（emotional/physical/attention）。",
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
            "description": "延迟回复风格：回复带有一点姗姗来迟的感觉。参数：minutes（1-60）。",
            "params": {
                "minutes": {"type": "int", "min": 1, "max": 60, "default": 5}
            },
            "intercept": False,
        },
        "emotional_outburst": {
            "description": "情绪爆发：突然情绪化地回应用户。参数：type（angry/sad/playful）。",
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
        """Validate and clamp tool parameters（宽松版，供旧调用方使用）。

        类型错误回退默认值；决策校验请用 strict 版本。
        """
        definition = cls.TOOLS.get(tool_name)
        if not definition:
            return {}

        param_specs = definition.get("params", {})
        result: dict[str, Any] = {}
        raw = params if isinstance(params, dict) else {}

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

    @classmethod
    def validate_params_strict(
        cls, tool_name: str, params: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """严格校验动作参数。返回 None 表示参数非法（调用方应整组拒绝）。

        - params 必须是 dict；None 仅表示调用方确认键缺失（按全默认
          处理），显式 null 由调用方区分并先行拒绝；
        - int 字段：缺失用默认值；提供时必须是真正的 int（bool 不算），
          越界按边界钳制；
        - str options 字段：缺失用默认值；提供时必须是指定选项之一；
        - 未知额外字段忽略。
        """
        definition = cls.TOOLS.get(tool_name)
        if not definition:
            return None
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return None

        result: dict[str, Any] = {}
        for key, spec in definition.get("params", {}).items():
            if key not in params:
                result[key] = spec.get("default")
                continue
            val = params[key]
            ptype = spec.get("type", "str")
            if ptype == "int":
                if not isinstance(val, int) or isinstance(val, bool):
                    return None
                result[key] = max(spec.get("min", val), min(spec.get("max", val), val))
            elif ptype == "str":
                if not isinstance(val, str):
                    return None
                if "options" in spec and val not in spec["options"]:
                    return None
                result[key] = val
            else:
                result[key] = val
        return result

    # ------------------------------------------------------------------ #
    #  软动作倾向提示（禁止预设台词，只描述情绪倾向）
    # ------------------------------------------------------------------ #

    @classmethod
    def get_prompt_injection(cls, tool_name: str, params: dict[str, Any]) -> str:
        """生成软动作的倾向提示文本。"""
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
        intensity = ["略显冷淡", "比较冷淡", "非常敷衍"][min(level, 3) - 1]
        return f"- 倾向：这条回复{intensity}、简短，提不起热情；用你自己的人格方式自然表现。"

    @staticmethod
    def _seek_comfort_injection(params: dict[str, Any]) -> str:
        ctype = params.get("type", "emotional")
        labels = {
            "emotional": "情感上的安慰",
            "physical": "身体层面的关心",
            "attention": "被关注和在意",
        }
        return f"- 倾向：你现在有点低落，想用自己的方式向用户索取{labels.get(ctype, '关怀')}。"

    @staticmethod
    def _delayed_reply_injection(_params: dict[str, Any]) -> str:
        return "- 倾向：你像是刚忙完才看到消息，回复自然带点“姗姗来迟”的感觉。"

    @staticmethod
    def _emotional_outburst_injection(params: dict[str, Any]) -> str:
        etype = params.get("type", "angry")
        labels = {"angry": "压不住的火气", "sad": "翻涌的委屈", "playful": "撒娇式的小发作"}
        return f"- 倾向：情绪突然上头，带着{labels.get(etype, '情绪')}回应，语气比平时强烈。"

    @staticmethod
    def _topic_shift_injection(_params: dict[str, Any]) -> str:
        return "- 倾向：你对用户的话题提不起兴趣，会自然地把话题引到自己关心的事情上。"


# --------------------------------------------------------------------------- #
#  决策合法性矩阵（第三阶段 JSON → RequestMoodDecision）
# --------------------------------------------------------------------------- #

def validate_decision(
    raw: Any,
    enabled_actions: set[str] | None = None,
) -> RequestMoodDecision:
    """把第三阶段输出严格校验为 RequestMoodDecision。

    失败分两级：
    - 决策级（valid=False）：整体结构、心境字段或解除字段非法 → 全部作废；
    - 动作组级（actions_rejected=True）：解除仍然有效，新动作整组不执行。

    enabled_actions 为 None 表示全部启用。
    本函数保证不抛异常：任何内部错误一律按决策级非法处理。
    """
    try:
        return _validate_decision_inner(raw, enabled_actions)
    except Exception as e:  # 防御：校验异常不得穿透到请求钩子
        return RequestMoodDecision(
            valid=False, reject_reason=f"validator_error:{type(e).__name__}",
        )


def _validate_decision_inner(
    raw: Any,
    enabled_actions: set[str] | None,
) -> RequestMoodDecision:
    if not isinstance(raw, dict) or not raw:
        return RequestMoodDecision(valid=False, reject_reason="decision_not_dict")

    enabled = set(MoodToolExecutor.TOOLS) if enabled_actions is None else set(enabled_actions)

    # ---- mood_update（固定字段，必填且完整合法）----
    mood_update = raw.get("mood_update")
    if not isinstance(mood_update, dict):
        return RequestMoodDecision(valid=False, reject_reason="mood_update_missing")
    status = mood_update.get("status")
    cause = mood_update.get("cause_category")
    if status not in STATUS_VALUES:
        return RequestMoodDecision(valid=False, reject_reason="mood_update_bad_status")
    if cause not in CAUSE_CATEGORIES:
        return RequestMoodDecision(valid=False, reject_reason="mood_update_bad_cause")
    # 布尔字段必须是真正的 bool（"false" 之类的字符串整份作废）
    for bool_key in ("improved", "fully_recovered"):
        if not isinstance(mood_update.get(bool_key), bool):
            return RequestMoodDecision(
                valid=False, reject_reason=f"mood_update_bad_bool:{bool_key}",
            )
    # 文本字段必须是字符串，缺失或类型错误都视为损坏（禁止强转）
    for text_key in ("summary", "latest_reason", "recovery_reason"):
        if not isinstance(mood_update.get(text_key), str):
            return RequestMoodDecision(
                valid=False, reject_reason=f"mood_update_bad_text:{text_key}",
            )
    improved = mood_update["improved"]
    fully_recovered = mood_update["fully_recovered"]
    # 状态组合一致性：完全恢复必须配 recovered，反之亦然
    if fully_recovered and status != "recovered":
        return RequestMoodDecision(
            valid=False, reject_reason="mood_update_contradiction:recovered",
        )
    if status == "recovered" and not fully_recovered:
        return RequestMoodDecision(
            valid=False, reject_reason="mood_update_contradiction:recovered",
        )
    mood_update = {
        "status": status,
        "summary": clamp_str(mood_update["summary"], 200),
        "cause_category": cause,
        "latest_reason": clamp_str(mood_update["latest_reason"], 200),
        "improved": improved,
        "fully_recovered": fully_recovered,
        "recovery_reason": clamp_str(mood_update["recovery_reason"], 200),
    }

    # ---- 顶层固定字段全量必填（固定输出契约：缺字段即整份作废，不更新）----
    for required_key in ("actions", "lift_actions", "silence_mode", "reasoning_summary"):
        if required_key not in raw:
            return RequestMoodDecision(
                valid=False, reject_reason=f"missing_field:{required_key}",
            )

    # ---- silence_mode ----
    silence_mode = raw.get("silence_mode", "none")
    if silence_mode not in SILENCE_MODES:
        return RequestMoodDecision(valid=False, reject_reason="bad_silence_mode")

    # ---- reasoning_summary（提供时必须为字符串）----
    reasoning_summary = raw.get("reasoning_summary", "")
    if not isinstance(reasoning_summary, str):
        return RequestMoodDecision(valid=False, reject_reason="bad_reasoning_summary")
    reasoning_summary = clamp_str(reasoning_summary, MAX_REASONING_CHARS)

    # ---- lift_actions（解除字段非法 → 整份决策作废）----
    lift_raw = raw.get("lift_actions", [])
    # 只有硬动作可解除（软动作只活当轮、永不持久化，解除无意义）；
    # 重复项视为非法整份作废——不由插件静默去重取舍
    if not isinstance(lift_raw, list) or any(
        not isinstance(n, str) or n not in HARD_ACTIONS for n in lift_raw
    ):
        return RequestMoodDecision(valid=False, reject_reason="bad_lift_actions")
    if len(lift_raw) != len(set(lift_raw)):
        return RequestMoodDecision(valid=False, reject_reason="lift_duplicate")
    lift_actions = list(lift_raw)

    # ---- actions（动作组非法 → 组拒绝，不影响合法解除）----
    def reject_group(reason: str) -> RequestMoodDecision:
        return RequestMoodDecision(
            valid=True,
            actions_rejected=True,
            reject_reason=reason,
            mood_update=mood_update,
            actions=[],
            lift_actions=lift_actions,
            silence_mode="none",
            reasoning_summary=reasoning_summary,
        )

    actions_raw = raw.get("actions", [])
    if not isinstance(actions_raw, list):
        return reject_group("actions_not_list")

    actions: list[dict] = []
    for item in actions_raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return reject_group("action_item_invalid")
        name = item["name"]
        if name not in MoodToolExecutor.TOOLS:
            return reject_group("action_unknown")
        if name not in enabled:
            return reject_group("action_disabled")
        # 显式 params: null 与缺失键不同：缺失按全默认处理，
        # 显式 null 属于非对象值，整组拒绝（不由插件擅自套默认值）
        if "params" in item and item["params"] is None:
            return reject_group("action_bad_params")
        params = MoodToolExecutor.validate_params_strict(name, item.get("params"))
        if params is None:
            return reject_group("action_bad_params")
        actions.append({"name": name, "params": params})

    names = [a["name"] for a in actions]
    if len(names) != len(set(names)):
        return reject_group("action_duplicate")
    if set(names) & set(lift_actions):
        return reject_group("action_lift_conflict")

    hard = [a for a in actions if a["name"] in HARD_ACTIONS]
    soft = [a for a in actions if a["name"] in SOFT_ACTIONS]
    if len(hard) > 1:
        return reject_group("multiple_hard_actions")
    if hard and soft:
        return reject_group("hard_soft_mix")
    # 完全恢复与"同时新激活硬动作"自相矛盾（软动作如求安慰不矛盾，放行）
    if mood_update["fully_recovered"] and hard:
        return reject_group("recovered_with_hard_action")

    # ---- silence_mode 与动作组合 ----
    if silence_mode != "none" and not hard:
        return reject_group("silence_without_hard_action")
    if hard and hard[0]["name"] == "cold_violence" and silence_mode == "none":
        return reject_group("cold_violence_requires_silence")
    if silence_mode == "after_expression" and (len(hard) != 1 or hard[0]["name"] != "cold_violence"):
        return reject_group("after_expression_requires_cold_violence")
    if hard and hard[0]["name"] == "read_no_reply" and silence_mode != "immediate":
        return reject_group("read_no_reply_requires_immediate")

    return RequestMoodDecision(
        valid=True,
        mood_update=mood_update,
        actions=actions,
        lift_actions=lift_actions,
        silence_mode=silence_mode,
        reasoning_summary=reasoning_summary,
    )
