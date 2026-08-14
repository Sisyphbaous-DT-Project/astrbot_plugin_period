"""Tests for core/mood_tools.py — MoodToolExecutor and decision legality matrix."""

import pytest

from core.mood_tools import MoodToolExecutor, validate_decision


class TestToolValidation:
    """Parameter validation and clamping."""

    def test_validate_cold_violence_defaults(self):
        params = MoodToolExecutor.validate_params("cold_violence", {})
        assert params == {"duration": 30}

    def test_validate_cold_violence_clamped(self):
        params = MoodToolExecutor.validate_params("cold_violence", {"duration": 9999})
        assert params["duration"] == 1440

        params = MoodToolExecutor.validate_params("cold_violence", {"duration": 0})
        assert params["duration"] == 1

    def test_validate_read_no_reply_rounds(self):
        assert MoodToolExecutor.validate_params("read_no_reply", {}) == {"rounds": 3}
        assert MoodToolExecutor.validate_params("read_no_reply", {"rounds": 99})["rounds"] == 10

    def test_validate_unknown_tool_returns_empty(self):
        params = MoodToolExecutor.validate_params("unknown", {"foo": "bar"})
        assert params == {}

    def test_validate_str_options_fallback(self):
        params = MoodToolExecutor.validate_params("seek_comfort", {"type": "invalid"})
        assert params["type"] == "emotional"

    def test_validate_perfunctory_level(self):
        params = MoodToolExecutor.validate_params("perfunctory_reply", {"level": 5})
        assert params["level"] == 3

        params = MoodToolExecutor.validate_params("perfunctory_reply", {"level": 0})
        assert params["level"] == 1


class TestPromptInjection:
    """Tendency prompts: 只描述情绪倾向，不含预设台词。"""

    def test_perfunctory_level_1(self):
        text = MoodToolExecutor.get_prompt_injection("perfunctory_reply", {"level": 1})
        assert "冷淡" in text

    def test_perfunctory_level_3(self):
        text = MoodToolExecutor.get_prompt_injection("perfunctory_reply", {"level": 3})
        assert "敷衍" in text

    def test_seek_comfort_emotional(self):
        text = MoodToolExecutor.get_prompt_injection("seek_comfort", {"type": "emotional"})
        assert "安慰" in text

    def test_emotional_outburst_angry(self):
        text = MoodToolExecutor.get_prompt_injection("emotional_outburst", {"type": "angry"})
        assert "火气" in text

    def test_topic_shift(self):
        text = MoodToolExecutor.get_prompt_injection("topic_shift", {})
        assert "话题" in text and "提不起兴趣" in text

    def test_unknown_tool_returns_empty(self):
        assert MoodToolExecutor.get_prompt_injection("unknown", {}) == ""

    def test_no_preset_lines_anywhere(self):
        """固定台词已删除：任何动作提示都不得包含旧预设句。"""
        banned = ["我现在不想说话", "……算了，别理我"]
        for name in ("perfunctory_reply", "seek_comfort", "delayed_reply",
                     "emotional_outburst", "topic_shift"):
            text = MoodToolExecutor.get_prompt_injection(name, {})
            for line in banned:
                assert line not in text

    def test_get_initial_message_removed(self):
        assert not hasattr(MoodToolExecutor, "get_initial_message")


class TestValidateDecision:
    """第三阶段 JSON 的合法性矩阵。"""

    def _valid_mood_update(self):
        return {
            "status": "recovering",
            "summary": "有所缓和但仍有些介意",
            "cause_category": "dismissive",
            "latest_reason": "感到回应不够被重视",
            "improved": True,
            "fully_recovered": False,
            "recovery_reason": "",
        }

    def _mk(self, **kw):
        """带完整固定字段的合法决策骨架，供动作矩阵用例覆盖默认字段。"""
        base = {
            "mood_update": self._valid_mood_update(),
            "actions": [],
            "lift_actions": [],
            "silence_mode": "none",
            "reasoning_summary": "摘要",
        }
        base.update(kw)
        return base

    def test_not_dict_invalid(self):
        d = validate_decision("not a dict")
        assert d.valid is False

    def test_empty_dict_invalid(self):
        d = validate_decision({})
        assert d.valid is False

    def test_valid_soft_combo(self):
        d = validate_decision({
            "mood_update": self._valid_mood_update(),
            "actions": [
                {"name": "seek_comfort", "params": {"type": "emotional"}},
                {"name": "topic_shift", "params": {}},
            ],
            "lift_actions": [],
            "silence_mode": "none",
            "reasoning_summary": "想被哄哄",
        })
        assert d.valid is True
        assert d.actions_rejected is False
        assert len(d.actions) == 2
        assert d.mood_update["status"] == "recovering"

    def test_bad_mood_update_invalidates_all(self):
        d = validate_decision({
            "mood_update": {"status": "bogus", "cause_category": "dismissive"},
            "actions": [],
            "lift_actions": ["cold_violence"],
        })
        assert d.valid is False
        assert d.lift_actions == []  # 决策级失败，解除也不执行

    def test_bad_lift_actions_invalidates_all(self):
        d = validate_decision({
            "mood_update": self._valid_mood_update(),
            "actions": [],
            "lift_actions": ["not_a_tool"],
        })
        assert d.valid is False

    def test_unknown_action_group_rejected_but_lift_applies(self):
        d = validate_decision(self._mk(
            actions=[{"name": "fly_away", "params": {}}],
            lift_actions=["cold_violence"],
        ))
        assert d.valid is True
        assert d.actions_rejected is True
        assert d.actions == []
        assert d.lift_actions == ["cold_violence"]

    def test_duplicate_actions_rejected(self):
        d = validate_decision(self._mk(
            actions=[
                {"name": "topic_shift"},
                {"name": "topic_shift"},
            ],
        ))
        assert d.actions_rejected and d.reject_reason == "action_duplicate"

    def test_lift_and_activate_same_action_rejected(self):
        d = validate_decision(self._mk(
            actions=[{"name": "cold_violence", "params": {"duration": 10}}],
            lift_actions=["cold_violence"],
            silence_mode="immediate",
        ))
        assert d.actions_rejected and d.reject_reason == "action_lift_conflict"

    def test_multiple_hard_actions_rejected(self):
        d = validate_decision(self._mk(
            actions=[
                {"name": "cold_violence", "params": {"duration": 10}},
                {"name": "read_no_reply", "params": {"rounds": 2}},
            ],
            silence_mode="immediate",
        ))
        assert d.actions_rejected and d.reject_reason == "multiple_hard_actions"

    def test_hard_soft_mix_rejected(self):
        d = validate_decision(self._mk(
            actions=[
                {"name": "cold_violence", "params": {"duration": 10}},
                {"name": "seek_comfort", "params": {}},
            ],
            silence_mode="immediate",
        ))
        assert d.actions_rejected and d.reject_reason == "hard_soft_mix"

    def test_after_expression_only_with_new_cold_violence(self):
        ok = validate_decision(self._mk(
            actions=[{"name": "cold_violence", "params": {"duration": 10}}],
            silence_mode="after_expression",
        ))
        assert ok.valid and not ok.actions_rejected

        bad = validate_decision(self._mk(
            actions=[{"name": "read_no_reply", "params": {"rounds": 2}}],
            silence_mode="after_expression",
        ))
        assert bad.actions_rejected

        no_action = validate_decision(self._mk(actions=[], silence_mode="after_expression"))
        assert no_action.actions_rejected

    def test_read_no_reply_requires_immediate(self):
        bad = validate_decision(self._mk(
            actions=[{"name": "read_no_reply", "params": {"rounds": 3}}],
            silence_mode="none",
        ))
        assert bad.actions_rejected
        assert bad.reject_reason == "read_no_reply_requires_immediate"

        ok = validate_decision(self._mk(
            actions=[{"name": "read_no_reply", "params": {"rounds": 3}}],
            silence_mode="immediate",
        ))
        assert ok.valid and not ok.actions_rejected

    def test_disabled_action_rejected(self):
        d = validate_decision(
            self._mk(actions=[{"name": "cold_violence", "params": {}}],
                     silence_mode="immediate"),
            enabled_actions={"topic_shift"},
        )
        assert d.actions_rejected and d.reject_reason == "action_disabled"

    def test_bad_silence_mode_invalid(self):
        d = validate_decision(self._mk(silence_mode="sometimes", actions=[]))
        assert d.valid is False

    def test_params_clamped_inside_decision(self):
        d = validate_decision(self._mk(
            actions=[{"name": "cold_violence", "params": {"duration": 99999}}],
            silence_mode="immediate",
        ))
        assert d.actions[0]["params"]["duration"] == 1440


    # ------------------------------------------------------------------ #
    #  严格模式（P1-④ 回归）
    # ------------------------------------------------------------------ #

    def test_string_false_rejected_as_bool(self):
        """"false" 字符串不得被 bool() 吞掉，整份决策作废。"""
        mu = self._valid_mood_update()
        mu["improved"] = "false"
        d = validate_decision({"mood_update": mu, "actions": []})
        assert d.valid is False
        assert d.reject_reason == "mood_update_bad_bool:improved"

    def test_params_wrong_type_rejected_not_crash(self):
        """params 为非对象时整组拒绝，且不得抛出 AttributeError 穿透。"""
        d = validate_decision(self._mk(
            actions=[{"name": "cold_violence", "params": "oops"}],
            silence_mode="immediate",
        ))
        assert d.valid is True
        assert d.actions_rejected is True
        assert d.reject_reason == "action_bad_params"

    def test_params_wrong_value_type_rejected(self):
        d = validate_decision(self._mk(
            actions=[{"name": "cold_violence", "params": {"duration": "30"}}],
            silence_mode="immediate",
        ))
        assert d.actions_rejected and d.reject_reason == "action_bad_params"

        d = validate_decision(self._mk(
            actions=[{"name": "read_no_reply", "params": {"rounds": True}}],
            silence_mode="immediate",
        ))
        assert d.actions_rejected and d.reject_reason == "action_bad_params"

    def test_cold_violence_requires_silence_mode(self):
        d = validate_decision(self._mk(
            actions=[{"name": "cold_violence", "params": {"duration": 10}}],
            silence_mode="none",
        ))
        assert d.actions_rejected
        assert d.reject_reason == "cold_violence_requires_silence"

    def test_missing_mood_update_invalid(self):
        """mood_update 是固定字段，缺失即整份作废。"""
        d = validate_decision({"actions": [], "lift_actions": []})
        assert d.valid is False
        assert d.reject_reason == "mood_update_missing"

    def test_mood_update_text_fields_not_coerced(self):
        """文本字段禁止把列表/对象强转成字符串。"""
        mu = self._valid_mood_update()
        mu["summary"] = ["被", "敷", "衍"]
        d = validate_decision({"mood_update": mu, "actions": []})
        assert d.valid is False
        assert d.reject_reason == "mood_update_bad_text:summary"

        mu = self._valid_mood_update()
        mu["latest_reason"] = {"x": 1}
        d = validate_decision({"mood_update": mu, "actions": []})
        assert d.valid is False

    def test_recovered_with_hard_action_rejected(self):
        """完全恢复同时新激活硬动作 = 自相矛盾，动作组整组拒绝。"""
        mu = self._valid_mood_update()
        mu["status"] = "recovered"
        mu["fully_recovered"] = True
        d = validate_decision(self._mk(
            mood_update=mu,
            actions=[{"name": "cold_violence", "params": {"duration": 10}}],
            silence_mode="immediate",
        ))
        assert d.actions_rejected
        assert d.reject_reason == "recovered_with_hard_action"

    def test_recovered_with_soft_action_allowed(self):
        """完全恢复 + 软动作（如求安慰）情绪上不矛盾，放行。"""
        mu = self._valid_mood_update()
        mu["status"] = "recovered"
        mu["fully_recovered"] = True
        d = validate_decision(self._mk(
            mood_update=mu,
            actions=[{"name": "seek_comfort", "params": {"type": "emotional"}}],
        ))
        assert d.valid and not d.actions_rejected

    def test_non_str_reasoning_summary_invalid(self):
        d = validate_decision(self._mk(actions=[], reasoning_summary=["x"]))
        assert d.valid is False
        assert d.reject_reason == "bad_reasoning_summary"

    def test_contradictory_recovery_state_invalid(self):
        mu = self._valid_mood_update()
        mu["status"] = "active"
        mu["fully_recovered"] = True
        d = validate_decision({"mood_update": mu, "actions": []})
        assert d.valid is False
        assert "contradiction" in d.reject_reason

        mu = self._valid_mood_update()
        mu["status"] = "recovered"
        mu["fully_recovered"] = False
        d = validate_decision({"mood_update": mu, "actions": []})
        assert d.valid is False

    def test_recovered_combo_accepted(self):
        mu = self._valid_mood_update()
        mu["status"] = "recovered"
        mu["fully_recovered"] = True
        mu["recovery_reason"] = "对方认真道歉"
        d = validate_decision(self._mk(mood_update=mu))
        assert d.valid is True

    def test_missing_top_level_fields_invalid(self):
        """actions/lift_actions/silence_mode/reasoning_summary 缺失即整份作废。"""
        for missing in ("actions", "lift_actions", "silence_mode", "reasoning_summary"):
            raw = self._mk()
            del raw[missing]
            d = validate_decision(raw)
            assert d.valid is False
            assert d.reject_reason == f"missing_field:{missing}"

    def test_validator_never_raises(self):
        """任何畸形输入都不得抛异常穿透到请求钩子。"""
        import json
        weird_inputs = [
            {"mood_update": [1, 2]},
            {"actions": [{"name": "cold_violence", "params": [1]}]},
            {"actions": [None, 42, "x"]},
            {"lift_actions": "cold_violence"},
            {"mood_update": {"status": "active", "cause_category": "other",
                             "improved": 1}},
        ]
        for raw in weird_inputs:
            d = validate_decision(json.loads(json.dumps(raw)))
            assert isinstance(d.valid, bool)

    def test_explicit_null_params_rejected(self):
        """显式 params: null 属于非对象值，整组拒绝；键缺失才按全默认。"""
        d = validate_decision(self._mk(
            actions=[{"name": "cold_violence", "params": None}],
            silence_mode="immediate",
        ))
        assert d.actions_rejected and d.reject_reason == "action_bad_params"

        ok = validate_decision(self._mk(
            actions=[{"name": "cold_violence"}],  # 键缺失 → 全默认
            silence_mode="immediate",
        ))
        assert ok.valid and not ok.actions_rejected
        assert ok.actions[0]["params"] == {"duration": 30}

    def test_duplicate_lift_actions_invalid(self):
        """lift_actions 重复项不得静默去重，整份决策作废。"""
        d = validate_decision(self._mk(
            lift_actions=["cold_violence", "cold_violence"],
        ))
        assert d.valid is False
        assert d.reject_reason == "lift_duplicate"

    def test_soft_action_lift_target_invalid(self):
        """软动作只活当轮、永不持久化，不能作为解除目标。"""
        d = validate_decision(self._mk(lift_actions=["seek_comfort"]))
        assert d.valid is False
        assert d.reject_reason == "bad_lift_actions"

    def test_hard_action_lift_accepted(self):
        d = validate_decision(self._mk(lift_actions=["read_no_reply"]))
        assert d.valid is True
        assert d.lift_actions == ["read_no_reply"]
