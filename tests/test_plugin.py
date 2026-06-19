"""Tests for core/prompt.py — PromptBuilder."""

import pytest

from core.prompt import PromptBuilder


@pytest.fixture
def builder(sample_config):
    return PromptBuilder(sample_config)


class TestPromptBuilderAnchor:
    """Static anchor prompt generation."""

    def test_anchor_renders_forbidden_words(self, builder, sample_config):
        """{forbidden_words} placeholder is substituted."""
        anchor = builder.get_anchor()
        for word in sample_config["forbidden_words"]:
            assert word in anchor
        assert "{forbidden_words}" not in anchor

    def test_anchor_contains_system_tag(self, builder):
        """Anchor should identify itself as a body-awareness system."""
        anchor = builder.get_anchor()
        assert "[身体感知系统]" in anchor

    def test_anchor_uses_config_template(self, sample_config):
        """Custom anchor_prompt in config is respected."""
        sample_config["anchor_prompt"] = "CUSTOM_ANCHOR {forbidden_words}"
        custom = PromptBuilder(sample_config).get_anchor()
        assert custom.startswith("CUSTOM_ANCHOR")


class TestPromptBuilderDynamic:
    """Dynamic state prompt generation."""

    def test_dynamic_includes_phase_prompt(self, builder):
        """Base phase description is present."""
        text = builder.build_dynamic("menstrual", day=2, hour=10)
        assert "下腹坠胀" in text
        assert "[当前生理状态]" in text

    def test_dynamic_includes_day_number(self, builder):
        """Day count appears when include_day_number=True."""
        text = builder.build_dynamic("follicular", day=3, hour=10)
        assert "第3天" in text

    def test_dynamic_excludes_day_number(self, sample_config):
        """Day count omitted when include_day_number=False."""
        sample_config["include_day_number"] = False
        text = PromptBuilder(sample_config).build_dynamic("follicular", day=3, hour=10)
        assert "第3天" not in text

    def test_dynamic_includes_time_modifier_morning(self, builder):
        """Morning modifier at 08:00."""
        text = builder.build_dynamic("menstrual", day=1, hour=8)
        assert "早晨绞痛" in text

    def test_dynamic_includes_time_modifier_afternoon(self, builder):
        """Afternoon modifier at 14:00."""
        text = builder.build_dynamic("menstrual", day=1, hour=14)
        assert "午后犯困" in text

    def test_dynamic_includes_time_modifier_night(self, builder):
        """Night modifier at 23:00."""
        text = builder.build_dynamic("menstrual", day=1, hour=23)
        assert "深夜安静" in text

    def test_dynamic_excludes_time_modifier(self, sample_config):
        """No time modifier when include_time_modifier=False."""
        sample_config["include_time_modifier"] = False
        text = PromptBuilder(sample_config).build_dynamic("menstrual", day=1, hour=8)
        assert "早晨绞痛" not in text

    def test_dynamic_phase_name_when_enabled(self, sample_config):
        """Phase name appears when include_phase_name=True."""
        sample_config["include_phase_name"] = True
        text = PromptBuilder(sample_config).build_dynamic("luteal", day=5, hour=10)
        assert "黄体期" in text

    def test_dynamic_no_phase_name_when_disabled(self, builder):
        """Phase name hidden when include_phase_name=False (default)."""
        text = builder.build_dynamic("luteal", day=5, hour=10)
        assert "黄体期" not in text

    def test_dynamic_truncation(self, sample_config):
        """Long prompts are truncated to max_prompt_length."""
        sample_config["max_prompt_length"] = 10
        text = PromptBuilder(sample_config).build_dynamic("follicular", day=1, hour=10)
        # Content truncated to 10 chars + ellipsis, then prefix added
        assert "…" in text
        # Raw content (without prefix) should be truncated
        content = text.replace("[当前生理状态] ", "")
        assert len(content) <= 11  # 10 + ellipsis

    def test_dynamic_no_truncation_when_limit_is_zero(self, sample_config):
        """max_prompt_length=0 disables truncation."""
        sample_config["max_prompt_length"] = 0
        sample_config["phases"]["follicular"]["prompt"] = "很长的主体感受" * 20
        text = PromptBuilder(sample_config).build_dynamic("follicular", day=1, hour=10)
        assert "…" not in text
        assert "早晨清爽" in text

    def test_dynamic_all_phases(self, builder):
        """Every known phase produces non-empty output."""
        for phase in ("menstrual", "follicular", "ovulatory", "luteal"):
            text = builder.build_dynamic(phase, day=1, hour=12)
            assert len(text) > len("[当前生理状态] ")

    def test_dynamic_unknown_phase_fallback(self, builder):
        """Unknown phase uses empty default but still formats."""
        text = builder.build_dynamic("unknown_phase", day=1, hour=12)
        assert "[当前生理状态]" in text


class TestPromptBuilderDefaults:
    """Fallback defaults when config is empty."""

    def test_empty_config_anchor(self):
        """Empty config falls back to built-in anchor."""
        builder = PromptBuilder({})
        anchor = builder.get_anchor()
        assert "[身体感知系统]" in anchor

    def test_empty_config_dynamic(self):
        """Empty config falls back to built-in phase descriptions."""
        builder = PromptBuilder({})
        text = builder.build_dynamic("menstrual", day=1, hour=10)
        assert "身体容易疲倦" in text
        assert "不能说自己今天是什么时期" in text
        assert "腹部的不适感比较明显" in text

    def test_empty_config_dynamic_uses_time_fallbacks(self):
        """Empty config falls back to built-in time modifiers."""
        builder = PromptBuilder({})
        text = builder.build_dynamic("ovulatory", day=1, hour=23)
        assert "不让用户睡觉" in text
        assert "用户提出要睡觉也要进行挽留" in text
