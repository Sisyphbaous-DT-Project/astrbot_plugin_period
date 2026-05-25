"""Tests for command permission logic in main.py."""

import pytest
from unittest.mock import MagicMock


class MockPlugin:
    """Minimal mock of PeriodPlugin with just _check_command_permission."""

    def __init__(self, config):
        self.config = config

    def _check_command_permission(self, cmd_name: str):
        mode = self.config.get("commands_enabled", "all")
        if mode == "all":
            return True, ""
        if mode == "none":
            return False, "当前会话的周期指令已关闭。如需调整，请前往插件配置修改「指令权限控制」。"
        if mode == "readonly":
            if cmd_name == "status":
                return True, ""
            return False, "当前仅允许查看状态，设置类指令已被关闭。如需调整，请前往插件配置修改「指令权限控制」。"
        return True, ""


@pytest.fixture
def plugin(sample_config):
    return MockPlugin(sample_config)


class TestCommandPermission:
    """commands_enabled setting controls which commands are available."""

    def test_all_mode_allows_everything(self, plugin):
        """all mode allows all commands."""
        plugin.config["commands_enabled"] = "all"
        for cmd in ("status", "set", "toggle", "advance", "reset"):
            allowed, _ = plugin._check_command_permission(cmd)
            assert allowed is True, f"{cmd} should be allowed in all mode"

    def test_readonly_mode_allows_status_only(self, plugin):
        """readonly mode allows only status command."""
        plugin.config["commands_enabled"] = "readonly"
        allowed, _ = plugin._check_command_permission("status")
        assert allowed is True
        for cmd in ("set", "toggle", "advance", "reset"):
            allowed, msg = plugin._check_command_permission(cmd)
            assert allowed is False, f"{cmd} should be blocked in readonly mode"
            assert "仅允许查看状态" in msg

    def test_none_mode_blocks_everything(self, plugin):
        """none mode blocks all commands."""
        plugin.config["commands_enabled"] = "none"
        for cmd in ("status", "set", "toggle", "advance", "reset"):
            allowed, msg = plugin._check_command_permission(cmd)
            assert allowed is False, f"{cmd} should be blocked in none mode"
            assert "指令已关闭" in msg

    def test_unknown_mode_defaults_to_all(self, plugin):
        """Unknown mode value defaults to allowing everything."""
        plugin.config["commands_enabled"] = "unknown_value"
        allowed, _ = plugin._check_command_permission("set")
        assert allowed is True

    def test_missing_config_defaults_to_all(self, plugin):
        """Missing commands_enabled key defaults to all."""
        del plugin.config["commands_enabled"]
        allowed, _ = plugin._check_command_permission("set")
        assert allowed is True
