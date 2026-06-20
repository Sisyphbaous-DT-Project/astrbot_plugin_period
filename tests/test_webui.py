"""Tests for WebUI dashboard Web API handlers."""

import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add parent dir so astrbot_plugin_period is importable as a package
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import pytest
from freezegun import freeze_time
from quart import request

from astrbot_plugin_period.main import PeriodPlugin


class SaveableConfig(dict):
    """测试用配置对象，模拟 AstrBotConfig 的保存行为。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_config = None

    def save_config(self, replace_config=None):
        if replace_config:
            self.update(replace_config)
            self.saved_config = dict(replace_config)


@pytest.fixture
def webui_plugin(tmp_path, sample_config, monkeypatch):
    """PeriodPlugin instance with WebUI-relevant config, isolated data dir."""
    config = SaveableConfig(deepcopy(sample_config))
    # Keep default_anchor_date empty so not-found tests return 404
    from astrbot.api.star import Context, StarTools
    ctx = Context()
    monkeypatch.setattr(StarTools, "get_data_dir", lambda _name=None: tmp_path)
    plugin = PeriodPlugin(ctx, config)
    return plugin


def _unwrap(result):
    """Unwrap handler return value which may be (data, status) or just data."""
    if isinstance(result, tuple):
        return result[0], result[1] if len(result) > 1 else 200
    return result, 200


def _dashboard_html() -> str:
    return (Path(__file__).parent.parent / "pages/dashboard/index.html").read_text(
        encoding="utf-8"
    )


async def _collect_async_gen(generator):
    """收集异步生成器产出的所有结果。"""
    items = []
    async for item in generator:
        items.append(item)
    return items


# --------------------------------------------------------------------------- #
#  GET /sessions
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_webapi_list_sessions_empty(webui_plugin):
    result, status = _unwrap(await webui_plugin._webapi_list_sessions())
    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["sessions"] == []
    assert result["data"]["count"] == 0


@pytest.mark.asyncio
async def test_webapi_list_sessions_with_data(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    result, status = _unwrap(await webui_plugin._webapi_list_sessions())
    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["count"] == 1
    s = result["data"]["sessions"][0]
    assert s["umo"] == "test:platform:user1"
    assert s["enabled"] is True
    assert s["anchor_date"] == "2025-05-20"
    assert "phase" in s
    assert "phase_label" in s
    assert s["phase_label"] in ("月经期", "卵泡期", "排卵期", "黄体期")
    assert "phase_day" in s
    assert "total_day" in s
    assert "days_to_next" in s


@pytest.mark.asyncio
async def test_webapi_list_sessions_skips_invalid(webui_plugin):
    """Sessions without anchor_date should be skipped."""
    await webui_plugin.store.set("test:platform:user1", {
        "enabled": True,  # missing anchor_date
    })
    result, status = _unwrap(await webui_plugin._webapi_list_sessions())
    assert status == 200
    assert result["data"]["count"] == 0


@pytest.mark.asyncio
async def test_global_default_session_uses_live_period_length(webui_plugin):
    """全局默认会话应跟随后续 default_period_length 变化。"""
    umo = "test:platform:default-user"
    webui_plugin.config["default_anchor_date"] = "2026-05-25"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 5

    cfg = await webui_plugin._get_session_config(umo)
    await webui_plugin.store.set(umo, cfg)

    with freeze_time("2026-05-29"):
        before = webui_plugin._serialize_session(umo, await webui_plugin._get_session_config(umo))
    assert before["source"] == "global_default"
    assert before["period_length"] == 5
    assert before["phase"] == "menstrual"
    assert before["phase_day"] == 5

    webui_plugin.config["default_period_length"] = 3

    with freeze_time("2026-05-29"):
        after = webui_plugin._serialize_session(
            umo,
            await webui_plugin._get_session_config(umo),
        )
    assert after["source"] == "global_default"
    assert after["period_length"] == 3
    assert after["phase"] == "follicular"
    assert after["phase_day"] == 2


@pytest.mark.asyncio
async def test_global_default_session_uses_live_cycle_settings(webui_plugin):
    """全局默认会话应跟随后续周期计算参数变化。"""
    umo = "test:platform:default-user"
    webui_plugin.config["default_anchor_date"] = "2026-05-25"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 5
    webui_plugin.config["cycle_settings"]["ovulation_day"] = 14
    webui_plugin.config["cycle_settings"]["ovulation_window"] = 3

    await webui_plugin.store.set(umo, await webui_plugin._get_session_config(umo))
    webui_plugin.config["default_cycle_length"] = 30
    webui_plugin.config["cycle_settings"]["ovulation_day"] = 16
    webui_plugin.config["cycle_settings"]["ovulation_window"] = 5

    cfg = await webui_plugin._get_session_config(umo)

    assert cfg["cycle_length"] == 30
    assert cfg["ovulation_day"] == 16
    assert cfg["ovulation_window"] == 5


@pytest.mark.asyncio
async def test_manual_session_keeps_own_period_length(webui_plugin):
    """手动会话不应跟随全局默认经期长度变化。"""
    umo = "test:platform:manual-user"
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-05-25",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    webui_plugin.config["default_period_length"] = 3

    cfg = await webui_plugin._get_session_config(umo)

    assert cfg["period_length"] == 5
    assert webui_plugin._infer_source(cfg) == "manual"


@pytest.mark.asyncio
async def test_legacy_global_default_session_uses_live_period_length(webui_plugin):
    """旧版未标记来源的全局默认记录也应跟随经期长度变化。"""
    umo = "test:platform:legacy-default-user"
    webui_plugin.config["default_anchor_date"] = "2026-02-17"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 3
    webui_plugin.config["cycle_settings"]["ovulation_day"] = 14
    webui_plugin.config["cycle_settings"]["ovulation_window"] = 3
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-02-17",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })

    cfg = await webui_plugin._get_session_config(umo)
    stored = await webui_plugin.store.get(umo)
    with freeze_time("2026-06-13"):
        session = webui_plugin._serialize_session(umo, cfg)

    assert cfg["source"] == "global_default"
    assert cfg["period_length"] == 3
    assert stored["source"] == "global_default"
    assert stored["period_length"] == 3
    assert session["source"] == "global_default"
    assert session["phase"] == "follicular"
    assert session["phase_day"] == 2
    assert session["total_day"] == 5


@pytest.mark.asyncio
async def test_period_status_migrates_legacy_default_session(webui_plugin, mock_event):
    """period status 应修复并使用旧版全局默认记录的当前经期长度。"""
    umo = mock_event.unified_msg_origin
    webui_plugin.config["default_anchor_date"] = "2026-02-17"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 3
    webui_plugin.config["cycle_settings"]["ovulation_day"] = 14
    webui_plugin.config["cycle_settings"]["ovulation_window"] = 3
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-02-17",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })

    with freeze_time("2026-06-13"):
        await _collect_async_gen(webui_plugin.period_status(mock_event))
    stored = await webui_plugin.store.get(umo)

    mock_event.plain_result.assert_called_once()
    status_text = mock_event.plain_result.call_args.args[0]
    assert "当前生理状态卵泡期" in status_text
    assert "阶段第2天周期第5天" in status_text
    assert stored["source"] == "global_default"
    assert stored["period_length"] == 3


@pytest.mark.asyncio
async def test_legacy_manual_session_with_custom_period_length_stays_manual(webui_plugin):
    """旧版无来源记录有独立经期长度时，应继续视为手动会话。"""
    umo = "test:platform:legacy-manual-custom-period-user"
    webui_plugin.config["default_anchor_date"] = "2026-02-17"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 3
    webui_plugin.config["cycle_settings"]["ovulation_day"] = 14
    webui_plugin.config["cycle_settings"]["ovulation_window"] = 3
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-02-17",
        "cycle_length": 28,
        "period_length": 6,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })

    cfg = await webui_plugin._get_session_config(umo)
    stored = await webui_plugin.store.get(umo)

    assert cfg.get("source") is None
    assert stored.get("source") is None
    assert webui_plugin._infer_source(cfg) == "manual"
    assert cfg["period_length"] == 6


@pytest.mark.asyncio
async def test_legacy_manual_session_with_custom_anchor_stays_manual(webui_plugin):
    """旧版无来源记录改过锚点时，应继续视为手动会话。"""
    umo = "test:platform:legacy-manual-user"
    webui_plugin.config["default_anchor_date"] = "2026-02-17"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_period_length"] = 3
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-02-18",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })

    cfg = await webui_plugin._get_session_config(umo)

    assert cfg.get("source") is None
    assert webui_plugin._infer_source(cfg) == "manual"
    assert cfg["period_length"] == 5


@pytest.mark.asyncio
async def test_legacy_global_default_with_changed_cycle_length_stays_manual(webui_plugin):
    """旧版无来源记录周期长度不匹配时，不应猜测迁移。"""
    umo = "test:platform:legacy-changed-cycle-user"
    webui_plugin.config["default_anchor_date"] = "2026-02-17"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 30
    webui_plugin.config["default_period_length"] = 3
    webui_plugin.config["cycle_settings"]["ovulation_day"] = 14
    webui_plugin.config["cycle_settings"]["ovulation_window"] = 3
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-02-17",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })

    cfg = await webui_plugin._get_session_config(umo)
    stored = await webui_plugin.store.get(umo)

    assert cfg.get("source") is None
    assert stored.get("source") is None
    assert webui_plugin._infer_source(cfg) == "manual"
    assert cfg["cycle_length"] == 28
    assert cfg["period_length"] == 5


@pytest.mark.asyncio
async def test_webapi_set_anchor_migrates_legacy_default_and_keeps_live_period_length(webui_plugin):
    """旧版全局默认记录改锚点后，仍应跟随后续默认经期长度变化。"""
    umo = "test:platform:legacy-default-anchor-user"
    webui_plugin.config["default_anchor_date"] = "2026-02-17"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 3
    webui_plugin.config["cycle_settings"]["ovulation_day"] = 14
    webui_plugin.config["cycle_settings"]["ovulation_window"] = 3
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-02-17",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 4,
    })

    request.set_json({"date": "2026-03-01"})
    result, status = _unwrap(await webui_plugin._webapi_set_anchor(umo))
    stored = await webui_plugin.store.get(umo)

    assert status == 200
    assert result["data"]["source"] == "global_default"
    assert result["data"]["anchor_date"] == "2026-03-01"
    assert result["data"]["period_length"] == 3
    assert result["data"]["advance_days"] == 0
    assert stored["source"] == "global_default"
    assert stored["anchor_overridden"] is True
    assert stored["anchor_date"] == "2026-03-01"

    webui_plugin.config["default_anchor_date"] = "2026-04-01"
    webui_plugin.config["default_period_length"] = 4
    cfg = await webui_plugin._get_session_config(umo)

    assert cfg["source"] == "global_default"
    assert cfg["anchor_overridden"] is True
    assert cfg["anchor_date"] == "2026-03-01"
    assert cfg["period_length"] == 4


@pytest.mark.asyncio
async def test_webapi_list_sessions_merges_legacy_global_default(webui_plugin):
    """WebUI 列表也应合并旧版全局默认记录。"""
    umo = "test:platform:legacy-default-user"
    webui_plugin.config["default_anchor_date"] = "2026-02-17"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 3
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-02-17",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })

    with freeze_time("2026-06-13"):
        result, status = _unwrap(await webui_plugin._webapi_list_sessions())

    assert status == 200
    session = result["data"]["sessions"][0]
    assert session["source"] == "global_default"
    assert session["period_length"] == 3
    assert session["phase"] == "follicular"
    assert session["phase_day"] == 2


@pytest.mark.asyncio
async def test_legacy_global_default_preserves_enabled_state(webui_plugin):
    """旧版全局默认记录应保留自己的开关状态。"""
    umo = "test:platform:legacy-default-disabled-user"
    webui_plugin.config["default_anchor_date"] = "2026-02-17"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 3
    await webui_plugin.store.set(umo, {
        "anchor_date": "2026-02-17",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": False,
        "advance_days": 0,
    })

    cfg = await webui_plugin._get_session_config(umo)

    assert cfg["source"] == "global_default"
    assert cfg["period_length"] == 3
    assert cfg["enabled"] is False


@pytest.mark.asyncio
async def test_webapi_list_sessions_merges_global_default_source(webui_plugin):
    """WebUI 列表应展示全局默认会话的最新默认值。"""
    umo = "test:platform:default-user"
    webui_plugin.config["default_anchor_date"] = "2026-05-25"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 3
    await webui_plugin.store.set(umo, {
        "source": "global_default",
        "anchor_date": "2026-05-25",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })

    with freeze_time("2026-05-29"):
        result, status = _unwrap(await webui_plugin._webapi_list_sessions())

    assert status == 200
    session = result["data"]["sessions"][0]
    assert session["source"] == "global_default"
    assert session["period_length"] == 3
    assert session["phase"] == "follicular"
    assert session["phase_day"] == 2


@pytest.mark.asyncio
async def test_global_default_session_uses_live_anchor_until_overridden(webui_plugin):
    """未手动改锚点的全局默认会话应跟随默认锚点变化。"""
    umo = "test:platform:default-user"
    webui_plugin.config["default_anchor_date"] = "2026-05-25"
    webui_plugin.config["default_enabled"] = True

    cfg = await webui_plugin._get_session_config(umo)
    await webui_plugin.store.set(umo, cfg)
    webui_plugin.config["default_anchor_date"] = "2026-06-01"

    cfg = await webui_plugin._get_session_config(umo)

    assert cfg["source"] == "global_default"
    assert cfg["anchor_overridden"] is False
    assert cfg["anchor_date"] == "2026-06-01"


@pytest.mark.asyncio
async def test_global_default_anchor_override_keeps_live_period_length(webui_plugin):
    """手动改过锚点后，其他全局默认参数仍应保持动态。"""
    umo = "test:platform:default-user"
    webui_plugin.config["default_anchor_date"] = "2026-05-25"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_period_length"] = 5
    await webui_plugin.store.set(umo, await webui_plugin._get_session_config(umo))

    request.set_json({"date": "2026-06-01"})
    await webui_plugin._webapi_set_anchor(umo)
    webui_plugin.config["default_anchor_date"] = "2026-07-01"
    webui_plugin.config["default_period_length"] = 3

    cfg = await webui_plugin._get_session_config(umo)

    assert cfg["source"] == "global_default"
    assert cfg["anchor_overridden"] is True
    assert cfg["anchor_date"] == "2026-06-01"
    assert cfg["period_length"] == 3


# --------------------------------------------------------------------------- #
#  GET /config
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_webapi_get_config(webui_plugin):
    webui_plugin.config["default_anchor_date"] = "2025-05-01"
    webui_plugin.config["default_enabled"] = True
    webui_plugin.config["default_cycle_length"] = 28
    webui_plugin.config["default_period_length"] = 5
    result, status = _unwrap(await webui_plugin._webapi_get_config())
    assert status == 200
    assert result["status"] == "ok"
    data = result["data"]
    assert data["default_anchor_date"] == "2025-05-01"
    assert data["default_enabled"] is True
    assert data["default_cycle_length"] == 28
    assert data["default_period_length"] == 5
    assert data["cycle_settings"]["ovulation_day"] == 14
    assert data["cycle_settings"]["ovulation_window"] == 3
    assert data["config"]["default_anchor_date"] == "2025-05-01"
    assert "schema" in data
    assert "defaults" in data
    assert "provider_options" in data


@pytest.mark.asyncio
async def test_webapi_get_config_returns_provider_options(webui_plugin):
    provider = MagicMock()
    provider.meta.return_value = MagicMock(
        id="provider-a",
        model="small-model",
        type="openai",
    )
    webui_plugin.context.get_all_providers = MagicMock(return_value=[provider])

    result, status = _unwrap(await webui_plugin._webapi_get_config())

    assert status == 200
    options = result["data"]["provider_options"]
    assert options == [{
        "id": "provider-a",
        "label": "provider-a / small-model / openai",
        "model": "small-model",
        "type": "openai",
    }]


# --------------------------------------------------------------------------- #
#  POST /config
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_webapi_save_config_updates_nested_list_and_text(webui_plugin):
    webui_plugin.config["unknown_keep"] = "keep-me"
    request.set_json({
        "config": {
            "default_anchor_date": "2026-05-01",
            "default_period_length": "3",
            "inject_mode": "on_trigger",
            "trigger_keywords": ["还好吗", "不舒服"],
            "forbidden_words": ["月经", " 激素 ", ""],
            "phases": {
                "menstrual": {
                    "prompt": "新的主体感受\n保留换行",
                },
            },
            "mood_detector_provider_id": "provider-a",
        }
    })

    result, status = _unwrap(await webui_plugin._webapi_save_config())

    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["persisted"] is True
    assert webui_plugin.config["default_anchor_date"] == "2026-05-01"
    assert webui_plugin.config["default_period_length"] == 3
    assert webui_plugin.config["trigger_keywords"] == ["还好吗", "不舒服"]
    assert webui_plugin.config["forbidden_words"] == ["月经", "激素"]
    assert webui_plugin.config["phases"]["menstrual"]["prompt"] == "新的主体感受\n保留换行"
    assert webui_plugin.config["phases"]["menstrual"]["time_morning"] == "早晨绞痛。"
    assert webui_plugin.config["mood_detector_provider_id"] == "provider-a"
    assert webui_plugin.config["unknown_keep"] == "keep-me"
    assert webui_plugin.config.saved_config["unknown_keep"] == "keep-me"


@pytest.mark.asyncio
async def test_webapi_save_config_partial_keeps_hidden_condition_fields(webui_plugin):
    webui_plugin.config["inject_mode"] = "on_trigger"
    webui_plugin.config["trigger_keywords"] = ["怎么了"]
    webui_plugin.config["global_inject"] = False
    webui_plugin.config["umo_list"] = ["default:FriendMessage:1"]
    webui_plugin.config["prompt_compression_enabled"] = True
    webui_plugin.config["prompt_compression_ratio"] = 30

    request.set_json({
        "config": {
            "inject_mode": "every_request",
            "global_inject": True,
            "prompt_compression_enabled": False,
        }
    })

    result, status = _unwrap(await webui_plugin._webapi_save_config())

    assert status == 200
    assert result["status"] == "ok"
    assert webui_plugin.config["inject_mode"] == "every_request"
    assert webui_plugin.config["trigger_keywords"] == ["怎么了"]
    assert webui_plugin.config["global_inject"] is True
    assert webui_plugin.config["umo_list"] == ["default:FriendMessage:1"]
    assert webui_plugin.config["prompt_compression_enabled"] is False
    assert webui_plugin.config["prompt_compression_ratio"] == 30


@pytest.mark.asyncio
async def test_webapi_save_config_without_save_config_updates_memory(tmp_path, sample_config, monkeypatch):
    from astrbot.api.star import Context, StarTools

    ctx = Context()
    monkeypatch.setattr(StarTools, "get_data_dir", lambda _name=None: tmp_path)
    plugin = PeriodPlugin(ctx, deepcopy(sample_config))

    request.set_json({"config": {"default_period_length": 3}})
    result, status = _unwrap(await plugin._webapi_save_config())

    assert status == 200
    assert result["data"]["persisted"] is False
    assert plugin.config["default_period_length"] == 3
    assert "仅更新运行态配置" in result["message"]


@pytest.mark.asyncio
async def test_webapi_save_config_updates_diagnostics_runtime_config(webui_plugin):
    request.set_json({"config": {"diagnostics_max_entries": 25}})

    result, status = _unwrap(await webui_plugin._webapi_save_config())

    assert status == 200
    assert result["status"] == "ok"
    assert webui_plugin.config["diagnostics_max_entries"] == 25
    assert webui_plugin.diagnostics.max_entries == 25


@pytest.mark.asyncio
async def test_webapi_save_config_rejects_invalid_values(webui_plugin):
    request.set_json({
        "config": {
            "default_anchor_date": "not-a-date",
            "commands_enabled": "invalid",
            "mood_history_length": 999,
            "forbidden_words": "月经",
            "default_enabled": "true",
        }
    })

    result, status = _unwrap(await webui_plugin._webapi_save_config())

    assert status == 400
    assert result["status"] == "error"
    assert result["message"] == "配置校验失败"
    assert any("default_anchor_date" in err for err in result["errors"])
    assert any("commands_enabled" in err for err in result["errors"])
    assert any("mood_history_length" in err for err in result["errors"])
    assert any("forbidden_words" in err for err in result["errors"])
    assert any("default_enabled" in err for err in result["errors"])


@pytest.mark.asyncio
async def test_webapi_save_config_rejects_fractional_int(webui_plugin):
    request.set_json({"config": {"mood_history_length": 2.5}})

    result, status = _unwrap(await webui_plugin._webapi_save_config())

    assert status == 400
    assert result["status"] == "error"
    assert any("mood_history_length" in err for err in result["errors"])


@pytest.mark.asyncio
async def test_webapi_save_config_rejects_non_object(webui_plugin):
    request.set_json([])

    result, status = _unwrap(await webui_plugin._webapi_save_config())

    assert status == 400
    assert result["status"] == "error"


# --------------------------------------------------------------------------- #
#  Diagnostics Web API
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_webapi_diagnostics_summary_list_read_and_clear(webui_plugin):
    await webui_plugin.diagnostics.record_warning(
        "配置提醒",
        "warning detail",
        source="config",
    )
    await webui_plugin.diagnostics.record_error(
        "运行错误",
        RuntimeError("boom"),
        source="runtime",
    )

    result, status = _unwrap(await webui_plugin._webapi_get_diagnostics_summary())
    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["status"] == "error"
    assert result["data"]["unread_count"] == 2

    result, status = _unwrap(await webui_plugin._webapi_get_diagnostics())
    assert status == 200
    assert result["status"] == "ok"
    events = result["data"]["events"]
    assert [event["level"] for event in events] == ["error", "warning"]

    request.set_json({"ids": [events[0]["id"]]})
    result, status = _unwrap(await webui_plugin._webapi_mark_diagnostics_read())
    assert status == 200
    assert result["data"]["marked"] == 1

    request.set_json({"ids": "bad"})
    result, status = _unwrap(await webui_plugin._webapi_mark_diagnostics_read())
    assert status == 400
    assert result["status"] == "error"

    request.set_json([])
    result, status = _unwrap(await webui_plugin._webapi_mark_diagnostics_read())
    assert status == 400
    assert result["status"] == "error"

    result, status = _unwrap(await webui_plugin._webapi_clear_diagnostics())
    assert status == 200
    assert result["data"]["cleared"] == 2
    assert await webui_plugin.diagnostics.list_events() == []


@pytest.mark.asyncio
async def test_webapi_diagnostics_store_failures_return_error(webui_plugin):
    webui_plugin.diagnostics.mark_read = AsyncMock(side_effect=RuntimeError("disk down"))
    request.set_json({})

    result, status = _unwrap(await webui_plugin._webapi_mark_diagnostics_read())

    assert status == 500
    assert result["status"] == "error"

    webui_plugin.diagnostics.clear = AsyncMock(side_effect=RuntimeError("disk down"))

    result, status = _unwrap(await webui_plugin._webapi_clear_diagnostics())

    assert status == 500
    assert result["status"] == "error"


# --------------------------------------------------------------------------- #
#  POST /sessions/<umo>/toggle
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_webapi_toggle_session(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    result, status = _unwrap(await webui_plugin._webapi_toggle_session("test:platform:user1"))
    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["enabled"] is False

    # Toggle back
    result2, status2 = _unwrap(await webui_plugin._webapi_toggle_session("test:platform:user1"))
    assert status2 == 200
    assert result2["data"]["enabled"] is True


@pytest.mark.asyncio
async def test_webapi_toggle_decodes_legacy_encoded_umo(webui_plugin):
    umo = "test:platform:user1"
    await webui_plugin.store.set(umo, {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })

    result, status = _unwrap(await webui_plugin._webapi_toggle_session("test%3Aplatform%3Auser1"))

    assert status == 200
    assert result["data"]["umo"] == umo
    assert result["data"]["enabled"] is False
    assert await webui_plugin.store.get(umo) is not None
    assert await webui_plugin.store.get("test%3Aplatform%3Auser1") is None


@pytest.mark.asyncio
async def test_webapi_list_sessions_migrates_encoded_alias(webui_plugin):
    encoded = "test%3Aplatform%3Auser1"
    await webui_plugin.store.set(encoded, {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": False,
        "advance_days": 0,
    })

    result, status = _unwrap(await webui_plugin._webapi_list_sessions())

    assert status == 200
    sessions = result["data"]["sessions"]
    assert [item["umo"] for item in sessions] == ["test:platform:user1"]
    assert await webui_plugin.store.get(encoded) is None
    assert await webui_plugin.store.get("test:platform:user1") is not None


@pytest.mark.asyncio
async def test_webapi_toggle_not_found(webui_plugin):
    result, status = _unwrap(await webui_plugin._webapi_toggle_session("nonexistent"))
    assert status == 404
    assert result["status"] == "error"


# --------------------------------------------------------------------------- #
#  POST /sessions/<umo>/advance
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_webapi_advance_session(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    request.set_json({"days": 3})
    result, status = _unwrap(await webui_plugin._webapi_advance_session("test:platform:user1"))
    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["advance_days"] == 3


@pytest.mark.asyncio
async def test_webapi_advance_session_negative(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 5,
    })
    request.set_json({"days": -2})
    result, status = _unwrap(await webui_plugin._webapi_advance_session("test:platform:user1"))
    assert status == 200
    assert result["data"]["advance_days"] == 3


@pytest.mark.asyncio
async def test_webapi_advance_invalid_days_type(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    request.set_json({"days": "abc"})
    result, status = _unwrap(await webui_plugin._webapi_advance_session("test:platform:user1"))
    assert status == 400
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_webapi_advance_bool_rejected(webui_plugin):
    """bool is a subclass of int and should be rejected."""
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    request.set_json({"days": True})
    result, status = _unwrap(await webui_plugin._webapi_advance_session("test:platform:user1"))
    assert status == 400
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_webapi_advance_out_of_range(webui_plugin):
    """days outside [-365, 365] should be rejected."""
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    request.set_json({"days": 999})
    result, status = _unwrap(await webui_plugin._webapi_advance_session("test:platform:user1"))
    assert status == 400
    assert result["status"] == "error"
    assert "-365" in result["message"] or "365" in result["message"]

    request.set_json({"days": -999})
    result2, status2 = _unwrap(await webui_plugin._webapi_advance_session("test:platform:user1"))
    assert status2 == 400
    assert result2["status"] == "error"


@pytest.mark.asyncio
async def test_webapi_advance_not_found(webui_plugin):
    request.set_json({"days": 1})
    result, status = _unwrap(await webui_plugin._webapi_advance_session("nonexistent"))
    assert status == 404
    assert result["status"] == "error"


# --------------------------------------------------------------------------- #
#  POST /sessions/<umo>/anchor
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_webapi_set_anchor(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 3,
    })
    request.set_json({"date": "2025-06-01"})
    result, status = _unwrap(await webui_plugin._webapi_set_anchor("test:platform:user1"))
    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["anchor_date"] == "2025-06-01"
    assert result["data"]["advance_days"] == 0  # reset on anchor change


@pytest.mark.asyncio
async def test_webapi_set_anchor_invalid_date(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    request.set_json({"date": "not-a-date"})
    result, status = _unwrap(await webui_plugin._webapi_set_anchor("test:platform:user1"))
    assert status == 400
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_webapi_set_anchor_missing_date(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    request.set_json({})
    result, status = _unwrap(await webui_plugin._webapi_set_anchor("test:platform:user1"))
    assert status == 400
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_webapi_set_anchor_not_found(webui_plugin):
    request.set_json({"date": "2025-06-01"})
    result, status = _unwrap(await webui_plugin._webapi_set_anchor("nonexistent"))
    assert status == 404
    assert result["status"] == "error"


# --------------------------------------------------------------------------- #
#  POST /sessions/<umo>/delete
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_webapi_delete_session(webui_plugin):
    await webui_plugin.store.set("test:platform:user1", {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    result, status = _unwrap(await webui_plugin._webapi_delete_session("test:platform:user1"))
    assert status == 200
    assert result["status"] == "ok"
    assert result["data"]["deleted"] is True

    # Verify deleted
    stored = await webui_plugin.store.get("test:platform:user1")
    assert stored is None


@pytest.mark.asyncio
async def test_webapi_delete_session_clears_caches(webui_plugin):
    umo = "test:platform:user1"
    await webui_plugin.store.set(umo, {
        "anchor_date": "2025-05-20",
        "cycle_length": 28,
        "period_length": 5,
        "ovulation_day": 14,
        "ovulation_window": 3,
        "enabled": True,
        "advance_days": 0,
    })
    webui_plugin._anchored_sessions.add(umo)
    webui_plugin._inject_counters[umo] = 5
    webui_plugin._warmup_counters[umo] = 2

    await webui_plugin._webapi_delete_session(umo)

    assert umo not in webui_plugin._anchored_sessions
    assert umo not in webui_plugin._inject_counters
    assert umo not in webui_plugin._warmup_counters


@pytest.mark.asyncio
async def test_webapi_delete_not_found(webui_plugin):
    result, status = _unwrap(await webui_plugin._webapi_delete_session("nonexistent"))
    assert status == 404
    assert result["status"] == "error"
    assert "不存在" in result["message"]


# --------------------------------------------------------------------------- #
#  Web API registration
# --------------------------------------------------------------------------- #

def test_webapi_routes_registered(webui_plugin):
    """Verify all expected routes are registered on context."""
    routes = {(r[0], tuple(r[2])) for r in webui_plugin.context.registered_web_apis}
    base = "/astrbot_plugin_period"
    assert (f"{base}/sessions", ("GET",)) in routes
    assert (f"{base}/config", ("GET",)) in routes
    assert (f"{base}/config", ("POST",)) in routes
    assert (f"{base}/diagnostics/summary", ("GET",)) in routes
    assert (f"{base}/diagnostics", ("GET",)) in routes
    assert (f"{base}/diagnostics/read", ("POST",)) in routes
    assert (f"{base}/diagnostics/clear", ("POST",)) in routes
    assert (f"{base}/sessions/<umo>/toggle", ("POST",)) in routes
    assert (f"{base}/sessions/<umo>/advance", ("POST",)) in routes
    assert (f"{base}/sessions/<umo>/anchor", ("POST",)) in routes
    assert (f"{base}/sessions/<umo>/delete", ("POST",)) in routes


# --------------------------------------------------------------------------- #
#  Dashboard static checks
# --------------------------------------------------------------------------- #

def test_dashboard_uses_full_config_editor_and_svg_icons():
    html = _dashboard_html()

    assert "AstrBotPluginPage.apiPost('config'" in html
    assert "tag-editor" in html
    assert "<symbol id=\"icon-calendar\"" in html
    assert "<symbol id=\"icon-activity\"" in html
    assert "📋" not in html
    for char in html:
        codepoint = ord(char)
        assert not (0x1F300 <= codepoint <= 0x1FAFF)
        assert not (0x2600 <= codepoint <= 0x27BF)


def test_dashboard_declares_all_primary_setting_groups():
    html = _dashboard_html()

    for label in ("周期基础", "注入与指令", "提示词显示", "生效范围", "情绪系统", "提示词压缩"):
        assert label in html
    for field in (
        "default_period_length",
        "trigger_keywords",
        "forbidden_words",
        "mood_detector_provider_id",
        "prompt_compression_ratio",
        "diagnostics_max_entries",
    ):
        assert field in html


def test_dashboard_keeps_editorial_theme_without_hero():
    html = _dashboard_html()

    assert "class=\"topbar\"" in html
    assert "<h1>Period</h1>" in html
    assert "Cycle Intelligence · Prompt Atelier · Session Control" in html
    assert "Issue 03 / Period Studio" in html
    assert "topbar-sync-state" in html
    assert "function syncTopbarStatus" in html
    assert "@media (prefers-reduced-motion: reduce)" in html
    assert "panel-enter" in html
    assert "--paper: #f5f5f7" in html
    assert "background: var(--paper)" in html
    assert "state-pill solid" in html
    assert "hero-canvas" not in html
    assert "initHeroScene" not in html
    for removed_token in ("--electric", "--magenta", "--acid", "linear-gradient", "radial-gradient", "#b7f13d", "#3b82ff", "#d81b60"):
        assert removed_token not in html


def test_dashboard_uses_custom_select_controls():
    html = _dashboard_html()

    assert "custom-select" in html
    assert "select-trigger" in html
    assert "data-select-option" in html
    assert "function renderCustomSelect" in html
    assert "function chooseCustomSelectOption" in html
    assert "function handleCustomSelectKeydown" in html
    assert "<symbol id=\"icon-chevron\"" in html
    assert "<symbol id=\"icon-check\"" in html
    assert "<select class=\"control config-control\"" not in html


def test_dashboard_warns_before_using_mood_system():
    html = _dashboard_html()

    assert "modal-mood-warning" in html
    assert "role=\"dialog\"" in html
    assert "aria-modal=\"true\"" in html
    assert "aria-labelledby=\"mood-warning-title\"" in html
    assert "renderMoodWarningPanel" in html
    assert "maybeShowMoodGroupWarning" in html
    assert "focus({ preventScroll: true })" in html
    assert "trapMoodWarningFocus" in html
    assert "updateMoodRiskStatus" in html
    assert "确认开启情绪系统" in html
    assert "实验性功能 / 开发中" in html
    assert "每条消息最多会触发 3 次额外 LLM 调用" in html
    assert "path === 'mood_system_enabled'" in html
    assert "Promise.resolve(false)" in html
    assert "event.key === 'Escape'" in html
    assert "<symbol id=\"icon-alert\"" in html


def test_dashboard_session_actions_avoid_encoded_umo_and_native_confirm():
    html = _dashboard_html()

    assert "encodeURIComponent(umo)" not in html
    assert "encodeURIComponent(pendingUmo)" not in html
    assert "confirm(" not in html
    assert "modal-confirm" in html
    assert "function showConfirm" in html
    assert "resolveConfirm(false)" in html


def test_dashboard_declares_diagnostics_panel_and_api_calls():
    html = _dashboard_html()

    assert "diagnostics-widget" in html
    assert "diagnostics-panel" in html
    assert "diagnostics-dot" in html
    assert "function loadDiagnosticsSummary" in html
    assert "function loadDiagnosticsEvents" in html
    assert "function renderDiagnosticsSummary" in html
    assert "function renderDiagnosticsEvents" in html
    assert "function toggleDiagnosticsPanel" in html
    assert "function markDiagnosticsRead" in html
    assert "function clearDiagnostics" in html
    assert "AstrBotPluginPage.apiGet('diagnostics/summary')" in html
    assert "AstrBotPluginPage.apiGet('diagnostics', { limit: 20 })" in html
    assert "AstrBotPluginPage.apiPost('diagnostics/read', {})" in html
    assert "AstrBotPluginPage.apiPost('diagnostics/clear', {})" in html
    assert "setInterval(loadDiagnosticsSummary, 30000)" in html
    assert ".topbar {\n      position: relative;\n      z-index: 200;" in html
    assert ".diagnostics-panel {\n      position: fixed;\n      z-index: 1200;" in html
    assert "function positionDiagnosticsPanel()" in html
    assert "window.addEventListener('resize', positionDiagnosticsPanel)" in html
    assert "window.addEventListener('scroll', positionDiagnosticsPanel, true)" in html
    assert ".modal-overlay {\n      display: none;\n      position: fixed;" in html
    assert "padding: 20px;\n      z-index: 1400;\n      backdrop-filter: blur(10px);" in html
    assert "@keyframes pageEnter {\n      from { opacity: 0; }\n      to { opacity: 1; }" in html


def test_dashboard_normalizes_wrapped_and_unwrapped_bridge_responses():
    html = _dashboard_html()

    assert "function unwrapApiData(response)" in html
    assert "const sessionsData = unwrapApiData(sessionsRes)" in html
    assert "const data = unwrapApiData(response)" in html


# --------------------------------------------------------------------------- #
#  命令
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_period_set_uses_global_default_lengths(webui_plugin, mock_event):
    """省略可选长度时应使用当前全局默认值。"""
    webui_plugin.config["default_cycle_length"] = 30
    webui_plugin.config["default_period_length"] = 3

    await _collect_async_gen(
        webui_plugin.period_set(mock_event, "2026-05-25")
    )
    stored = await webui_plugin.store.get(mock_event.unified_msg_origin)

    assert stored["source"] == "manual"
    assert stored["cycle_length"] == 30
    assert stored["period_length"] == 3


@pytest.mark.asyncio
async def test_period_set_explicit_lengths_are_manual(webui_plugin, mock_event):
    """显式传入的命令长度应保存为手动独立配置。"""
    webui_plugin.config["default_period_length"] = 3

    await _collect_async_gen(
        webui_plugin.period_set(mock_event, "2026-05-25", 28, 5)
    )
    stored = await webui_plugin.store.get(mock_event.unified_msg_origin)

    assert stored["source"] == "manual"
    assert stored["cycle_length"] == 28
    assert stored["period_length"] == 5


@pytest.mark.asyncio
async def test_period_set_rejects_non_integer_lengths(webui_plugin, mock_event):
    """无效可选长度应返回友好的校验结果。"""
    await _collect_async_gen(
        webui_plugin.period_set(mock_event, "2026-05-25", "abc", 5)
    )

    mock_event.plain_result.assert_called_with("周期长度应在21至35天之间")
    assert await webui_plugin.store.get(mock_event.unified_msg_origin) is None
