"""Tests for WebUI dashboard Web API handlers."""

import sys
from pathlib import Path

# Add parent dir so astrbot_plugin_period is importable as a package
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import pytest
from freezegun import freeze_time
from quart import request

from astrbot_plugin_period.main import PeriodPlugin


@pytest.fixture
def webui_plugin(tmp_path, sample_config, monkeypatch):
    """PeriodPlugin instance with WebUI-relevant config, isolated data dir."""
    config = dict(sample_config)
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
    routes = {r[0] for r in webui_plugin.context.registered_web_apis}
    base = "/astrbot_plugin_period"
    assert f"{base}/sessions" in routes
    assert f"{base}/config" in routes
    assert f"{base}/sessions/<umo>/toggle" in routes
    assert f"{base}/sessions/<umo>/advance" in routes
    assert f"{base}/sessions/<umo>/anchor" in routes
    assert f"{base}/sessions/<umo>/delete" in routes


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
