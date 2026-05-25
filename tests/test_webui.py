"""Tests for WebUI dashboard Web API handlers."""

import sys
from pathlib import Path

# Add parent dir so astrbot_plugin_period is importable as a package
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import pytest
from quart import request

from astrbot_plugin_period.main import PeriodPlugin


@pytest.fixture
def webui_plugin(tmp_path, sample_config, monkeypatch):
    """PeriodPlugin instance with WebUI-relevant config, isolated data dir."""
    config = dict(sample_config)
    # Keep default_anchor_date empty so not-found tests return 404
    from astrbot.api.star import Context, StarTools
    ctx = Context()
    monkeypatch.setattr(StarTools, "get_data_dir", lambda: tmp_path)
    plugin = PeriodPlugin(ctx, config)
    return plugin


def _unwrap(result):
    """Unwrap handler return value which may be (data, status) or just data."""
    if isinstance(result, tuple):
        return result[0], result[1] if len(result) > 1 else 200
    return result, 200


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
