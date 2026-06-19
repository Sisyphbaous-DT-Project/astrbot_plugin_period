"""Tests for core/diagnostics.py."""

import json
import sys
from pathlib import Path

import pytest

_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from astrbot_plugin_period.core.diagnostics import DiagnosticsStore


@pytest.mark.asyncio
async def test_diagnostics_records_summary_list_mark_read_and_clear(tmp_path):
    store = DiagnosticsStore({"diagnostics_max_entries": 20}, tmp_path)

    await store.record_warning(
        "配置提醒",
        "warning detail",
        source="config",
        context={"prompt": "x" * 400, "count": 2},
    )
    await store.record_error("运行错误", ValueError("secret message"), source="runtime")

    summary = await store.get_summary()
    assert summary["status"] == "error"
    assert summary["unread_warning_count"] == 1
    assert summary["unread_error_count"] == 1
    assert summary["unread_count"] == 2
    assert summary["total"] == 2

    events = await store.list_events()
    assert [event["level"] for event in events] == ["error", "warning"]
    assert events[0]["message"] == "ValueError occurred"
    assert events[1]["context"]["prompt"] == "[redacted]"
    assert events[1]["context"]["count"] == 2

    count, saved = await store.mark_read(ids=[events[0]["id"]])
    assert saved is True
    assert count == 1
    summary = await store.get_summary()
    assert summary["unread_error_count"] == 0
    assert summary["unread_warning_count"] == 1

    count, saved = await store.mark_read()
    assert saved is True
    assert count == 1
    summary = await store.get_summary()
    assert summary["status"] == "ok"
    assert summary["unread_count"] == 0

    count, saved = await store.clear()
    assert saved is True
    assert count == 2
    assert await store.list_events() == []


@pytest.mark.asyncio
async def test_diagnostics_redacts_sensitive_messages_and_context(tmp_path):
    store = DiagnosticsStore({"diagnostics_max_entries": 20}, tmp_path)

    await store.record_warning(
        "敏感提醒",
        "api_key=sk-test system_prompt=\"一段很长的人设提示词内容\"",
        source="test",
        context={
            "api_key": "sk-secret",
            "provider_config": {"token": "secret"},
            "safe_count": 3,
            "note": "token=abc123",
        },
    )
    await store.record_warning(
        "异常对象",
        RuntimeError("password=abc123"),
        source="test",
    )

    events = await store.list_events()

    assert events[0]["message"] == "RuntimeError occurred"
    assert "abc123" not in events[1]["message"]
    assert "一段很长的人设提示词内容" not in events[1]["message"]
    assert events[1]["context"]["api_key"] == "[redacted]"
    assert events[1]["context"]["provider_config"] == "[redacted]"
    assert events[1]["context"]["safe_count"] == 3
    assert "abc123" not in events[1]["context"]["note"]


@pytest.mark.asyncio
async def test_diagnostics_trims_to_max_entries_and_loads(tmp_path):
    store = DiagnosticsStore({"diagnostics_max_entries": 21}, tmp_path)

    for index in range(25):
        await store.record_warning(f"事件 {index}", "message", source="test")

    events = await store.list_events(limit=100)
    assert len(events) == 21
    assert events[0]["title"] == "事件 24"
    assert events[-1]["title"] == "事件 4"

    reloaded = DiagnosticsStore({"diagnostics_max_entries": 21}, tmp_path)
    await reloaded.load()
    reloaded_events = await reloaded.list_events(limit=100)
    assert [event["title"] for event in reloaded_events] == [
        event["title"] for event in events
    ]


@pytest.mark.asyncio
async def test_diagnostics_ignores_invalid_file(tmp_path):
    (tmp_path / "diagnostics.json").write_text("{bad json", encoding="utf-8")
    store = DiagnosticsStore({}, tmp_path)

    await store.load()

    assert await store.list_events() == []


@pytest.mark.asyncio
async def test_diagnostics_loads_only_dict_events(tmp_path):
    (tmp_path / "diagnostics.json").write_text(
        json.dumps({"events": [{"level": "warning"}, "bad"]}),
        encoding="utf-8",
    )
    store = DiagnosticsStore({}, tmp_path)

    await store.load()

    assert await store.list_events() == [{"level": "warning"}]
