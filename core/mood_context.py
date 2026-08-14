"""请求上下文解析、门禁纯函数与安全注入器。

对齐 AstrBot 4.27.2 的真实历史形态：
- contexts 条目为 OpenAI 兼容 dict（也可能是个别路径传入的 Message 对象）；
- role 可能是 system/user/assistant/tool/_checkpoint；
- content 可能是字符串或多段 part 列表（text/think/image_url 等）；
- 临时内容带 _no_save 标记（消息级或 part 级），不得进入情绪上下文。
"""

from __future__ import annotations

from typing import Any

from astrbot.core.agent.message import TextPart

# 情绪状态/日记允许的注入位置；fake_tool_call 与 user_message_before 已废弃，
# 统一降级为 extra_user_content_parts（user_message_before 会改写 req.prompt
# 并被 AstrBot 写入普通聊天历史，违反临时注入硬约束，故移除）。
SAFE_INJECT_LOCATIONS = (
    "extra_user_content_parts",
    "system_prompt_append",
)
DEFAULT_INJECT_LOCATION = "extra_user_content_parts"


# --------------------------------------------------------------------------- #
#  真实历史解析
# --------------------------------------------------------------------------- #

def _part_text(part: Any) -> str | None:
    """从单个 content part 提取文本；非文本或临时 part 返回 None。"""
    if isinstance(part, dict):
        if part.get("_no_save"):
            return None
        if part.get("type") == "text":
            text = part.get("text")
            return text if isinstance(text, str) else None
        return None
    # astrbot Message part 对象（如 TextPart）
    if getattr(part, "_no_save", False):
        return None
    if part.__class__.__name__ == "TextPart":
        text = getattr(part, "text", None)
        return text if isinstance(text, str) else None
    return None


def extract_text_content(content: Any) -> str:
    """把 str 或 part 列表形态的 content 规整为纯文本。

    多段内容只拼接 text part；忽略 think、图片、音频等正文。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [t for t in (_part_text(p) for p in content) if t]
        return "\n".join(texts)
    return ""


def _entry_role_content(entry: Any) -> tuple[str, Any, bool]:
    """从 dict 或 Message 对象中取 (role, content, no_save)。"""
    if isinstance(entry, dict):
        return (
            str(entry.get("role", "")),
            entry.get("content", ""),
            bool(entry.get("_no_save")),
        )
    if hasattr(entry, "role") and hasattr(entry, "content"):
        return (
            str(getattr(entry, "role", "") or ""),
            getattr(entry, "content", ""),
            bool(getattr(entry, "_no_save", False)),
        )
    return "", "", False


def parse_history(contexts: Any, limit: int) -> list[dict]:
    """从 req.contexts 提取最近 user/assistant 文本历史。

    - 跳过 system/tool/_checkpoint 角色与 _no_save 消息；
    - 多段 content 只保留 text part；
    - 空文本消息跳过；
    - 按最终提取出的条数从尾部裁剪到 limit（0 表示不携带历史）；
    - 当前轮用户消息不在 contexts 内，由调用方单独传入，本函数不去重。

    返回 [{"role": "user"|"assistant", "content": str}, ...]
    """
    if limit <= 0 or not contexts:
        return []

    history: list[dict] = []
    for entry in contexts:
        role, content, no_save = _entry_role_content(entry)
        if no_save or role not in ("user", "assistant"):
            continue
        text = extract_text_content(content).strip()
        if not text:
            continue
        history.append({"role": role, "content": text})

    return history[-limit:]


def history_to_contexts(history: list[dict]) -> list[dict]:
    """把解析后的精简历史还原为 OpenAI 兼容 contexts（供 text_chat(contexts=...)）。"""
    return [{"role": h["role"], "content": h["content"]} for h in history]


# --------------------------------------------------------------------------- #
#  门禁纯函数
# --------------------------------------------------------------------------- #

def is_umo_allowed(config: dict, umo: str) -> bool:
    """UMO 白/黑名单过滤（global_inject/umo_list/umo_mode）。"""
    if config.get("global_inject", False):
        return True
    umo_list = config.get("umo_list", []) or []
    umo_mode = config.get("umo_mode", "whitelist")
    if umo_mode == "blacklist":
        return umo not in umo_list
    return umo in umo_list


def should_show_body_hint(
    config: dict,
    umo: str,
    message_str: str,
    warmup_counters: dict,
    inject_counters: dict,
) -> bool:
    """身体状态提示的展示门禁：warmup_rounds / inject_mode / 触发关键词。

    只控制身体提示是否展示，不得用于阻断情绪系统。会推进计数器。
    """
    warmup = config.get("warmup_rounds", 0)
    if warmup > 0:
        count = warmup_counters.get(umo, 0) + 1
        warmup_counters[umo] = count
        if count <= warmup:
            return False

    mode = config.get("inject_mode", "every_request")
    if mode == "only_status":
        return False
    if mode == "interval_3":
        count = inject_counters.get(umo, 0) + 1
        inject_counters[umo] = count
        if count % 3 != 1:  # 第 1、4、7... 次请求注入
            return False
    elif mode == "on_trigger":
        keywords = config.get(
            "trigger_keywords",
            ["怎么了", "还好吗", "不舒服", "心情不好", "你没事吧"],
        )
        if not any(kw in message_str for kw in keywords):
            return False
    return True


# --------------------------------------------------------------------------- #
#  安全注入器
# --------------------------------------------------------------------------- #

def normalize_inject_location(value: Any) -> tuple[str, bool]:
    """把配置值规整为安全位置。

    返回 (location, downgraded)；fake_tool_call、user_message_before 及未知值
    降级为 extra_user_content_parts，downgraded=True 供调用方记一次诊断。
    """
    if value in SAFE_INJECT_LOCATIONS:
        return value, False
    return DEFAULT_INJECT_LOCATION, True


def apply_injection(req: Any, text: str, location: str) -> None:
    """按位置把文本注入 ProviderRequest。

    - extra_user_content_parts：临时 TextPart（mark_as_temp），不落历史；
    - system_prompt_append：追加到 system_prompt（不写入单条历史）。
    """
    if not text:
        return
    if location == "system_prompt_append":
        sep = "\n\n" if req.system_prompt else ""
        req.system_prompt = (req.system_prompt or "") + sep + text
    else:
        req.extra_user_content_parts.append(TextPart(text=text).mark_as_temp())
