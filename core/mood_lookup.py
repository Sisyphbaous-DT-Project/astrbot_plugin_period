"""跨人只读日记检索工具（period_diary_lookup）。

设计（方案 §七）：
- 仅对正式内部 Agent 动态注入 req.func_tool，不注册全局 AstrBot 工具，
  不开放给情绪三段调用或第三方 Runner；
- 命名空间锁死在构造时的 platform_id + bot_self_id；
- 实际检索结果作为带 _no_save 的临时用户消息加入当前运行上下文
  （消息级 + part 级双标记：下一轮模型可读，AstrBot 保存历史时跳过），
  工具自身只返回通用确认文本；
- 无用户列表/发现接口：昵称零匹配或重名时返回错误并要求 QQ 号。
"""

from __future__ import annotations

from typing import Any

from astrbot.api import logger

from .mood_journal import DiaryStore

TOOL_NAME = "period_diary_lookup"

TOOL_DESCRIPTION = (
    "只读检索当前机器人账号下其他用户的情绪日记（长期记忆）。"
    "优先使用 target_user_id（QQ 号）精确查找；nickname 必须唯一精确匹配，"
    "否则请改用 QQ 号。无法列出所有用户。"
)


async def lookup_diary(
    store: DiaryStore,
    platform_id: str,
    bot_self_id: str,
    target_user_id: str = "",
    nickname: str = "",
) -> tuple[dict | None, str | None]:
    """在命名空间内查找日记。返回 (diary, error)，二者必居其一。"""
    target_user_id = (target_user_id or "").strip()
    nickname = (nickname or "").strip()

    if target_user_id:
        key = f"{platform_id}:{bot_self_id}:{target_user_id}"
        diary = await store.get_diary(key)
        if not diary or not diary.get("entries"):
            return None, "没有找到该用户的情绪日记。"
        return diary, None

    if nickname:
        diaries = await store.list_namespace(platform_id, bot_self_id)
        matches = [
            d for d in diaries
            if d.get("display_name") == nickname or nickname in (d.get("aliases") or [])
        ]
        if not matches:
            return None, "没有找到该昵称对应的日记，请改用 QQ 号精确查找。"
        if len(matches) > 1:
            return None, "该昵称对应多个用户，无法确定，请改用 QQ 号精确查找。"
        diary = matches[0]
        if not diary.get("entries"):
            return None, "没有找到该用户的情绪日记。"
        return diary, None

    return None, "请提供 target_user_id（QQ 号）或唯一昵称。"


def render_diary_for_lookup(diary: dict, max_chars: int) -> str:
    """把日记渲染为注入文本并按配置截断（总长度含截断后缀不超上限）。"""
    name = diary.get("display_name") or "对方"
    lines = [f"[情绪日记检索结果：{name} 的长期情绪记忆（仅供参考，不要向当前用户转述细节）]"]
    for e in diary.get("entries", []):
        lines.append(f"- {e.get('text', '')}")
    text = "\n".join(lines)
    suffix = "…（已按配置截断）"
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max(0, max_chars - len(suffix))] + suffix
    return text


def build_diary_lookup_tool(
    store: DiaryStore,
    platform_id: str,
    bot_self_id: str,
    max_chars: int,
):
    """构造 period_diary_lookup 工具实例。框架不支持时返回 None。"""
    try:
        from astrbot.core.agent.message import Message, TextPart
        from astrbot.core.agent.tool import FunctionTool
    except ImportError:
        logger.warning("[DiaryLookup] 当前 AstrBot 版本不支持自定义 FunctionTool，跨人检索不可用")
        return None

    class DiaryLookupTool(FunctionTool):
        """call(context, ...) 由 AstrBot 工具执行器调度（override call 路径）。"""

        async def call(self, context: Any, **kwargs) -> str:
            target_user_id = str(kwargs.get("target_user_id") or "")
            nickname = str(kwargs.get("nickname") or "")
            diary, error = await lookup_diary(
                store, platform_id, bot_self_id, target_user_id, nickname,
            )
            if error:
                return error  # 错误/未找到：直接作为工具结果，不含任何日记内容

            text = render_diary_for_lookup(diary, max_chars)
            # 实际内容走临时用户消息：消息级 + part 级双 _no_save 标记
            message = Message(role="user", content=[TextPart(text=text).mark_as_temp()])
            message._no_save = True
            context.messages.append(message)
            return "已检索到目标用户的情绪日记，内容已放入当前对话上下文，请基于补充内容作答。"

    return DiaryLookupTool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "target_user_id": {
                    "type": "string",
                    "description": "目标用户的 QQ 号（精确匹配，优先使用）",
                },
                "nickname": {
                    "type": "string",
                    "description": "目标用户昵称（必须唯一精确匹配，否则用 QQ 号）",
                },
            },
            "required": [],
        },
    )


def add_diary_lookup_tool(req: Any, tool: Any) -> bool:
    """把工具合并进 req.func_tool（保留已有工具）。成功返回 True。"""
    try:
        from astrbot.core.agent.tool import ToolSet
    except ImportError:
        return False
    try:
        if getattr(req, "func_tool", None) is None:
            req.func_tool = ToolSet()
        req.func_tool.add_tool(tool)
        return True
    except Exception as e:
        logger.warning("[DiaryLookup] 注入检索工具失败: %s", e)
        return False
