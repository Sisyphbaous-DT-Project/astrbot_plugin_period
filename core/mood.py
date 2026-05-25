"""Mood state model - tracks active tools and interaction history."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MoodState:
    """Emotional state - no numerical scores, only active tools and history."""

    active_tools: list[dict] = field(default_factory=list)
    # active_tools item: {
    #   "name": str,
    #   "params": dict,
    #   "expires_at": str | None,   # ISO timestamp for time-limited tools
    #   "rounds_left": int | None,  # Remaining rounds for read_no_reply
    #   "initiated": bool,          # Whether initial message was sent (cold_violence)
    # }

    history: list[dict] = field(default_factory=list)
    # history item: {
    #   "timestamp": str,
    #   "event": str,          # e.g. "screen:yes", "tool:cold_violence", "lift:cold_violence"
    #   "reasoning": str,      # Decision reasoning
    #   "user_message": str,   # Trigger message summary
    # }

    last_interaction: str = ""  # ISO timestamp

    @classmethod
    def from_dict(cls, data: dict | None) -> "MoodState":
        """Deserialize from dict. Returns default state if data is None/empty."""
        if not data:
            return cls()
        return cls(
            active_tools=list(data.get("active_tools", [])),
            history=list(data.get("history", [])),
            last_interaction=data.get("last_interaction", ""),
        )

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "active_tools": list(self.active_tools),
            "history": list(self.history),
            "last_interaction": self.last_interaction,
        }

    def is_tool_active(self, name: str) -> bool:
        """Check if a tool with the given name is currently active."""
        return any(t.get("name") == name for t in self.active_tools)

    def expire_tools(self, now_iso: str) -> list[dict]:
        """Remove expired tools. Returns list of removed tools."""
        expired = []
        remaining = []
        for tool in self.active_tools:
            exp = tool.get("expires_at")
            if exp and isinstance(exp, str) and exp <= now_iso:
                expired.append(tool)
            else:
                remaining.append(tool)
        self.active_tools = remaining
        return expired

    def add_tool(
        self,
        name: str,
        params: dict,
        *,
        expires_at: str | None = None,
        rounds_left: int | None = None,
        initiated: bool = False,
    ) -> None:
        """Add or replace a tool. Removes any existing tool with the same name first."""
        self.remove_tool(name)
        self.active_tools.append({
            "name": name,
            "params": dict(params),
            "expires_at": expires_at,
            "rounds_left": rounds_left,
            "initiated": initiated,
        })

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name. Returns True if a tool was removed."""
        original_len = len(self.active_tools)
        self.active_tools = [t for t in self.active_tools if t.get("name") != name]
        return len(self.active_tools) < original_len

    def add_history(
        self,
        event: str,
        reasoning: str,
        user_message: str,
        max_length: int = 20,
    ) -> None:
        """Record a history entry."""
        from datetime import datetime, timezone

        self.history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "reasoning": reasoning,
            "user_message": user_message,
        })
        if len(self.history) > max_length:
            self.history = self.history[-max_length:]
