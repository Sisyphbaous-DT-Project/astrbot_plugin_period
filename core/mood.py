"""Mood state model for the emotion management system."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MoodState:
    """Emotional state for a single UMO session."""

    mood_score: float = 0.0
    """-10 ~ 10, higher is better."""

    energy: float = 5.0
    """0 ~ 10."""

    intimacy: float = 5.0
    """0 ~ 10, closeness with the current user."""

    dominant_emotion: Literal[
        "happy", "calm", "irritable", "depressed", "angry", "playful"
    ] = "calm"

    active_tools: list[dict] = field(default_factory=list)
    """Each item: {
        "name": str,
        "expires_at": str | None,   # ISO timestamp for time-based tools
        "params": dict,
        "rounds_left": int | None,  # For read_no_reply
        "initiated": bool,          # For cold_violence (has sent initial message)
    }
    """

    history: list[dict] = field(default_factory=list)
    """Each item: {
        "timestamp": str,
        "event": str,
        "mood_change": float,
        "tool_used": str | None,
        "user_message": str,
    }
    """

    last_interaction: str = ""
    """ISO timestamp."""

    consecutive_unpleasant: int = 0
    """Counter for successive negative interactions."""

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON storage."""
        return {
            "mood_score": self.mood_score,
            "energy": self.energy,
            "intimacy": self.intimacy,
            "dominant_emotion": self.dominant_emotion,
            "active_tools": copy.deepcopy(self.active_tools),
            "history": copy.deepcopy(self.history),
            "last_interaction": self.last_interaction,
            "consecutive_unpleasant": self.consecutive_unpleasant,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> MoodState | None:
        """Deserialize from plain dict."""
        if data is None:
            return None
        if not isinstance(data, dict):
            return None
        return cls(
            mood_score=float(data.get("mood_score", 0.0)),
            energy=float(data.get("energy", 5.0)),
            intimacy=float(data.get("intimacy", 5.0)),
            dominant_emotion=data.get("dominant_emotion", "calm"),
            active_tools=list(data.get("active_tools", [])),
            history=list(data.get("history", [])),
            last_interaction=data.get("last_interaction", ""),
            consecutive_unpleasant=int(data.get("consecutive_unpleasant", 0)),
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def is_tool_active(self, name: str) -> bool:
        """Check whether a given tool is currently active."""
        return any(t["name"] == name for t in self.active_tools)

    def get_active_tool(self, name: str) -> dict | None:
        """Return the active tool dict by name, or None."""
        for t in self.active_tools:
            if t["name"] == name:
                return t
        return None

    def expire_tools(self, now_iso: str) -> list[dict]:
        """Remove expired tools and return the removed items.

        A tool is expired when:
        - it has an "expires_at" and now_iso >= expires_at, or
        - it is "read_no_reply" with rounds_left <= 0.
        """
        remaining: list[dict] = []
        expired: list[dict] = []
        for t in self.active_tools:
            if t.get("expires_at") and now_iso >= t["expires_at"]:
                expired.append(t)
                continue
            if t["name"] == "read_no_reply" and t.get("rounds_left", 1) <= 0:
                expired.append(t)
                continue
            remaining.append(t)
        self.active_tools = remaining
        return expired

    def clamp(self) -> None:
        """Clamp all numeric fields to their valid ranges."""
        self.mood_score = max(-10.0, min(10.0, self.mood_score))
        self.energy = max(0.0, min(10.0, self.energy))
        self.intimacy = max(0.0, min(10.0, self.intimacy))

    def add_history(
        self,
        event: str,
        mood_change: float = 0.0,
        tool_used: str | None = None,
        user_message: str = "",
        max_length: int = 10,
    ) -> None:
        """Append a history entry and trim to max_length."""
        from datetime import datetime

        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": event,
                "mood_change": mood_change,
                "tool_used": tool_used,
                "user_message": user_message,
            }
        )
        if len(self.history) > max_length:
            self.history = self.history[-max_length:]
