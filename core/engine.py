"""Period cycle calculation engine (pure functions)."""

from dataclasses import dataclass
from typing import Literal
from datetime import datetime, timedelta


@dataclass
class PhaseInfo:
    """Information about the current cycle phase."""

    phase: Literal["menstrual", "follicular", "ovulatory", "luteal"]
    day: int              # Day within the current phase (1-indexed)
    days_to_next: int     # Days until next period starts
    total_day: int        # Day within the full cycle (1-indexed)


class CycleEngine:
    """Pure date calculation engine for menstrual cycle phases."""

    @staticmethod
    def get_phase(
        anchor_date: str,
        cycle_length: int = 28,
        period_length: int = 5,
        ovulation_day: int = 14,
        ovulation_window: int = 3,
        advance_days: int = 0,
    ) -> PhaseInfo:
        """Calculate current cycle phase based on anchor date.

        Args:
            anchor_date: The first day of last period, format "YYYY-MM-DD".
            cycle_length: Total cycle length in days (default 28).
            period_length: Menstrual period duration in days (default 5).
            ovulation_day: Day of ovulation within cycle (default 14).
            ovulation_window: Days around ovulation considered ovulatory (default 3).
            advance_days: Fast-forward days for debugging (default 0).

        Returns:
            PhaseInfo with current phase details.
        """
        anchor = datetime.strptime(anchor_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        
        # Apply advance days for debugging
        effective_today = today + timedelta(days=advance_days)
        
        # Calculate days since anchor, normalized to cycle
        days_diff = (effective_today - anchor).days
        
        # Handle cycles before anchor by using modulo
        if days_diff < 0:
            # Calculate how many full cycles back
            cycles_back = (-days_diff // cycle_length) + 1
            days_diff += cycles_back * cycle_length
        
        # Normalize to current cycle (1-indexed)
        total_day = (days_diff % cycle_length) + 1
        
        # Determine phase
        ovulation_half = (ovulation_window - 1) // 2
        ovulation_start = ovulation_day - ovulation_half
        ovulation_end = ovulation_start + ovulation_window - 1

        if total_day <= period_length:
            phase = "menstrual"
            day = total_day
        elif total_day < ovulation_start:
            phase = "follicular"
            day = total_day - period_length
        elif ovulation_start <= total_day <= ovulation_end:
            phase = "ovulatory"
            day = total_day - ovulation_start + 1
        else:
            phase = "luteal"
            day = total_day - ovulation_end
        
        # Days until next period
        if total_day <= period_length:
            days_to_next = 0  # Currently in period
        else:
            days_to_next = cycle_length - total_day + 1
        
        return PhaseInfo(
            phase=phase,
            day=day,
            days_to_next=days_to_next,
            total_day=total_day,
        )
