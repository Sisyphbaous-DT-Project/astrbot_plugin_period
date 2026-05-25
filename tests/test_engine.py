"""Tests for core/engine.py — CycleEngine date calculations."""

from datetime import datetime, timedelta
from freezegun import freeze_time
import pytest

from core.engine import CycleEngine, PhaseInfo


class TestCycleEngineBasic:
    """Unit tests for standard cycle calculations."""

    def test_menstrual_phase_day1(self):
        """Anchor is today → should be menstrual day 1."""
        with freeze_time("2026-05-25"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=28, period_length=5, ovulation_day=14)
        assert info.phase == "menstrual"
        assert info.day == 1
        assert info.total_day == 1
        assert info.days_to_next == 0

    def test_menstrual_phase_day5(self):
        """4 days after anchor → menstrual day 5."""
        with freeze_time("2026-05-29"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=28, period_length=5, ovulation_day=14)
        assert info.phase == "menstrual"
        assert info.day == 5
        assert info.total_day == 5

    def test_follicular_phase(self):
        """Day 6 → follicular day 1."""
        with freeze_time("2026-05-30"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=28, period_length=5, ovulation_day=14)
        assert info.phase == "follicular"
        assert info.day == 1
        assert info.total_day == 6

    def test_ovulatory_phase(self):
        """Day 13 → ovulatory day 1 (window starts at ovulation_day - 1)."""
        with freeze_time("2026-06-06"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=28, period_length=5, ovulation_day=14)
        assert info.phase == "ovulatory"
        assert info.day == 1
        assert info.total_day == 13

    def test_ovulatory_phase_day3(self):
        """Day 15 → ovulatory day 3."""
        with freeze_time("2026-06-08"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=28, period_length=5, ovulation_day=14)
        assert info.phase == "ovulatory"
        assert info.day == 3
        assert info.total_day == 15

    def test_luteal_phase(self):
        """Day 16 → luteal day 1."""
        with freeze_time("2026-06-09"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=28, period_length=5, ovulation_day=14)
        assert info.phase == "luteal"
        assert info.day == 1
        assert info.total_day == 16

    def test_days_to_next_in_luteal(self):
        """Luteal phase should report days until next period."""
        with freeze_time("2026-06-10"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=28, period_length=5, ovulation_day=14)
        # Day 17 of 28, next period starts day 29 → 12 days remaining
        assert info.days_to_next == 12

    def test_full_cycle_rollover(self):
        """Day 29 (next cycle day 1) should be menstrual."""
        with freeze_time("2026-06-22"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=28, period_length=5, ovulation_day=14)
        assert info.phase == "menstrual"
        assert info.day == 1
        assert info.total_day == 1


class TestCycleEngineAdvance:
    """Tests for advance_days debug feature."""

    def test_advance_3_days(self):
        """Advance moves the effective date forward."""
        with freeze_time("2026-05-25"):
            info = CycleEngine.get_phase("2026-05-25", advance_days=3)
        assert info.phase == "menstrual"
        assert info.day == 4

    def test_advance_into_next_phase(self):
        """Large advance can cross phase boundaries."""
        with freeze_time("2026-05-25"):
            info = CycleEngine.get_phase("2026-05-25", advance_days=10)
        assert info.phase == "follicular"


class TestCycleEngineEdgeCases:
    """Boundary and edge-case tests."""

    def test_short_cycle_21_days(self):
        """Minimum realistic cycle length."""
        with freeze_time("2026-06-04"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=21, period_length=3, ovulation_day=10)
        assert info.total_day == 11

    def test_long_cycle_35_days(self):
        """Maximum realistic cycle length."""
        with freeze_time("2026-06-28"):
            info = CycleEngine.get_phase("2026-05-25", cycle_length=35, period_length=7, ovulation_day=18)
        assert info.total_day == 35

    def test_anchor_in_future(self):
        """Anchor date after today should still compute correctly via modulo."""
        with freeze_time("2026-05-25"):
            info = CycleEngine.get_phase("2026-06-01", cycle_length=28)
        # 7 days before anchor → modulo wraps to day 22 of previous cycle
        assert info.total_day == 22

    def test_cross_year_boundary(self):
        """Cycle calculations must work across December-January."""
        with freeze_time("2026-01-05"):
            info = CycleEngine.get_phase("2025-12-25", cycle_length=28)
        assert info.total_day == 12

    def test_zero_days_diff(self):
        """Anchor equals today exactly."""
        with freeze_time("2026-05-25"):
            info = CycleEngine.get_phase("2026-05-25")
        assert info.total_day == 1
        assert info.phase == "menstrual"

    def test_phase_transition_boundaries(self):
        """Verify exact boundary days between phases."""
        with freeze_time("2026-06-08"):
            # Day 15 = ovulatory day 2
            info = CycleEngine.get_phase("2026-05-25")
        assert info.phase == "ovulatory"

        with freeze_time("2026-06-11"):
            # Day 18 = luteal day 1
            info = CycleEngine.get_phase("2026-05-25")
        assert info.phase == "luteal"

    def test_custom_ovulation_window(self):
        """ovulation_window parameter actually changes the ovulatory range."""
        with freeze_time("2026-06-10"):
            # Day 17 with default window 3 → luteal (window was 13-15)
            info = CycleEngine.get_phase("2026-05-25", ovulation_window=3)
        assert info.phase == "luteal"

        with freeze_time("2026-06-10"):
            # Day 17 with window 7 → ovulatory (window is 11-17)
            info = CycleEngine.get_phase("2026-05-25", ovulation_window=7)
        assert info.phase == "ovulatory"
        assert info.day == 7  # 17 - 11 + 1 = 7

    def test_even_ovulation_window(self):
        """Even ovulation_window values must produce exactly that many days."""
        with freeze_time("2026-06-09"):
            # Day 16, window 4 → ovulatory (window 13-16, 4 days)
            info = CycleEngine.get_phase("2026-05-25", ovulation_window=4)
        assert info.phase == "ovulatory"
        assert info.day == 4  # 16 - 13 + 1 = 4

        with freeze_time("2026-06-12"):
            # Day 19, window 4 → luteal (window ended at 16)
            info = CycleEngine.get_phase("2026-05-25", ovulation_window=4)
        assert info.phase == "luteal"
