"""Data models for user inputs and the generated workout plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Goal(str, Enum):
    BUILD_MUSCLE = "Build muscle"
    LOSE_FAT = "Lose fat"
    GENERAL_FITNESS = "General fitness"
    IMPROVE_ENDURANCE = "Improve endurance"


class Experience(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class Equipment(str, Enum):
    NONE = "No equipment"
    DUMBBELLS = "Home dumbbells"
    FULL_GYM = "Full gym"


@dataclass(frozen=True)
class UserProfile:
    """Validated, structured inputs collected from the UI."""

    goal: Goal
    experience: Experience
    days_per_week: int
    equipment: tuple[Equipment, ...]
    session_minutes: int = 45
    limitations: str = ""

    @property
    def has_limitations(self) -> bool:
        return bool(self.limitations.strip())


@dataclass
class Exercise:
    name: str
    sets: str
    reps: str
    rest: str = ""
    notes: str = ""


@dataclass
class WorkoutDay:
    day: int
    focus: str
    warmup: str
    exercises: list[Exercise]
    cooldown: str = ""


@dataclass
class WorkoutPlan:
    title: str
    summary: str
    days: list[WorkoutDay]
    progression: str = ""
    rest_day_guidance: str = ""
    limitation_adjustments: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InputValidationError(ValueError):
    """Raised when the user's inputs can't be turned into a valid profile."""


class PlanGenerationError(RuntimeError):
    """Raised with a user-friendly message when a plan can't be produced."""
