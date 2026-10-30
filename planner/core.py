"""Core logic: input validation, Groq call, response parsing and constraint checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .models import (
    Equipment,
    Exercise,
    Experience,
    Goal,
    InputValidationError,
    PlanGenerationError,
    UserProfile,
    WorkoutDay,
    WorkoutPlan,
)
from .prompts import (
    PLAN_JSON_SCHEMA,
    SWAP_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_repair_prompt,
    build_swap_prompt,
    build_user_prompt,
)

# llama-3.3-70b-versatile was retired by Groq in Aug 2026; gpt-oss-120b is the recommended replacement.
DEFAULT_MODEL = "openai/gpt-oss-120b"
# Groq counts prompt + max_completion_tokens against tokens-per-minute up front, and the free tier
# allows 8000 TPM, so a larger request is rejected with HTTP 413 before the model even runs.
MAX_REQUEST_TOKENS = 7500
MIN_COMPLETION_TOKENS = 2000
MIN_DAYS, MAX_DAYS = 1, 7
MIN_MINUTES, MAX_MINUTES = 15, 180

INJURY_DISCLAIMER = (
    "You mentioned an injury or limitation. This plan is general fitness guidance, not medical "
    "advice. Please get it cleared by a doctor or physiotherapist, and stop any exercise that "
    "causes pain."
)
GENERAL_DISCLAIMER = (
    "General fitness guidance only. Check with a healthcare professional before starting a new "
    "exercise programme, especially if you have a medical condition."
)


@dataclass
class GenerationResult:
    plan: WorkoutPlan
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1


# --------------------------------------------------------------------------- inputs

def _enum_value(enum_cls: type, value: Any, label: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        options = ", ".join(e.value for e in enum_cls)
        raise InputValidationError(f"Please choose a {label} ({options}).") from exc


def validate_inputs(
    goal: str | Goal | None,
    experience: str | Experience | None,
    days_per_week: int | float | None,
    equipment: Iterable[str | Equipment] | None,
    limitations: str | None = "",
    session_minutes: int | float | None = 45,
) -> UserProfile:
    """Validate raw UI inputs and return a UserProfile, or raise InputValidationError."""
    if goal in (None, ""):
        raise InputValidationError("Please choose a fitness goal.")
    if experience in (None, ""):
        raise InputValidationError("Please choose your experience level.")
    goal_enum = _enum_value(Goal, goal, "fitness goal")
    exp_enum = _enum_value(Experience, experience, "experience level")

    if days_per_week is None:
        raise InputValidationError("Please choose how many days per week you can train.")
    try:
        days = int(days_per_week)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError("Days per week must be a whole number.") from exc
    if days != days_per_week or not MIN_DAYS <= days <= MAX_DAYS:
        raise InputValidationError(
            f"Days per week must be between {MIN_DAYS} and {MAX_DAYS}. "
            "Even one day a week is a great start!"
        )

    equipment_list = [_enum_value(Equipment, e, "equipment option") for e in (equipment or [])]
    if not equipment_list:
        raise InputValidationError(
            "Please select at least one equipment option (choose 'No equipment' for bodyweight)."
        )

    try:
        minutes = int(session_minutes if session_minutes is not None else 45)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError("Session length must be a number of minutes.") from exc
    if not MIN_MINUTES <= minutes <= MAX_MINUTES:
        raise InputValidationError(
            f"Session length must be between {MIN_MINUTES} and {MAX_MINUTES} minutes."
        )

    return UserProfile(
        goal=goal_enum,
        experience=exp_enum,
        days_per_week=days,
        equipment=tuple(dict.fromkeys(equipment_list)),  # dedupe, keep order
        session_minutes=minutes,
        limitations=(limitations or "").strip(),
    )


# --------------------------------------------------------------------------- parsing

def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of an LLM response (tolerates code fences / stray prose)."""
    if not text or not text.strip():
        raise ValueError("empty response")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found") from None
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def _s(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(_s(v) for v in value)
    return str(value).strip()


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return [_s(value)] if _s(value) else []
    return [_s(v) for v in value if _s(v)]


def parse_exercise(raw: Any) -> Exercise | None:
    if not isinstance(raw, dict) or not _s(raw.get("name")):
        return None
    return Exercise(
        name=_s(raw.get("name")),
        sets=_s(raw.get("sets")) or "-",
        reps=_s(raw.get("reps")) or "-",
        rest=_s(raw.get("rest")),
        notes=_s(raw.get("notes")),
    )


def parse_plan(data: dict[str, Any]) -> WorkoutPlan:
    """Convert loosely-typed model JSON into a WorkoutPlan; raise ValueError if unusable."""
    raw_days = data.get("days") or data.get("weekly_schedule")
    if not isinstance(raw_days, list) or not raw_days:
        raise ValueError("plan has no days")

    days: list[WorkoutDay] = []
    for raw in raw_days:
        if not isinstance(raw, dict):
            continue
        raw_exercises = raw.get("exercises")
        if not isinstance(raw_exercises, list):
            continue
        exercises = [ex for ex in map(parse_exercise, raw_exercises) if ex]
        if not exercises:
            continue
        days.append(
            WorkoutDay(
                day=len(days) + 1,  # renumber so days are always 1..N
                focus=_s(raw.get("focus")) or f"Day {len(days) + 1}",
                warmup=_s(raw.get("warmup")),
                exercises=exercises,
                cooldown=_s(raw.get("cooldown")),
            )
        )
    if not days:
        raise ValueError("no day contains valid exercises")

    return WorkoutPlan(
        title=_s(data.get("title")) or "Your Weekly Workout Plan",
        summary=_s(data.get("summary")),
        days=days,
        progression=_s(data.get("progression")),
        rest_day_guidance=_s(data.get("rest_day_guidance")),
        limitation_adjustments=_str_list(data.get("limitation_adjustments")),
        safety_notes=_str_list(data.get("safety_notes")),
    )


# --------------------------------------------------------------------------- constraint checks

_GYM_ONLY = [
    r"barbell", r"\bcable", r"machine", r"smith", r"lat pull-?down", r"leg press",
    r"leg extension", r"leg curl machine", r"pec deck", r"\bsled\b", r"trap bar", r"ez[- ]bar",
    r"t-bar", r"hack squat", r"kettlebell", r"\bband(ed)?\b", r"resistance band",
    r"pull-?ups?\b", r"chin-?ups?\b", r"treadmill", r"\brower\b", r"\berg\b", r"\bbike\b",
    r"\bbench\b", r"ab wheel", r"\btrx\b", r"suspension", r"medicine ball",
    r"(jump|skipping) rope",
]
_DUMBBELL = [r"dumbbells?", r"\bdb\b"]


def forbidden_patterns(equipment: tuple[Equipment, ...]) -> list[str]:
    if Equipment.FULL_GYM in equipment:
        return []
    if Equipment.DUMBBELLS in equipment:
        return _GYM_ONLY
    return _GYM_ONLY + _DUMBBELL


def find_violations(plan: WorkoutPlan, profile: UserProfile) -> list[str]:
    """Deterministic checks for the constraints the model most often ignores."""
    problems: list[str] = []
    if len(plan.days) != profile.days_per_week:
        problems.append(
            f"The client asked for exactly {profile.days_per_week} training days, "
            f"but the plan has {len(plan.days)}."
        )
    patterns = forbidden_patterns(profile.equipment)
    for day in plan.days:
        for ex in day.exercises:
            if any(re.search(p, ex.name, re.IGNORECASE) for p in patterns):
                problems.append(
                    f"Day {day.day}: '{ex.name}' needs equipment the client does not have "
                    f"(available: {', '.join(e.value for e in profile.equipment)})."
                )
    if profile.has_limitations and not plan.limitation_adjustments:
        problems.append(
            "The client listed limitations but 'limitation_adjustments' is empty - explain "
            "what you avoided and what you substituted."
        )
    return problems


# --------------------------------------------------------------------------- Groq

def _friendly_api_error(exc: Exception) -> PlanGenerationError:
    """Map Groq / network exceptions to messages a user can act on."""
    name = type(exc).__name__
    text = str(exc).lower()
    if name == "AuthenticationError" or "invalid api key" in text:
        msg = "Your Groq API key was rejected. Check that it is correct and still active."
    elif name == "PermissionDeniedError":
        msg = "This Groq API key doesn't have access to the selected model."
    elif name == "RateLimitError" or getattr(exc, "status_code", 0) == 413 or "tokens per minute" in text:
        msg = "Groq's rate limit was hit. Wait a minute and try again."
    elif name in ("APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError"):
        msg = "Couldn't reach Groq. Check your internet connection and try again."
    elif name == "NotFoundError" or "decommissioned" in text or "model_not_found" in text:
        msg = "The selected model isn't available on Groq any more. Pick another model in the sidebar."
    elif name == "InternalServerError" or getattr(exc, "status_code", 0) in (500, 502, 503):
        msg = "Groq is having trouble right now. Please try again shortly."
    else:
        msg = "Something went wrong while generating your plan. Please try again."
    return PlanGenerationError(msg)


def make_client(api_key: str) -> Any:
    if not api_key or not api_key.strip():
        raise PlanGenerationError(
            "No Groq API key found. Add GROQ_API_KEY to .streamlit/secrets.toml or your "
            "environment, or paste it in the sidebar."
        )
    try:
        from groq import Groq  # imported lazily so the rest of the module is testable offline
    except ImportError as exc:
        raise PlanGenerationError("The 'groq' package isn't installed. Run: pip install -r requirements.txt") from exc
    return Groq(api_key=api_key.strip(), timeout=60.0, max_retries=2)


def _chat(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int = 8000,
    schema: dict[str, Any] | None = None,
) -> str:
    """Single chat-completion call returning the message text (may be empty)."""
    prompt_tokens = sum(len(m["content"]) for m in messages) // 4  # rough estimate
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": min(max_tokens, max(MIN_COMPLETION_TOKENS, MAX_REQUEST_TOKENS - prompt_tokens)),
        "response_format": {"type": "json_object"},
    }
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "medium"
        if schema:  # Groq's strict structured outputs only cover the gpt-oss models
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "workout_plan", "strict": True, "schema": schema},
            }
    response = client.chat.completions.create(**kwargs)
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return (getattr(choices[0].message, "content", None) or "").strip()


def _is_json_mode_failure(exc: Exception) -> bool:
    # Groq returns HTTP 400 "json_validate_failed" when the model emits invalid JSON in JSON mode.
    return "json_validate_failed" in str(exc).lower()


def request_plan(
    profile: UserProfile,
    client: Any,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_attempts: int = 2,
) -> GenerationResult:
    """Call the LLM, validate the plan, and send one repair round if constraints are broken."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(profile)},
    ]
    best: WorkoutPlan | None = None
    best_problems: list[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            content = _chat(client, model, messages, temperature, schema=PLAN_JSON_SCHEMA)
        except PlanGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - every API failure becomes a friendly message
            if _is_json_mode_failure(exc) and attempt < max_attempts:
                continue
            if best is not None:  # a flawed plan with warnings beats losing it to a failed repair
                break
            raise _friendly_api_error(exc) from exc

        try:
            plan = parse_plan(extract_json(content))
        except (ValueError, json.JSONDecodeError):
            if attempt < max_attempts:
                messages.append({"role": "assistant", "content": content[:4000]})
                messages.append({
                    "role": "user",
                    "content": "That was not a valid plan object. Return ONLY the JSON object in the required shape.",
                })
                continue
            break

        problems = find_violations(plan, profile)
        if best is None or len(problems) < len(best_problems):
            best, best_problems = plan, problems
        if not problems:
            return GenerationResult(plan=plan, attempts=attempt)
        if attempt < max_attempts:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": build_repair_prompt(problems)})

    if best is None:
        raise PlanGenerationError(
            "The AI returned an empty or unreadable plan. Please click Generate again - "
            "it usually works on the next try."
        )
    return GenerationResult(plan=best, warnings=best_problems, attempts=max_attempts)


def generate_workout_plan(
    goal: str,
    experience: str,
    days_per_week: int,
    equipment: list[str],
    limitations: str = "",
    session_minutes: int = 45,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    client: Any | None = None,
) -> GenerationResult:
    """Validate inputs, build the prompt, call Groq and return a parsed, checked plan.

    Raises InputValidationError for bad inputs and PlanGenerationError (with a message that
    is safe to show the user) for API failures or empty/malformed responses.
    """
    profile = validate_inputs(goal, experience, days_per_week, equipment, limitations, session_minutes)
    try:
        llm = client or make_client(api_key)
        return request_plan(profile, llm, model=model, temperature=temperature)
    except (InputValidationError, PlanGenerationError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise _friendly_api_error(exc) from exc


def swap_exercise(
    profile: UserProfile,
    plan: WorkoutPlan,
    day_index: int,
    exercise_index: int,
    reason: str = "",
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> Exercise:
    """Ask the LLM for one replacement exercise that still respects the constraints."""
    try:
        day = plan.days[day_index]
        current = day.exercises[exercise_index]
    except IndexError as exc:
        raise InputValidationError("Please pick an exercise to swap.") from exc

    others = [e.name for i, e in enumerate(day.exercises) if i != exercise_index]
    taken = {name.lower() for name in [current.name, *others]}
    patterns = forbidden_patterns(profile.equipment)
    messages = [
        {"role": "system", "content": SWAP_SYSTEM_PROMPT},
        {"role": "user", "content": build_swap_prompt(profile, day.focus, current.name, others, reason)},
    ]
    try:
        llm = client or make_client(api_key)
        for _ in range(2):
            content = _chat(llm, model, messages, 0.8, max_tokens=2000)
            try:
                new = parse_exercise(extract_json(content))
            except ValueError:  # includes json.JSONDecodeError
                new = None
            if new and new.name.lower() not in taken and not any(
                re.search(p, new.name, re.IGNORECASE) for p in patterns
            ):
                return new
            messages.append({
                "role": "user",
                "content": "That suggestion is invalid (unreadable, already in the session, or not "
                "allowed with this equipment). Try a different one.",
            })
    except PlanGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _friendly_api_error(exc) from exc
    raise PlanGenerationError("Couldn't find a suitable swap this time. Please try again.")


# --------------------------------------------------------------------------- export

def plan_to_markdown(plan: WorkoutPlan, profile: UserProfile) -> str:
    lines = [f"# {plan.title}", ""]
    lines.append(
        f"**Goal:** {profile.goal.value} | **Level:** {profile.experience.value} | "
        f"**Days/week:** {profile.days_per_week} | **Session:** {profile.session_minutes} min | "
        f"**Equipment:** {', '.join(e.value for e in profile.equipment)}"
    )
    if profile.has_limitations:
        lines.append(f"**Limitations:** {profile.limitations}")
    lines.append("")
    if plan.summary:
        lines += [plan.summary, ""]
    for day in plan.days:
        lines += [f"## Day {day.day}: {day.focus}", ""]
        if day.warmup:
            lines += [f"**Warm-up:** {day.warmup}", ""]
        lines += ["| Exercise | Sets | Reps / Time | Rest | Notes |", "|---|---|---|---|---|"]
        for ex in day.exercises:
            row = [ex.name, ex.sets, ex.reps, ex.rest, ex.notes]
            lines.append("| " + " | ".join(c.replace("|", "/") for c in row) + " |")
        lines.append("")
        if day.cooldown:
            lines += [f"**Cool-down:** {day.cooldown}", ""]
    if plan.rest_day_guidance:
        lines += ["## Rest days", plan.rest_day_guidance, ""]
    if plan.progression:
        lines += ["## Progression", plan.progression, ""]
    if plan.limitation_adjustments:
        lines += ["## How your limitations were handled"] + [f"- {a}" for a in plan.limitation_adjustments] + [""]
    if plan.safety_notes:
        lines += ["## Safety notes"] + [f"- {n}" for n in plan.safety_notes] + [""]
    lines += ["---", f"_{INJURY_DISCLAIMER if profile.has_limitations else GENERAL_DISCLAIMER}_", ""]
    return "\n".join(lines)
