"""Streamlit UI for the Workout Plan Generator."""

from __future__ import annotations

import os
from datetime import date

import streamlit as st

from planner import (
    DEFAULT_MODEL,
    GENERAL_DISCLAIMER,
    INJURY_DISCLAIMER,
    Equipment,
    Experience,
    GenerationResult,
    Goal,
    InputValidationError,
    PlanGenerationError,
    UserProfile,
    WorkoutPlan,
    generate_workout_plan,
    plan_to_markdown,
    swap_exercise,
    validate_inputs,
)

MODEL_OPTIONS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

st.set_page_config(page_title="Workout Plan Generator", page_icon="🏋️", layout="wide")


# --------------------------------------------------------------------------- helpers

def get_secret_api_key() -> str:
    """Read GROQ_API_KEY from Streamlit secrets or the environment (never crashes)."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets.toml present
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


def init_state() -> None:
    for key, default in {
        "plan": None, "profile": None, "warnings": [], "variation": 0, "swap_msg": None,
    }.items():
        st.session_state.setdefault(key, default)


def run_generation(inputs: dict, api_key: str, model: str, temperature: float) -> None:
    """Validate, call the LLM and store the result in session_state; show friendly errors."""
    try:
        profile = validate_inputs(**inputs)
    except InputValidationError as err:
        st.warning(f"⚠️ {err}")
        return

    with st.spinner("Designing your plan... (usually 5-20 seconds)"):
        try:
            result: GenerationResult = generate_workout_plan(
                **inputs, api_key=api_key, model=model, temperature=temperature
            )
        except InputValidationError as err:
            st.warning(f"⚠️ {err}")
            return
        except PlanGenerationError as err:
            st.error(f"😕 {err}")
            return
        except Exception:  # last-resort guard so the app never shows a traceback
            st.error("😕 Something unexpected went wrong. Please try again.")
            return

    st.session_state.plan = result.plan
    st.session_state.profile = profile
    st.session_state.warnings = result.warnings
    st.session_state.variation += 1
    st.session_state.swap_msg = None


def render_plan(plan: WorkoutPlan, profile: UserProfile) -> None:
    st.header(plan.title)
    if profile.has_limitations:
        st.warning(f"⚕️ {INJURY_DISCLAIMER}")
    if st.session_state.warnings:
        with st.expander("⚠️ Double-check these items", expanded=True):
            st.caption("Our automatic checks found possible issues the AI couldn't fully fix:")
            for w in st.session_state.warnings:
                st.write(f"- {w}")
    if plan.summary:
        st.write(plan.summary)

    tabs = st.tabs([f"Day {d.day}" for d in plan.days])
    for tab, day in zip(tabs, plan.days):
        with tab:
            st.subheader(day.focus)
            if day.warmup:
                st.markdown(f"**Warm-up:** {day.warmup}")
            st.dataframe(
                [
                    {"Exercise": e.name, "Sets": e.sets, "Reps / Time": e.reps, "Rest": e.rest, "Notes": e.notes}
                    for e in day.exercises
                ],
                hide_index=True,
                width="stretch",
            )
            if day.cooldown:
                st.markdown(f"**Cool-down:** {day.cooldown}")

    col1, col2 = st.columns(2)
    with col1:
        if plan.rest_day_guidance:
            st.markdown("#### 🛌 Rest days")
            st.write(plan.rest_day_guidance)
        if plan.progression:
            st.markdown("#### 📈 Progression")
            st.write(plan.progression)
    with col2:
        if plan.limitation_adjustments:
            st.markdown("#### 🩹 How your limitations were handled")
            for a in plan.limitation_adjustments:
                st.write(f"- {a}")
        if plan.safety_notes:
            st.markdown("#### 🛡️ Safety notes")
            for n in plan.safety_notes:
                st.write(f"- {n}")
    if not profile.has_limitations:
        st.caption(GENERAL_DISCLAIMER)


def render_swap(plan: WorkoutPlan, profile: UserProfile, api_key: str, model: str) -> None:
    with st.expander("🔁 Swap an exercise"):
        # Keys are per plan and per day so a stale index can't outlive the list it pointed into.
        variation = st.session_state.variation
        day_idx = st.selectbox(
            "Day", range(len(plan.days)),
            format_func=lambda i: f"Day {plan.days[i].day}: {plan.days[i].focus}",
            key=f"swap_day_{variation}",
        )
        exercises = plan.days[day_idx].exercises
        ex_idx = st.selectbox(
            "Exercise", range(len(exercises)), format_func=lambda i: exercises[i].name,
            key=f"swap_ex_{variation}_{day_idx}",
        )
        reason = st.text_input("Why? (optional)", placeholder="e.g. too hard, don't like it, hurts my wrist")
        if st.button("Swap it"):
            with st.spinner("Finding an alternative..."):
                try:
                    old_name = exercises[ex_idx].name
                    new = swap_exercise(profile, plan, day_idx, ex_idx, reason, api_key=api_key, model=model)
                    plan.days[day_idx].exercises[ex_idx] = new
                    st.session_state.swap_msg = ("success", f"Swapped **{old_name}** → **{new.name}**")
                except (InputValidationError, PlanGenerationError) as err:
                    st.session_state.swap_msg = ("error", str(err))
                except Exception:
                    st.session_state.swap_msg = ("error", "Couldn't swap that exercise. Please try again.")
            st.rerun()
        if st.session_state.swap_msg:
            kind, msg = st.session_state.swap_msg
            (st.success if kind == "success" else st.error)(msg)


# --------------------------------------------------------------------------- page

init_state()

with st.sidebar:
    st.header("⚙️ Settings")
    secret_key = get_secret_api_key()
    if secret_key:
        st.success("Groq API key loaded from secrets/env.")
        api_key = secret_key
    else:
        api_key = st.text_input(
            "Groq API key", type="password", help="Get one free at console.groq.com/keys"
        )
    model = st.selectbox("Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    temperature = st.slider("Creativity", 0.0, 1.0, 0.6, 0.1, help="Higher = more varied plans")

st.title("🏋️ Workout Plan Generator")
st.caption("Answer a few questions a personal trainer would ask, and get a weekly plan you can actually follow.")

left, right = st.columns(2)
with left:
    goal = st.selectbox("Fitness goal", [g.value for g in Goal], index=None, placeholder="Choose a goal")
    experience = st.selectbox(
        "Experience level", [e.value for e in Experience], index=None, placeholder="Choose your level",
        help="Beginner: <6 months consistent training. Intermediate: 6 months-2 years. Advanced: 2+ years.",
    )
    days = st.slider("Days available per week", 0, 7, 3)
with right:
    equipment = st.multiselect("Equipment access", [e.value for e in Equipment], placeholder="Select all that apply")
    minutes = st.select_slider("Time per session (minutes)", [20, 30, 45, 60, 75, 90], value=45)
    limitations = st.text_area(
        "Injuries or limitations (optional)",
        placeholder="e.g. bad knees, no overhead pressing, lower-back pain when bending",
        max_chars=300,
    )

inputs = {
    "goal": goal, "experience": experience, "days_per_week": days,
    "equipment": equipment, "limitations": limitations, "session_minutes": minutes,
}

b1, b2, _ = st.columns([1, 1, 3])
generate = b1.button("Generate Plan", type="primary", width="stretch")
regenerate = b2.button(
    "🔄 Regenerate", width="stretch", disabled=st.session_state.plan is None,
    help="Get a different variation with the same inputs",
)

if generate:
    run_generation(inputs, api_key, model, temperature)
elif regenerate:
    # Nudge temperature up a little so the variation is genuinely different.
    run_generation(inputs, api_key, model, min(1.0, temperature + 0.2))

plan: WorkoutPlan | None = st.session_state.plan
profile: UserProfile | None = st.session_state.profile
if plan and profile:
    st.divider()
    render_plan(plan, profile)
    render_swap(plan, profile, api_key, model)

    md = plan_to_markdown(plan, profile)
    stamp = date.today().isoformat()
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇️ Download .md", md, f"workout-plan-{stamp}.md", "text/markdown", width="stretch")
    d2.download_button("⬇️ Download .txt", md, f"workout-plan-{stamp}.txt", "text/plain", width="stretch")
else:
    st.info("Fill in your details and click **Generate Plan**.")
