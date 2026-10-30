"""UI tests: drive the real Streamlit script with a fake Groq client."""

from __future__ import annotations

from typing import Any

import pytest
from streamlit.testing.v1 import AppTest
from test_planner import FakeClient, make_plan

import planner.core


def by_label(widgets: Any, label: str) -> Any:
    return next(w for w in widgets if label in w.label)


def click_generate(at: AppTest, days: int) -> None:
    by_label(at.slider, "Days available").set_value(days)
    by_label(at.button, "Generate Plan").click().run()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return (filled-in app, list of queued LLM responses)."""
    responses: list[Any] = []
    monkeypatch.setattr(planner.core, "make_client", lambda _key: FakeClient(responses))
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.secrets["GROQ_API_KEY"] = "gsk_fake"
    at.run()
    by_label(at.selectbox, "Fitness goal").set_value("Build muscle")
    by_label(at.selectbox, "Experience level").set_value("Beginner")
    at.multiselect[0].set_value(["No equipment"])
    return at.run(), responses


def test_generate_renders_one_tab_per_day(app: Any) -> None:
    at, responses = app
    responses.append(make_plan(3))
    click_generate(at, 3)
    assert not at.exception and len(at.tabs) == 3


def test_new_plan_with_fewer_days_does_not_crash_swap_picker(app: Any) -> None:
    at, responses = app
    responses += [make_plan(5), make_plan(2)]
    click_generate(at, 5)
    by_label(at.selectbox, "Day").set_value(4).run()

    click_generate(at, 2)

    assert not at.exception and len(at.tabs) == 2
    assert by_label(at.selectbox, "Day").value == 0


def test_day_with_fewer_exercises_resets_exercise_picker(app: Any) -> None:
    at, responses = app
    plan = make_plan(2)
    plan["days"][0]["exercises"] *= 6
    responses.append(plan)
    click_generate(at, 2)
    by_label(at.selectbox, "Exercise").set_value(5).run()

    by_label(at.selectbox, "Day").set_value(1).run()

    assert not at.exception
    assert by_label(at.selectbox, "Exercise").value == 0
