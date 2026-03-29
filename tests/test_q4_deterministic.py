from __future__ import annotations

import pytest

from ml26_grader.q4.deterministic import (
    assign_leaderboard_ranks,
    compute_binary_f1,
    normalize_binary_label,
    validate_binary_predictions,
    validate_prediction_count,
)
from ml26_grader.q4.models import LeaderboardEntry


def test_validate_binary_predictions_accepts_bool_and_int_values() -> None:
    assert validate_binary_predictions([True, False, 1, 0]) == [1, 0, 1, 0]


def test_validate_binary_predictions_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError):
        validate_binary_predictions([0, 2, 1])


def test_compute_binary_f1_matches_expected_score() -> None:
    score = compute_binary_f1([1, 0, 1, 1], [1, 0, 0, 1])

    assert score == pytest.approx(0.8)


def test_normalize_binary_label_accepts_yes_no_values() -> None:
    assert normalize_binary_label("Yes") == 1
    assert normalize_binary_label("no") == 0


def test_validate_prediction_count_rejects_mismatch() -> None:
    with pytest.raises(ValueError):
        validate_prediction_count([1, 0], expected_count=3)


def test_assign_leaderboard_ranks_is_deterministic_for_ties() -> None:
    entries = [
        LeaderboardEntry(submission_id="student_b", f1_score=0.91),
        LeaderboardEntry(submission_id="student_c", f1_score=0.87),
        LeaderboardEntry(submission_id="student_a", f1_score=0.91),
    ]

    ranked = assign_leaderboard_ranks(entries)

    assert [entry.submission_id for entry in ranked] == ["student_a", "student_b", "student_c"]
    assert [entry.rank for entry in ranked] == [1, 1, 3]
