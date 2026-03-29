from __future__ import annotations

from collections.abc import Sequence
from math import isclose

from .models import LeaderboardEntry


def validate_binary_predictions(predictions: Sequence[int | bool]) -> list[int]:
    if not predictions:
        raise ValueError("Predictions must not be empty.")

    normalized: list[int] = []
    for value in predictions:
        if isinstance(value, bool):
            normalized.append(int(value))
            continue
        if value in (0, 1):
            normalized.append(int(value))
            continue
        raise ValueError("Predictions must contain only binary values 0 or 1.")
    return normalized


def normalize_binary_label(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "yes", "true", "y"}:
            return 1
        if normalized in {"0", "no", "false", "n"}:
            return 0
    raise ValueError("Labels must be coercible to binary values.")


def compute_binary_f1(
    truth: Sequence[int | bool],
    predictions: Sequence[int | bool],
) -> float:
    y_true = validate_binary_predictions(truth)
    y_pred = validate_binary_predictions(predictions)

    if len(y_true) != len(y_pred):
        raise ValueError("Prediction count must match the number of labels.")

    true_positive = sum(1 for expected, actual in zip(y_true, y_pred) if expected == 1 and actual == 1)
    false_positive = sum(1 for expected, actual in zip(y_true, y_pred) if expected == 0 and actual == 1)
    false_negative = sum(1 for expected, actual in zip(y_true, y_pred) if expected == 1 and actual == 0)

    if true_positive == 0:
        return 0.0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return (2 * precision * recall) / (precision + recall)


def validate_prediction_count(
    predictions: Sequence[int | bool],
    expected_count: int,
) -> None:
    if expected_count < 1:
        raise ValueError("Expected prediction count must be at least 1.")
    if len(predictions) != expected_count:
        raise ValueError("Prediction count must match the number of input rows.")


def assign_leaderboard_ranks(entries: Sequence[LeaderboardEntry]) -> list[LeaderboardEntry]:
    ordered_entries = sorted(entries, key=lambda item: (-item.f1_score, item.submission_id))
    ranked_entries: list[LeaderboardEntry] = []
    current_rank = 0
    previous_score: float | None = None

    for index, entry in enumerate(ordered_entries, start=1):
        if previous_score is None or not isclose(entry.f1_score, previous_score):
            current_rank = index
            previous_score = entry.f1_score
        ranked_entries.append(entry.model_copy(update={"rank": current_rank}))

    return ranked_entries
