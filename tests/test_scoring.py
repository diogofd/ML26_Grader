from __future__ import annotations

from ml26_grader.llm.schemas import JudgeQuestionResult
from ml26_grader.scoring.aggregation import aggregate_submission_scorecard, summarise_question_result


def build_review_result() -> JudgeQuestionResult:
    return JudgeQuestionResult.model_validate(
        {
            "question_id": "Q3",
            "score": 2.5,
            "max_score": 4.0,
            "confidence": 8.0,
            "student_feedback_overall": "Metric selection is reasonable, but deployment justification needs review.",
            "internal_notes_overall": "Low confidence on Q3.2.",
            "review_recommended": True,
            "review_reasons": ["question_confidence_below_threshold"],
            "subquestions": {
                "Q3.1": {
                    "score": 1.5,
                    "max_score": 2.0,
                    "confidence": 9.1,
                    "student_feedback": "Appropriate metric choice is stated clearly.",
                    "internal_notes": "Good business-risk framing.",
                    "evidence_used": ["md-8"],
                    "missing_requirements": [],
                },
                "Q3.2": {
                    "score": 1.0,
                    "max_score": 2.0,
                    "confidence": 7.8,
                    "student_feedback": "Model comparison exists, but the deployment recommendation is thin.",
                    "internal_notes": "Needs manual review.",
                    "evidence_used": ["md-10"],
                    "missing_requirements": ["deployment_justification"],
                },
            },
        }
    )


def test_summarise_question_result_marks_reviewed_scores_as_provisional() -> None:
    summary = summarise_question_result(build_review_result())

    assert summary.review_required is True
    assert summary.provisional is True
    assert summary.review_reasons == ["question_confidence_below_threshold"]
    assert summary.student_feedback_overall.startswith("Metric selection")
    assert summary.subquestions["Q3.2"].student_feedback.startswith("Model comparison")


def test_aggregate_submission_scorecard_keeps_question_slots() -> None:
    scorecard = aggregate_submission_scorecard("S123", q3_result=build_review_result())

    assert scorecard.submission_id == "S123"
    assert scorecard.q2 is None
    assert scorecard.q3 is not None
    assert scorecard.q3.question_id == "Q3"


def test_summarise_question_result_requires_explicit_threshold_for_confidence_gating() -> None:
    payload = build_review_result().model_dump(mode="json")
    payload["review_recommended"] = False
    payload["review_reasons"] = []
    payload["confidence"] = 7.5

    summary = summarise_question_result(
        JudgeQuestionResult.model_validate(payload),
        threshold=8.5,
    )

    assert summary.review_required is False
    assert summary.provisional is False
