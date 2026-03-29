from __future__ import annotations

from ..llm.schemas import JudgeQuestionResult
from ..q4.models import Q4EvaluationResult
from .models import QuestionScoreSummary, SubmissionScorecard, SubquestionScoreSummary


def summarise_question_result(
    result: JudgeQuestionResult,
    threshold: float | None = None,
) -> QuestionScoreSummary:
    del threshold
    review_required = result.review_recommended or bool(result.review_reasons)
    return QuestionScoreSummary(
        question_id=result.question_id,
        score=result.score,
        max_score=result.max_score,
        confidence=result.confidence,
        review_required=review_required,
        review_reasons=result.review_reasons,
        student_feedback_overall=result.student_feedback_overall,
        internal_notes_overall=result.internal_notes_overall,
        subquestions={
            subquestion_id: SubquestionScoreSummary(
                score=subquestion.score,
                max_score=subquestion.max_score,
                confidence=subquestion.confidence,
                student_feedback=subquestion.student_feedback,
                internal_notes=subquestion.internal_notes,
                evidence_used=subquestion.evidence_used,
                missing_requirements=subquestion.missing_requirements,
            )
            for subquestion_id, subquestion in result.subquestions.items()
        },
        provisional=review_required,
    )


def aggregate_submission_scorecard(
    submission_id: str,
    q2_result: JudgeQuestionResult | None = None,
    q3_result: JudgeQuestionResult | None = None,
    q4_result: Q4EvaluationResult | None = None,
    confidence_threshold: float | None = None,
) -> SubmissionScorecard:
    return SubmissionScorecard(
        submission_id=submission_id,
        q2=(
            summarise_question_result(q2_result, threshold=confidence_threshold)
            if q2_result is not None
            else None
        ),
        q3=(
            summarise_question_result(q3_result, threshold=confidence_threshold)
            if q3_result is not None
            else None
        ),
        q4=q4_result,
    )
