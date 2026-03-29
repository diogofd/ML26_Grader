from __future__ import annotations

import pytest
from pydantic import ValidationError

from ml26_grader.llm.schemas import EvidencePacket, JudgeQuestionResult, JudgeRequest


def build_valid_request() -> dict:
    return {
        "question_id": "Q2",
        "question_text": "Full official wording for Q2.",
        "subquestions": [
            {
                "subquestion_id": "Q2.1",
                "question_text": "Identify the machine learning problem.",
                "max_score": 1.0,
            },
            {
                "subquestion_id": "Q2.2",
                "question_text": "Build two predictive models.",
                "max_score": 3.0,
            },
        ],
        "max_scores": {
            "overall": 4.0,
            "subquestions": {
                "Q2.1": 1.0,
                "Q2.2": 3.0,
            },
        },
        "rubric_blocks": [
            {
                "subquestion_id": "Q2.1",
                "required_evidence": ["Problem statement"],
                "partial_credit_guidance": ["Partial if task type is vague."],
                "common_failure_modes": ["Incorrect task framing"],
                "score_band_guidance": ["Full marks require explicit binary classification framing."],
                "feedback_guidance": ["Mention whether the task type is explicit and correct."],
            },
            {
                "subquestion_id": "Q2.2",
                "required_evidence": ["Two models", "Preprocessing", "Tuning"],
                "partial_credit_guidance": ["Partial if only one model is supported."],
                "common_failure_modes": ["No tuning evidence"],
                "score_band_guidance": ["Full marks require explicit technical workflow evidence."],
                "feedback_guidance": ["Tie comments to concrete evidence and missing workflow pieces."],
            },
        ],
        "evidence_packet": {
            "question_id": "Q2",
            "markdown_snippets": [
                {
                    "snippet_id": "md-1",
                    "source_ref": "notebook:12",
                    "content": "We solve a binary classification problem.",
                }
            ],
            "code_snippets": [],
            "output_snippets": [],
            "extracted_signals": [],
            "extraction_warnings": [],
            "detected_models": ["logistic_regression", "random_forest"],
            "detected_metrics": ["f1"],
            "preprocessing_signals": ["missing_value_imputation"],
            "tuning_signals": ["grid_search"],
            "comparison_signals": ["model_comparison_table"],
            "business_justification_signals": ["retention_risk"],
        },
        "metadata": {
            "submission_id": "S123",
            "rubric_version": "draft",
            "prompt_version": "draft",
            "extraction_version": "placeholder",
        },
    }


def build_valid_result(review_recommended: bool = False) -> dict:
    return {
        "question_id": "Q2",
        "score": 3.0,
        "max_score": 4.0,
        "confidence": 8.8,
        "student_feedback_overall": "The response identifies the task and supports the models reasonably well.",
        "internal_notes_overall": "Overall evidence is mostly explicit.",
        "review_recommended": review_recommended,
        "review_reasons": [],
        "subquestions": {
            "Q2.1": {
                "score": 1.0,
                "max_score": 1.0,
                "confidence": 9.2,
                "student_feedback": "The task is clearly framed as binary classification.",
                "internal_notes": "Explicit statement found in markdown.",
                "evidence_used": ["md-1"],
                "missing_requirements": [],
            },
            "Q2.2": {
                "score": 2.0,
                "max_score": 3.0,
                "confidence": 9.0,
                "student_feedback": "Two models are present, but tuning support is limited.",
                "internal_notes": "Need stronger hyperparameter evidence.",
                "evidence_used": ["code-4", "out-2"],
                "missing_requirements": ["hyperparameter_tuning_detail"],
            },
        },
    }


def test_judge_request_accepts_valid_q2_payload() -> None:
    request = JudgeRequest.model_validate(build_valid_request())

    assert request.question_id == "Q2"
    assert request.max_scores.overall == 4.0
    assert set(request.max_scores.subquestions) == {"Q2.1", "Q2.2"}


def test_judge_request_rejects_mismatched_subquestions() -> None:
    payload = build_valid_request()
    payload["subquestions"][1]["subquestion_id"] = "Q3.2"

    with pytest.raises(ValidationError):
        JudgeRequest.model_validate(payload)


def test_judge_request_rejects_empty_evidence_packets() -> None:
    payload = build_valid_request()
    payload["evidence_packet"] = EvidencePacket(question_id="Q2").model_dump(mode="json")

    with pytest.raises(ValidationError):
        JudgeRequest.model_validate(payload)


def test_judge_question_result_accepts_policy_compliant_payload() -> None:
    result = JudgeQuestionResult.model_validate(build_valid_result())

    assert result.review_recommended is False
    assert result.subquestions["Q2.2"].score == 2.0


def test_judge_question_result_does_not_own_confidence_threshold_policy() -> None:
    payload = build_valid_result(review_recommended=False)
    payload["confidence"] = 7.5

    result = JudgeQuestionResult.model_validate(payload)

    assert result.review_recommended is False
    assert result.review_reasons == []


def test_judge_question_result_repairs_score_consistency_and_marks_review() -> None:
    payload = build_valid_result()
    payload["score"] = 2.2

    result = JudgeQuestionResult.model_validate(payload)

    assert result.score == 3.0
    assert result.review_recommended is True
    assert "score_consistency_issue" in result.review_reasons
