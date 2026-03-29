from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from pydantic import model_validator

from ..extraction.service import ExtractionResult
from ..llm.interface import JudgeEvaluationAudit
from ..llm.schemas import QuestionId, SubquestionId
from ..llm.schemas import JudgeQuestionResult, JudgeRequest
from ..q4.models import Q4EvaluationResult


class SubquestionScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    confidence: float = Field(ge=0, le=10)
    student_feedback: str = Field(min_length=1)
    internal_notes: str = Field(min_length=1)
    evidence_used: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)


class QuestionScoreSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: QuestionId
    score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    confidence: float = Field(ge=0, le=10)
    review_required: bool
    review_reasons: list[str] = Field(default_factory=list)
    student_feedback_overall: str = Field(min_length=1)
    internal_notes_overall: str = Field(min_length=1)
    subquestions: dict[SubquestionId, SubquestionScoreSummary] = Field(
        min_length=2,
        max_length=2,
    )
    provisional: bool = True


class QuestionGradingStatus(StrEnum):
    SCORED = "scored"
    REVIEW = "review"
    FAILED = "failed"


class QuestionReviewTier(StrEnum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


class QuestionGradingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: QuestionId
    status: QuestionGradingStatus
    review_tier: QuestionReviewTier = QuestionReviewTier.NONE
    review_required: bool
    review_reasons: list[str] = Field(default_factory=list)
    hard_review_reasons: list[str] = Field(default_factory=list)
    soft_review_reasons: list[str] = Field(default_factory=list)
    soft_auto_pass_applied: bool = False
    failure_reason: str | None = None
    extraction_result: ExtractionResult
    judge_request: JudgeRequest | None = None
    judge_result: JudgeQuestionResult | None = None
    judge_audit: JudgeEvaluationAudit | None = None
    score_summary: QuestionScoreSummary | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "QuestionGradingResult":
        if self.extraction_result.question_id != self.question_id:
            raise ValueError("Extraction result question_id must match the grading result.")
        if self.judge_request is not None and self.judge_request.question_id != self.question_id:
            raise ValueError("Judge request question_id must match the grading result.")
        if self.judge_result is not None and self.judge_result.question_id != self.question_id:
            raise ValueError("Judge result question_id must match the grading result.")
        if self.score_summary is not None and self.score_summary.question_id != self.question_id:
            raise ValueError("Score summary question_id must match the grading result.")

        if self.status == QuestionGradingStatus.FAILED:
            if not self.review_required:
                raise ValueError("Failed grading results must require review.")
            if not self.failure_reason:
                raise ValueError("Failed grading results must include a failure_reason.")
            if self.review_tier != QuestionReviewTier.HARD:
                raise ValueError("Failed grading results must use the hard review tier.")
            if self.soft_auto_pass_applied:
                raise ValueError("Failed grading results cannot apply soft auto-pass.")
            if self.review_reasons != self.hard_review_reasons:
                raise ValueError("Failed grading review_reasons must match hard_review_reasons.")
            return self

        if self.failure_reason is not None:
            raise ValueError("Non-failed grading results must not include a failure_reason.")
        if self.judge_request is None or self.judge_result is None or self.score_summary is None:
            raise ValueError("Scored or review grading results require request, result, and summary.")
        if self.status == QuestionGradingStatus.SCORED and self.review_required:
            raise ValueError("Scored grading results cannot require review.")
        if self.status == QuestionGradingStatus.REVIEW and not self.review_required:
            raise ValueError("Review grading results must require review.")
        if self.status == QuestionGradingStatus.SCORED:
            if self.review_tier != QuestionReviewTier.NONE:
                raise ValueError("Scored grading results must not have a review tier.")
            if self.review_reasons:
                raise ValueError("Scored grading results must not expose active review reasons.")
            if self.hard_review_reasons:
                raise ValueError("Scored grading results cannot include hard review reasons.")
        if self.status == QuestionGradingStatus.REVIEW:
            if self.review_tier != QuestionReviewTier.SOFT:
                raise ValueError("Review grading results must use the soft review tier.")
            if self.review_reasons != self.soft_review_reasons:
                raise ValueError("Review grading review_reasons must match soft_review_reasons.")
            if self.hard_review_reasons:
                raise ValueError("Review grading results cannot include hard review reasons.")
            if self.soft_auto_pass_applied:
                raise ValueError("Review grading results cannot apply soft auto-pass.")
        return self


class SubmissionScorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str = Field(min_length=1)
    q2: QuestionScoreSummary | None = None
    q3: QuestionScoreSummary | None = None
    q4: Q4EvaluationResult | None = None
