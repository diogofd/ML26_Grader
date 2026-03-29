from __future__ import annotations

from math import isclose
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..constants import QUESTION_SUBQUESTION_IDS

QuestionId = Literal["Q2", "Q3"]
SubquestionId = Literal["Q2.1", "Q2.2", "Q3.1", "Q3.2"]


def _expected_subquestions(question_id: QuestionId) -> tuple[str, str]:
    return QUESTION_SUBQUESTION_IDS[question_id]


def _normalize_review_reasons(raw_reasons: Any) -> list[str]:
    if not isinstance(raw_reasons, list):
        return []
    normalized: list[str] = []
    for reason in raw_reasons:
        if isinstance(reason, str) and reason and reason not in normalized:
            normalized.append(reason)
    return normalized


def _append_review_reason(reasons: list[str], reason: str) -> list[str]:
    if reason in reasons:
        return reasons
    return [*reasons, reason]


class QuestionSubquestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subquestion_id: SubquestionId
    question_text: str = Field(min_length=1)
    max_score: float = Field(gt=0)


class MaxScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: float = Field(gt=0)
    subquestions: dict[SubquestionId, float] = Field(min_length=2, max_length=2)


class RubricBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subquestion_id: SubquestionId
    required_evidence: list[str] = Field(min_length=1)
    partial_credit_guidance: list[str] = Field(min_length=1)
    common_failure_modes: list[str] = Field(min_length=1)
    score_band_guidance: list[str] = Field(min_length=1)
    feedback_guidance: list[str] = Field(min_length=1)


class EvidenceSnippet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snippet_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ExtractedSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str = Field(min_length=1)
    value: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class ExtractionWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class EvidencePacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: QuestionId
    markdown_snippets: list[EvidenceSnippet] = Field(default_factory=list)
    code_snippets: list[EvidenceSnippet] = Field(default_factory=list)
    output_snippets: list[EvidenceSnippet] = Field(default_factory=list)
    extracted_signals: list[ExtractedSignal] = Field(default_factory=list)
    extraction_warnings: list[ExtractionWarning] = Field(default_factory=list)
    detected_models: list[str] = Field(default_factory=list)
    detected_metrics: list[str] = Field(default_factory=list)
    preprocessing_signals: list[str] = Field(default_factory=list)
    tuning_signals: list[str] = Field(default_factory=list)
    comparison_signals: list[str] = Field(default_factory=list)
    business_justification_signals: list[str] = Field(default_factory=list)

    def has_evidence(self) -> bool:
        return any(
            (
                self.markdown_snippets,
                self.code_snippets,
                self.output_snippets,
                self.extracted_signals,
                self.detected_models,
                self.detected_metrics,
                self.preprocessing_signals,
                self.tuning_signals,
                self.comparison_signals,
                self.business_justification_signals,
            )
        )


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: QuestionId
    question_text: str = Field(min_length=1)
    subquestions: list[QuestionSubquestion] = Field(min_length=2, max_length=2)
    max_scores: MaxScores
    rubric_blocks: list[RubricBlock] = Field(min_length=2, max_length=2)
    evidence_packet: EvidencePacket
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        expected = set(_expected_subquestions(self.question_id))
        subquestions = {item.subquestion_id for item in self.subquestions}
        rubric_blocks = {item.subquestion_id for item in self.rubric_blocks}
        score_keys = set(self.max_scores.subquestions)

        if self.evidence_packet.question_id != self.question_id:
            raise ValueError("Evidence packet question_id must match the request question_id.")
        if subquestions != expected:
            raise ValueError(f"Subquestions must be exactly {sorted(expected)}.")
        if rubric_blocks != expected:
            raise ValueError(f"Rubric blocks must be exactly {sorted(expected)}.")
        if score_keys != expected:
            raise ValueError(f"Max score keys must be exactly {sorted(expected)}.")
        if not self.evidence_packet.has_evidence():
            raise ValueError(
                "Evidence packet must contain explicit extracted evidence before judging."
            )

        subquestion_max_scores = {item.subquestion_id: item.max_score for item in self.subquestions}
        if subquestion_max_scores != self.max_scores.subquestions:
            raise ValueError("Subquestion max_score values must match max_scores.subquestions.")

        total = sum(self.max_scores.subquestions.values())
        if not isclose(total, self.max_scores.overall):
            raise ValueError("max_scores.overall must equal the sum of subquestion max scores.")
        return self


class JudgeSubquestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    confidence: float = Field(ge=0, le=10)
    student_feedback: str = Field(min_length=1)
    internal_notes: str = Field(min_length=1)
    evidence_used: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_score_bounds(self) -> Self:
        if self.score > self.max_score:
            raise ValueError("Subquestion score cannot exceed max_score.")
        return self


class JudgeQuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: QuestionId
    score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    confidence: float = Field(ge=0, le=10)
    student_feedback_overall: str = Field(min_length=1)
    internal_notes_overall: str = Field(min_length=1)
    review_recommended: bool
    review_reasons: list[str] = Field(default_factory=list)
    subquestions: dict[SubquestionId, JudgeSubquestionResult] = Field(min_length=2, max_length=2)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_scores_and_review_state(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        review_reasons = _normalize_review_reasons(normalized.get("review_reasons"))
        raw_subquestions = normalized.get("subquestions")

        if isinstance(raw_subquestions, dict) and raw_subquestions:
            score_total = 0.0
            max_total = 0.0
            complete_subquestion_payload = True

            for subquestion in raw_subquestions.values():
                if not isinstance(subquestion, dict):
                    complete_subquestion_payload = False
                    break
                score = subquestion.get("score")
                max_score = subquestion.get("max_score")
                if not isinstance(score, (int, float)) or not isinstance(max_score, (int, float)):
                    complete_subquestion_payload = False
                    break
                score_total += float(score)
                max_total += float(max_score)

            if complete_subquestion_payload:
                existing_score = normalized.get("score")
                if isinstance(existing_score, (int, float)) and not isclose(
                    float(existing_score),
                    score_total,
                ):
                    review_reasons = _append_review_reason(
                        review_reasons,
                        "score_consistency_issue",
                    )
                existing_max_score = normalized.get("max_score")
                if isinstance(existing_max_score, (int, float)) and not isclose(
                    float(existing_max_score),
                    max_total,
                ):
                    review_reasons = _append_review_reason(
                        review_reasons,
                        "score_consistency_issue",
                    )
                normalized["score"] = score_total
                normalized["max_score"] = max_total

        normalized["review_reasons"] = review_reasons
        if review_reasons:
            normalized["review_recommended"] = True
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        expected = set(_expected_subquestions(self.question_id))
        actual = set(self.subquestions)
        if actual != expected:
            raise ValueError(f"Subquestion results must be exactly {sorted(expected)}.")

        total_score = sum(result.score for result in self.subquestions.values())
        total_max = sum(result.max_score for result in self.subquestions.values())
        if not isclose(total_score, self.score):
            raise ValueError("Question score must equal the sum of subquestion scores.")
        if not isclose(total_max, self.max_score):
            raise ValueError("Question max_score must equal the sum of subquestion max scores.")
        if self.score > self.max_score:
            raise ValueError("Question score cannot exceed max_score.")
        if self.review_reasons and not self.review_recommended:
            raise ValueError("review_reasons require review_recommended=true.")
        return self
