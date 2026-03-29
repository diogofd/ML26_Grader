from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from ..config import GradingConfig
from ..extraction.service import (
    EvidenceExtractor,
    ExtractionResult,
    ExtractionStatus,
    NotebookEvidenceExtractor,
)
from ..llm.factory import build_llm_judge
from ..llm.interface import JudgeEvaluationAudit
from ..llm.interface import LLMJudge
from ..llm.schemas import JudgeQuestionResult, JudgeRequest, QuestionId
from ..specs import LoadedQuestionSpec, load_llm_question_specs
from .aggregation import summarise_question_result
from .models import (
    QuestionGradingResult,
    QuestionGradingStatus,
    QuestionReviewTier,
    QuestionScoreSummary,
)

MATERIAL_EVIDENCE_WARNING_CODES = {
    "limited_code_evidence",
    "limited_output_evidence",
    "missing_preprocessing_signal",
    "missing_feature_engineering_signal",
    "missing_data_split_signal",
    "missing_tuning_signal",
    "missing_metric_choice_signal",
    "missing_model_comparison_signal",
    "missing_deployment_recommendation_signal",
    "fewer_than_two_models_detected",
}
SOFT_EXTRACTION_WARNING_CODES = MATERIAL_EVIDENCE_WARNING_CODES | {
    "configured_notebook_pattern_miss",
    "multiple_notebooks_found",
    "multiple_relevant_notebooks_found",
    "notebook_parse_error",
    "question_section_inferred_from_content",
    "subquestion_anchor_missing",
}
HARD_BLOCKING_REASON_CODES = {
    "evidence_extraction_failed",
    "empty_evidence_packet",
    "question_section_not_found",
    "invalid_rubric_spec",
    "question_spec_missing",
    "invalid_judge_request",
    "judge_unavailable",
    "judge_evaluation_failed",
    "invalid_judge_output",
    "score_consistency_issue",
}


@dataclass(frozen=True)
class _ReviewPolicyAssessment:
    status: QuestionGradingStatus
    review_tier: QuestionReviewTier
    review_required: bool
    hard_review_reasons: list[str]
    soft_review_reasons: list[str]
    final_review_reasons: list[str]
    soft_auto_pass_applied: bool


class Q23GradingPipeline:
    def __init__(
        self,
        grading_config: GradingConfig,
        question_specs: Mapping[QuestionId, LoadedQuestionSpec] | None = None,
        *,
        extractor: EvidenceExtractor | None = None,
        judge: LLMJudge | None = None,
        spec_load_error: str | None = None,
    ) -> None:
        self._grading_config = grading_config
        self._question_specs = dict(question_specs or {})
        self._extractor = extractor or NotebookEvidenceExtractor(grading_config.submission)
        self._judge = judge or self._build_configured_judge(grading_config)
        self._spec_load_error = spec_load_error

    @classmethod
    def from_paths(
        cls,
        grading_config: GradingConfig,
        questions_path: Path,
        rubrics_path: Path,
        *,
        extractor: EvidenceExtractor | None = None,
        judge: LLMJudge | None = None,
    ) -> "Q23GradingPipeline":
        try:
            question_specs = load_llm_question_specs(
                grading_config,
                questions_path,
                rubrics_path,
            )
        except (ValidationError, ValueError, KeyError, FileNotFoundError, OSError) as exc:
            return cls(
                grading_config,
                question_specs={},
                extractor=extractor,
                judge=judge,
                spec_load_error=str(exc),
            )

        return cls(
            grading_config,
            question_specs=question_specs,
            extractor=extractor,
            judge=judge,
        )

    def grade_question(
        self,
        submission_root: Path,
        question_id: QuestionId,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QuestionGradingResult:
        extraction_result = self._extractor.extract(submission_root, question_id)
        return self.grade_extraction_result(extraction_result, metadata=metadata)

    def grade_extraction_result(
        self,
        extraction_result: ExtractionResult,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QuestionGradingResult:
        question_id = extraction_result.question_id

        if self._spec_load_error is not None:
            return self._failed_result(
                extraction_result,
                "invalid_rubric_spec",
                f"Question specifications could not be loaded: {self._spec_load_error}",
            )

        if extraction_result.status != ExtractionStatus.READY:
            return self._failed_result(
                extraction_result,
                "evidence_extraction_failed",
                "Evidence extraction did not produce a usable packet.",
            )

        if not extraction_result.evidence_packet.has_evidence():
            return self._failed_result(
                extraction_result,
                "empty_evidence_packet",
                "Evidence extraction returned an empty packet.",
            )

        question_spec = self._question_specs.get(question_id)
        if question_spec is None:
            return self._failed_result(
                extraction_result,
                "question_spec_missing",
                f"No loaded question specification is available for {question_id}.",
            )

        try:
            judge_request = question_spec.build_request(
                extraction_result.evidence_packet,
                metadata=self._build_request_metadata(extraction_result, metadata),
            )
        except (ValidationError, ValueError, KeyError) as exc:
            return self._failed_result(
                extraction_result,
                "invalid_judge_request",
                f"Judge request construction failed: {exc}",
            )

        if self._judge is None:
            return self._failed_result(
                extraction_result,
                "judge_unavailable",
                "No LLM judge implementation was provided.",
                judge_request=judge_request,
            )

        try:
            raw_result = self._judge.evaluate(judge_request)
        except Exception as exc:
            return self._failed_result(
                extraction_result,
                "judge_evaluation_failed",
                f"Judge evaluation failed: {exc}",
                judge_request=judge_request,
                judge_audit=self._current_judge_audit(),
            )

        try:
            judge_result = (
                raw_result
                if isinstance(raw_result, JudgeQuestionResult)
                else JudgeQuestionResult.model_validate(raw_result)
            )
        except ValidationError as exc:
            return self._failed_result(
                extraction_result,
                "invalid_judge_output",
                f"Judge output failed schema validation: {exc}",
                judge_request=judge_request,
                judge_audit=self._current_judge_audit(),
            )

        score_summary = summarise_question_result(
            judge_result,
        )
        policy = self._assess_review_policy(
            extraction_result,
            judge_result,
            score_summary,
        )
        merged_summary = None
        if policy.status != QuestionGradingStatus.FAILED:
            merged_summary = score_summary.model_copy(
                update={
                    "review_required": policy.review_required,
                    "review_reasons": policy.final_review_reasons,
                    "provisional": policy.review_required,
                }
            )

        if policy.status == QuestionGradingStatus.FAILED:
            failure_reason = (
                "Judge output contained a score consistency issue and was failed closed."
                if "score_consistency_issue" in policy.hard_review_reasons
                else "Judge output could not be accepted safely after structured validation."
            )
            return self._failed_result(
                extraction_result,
                policy.hard_review_reasons[0],
                failure_reason,
                judge_request=judge_request,
                judge_result=judge_result,
                judge_audit=self._current_judge_audit(),
                score_summary=merged_summary or score_summary,
                hard_review_reasons=policy.hard_review_reasons,
                soft_review_reasons=policy.soft_review_reasons,
            )

        return QuestionGradingResult(
            question_id=question_id,
            status=policy.status,
            review_tier=policy.review_tier,
            review_required=policy.review_required,
            review_reasons=policy.final_review_reasons,
            hard_review_reasons=policy.hard_review_reasons,
            soft_review_reasons=policy.soft_review_reasons,
            soft_auto_pass_applied=policy.soft_auto_pass_applied,
            extraction_result=extraction_result,
            judge_request=judge_request,
            judge_result=judge_result,
            judge_audit=self._current_judge_audit(),
            score_summary=merged_summary,
        )

    def _failed_result(
        self,
        extraction_result: ExtractionResult,
        reason_code: str,
        failure_reason: str,
        *,
        judge_request: JudgeRequest | None = None,
        judge_result: JudgeQuestionResult | None = None,
        judge_audit: JudgeEvaluationAudit | None = None,
        score_summary: QuestionScoreSummary | None = None,
        hard_review_reasons: list[str] | None = None,
        soft_review_reasons: list[str] | None = None,
    ) -> QuestionGradingResult:
        classified_hard_reasons = hard_review_reasons or _dedupe_reason_codes(
            [reason_code, *self._hard_extraction_reason_codes(extraction_result)]
        )
        classified_soft_reasons = soft_review_reasons or self._soft_extraction_reason_codes(
            extraction_result
        )
        return QuestionGradingResult(
            question_id=extraction_result.question_id,
            status=QuestionGradingStatus.FAILED,
            review_tier=QuestionReviewTier.HARD,
            review_required=True,
            review_reasons=classified_hard_reasons,
            hard_review_reasons=classified_hard_reasons,
            soft_review_reasons=classified_soft_reasons,
            failure_reason=failure_reason,
            extraction_result=extraction_result,
            judge_request=judge_request,
            judge_result=judge_result,
            judge_audit=judge_audit,
            score_summary=score_summary,
        )

    def _build_request_metadata(
        self,
        extraction_result: ExtractionResult,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = dict(metadata or {})
        payload.setdefault("extraction_status", extraction_result.status.value)
        payload.setdefault(
            "extraction_warning_codes",
            [warning.code for warning in extraction_result.evidence_packet.extraction_warnings],
        )
        payload.setdefault("extraction_notes", list(extraction_result.notes))
        return payload

    def _hard_extraction_reason_codes(
        self,
        extraction_result: ExtractionResult,
    ) -> list[str]:
        return _dedupe_reason_codes(
            [
                warning.code
                for warning in extraction_result.evidence_packet.extraction_warnings
                if warning.code in HARD_BLOCKING_REASON_CODES
            ]
        )

    def _soft_extraction_reason_codes(
        self,
        extraction_result: ExtractionResult,
    ) -> list[str]:
        return _dedupe_reason_codes(
            [
                warning.code
                for warning in extraction_result.evidence_packet.extraction_warnings
                if warning.code in SOFT_EXTRACTION_WARNING_CODES
            ]
        )

    def _evidence_quality_review_reason_codes(
        self,
        extraction_result: ExtractionResult,
    ) -> list[str]:
        warning_codes = [
            warning.code
            for warning in extraction_result.evidence_packet.extraction_warnings
        ]
        material_warning_codes = [
            warning_code
            for warning_code in warning_codes
            if warning_code in MATERIAL_EVIDENCE_WARNING_CODES
        ]

        review_reason_codes: list[str] = []
        if {
            "limited_code_evidence",
            "limited_output_evidence",
        }.issubset(set(material_warning_codes)):
            review_reason_codes.append("narrative_only_evidence_packet")
        if len(set(material_warning_codes)) >= 2:
            review_reason_codes.append("warning_heavy_evidence_packet")
        return review_reason_codes

    def _assess_review_policy(
        self,
        extraction_result: ExtractionResult,
        judge_result: JudgeQuestionResult,
        score_summary: QuestionScoreSummary,
    ) -> _ReviewPolicyAssessment:
        hard_review_reasons = _dedupe_reason_codes(
            [
                *self._hard_extraction_reason_codes(extraction_result),
                *(
                    ["score_consistency_issue"]
                    if "score_consistency_issue" in judge_result.review_reasons
                    else []
                ),
            ]
        )
        soft_review_reasons = _dedupe_reason_codes(
            [
                *self._soft_extraction_reason_codes(extraction_result),
                *self._evidence_quality_review_reason_codes(extraction_result),
                *(
                    ["question_confidence_below_threshold"]
                    if judge_result.confidence < self._grading_config.llm.auto_accept_confidence
                    else []
                ),
                *(
                    ["judge_review_recommended"]
                    if (
                        judge_result.review_recommended
                        and judge_result.confidence >= self._grading_config.llm.auto_accept_confidence
                    )
                    else []
                ),
            ]
        )

        if hard_review_reasons:
            return _ReviewPolicyAssessment(
                status=QuestionGradingStatus.FAILED,
                review_tier=QuestionReviewTier.HARD,
                review_required=True,
                hard_review_reasons=hard_review_reasons,
                soft_review_reasons=soft_review_reasons,
                final_review_reasons=hard_review_reasons,
                soft_auto_pass_applied=False,
            )

        if self._should_auto_pass_strong_question(judge_result, soft_review_reasons):
            return _ReviewPolicyAssessment(
                status=QuestionGradingStatus.SCORED,
                review_tier=QuestionReviewTier.NONE,
                review_required=False,
                hard_review_reasons=[],
                soft_review_reasons=soft_review_reasons,
                final_review_reasons=[],
                soft_auto_pass_applied=False,
            )

        if self._can_soft_auto_pass(judge_result, score_summary, soft_review_reasons):
            return _ReviewPolicyAssessment(
                status=QuestionGradingStatus.SCORED,
                review_tier=QuestionReviewTier.NONE,
                review_required=False,
                hard_review_reasons=[],
                soft_review_reasons=soft_review_reasons,
                final_review_reasons=[],
                soft_auto_pass_applied=True,
            )

        return _ReviewPolicyAssessment(
            status=QuestionGradingStatus.REVIEW,
            review_tier=QuestionReviewTier.SOFT,
            review_required=True,
            hard_review_reasons=[],
            soft_review_reasons=soft_review_reasons,
            final_review_reasons=soft_review_reasons,
            soft_auto_pass_applied=False,
        )

    def _should_auto_pass_strong_question(
        self,
        judge_result: JudgeQuestionResult,
        soft_review_reasons: list[str],
    ) -> bool:
        if judge_result.confidence < self._grading_config.llm.auto_accept_confidence:
            return False
        if "judge_review_recommended" in soft_review_reasons:
            return False
        return not self._has_disallowed_soft_reasons(soft_review_reasons)

    def _can_soft_auto_pass(
        self,
        judge_result: JudgeQuestionResult,
        score_summary: QuestionScoreSummary,
        soft_review_reasons: list[str],
    ) -> bool:
        llm_config = self._grading_config.llm
        if not llm_config.soft_auto_pass_enabled:
            return False
        if not (
            llm_config.soft_auto_pass_min_confidence
            <= judge_result.confidence
            < llm_config.auto_accept_confidence
        ):
            return False
        if self._has_disallowed_soft_reasons(soft_review_reasons):
            return False
        if (score_summary.score / score_summary.max_score) < llm_config.soft_auto_pass_min_score_ratio:
            return False
        if llm_config.soft_auto_pass_requires_no_failures and self._score_summary_has_failures(
            score_summary
        ):
            return False
        return True

    def _has_disallowed_soft_reasons(self, soft_review_reasons: list[str]) -> bool:
        disallowed_reason_codes = set(self._grading_config.llm.soft_auto_pass_disallowed_reasons)
        return any(reason in disallowed_reason_codes for reason in soft_review_reasons)

    def _score_summary_has_failures(self, score_summary: QuestionScoreSummary) -> bool:
        return any(
            subquestion_summary.missing_requirements
            for subquestion_summary in score_summary.subquestions.values()
        )

    def _build_configured_judge(
        self,
        grading_config: GradingConfig,
    ) -> LLMJudge | None:
        if not grading_config.llm.enabled:
            return None
        try:
            return build_llm_judge(grading_config)
        except Exception:
            return None

    def _current_judge_audit(self) -> JudgeEvaluationAudit | None:
        if self._judge is None:
            return None
        audit = getattr(self._judge, "last_evaluation_audit", None)
        if audit is None:
            return None
        if isinstance(audit, JudgeEvaluationAudit):
            return audit
        try:
            return JudgeEvaluationAudit.model_validate(audit)
        except ValidationError:
            return None


def _dedupe_reason_codes(reason_codes: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for reason_code in reason_codes:
        if not reason_code or reason_code in seen:
            continue
        deduped.append(reason_code)
        seen.add(reason_code)
    return deduped
