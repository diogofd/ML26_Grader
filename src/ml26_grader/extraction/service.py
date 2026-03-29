from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..ingest.submission import (
    SubmissionArtifactPatterns,
    is_analysis_likely_notebook,
    is_discoverable_notebook,
    notebook_analysis_preference_score,
    scan_submission_directory,
)
from ..llm.schemas import EvidencePacket, ExtractionWarning, QuestionId
from .notebooks import (
    REVIEW_REQUIRED_WARNING_CODES,
    NotebookQuestionAnalysis,
    analyze_notebook_for_question,
)

UNUSABLE_NOTEBOOK_WARNING_CODES = {
    "question_section_not_found",
    "fallback_content_not_corroborated",
    "heading_content_not_corroborated",
    "notebook_parse_error",
}


class ExtractionStatus(StrEnum):
    READY = "ready"
    FAILED = "failed"


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: QuestionId
    status: ExtractionStatus
    review_required: bool
    evidence_packet: EvidencePacket
    notes: list[str] = Field(default_factory=list)


class EvidenceExtractor(Protocol):
    def extract(self, submission_root: Path, question_id: QuestionId) -> ExtractionResult:
        ...


@dataclass(frozen=True)
class _NotebookCandidatePools:
    configured_candidates: list[Path]
    fallback_candidates: list[Path]
    discovery_warnings: list[ExtractionWarning]


@dataclass(frozen=True)
class _NotebookAttempt:
    notebook_path: Path
    analysis: NotebookQuestionAnalysis | None
    warnings: list[ExtractionWarning]
    source: str

    @property
    def has_candidate_evidence(self) -> bool:
        return self.analysis.has_candidate_evidence if self.analysis is not None else False

    @property
    def is_usable(self) -> bool:
        return bool(self.analysis is not None and self.analysis.packet.has_evidence())

    @property
    def unusable_reason_codes(self) -> list[str]:
        warning_codes = [
            warning.code
            for warning in self.warnings
            if warning.code in UNUSABLE_NOTEBOOK_WARNING_CODES
        ]
        if not self.is_usable:
            warning_codes.append("empty_evidence_packet")
        return _dedupe_reason_codes(warning_codes)


class NotebookEvidenceExtractor:
    def __init__(
        self,
        submission_patterns: SubmissionArtifactPatterns | None = None,
    ) -> None:
        self._submission_patterns = submission_patterns or SubmissionArtifactPatterns()

    def extract(self, submission_root: Path, question_id: QuestionId) -> ExtractionResult:
        candidate_pools = self._discover_notebooks(submission_root)
        configured_candidates = sorted(
            _plausible_notebook_paths(candidate_pools.configured_candidates),
            key=_notebook_discovery_sort_key,
            reverse=True,
        )
        fallback_candidates = sorted(
            _plausible_notebook_paths(candidate_pools.fallback_candidates),
            key=_notebook_discovery_sort_key,
            reverse=True,
        )
        notebook_paths = [*configured_candidates, *fallback_candidates]
        discovery_warnings = candidate_pools.discovery_warnings
        if not notebook_paths:
            warnings = discovery_warnings + [
                ExtractionWarning(
                    code="notebook_missing",
                    message="No Jupyter notebook files were found for evidence extraction.",
                )
            ]
            return ExtractionResult(
                question_id=question_id,
                status=ExtractionStatus.FAILED,
                review_required=True,
                evidence_packet=EvidencePacket(
                    question_id=question_id,
                    extraction_warnings=warnings,
                ),
                notes=["Extraction failed because no notebook could be discovered."],
            )

        warnings = list(discovery_warnings)
        notes = ["Q2 and Q3 are extracted from notebook evidence packets only."]
        configured_attempts = self._analyze_candidates(configured_candidates, question_id, source="configured")
        attempts = list(configured_attempts)
        selected_attempt = _best_usable_attempt(configured_attempts)
        if selected_attempt is None and fallback_candidates:
            if configured_candidates:
                notes.append(
                    f"Trying fallback notebook candidates for {question_id} because configured candidates were unusable."
                )
            fallback_attempts = self._analyze_candidates(fallback_candidates, question_id, source="fallback")
            attempts.extend(fallback_attempts)
            selected_attempt = _best_usable_attempt(fallback_attempts)

        candidate_analyses = [
            attempt.analysis
            for attempt in attempts
            if attempt.analysis is not None and attempt.has_candidate_evidence
        ]
        ambiguous_analysis_notebook_paths = _analysis_notebook_warning_paths(notebook_paths)
        if selected_attempt is None:
            warnings.extend(_attempt_warnings(attempts))
            warnings.extend(
                _candidate_rejection_warnings(
                    attempts,
                    question_id=question_id,
                    selected_notebook_path=None,
                )
            )
            warnings.append(
                ExtractionWarning(
                    code="empty_evidence_packet",
                    message=f"No usable {question_id} evidence could be extracted from the available notebooks.",
                )
            )
            notes.extend(
                _candidate_rejection_notes(
                    attempts,
                    question_id=question_id,
                    selected_notebook_path=None,
                )
            )
            return ExtractionResult(
                question_id=question_id,
                status=ExtractionStatus.FAILED,
                review_required=True,
                evidence_packet=EvidencePacket(
                    question_id=question_id,
                    extraction_warnings=_dedupe_warnings(warnings),
                ),
                notes=notes + ["Extraction failed closed because the evidence packet is empty or unusable."],
            )

        selected = selected_attempt.analysis
        assert selected is not None
        warnings.extend(
            _candidate_rejection_warnings(
                attempts,
                question_id=question_id,
                selected_notebook_path=selected.notebook_path,
            )
        )
        notes.extend(
            _candidate_rejection_notes(
                attempts,
                question_id=question_id,
                selected_notebook_path=selected.notebook_path,
            )
        )
        if _preceding_rejected_attempts(attempts, selected.notebook_path):
            warnings.append(
                ExtractionWarning(
                    code="fallback_notebook_selected",
                    message=(
                        f"Selected {selected.notebook_path.name} for {question_id} after higher-ranked "
                        "notebook candidates were rejected as unusable."
                    ),
                )
            )
            notes.append(
                f"Selected lower-ranked notebook {selected.notebook_path.name} for {question_id} after rejecting higher-ranked unusable candidates."
            )
        warnings.extend(selected.warnings)
        notes.extend(selected.notes)
        notes.append(f"Selected notebook: {selected.notebook_path.name}")

        if len(ambiguous_analysis_notebook_paths) > 1:
            warnings.append(
                ExtractionWarning(
                    code="multiple_notebooks_found",
                    message=(
                        f"Found {len(ambiguous_analysis_notebook_paths)} plausible notebook candidates "
                        f"(from {len(notebook_paths)} discovered notebooks) and selected "
                        f"{selected.notebook_path.name} as the strongest {question_id} evidence source."
                    ),
                )
            )
        if len(candidate_analyses) > 1:
            warnings.append(
                ExtractionWarning(
                    code="multiple_relevant_notebooks_found",
                    message=(
                        f"More than one notebook contained plausible {question_id} evidence; "
                        f"{selected.notebook_path.name} was selected."
                    ),
                )
            )

        deduped_warnings = _dedupe_warnings(warnings)
        packet = selected.packet.model_copy(update={"extraction_warnings": deduped_warnings})
        review_required = any(
            warning.code in REVIEW_REQUIRED_WARNING_CODES for warning in deduped_warnings
        )
        return ExtractionResult(
            question_id=question_id,
            status=ExtractionStatus.READY,
            review_required=review_required,
            evidence_packet=packet,
            notes=notes,
        )

    def _analyze_candidates(
        self,
        notebook_paths: list[Path],
        question_id: QuestionId,
        *,
        source: str,
    ) -> list[_NotebookAttempt]:
        attempts: list[_NotebookAttempt] = []
        for notebook_path in notebook_paths:
            try:
                analysis = analyze_notebook_for_question(notebook_path, question_id)
                attempts.append(
                    _NotebookAttempt(
                        notebook_path=notebook_path,
                        analysis=analysis,
                        warnings=list(analysis.warnings),
                        source=source,
                    )
                )
            except Exception as exc:
                attempts.append(
                    _NotebookAttempt(
                        notebook_path=notebook_path,
                        analysis=None,
                        warnings=[
                            ExtractionWarning(
                                code="notebook_parse_error",
                                message=f"Failed to parse {notebook_path.name}: {exc}",
                            )
                        ],
                        source=source,
                    )
                )
        return attempts

    def _discover_notebooks(self, submission_root: Path) -> _NotebookCandidatePools:
        path = submission_root.resolve()
        warnings: list[ExtractionWarning] = []

        if not path.exists():
            return _NotebookCandidatePools(
                configured_candidates=[],
                fallback_candidates=[],
                discovery_warnings=[
                    ExtractionWarning(
                        code="notebook_missing",
                        message=f"{path} does not exist.",
                    )
                ],
            )

        if path.is_file():
            if not is_discoverable_notebook(path):
                return _NotebookCandidatePools(
                    configured_candidates=[],
                    fallback_candidates=[],
                    discovery_warnings=[
                        ExtractionWarning(
                            code="notebook_missing",
                            message=f"{path.name} is not an eligible Jupyter notebook file for discovery.",
                        )
                    ],
                )
            return _NotebookCandidatePools(
                configured_candidates=[path],
                fallback_candidates=[],
                discovery_warnings=warnings,
            )

        artifacts = scan_submission_directory(path, self._submission_patterns)
        configured_candidates = list(artifacts.notebooks)
        configured_candidate_set = set(configured_candidates)
        fallback_candidates = sorted(
            candidate
            for candidate in path.rglob("*.ipynb")
            if is_discoverable_notebook(candidate) and candidate not in configured_candidate_set
        )

        if fallback_candidates and not configured_candidates:
            warnings.append(
                ExtractionWarning(
                    code="configured_notebook_pattern_miss",
                    message=(
                        "No notebook matched the configured submission pattern; "
                        "falling back to all .ipynb files."
                    ),
                )
            )
        return _NotebookCandidatePools(
            configured_candidates=configured_candidates,
            fallback_candidates=fallback_candidates,
            discovery_warnings=warnings,
        )


PlaceholderEvidenceExtractor = NotebookEvidenceExtractor


def _analysis_selection_key(analysis: NotebookQuestionAnalysis) -> tuple[int, int, int, int, int, str]:
    packet = analysis.packet
    direct_evidence_score = (
        (3 * len(packet.code_snippets))
        + (2 * len(packet.output_snippets))
        + (2 * len(packet.extracted_signals))
    )
    warning_codes = {warning.code for warning in analysis.warnings}
    structure_bonus = 0
    if "question_section_inferred_from_content" not in warning_codes:
        structure_bonus += 6
    if "subquestion_anchor_missing" not in warning_codes:
        structure_bonus += 2
    narrative_bonus = min(len(packet.markdown_snippets), 4)
    structure_penalty = sum(
        1
        for code in (
            "question_section_inferred_from_content",
            "subquestion_anchor_missing",
            "limited_code_evidence",
            "limited_output_evidence",
        )
        if code in warning_codes
    )
    notebook_bias = 2 * notebook_analysis_preference_score(analysis.notebook_path)
    selection_total = (
        analysis.score
        + direct_evidence_score
        + structure_bonus
        + narrative_bonus
        + notebook_bias
        - structure_penalty
    )
    return (
        selection_total,
        direct_evidence_score,
        analysis.score,
        notebook_bias,
        -len(analysis.warnings),
        analysis.notebook_path.name.lower(),
    )


def _notebook_discovery_sort_key(path: Path) -> tuple[int, str]:
    return notebook_analysis_preference_score(path), path.name.lower()


def _plausible_notebook_paths(notebook_paths: list[Path]) -> list[Path]:
    plausible_paths = [path for path in notebook_paths if is_analysis_likely_notebook(path)]
    return plausible_paths or notebook_paths


def _analysis_notebook_warning_paths(notebook_paths: list[Path]) -> list[Path]:
    return [path for path in notebook_paths if is_analysis_likely_notebook(path)]


def _best_usable_attempt(attempts: list[_NotebookAttempt]) -> _NotebookAttempt | None:
    usable_attempts = [attempt for attempt in attempts if attempt.is_usable and attempt.analysis is not None]
    if not usable_attempts:
        return None
    return max(
        usable_attempts,
        key=lambda attempt: _analysis_selection_key(attempt.analysis),
    )


def _attempt_warnings(attempts: list[_NotebookAttempt]) -> list[ExtractionWarning]:
    return [
        warning
        for attempt in attempts
        for warning in attempt.warnings
    ]


def _candidate_rejection_warnings(
    attempts: list[_NotebookAttempt],
    *,
    question_id: QuestionId,
    selected_notebook_path: Path | None,
) -> list[ExtractionWarning]:
    warnings: list[ExtractionWarning] = []
    for attempt in _preceding_rejected_attempts(attempts, selected_notebook_path):
        warning_codes = ", ".join(attempt.unusable_reason_codes)
        warnings.append(
            ExtractionWarning(
                code="notebook_candidate_rejected_unusable",
                message=(
                    f"Rejected notebook candidate {attempt.notebook_path.name} for {question_id} "
                    f"because extraction was unusable: {warning_codes}."
                ),
            )
        )
    return warnings


def _candidate_rejection_notes(
    attempts: list[_NotebookAttempt],
    *,
    question_id: QuestionId,
    selected_notebook_path: Path | None,
) -> list[str]:
    notes: list[str] = []
    for attempt in _preceding_rejected_attempts(attempts, selected_notebook_path):
        warning_codes = ", ".join(attempt.unusable_reason_codes)
        notes.append(
            f"Rejected notebook candidate {attempt.notebook_path.name} for {question_id} because extraction was unusable ({warning_codes})."
        )
    return notes


def _preceding_rejected_attempts(
    attempts: list[_NotebookAttempt],
    selected_notebook_path: Path | None,
) -> list[_NotebookAttempt]:
    if selected_notebook_path is None:
        return [attempt for attempt in attempts if not attempt.is_usable]

    rejected_attempts: list[_NotebookAttempt] = []
    for attempt in attempts:
        if attempt.notebook_path == selected_notebook_path:
            break
        if not attempt.is_usable:
            rejected_attempts.append(attempt)
    return rejected_attempts


def _dedupe_reason_codes(reason_codes: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for reason_code in reason_codes:
        if not reason_code or reason_code in seen:
            continue
        deduped.append(reason_code)
        seen.add(reason_code)
    return deduped


def _dedupe_warnings(warnings: list[ExtractionWarning]) -> list[ExtractionWarning]:
    seen: set[tuple[str, str]] = set()
    deduped: list[ExtractionWarning] = []
    for warning in warnings:
        key = (warning.code, warning.message)
        if key in seen:
            continue
        deduped.append(warning)
        seen.add(key)
    return deduped
