from __future__ import annotations

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


class NotebookEvidenceExtractor:
    def __init__(
        self,
        submission_patterns: SubmissionArtifactPatterns | None = None,
    ) -> None:
        self._submission_patterns = submission_patterns or SubmissionArtifactPatterns()

    def extract(self, submission_root: Path, question_id: QuestionId) -> ExtractionResult:
        notebook_paths, discovery_warnings = self._discover_notebooks(submission_root)
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

        notebook_paths = sorted(
            notebook_paths,
            key=_notebook_discovery_sort_key,
            reverse=True,
        )
        analyses: list[NotebookQuestionAnalysis] = []
        warnings = list(discovery_warnings)
        notes = ["Q2 and Q3 are extracted from notebook evidence packets only."]

        for notebook_path in notebook_paths:
            try:
                analyses.append(analyze_notebook_for_question(notebook_path, question_id))
            except Exception as exc:
                warnings.append(
                    ExtractionWarning(
                        code="notebook_parse_error",
                        message=f"Failed to parse {notebook_path.name}: {exc}",
                    )
                )

        candidate_analyses = [
            analysis for analysis in analyses if analysis.has_candidate_evidence
        ]
        plausible_notebook_paths = _plausible_notebook_paths(notebook_paths)
        plausible_notebook_set = set(plausible_notebook_paths)
        plausible_candidate_analyses = [
            analysis
            for analysis in candidate_analyses
            if analysis.notebook_path in plausible_notebook_set
        ] or candidate_analyses
        usable_analyses = [analysis for analysis in analyses if analysis.packet.has_evidence()]
        if not usable_analyses:
            warnings.extend(
                analysis_warning
                for analysis in analyses
                for analysis_warning in analysis.warnings
            )
            warnings.append(
                ExtractionWarning(
                    code="empty_evidence_packet",
                    message=f"No usable {question_id} evidence could be extracted from the available notebooks.",
                )
            )
            if analyses:
                notes.extend(analyses[0].notes)
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

        selected = max(usable_analyses, key=_analysis_selection_key)
        warnings.extend(selected.warnings)
        notes.extend(selected.notes)
        notes.append(f"Selected notebook: {selected.notebook_path.name}")

        if len(plausible_notebook_paths) > 1:
            warnings.append(
                ExtractionWarning(
                    code="multiple_notebooks_found",
                    message=(
                        f"Found {len(plausible_notebook_paths)} plausible notebook candidates "
                        f"(from {len(notebook_paths)} discovered notebooks) and selected "
                        f"{selected.notebook_path.name} as the strongest {question_id} evidence source."
                    ),
                )
            )
        if len(plausible_candidate_analyses) > 1:
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

    def _discover_notebooks(self, submission_root: Path) -> tuple[list[Path], list[ExtractionWarning]]:
        path = submission_root.resolve()
        warnings: list[ExtractionWarning] = []

        if not path.exists():
            return [], [
                ExtractionWarning(
                    code="notebook_missing",
                    message=f"{path} does not exist.",
                )
            ]

        if path.is_file():
            if not is_discoverable_notebook(path):
                return [], [
                    ExtractionWarning(
                        code="notebook_missing",
                        message=f"{path.name} is not an eligible Jupyter notebook file for discovery.",
                    )
                ]
            return [path], warnings

        artifacts = scan_submission_directory(path, self._submission_patterns)
        notebook_paths = list(artifacts.notebooks)
        if notebook_paths:
            return notebook_paths, warnings

        fallback_notebooks = sorted(
            candidate
            for candidate in path.rglob("*.ipynb")
            if is_discoverable_notebook(candidate)
        )
        if fallback_notebooks:
            warnings.append(
                ExtractionWarning(
                    code="configured_notebook_pattern_miss",
                    message=(
                        "No notebook matched the configured submission pattern; "
                        "falling back to all .ipynb files."
                    ),
                )
            )
        return fallback_notebooks, warnings


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
