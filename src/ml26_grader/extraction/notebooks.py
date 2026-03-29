from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nbformat

from ..llm.schemas import (
    EvidencePacket,
    EvidenceSnippet,
    ExtractedSignal,
    ExtractionWarning,
    QuestionId,
)
from ..constants import PUBLIC_DATASET_COLUMNS

SNIPPET_CHAR_LIMIT = 1200
REVIEW_REQUIRED_WARNING_CODES = {
    "configured_notebook_pattern_miss",
    "multiple_notebooks_found",
    "multiple_relevant_notebooks_found",
    "notebook_parse_error",
    "question_section_not_found",
    "question_section_inferred_from_content",
    "empty_evidence_packet",
}

QUESTION_ONLY_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:question|q)\s*[23](?:(?:\s*[\.\-]\s*|\s+)[12])?\s*$",
    re.IGNORECASE,
)
QUESTION_ANCHOR_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:question|q)\s*(?P<question_number>[1-5])"
    r"(?:(?:\s*[\.\-]\s*|\s+)(?P<subquestion_number>\d+))?\b",
    re.IGNORECASE,
)
LEADING_HTML_ANCHOR_PATTERN = re.compile(
    r"^\s*(?:<a\b[^>]*>\s*</a>\s*)+",
    re.IGNORECASE,
)
LEADING_SEPARATOR_PATTERN = re.compile(
    r"^\s*(?:(?:[-*_]{3,})\s*)+",
)
FIT_OR_PREDICT_PATTERN = re.compile(r"(\.fit\s*\()|(\.predict(?:_proba)?\s*\()", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\n+|(?<=[.!?])\s+|(?<=:)\s+)")
FEATURE_ASSIGNMENT_PATTERN = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\s*\[\s*['\"](?P<column>[^'\"]+)['\"]\s*\]\s*=",
    re.IGNORECASE,
)
FEATURE_ASSIGN_CALL_PATTERN = re.compile(
    r"\.assign\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*=",
    re.IGNORECASE,
)
FEATURE_NAME_HINT_PATTERN = re.compile(
    r"(?:_days?$|_flag$|_bucket$|_bin$|_interaction$|_ratio$|^is_|^has_|_count$)",
    re.IGNORECASE,
)
TARGET_COLUMN_HINT_PATTERN = re.compile(
    r"(?:^target$|^label$|^y$|consumer_disputed|consumer disputed|prediction|predicted)",
    re.IGNORECASE,
)


def _compile(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


MAIN_QUESTION_PATTERNS = {
    f"Q{question_number}": re.compile(
        rf"\b(?:question|q)\s*{question_number}\b"
        rf"(?!\s*(?:[\.\-]\s*|\s+)(?:1|2)\b)",
        re.IGNORECASE,
    )
    for question_number in range(1, 6)
}

SUBQUESTION_PATTERNS = {
    "Q2.1": re.compile(
        r"\b(?:question|q)\s*2(?:\s*[\.\-]\s*|\s+)1\b",
        re.IGNORECASE,
    ),
    "Q2.2": re.compile(
        r"\b(?:question|q)\s*2(?:\s*[\.\-]\s*|\s+)2\b",
        re.IGNORECASE,
    ),
    "Q3.1": re.compile(
        r"\b(?:question|q)\s*3(?:\s*[\.\-]\s*|\s+)1\b",
        re.IGNORECASE,
    ),
    "Q3.2": re.compile(
        r"\b(?:question|q)\s*3(?:\s*[\.\-]\s*|\s+)2\b",
        re.IGNORECASE,
    ),
}

MODEL_PATTERNS = {
    "logistic_regression": _compile([r"\blogisticregression\b", r"\blogistic regression\b"]),
    "random_forest": _compile([r"\brandomforest(classifier)?\b", r"\brandom forest\b"]),
    "gradient_boosting": _compile(
        [
            r"\bgradientboosting(classifier)?\b",
            r"\bgradient boosting\b",
            r"\bhistgradientboosting(classifier)?\b",
        ]
    ),
    "xgboost": _compile([r"\bxgboost\b", r"\bxgb(classifier)?\b"]),
    "decision_tree": _compile([r"\bdecisiontree(classifier)?\b", r"\bdecision tree\b"]),
    "svm": _compile([r"\bsvc\b", r"\bsupport vector\b"]),
    "naive_bayes": _compile([r"\bgaussiannb\b", r"\bmultinomialnb\b", r"\bnaive bayes\b"]),
}

METRIC_PATTERNS = {
    "f1_score": _compile([r"\bf1[_\s-]?score\b", r"\bf1\b"]),
    "precision": _compile([r"\bprecision\b"]),
    "recall": _compile([r"\brecall\b"]),
    "roc_auc": _compile([r"\broc[_\s-]?auc\b", r"\broc auc\b"]),
    "pr_auc": _compile([r"\bpr[_\s-]?auc\b", r"\bprecision-recall auc\b"]),
    "accuracy": _compile([r"\baccuracy\b"]),
    "confusion_matrix": _compile([r"\bconfusion matrix\b", r"\bconfusion_matrix\b"]),
    "classification_report": _compile([r"\bclassification report\b", r"\bclassification_report\b"]),
}

PREPROCESSING_RULES = {
    "missing value handling": _compile(
        [r"\bimput", r"\bfillna\b", r"\bdropna\b", r"\bmissing values?\b"]
    ),
    "categorical encoding": _compile(
        [r"\bonehotencoder\b", r"\bget_dummies\b", r"\bencode", r"\bcategorical\b"]
    ),
    "feature scaling": _compile(
        [r"\bstandardscaler\b", r"\bminmaxscaler\b", r"\bscal", r"\bnormaliz"]
    ),
    "preprocessing pipeline": _compile(
        [r"\bcolumntransformer\b", r"\bpreprocess", r"\bpipeline\b", r"\btransformer\b"]
    ),
    "class imbalance handling": _compile(
        [r"\bclass_weight\b", r"\bbalanced\b", r"\bsmote\b", r"\bimbalance\b"]
    ),
}

FEATURE_ENGINEERING_RULES = {
    "engineered features": _compile(
        [
            r"\bfeature engineering\b",
            r"\bfeature_engineering\b",
            r"\bengineered features?\b",
            r"\bderived features?\b",
            r"\bcreated? (?:an?|new|additional) (?:feature|variable|column)s?\b",
            r"\bconstruct(?:ed|ing)? (?:an?|new|derived) (?:feature|variable|column)s?\b",
            r"\binteraction feature\b",
            r"\binteraction term\b",
            r"\bbinary flag\b",
            r"\bderived column\b",
            r"\bfeature transformation\b",
            r"\bresponse_time_days\b",
            r"\bresponse_delay_days\b",
            r"\bnew feature\b",
        ]
    ),
}

DATA_SPLIT_RULES = {
    "train_test_split": _compile(
        [r"\btrain_test_split\b", r"\btrain/test split\b", r"\btest_size\b"]
    ),
    "validation_split": _compile(
        [r"\bvalidation split\b", r"\bcross[-\s]?validation\b", r"\bstratifiedkfold\b", r"\bkfold\b"]
    ),
}

TUNING_RULES = {
    "grid search": _compile([r"\bgridsearchcv\b", r"\bgrid search\b", r"\bparam_grid\b"]),
    "randomized search": _compile(
        [r"\brandomizedsearchcv\b", r"\brandomized search\b", r"\bparam_distributions\b"]
    ),
    "hyperparameter tuning": _compile(
        [r"\bhyperparameter", r"\btuning\b", r"\bbest params?\b", r"\bcv_results_\b"]
    ),
}

COMPARISON_RULES = {
    "side-by-side metric comparison": _compile(
        [r"\bmodel comparison\b", r"\bcompare", r"\bcomparison\b"]
    ),
    "best model selection": _compile(
        [r"\bselected as the model\b", r"\bselected model\b", r"\bbest suited\b", r"\bbest model\b"]
    ),
}

RECOMMENDATION_PATTERNS = _compile(
    [
        r"\brecommend(?:ed|ation)?\b",
        r"\bbest suited\b",
        r"\bbest model\b",
        r"\bselected model\b",
        r"\bchosen\b",
        r"\bpreferred\b",
    ]
)
DEPLOYMENT_PATTERNS = _compile(
    [
        r"\bdeploy(?:ed|ment|ing)?\b",
        r"\bimplement(?:ed|ation)?\b",
        r"\bproduction\b",
        r"\blive use\b",
        r"\boperational use\b",
        r"\brecommend(?:ed|ation)?\b",
    ]
)
BUSINESS_JUSTIFICATION_RULES = {
    "business-risk rationale": _compile(
        [r"\bbusiness\b", r"\brisk\b", r"\bcost\b", r"\bdecision[-\s]?making\b"]
    ),
    "regulatory rationale": _compile([r"\bregulator", r"\bregulatory\b", r"\bcompliance\b"]),
    "operational trade-off rationale": _compile(
        [r"\boperational\b", r"\bresource", r"\bfalse positive\b", r"\bfalse negative\b"]
    ),
    "customer impact rationale": _compile(
        [r"\bcustomer\b", r"\breputation", r"\bretention\b", r"\bescalation\b"]
    ),
}

Q2_FALLBACK_RULES = {
    "problem_type": _compile(
        [r"\bbinary classification\b", r"\bsupervised binary classification\b", r"\bclassification problem\b"]
    ),
    "feature_engineering": tuple(FEATURE_ENGINEERING_RULES["engineered features"]),
    "preprocessing": tuple(
        pattern for patterns in PREPROCESSING_RULES.values() for pattern in patterns
    ),
    "data_split": tuple(pattern for patterns in DATA_SPLIT_RULES.values() for pattern in patterns),
    "tuning": tuple(pattern for patterns in TUNING_RULES.values() for pattern in patterns),
    "models": tuple(pattern for patterns in MODEL_PATTERNS.values() for pattern in patterns),
}

Q3_FALLBACK_RULES = {
    "metrics": tuple(pattern for patterns in METRIC_PATTERNS.values() for pattern in patterns),
    "comparison": tuple(pattern for patterns in COMPARISON_RULES.values() for pattern in patterns),
    "deployment": tuple(DEPLOYMENT_PATTERNS),
    "business": tuple(
        pattern for patterns in BUSINESS_JUSTIFICATION_RULES.values() for pattern in patterns
    ),
}


@dataclass(frozen=True)
class NotebookCell:
    notebook_path: Path
    index: int
    cell_type: str
    source: str
    output_text: str
    normalized_source: str
    normalized_output: str
    lower_source: str
    lower_combined: str

    @property
    def source_ref(self) -> str:
        return f"{self.notebook_path.name}#cell-{self.index}"

    @property
    def combined_text(self) -> str:
        return _normalize_text(f"{self.normalized_source}\n{self.normalized_output}")


@dataclass(frozen=True)
class NotebookQuestionAnalysis:
    notebook_path: Path
    packet: EvidencePacket
    warnings: list[ExtractionWarning]
    notes: list[str]
    has_candidate_evidence: bool = False

    @property
    def score(self) -> int:
        return (
            (2 * len(self.packet.markdown_snippets))
            + (3 * len(self.packet.code_snippets))
            + (2 * len(self.packet.output_snippets))
            + (3 * len(self.packet.extracted_signals))
            + len(self.packet.detected_models)
            + len(self.packet.detected_metrics)
            + len(self.packet.preprocessing_signals)
            + len(self.packet.tuning_signals)
            + len(self.packet.comparison_signals)
            + len(self.packet.business_justification_signals)
        )


@dataclass(frozen=True)
class _QuestionSpan:
    start_index: int
    end_index: int
    mode: str
    subquestion_anchor_count: int


def analyze_notebook_for_question(
    notebook_path: Path,
    question_id: QuestionId,
) -> NotebookQuestionAnalysis:
    cells = load_notebook_cells(notebook_path)
    warnings: list[ExtractionWarning] = []
    notes: list[str] = []

    span = _detect_question_span(cells, question_id)
    if span is None:
        warnings.append(
            ExtractionWarning(
                code="question_section_not_found",
                message=f"Could not locate a reliable {question_id} section in {notebook_path.name}.",
            )
        )
        return NotebookQuestionAnalysis(
            notebook_path=notebook_path,
            packet=EvidencePacket(question_id=question_id, extraction_warnings=_dedupe_warnings(warnings)),
            warnings=_dedupe_warnings(warnings),
            notes=[f"No usable {question_id} section was extracted from {notebook_path.name}."],
            has_candidate_evidence=False,
        )

    selected_cells = cells[span.start_index : span.end_index]
    notes.append(
        f"Selected {question_id} span from cells {span.start_index + 1}-{span.end_index} in {notebook_path.name} using {span.mode}."
    )
    if span.mode == "subquestion_headings":
        warnings.append(
            ExtractionWarning(
                code="subquestion_anchor_missing",
                message=(
                    f"{question_id} section in {notebook_path.name} was located from subquestion headings without a clear top-level question heading."
                ),
            )
        )
    if span.mode == "fallback_content":
        warnings.append(
            ExtractionWarning(
                code="question_section_inferred_from_content",
                message=(
                    f"{question_id} section in {notebook_path.name} was inferred from notebook content rather than explicit headings."
                ),
            )
        )
    if span.subquestion_anchor_count < 2:
        warnings.append(
            ExtractionWarning(
                code="subquestion_anchor_missing",
                message=(
                    f"{question_id} section in {notebook_path.name} does not clearly mark both subquestions."
                ),
            )
        )

    markdown_snippets = _collect_markdown_snippets(selected_cells)
    code_snippets = _collect_code_snippets(selected_cells, question_id)
    output_snippets = _collect_output_snippets(selected_cells, question_id)
    signal_bundle = _collect_signals(selected_cells, question_id)

    packet = EvidencePacket(
        question_id=question_id,
        markdown_snippets=markdown_snippets,
        code_snippets=code_snippets,
        output_snippets=output_snippets,
        extracted_signals=signal_bundle["extracted_signals"],
        extraction_warnings=[],
        detected_models=signal_bundle["detected_models"],
        detected_metrics=signal_bundle["detected_metrics"],
        preprocessing_signals=signal_bundle["preprocessing_signals"],
        tuning_signals=signal_bundle["tuning_signals"],
        comparison_signals=signal_bundle["comparison_signals"],
        business_justification_signals=signal_bundle["business_justification_signals"],
    )
    has_candidate_evidence = packet.has_evidence()

    if span.mode == "fallback_content" and not _fallback_content_packet_is_usable(packet, question_id):
        warnings.append(
            ExtractionWarning(
                code="fallback_content_not_corroborated",
                message=(
                    f"Inferred {question_id} content in {notebook_path.name} did not contain enough corroborating evidence "
                    "to classify the section as usable."
                ),
            )
        )
        return NotebookQuestionAnalysis(
            notebook_path=notebook_path,
            packet=EvidencePacket(
                question_id=question_id,
                extraction_warnings=_dedupe_warnings(warnings),
            ),
            warnings=_dedupe_warnings(warnings),
            notes=[
                *notes,
                f"Fallback-inferred {question_id} span in {notebook_path.name} was rejected because corroborating grading evidence was too weak.",
            ],
            has_candidate_evidence=has_candidate_evidence,
        )

    if span.mode in {"question_heading", "subquestion_headings"} and not _heading_based_packet_is_usable(
        packet,
        question_id,
    ):
        warnings.append(
            ExtractionWarning(
                code="heading_content_not_corroborated",
                message=(
                    f"Headed {question_id} content in {notebook_path.name} did not contain enough substantive "
                    "grading evidence to classify the section as usable."
                ),
            )
        )
        return NotebookQuestionAnalysis(
            notebook_path=notebook_path,
            packet=EvidencePacket(
                question_id=question_id,
                extraction_warnings=_dedupe_warnings(warnings),
            ),
            warnings=_dedupe_warnings(warnings),
            notes=[
                *notes,
                f"Headed {question_id} span in {notebook_path.name} was rejected because corroborating grading evidence was too weak.",
            ],
            has_candidate_evidence=has_candidate_evidence,
        )

    warnings.extend(_coverage_warnings(packet, question_id, notebook_path))
    if not markdown_snippets:
        warnings.append(
            ExtractionWarning(
                code="limited_markdown_evidence",
                message=f"No substantive markdown evidence was extracted for {question_id} from {notebook_path.name}.",
            )
        )
    if not code_snippets:
        warnings.append(
            ExtractionWarning(
                code="limited_code_evidence",
                message=f"No relevant code evidence was extracted for {question_id} from {notebook_path.name}.",
            )
        )
    if not output_snippets:
        warnings.append(
            ExtractionWarning(
                code="limited_output_evidence",
                message=f"No relevant output evidence was extracted for {question_id} from {notebook_path.name}.",
            )
        )

    packet = packet.model_copy(update={"extraction_warnings": _dedupe_warnings(warnings)})
    return NotebookQuestionAnalysis(
        notebook_path=notebook_path,
        packet=packet,
        warnings=packet.extraction_warnings,
        notes=notes,
        has_candidate_evidence=has_candidate_evidence,
    )


def load_notebook_cells(notebook_path: Path) -> list[NotebookCell]:
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    cells: list[NotebookCell] = []
    for index, cell in enumerate(notebook.cells, start=1):
        source = _coerce_text(cell.get("source", ""))
        output_text = _collect_output_text(cell)
        normalized_source = _normalize_text(source)
        normalized_output = _normalize_text(output_text)
        cells.append(
            NotebookCell(
                notebook_path=notebook_path,
                index=index,
                cell_type=cell.get("cell_type", "unknown"),
                source=source,
                output_text=output_text,
                normalized_source=normalized_source,
                normalized_output=normalized_output,
                lower_source=normalized_source.lower(),
                lower_combined=_normalize_text(f"{source}\n{output_text}").lower(),
            )
        )
    return cells


def _detect_question_span(
    cells: list[NotebookCell],
    question_id: QuestionId,
) -> _QuestionSpan | None:
    target_main_anchor = _find_main_anchor(cells, question_id)
    target_subanchors = _find_subquestion_anchors(cells, question_id)

    if target_main_anchor is not None:
        end_index = _find_next_question_boundary(cells, target_main_anchor + 1, question_id)
        return _QuestionSpan(
            start_index=target_main_anchor,
            end_index=end_index,
            mode="question_heading",
            subquestion_anchor_count=len(target_subanchors),
        )

    if target_subanchors:
        start_index = target_subanchors[0]
        end_index = _find_next_question_boundary(cells, start_index + 1, question_id)
        return _QuestionSpan(
            start_index=start_index,
            end_index=end_index,
            mode="subquestion_headings",
            subquestion_anchor_count=len(target_subanchors),
        )

    return _infer_span_from_content(cells, question_id)


def _find_main_anchor(cells: list[NotebookCell], question_id: QuestionId) -> int | None:
    target_question_number = _question_number(question_id)
    for idx, cell in enumerate(cells):
        if cell.cell_type != "markdown":
            continue
        anchor = _extract_question_anchor(cell)
        if anchor == (target_question_number, None):
            return idx
    return None


def _find_subquestion_anchors(cells: list[NotebookCell], question_id: QuestionId) -> list[int]:
    target_question_number = _question_number(question_id)
    indices: list[int] = []
    for idx, cell in enumerate(cells):
        if cell.cell_type != "markdown":
            continue
        anchor = _extract_question_anchor(cell)
        if anchor is None:
            continue
        question_number, subquestion_number = anchor
        if question_number == target_question_number and subquestion_number in {1, 2}:
            indices.append(idx)
    return indices


def _has_subquestion_anchor(cell: NotebookCell, question_id: QuestionId) -> bool:
    target_question_number = _question_number(question_id)
    anchor = _extract_question_anchor(cell)
    return anchor is not None and anchor[0] == target_question_number and anchor[1] in {1, 2}


def _find_next_question_boundary(
    cells: list[NotebookCell],
    start_index: int,
    question_id: QuestionId,
) -> int:
    target_question_number = _question_number(question_id)
    for idx in range(start_index, len(cells)):
        cell = cells[idx]
        if cell.cell_type != "markdown":
            continue
        anchor = _extract_question_anchor(cell)
        if anchor is None:
            continue
        question_number, _ = anchor
        if question_number > target_question_number:
            return idx
    return len(cells)


def _find_previous_question_boundary(
    cells: list[NotebookCell],
    end_index: int,
    question_id: QuestionId,
) -> int:
    target_question_number = _question_number(question_id)
    for idx in range(end_index - 1, -1, -1):
        cell = cells[idx]
        if cell.cell_type != "markdown":
            continue
        anchor = _extract_question_anchor(cell)
        if anchor is None:
            continue
        question_number, _ = anchor
        if question_number != target_question_number:
            return idx + 1
    return 0


def _is_any_subquestion_anchor(cell: NotebookCell) -> bool:
    anchor = _extract_question_anchor(cell)
    return anchor is not None and anchor[1] is not None


def _extract_question_anchor(cell: NotebookCell) -> tuple[int, int | None] | None:
    match = QUESTION_ANCHOR_PATTERN.match(_normalize_question_anchor_source(cell.normalized_source))
    if match is None:
        return None

    question_number = int(match.group("question_number"))
    subquestion_value = match.group("subquestion_number")
    return question_number, (int(subquestion_value) if subquestion_value is not None else None)


def _question_number(question_id: QuestionId) -> int:
    return int(question_id[1:])


def _infer_span_from_content(cells: list[NotebookCell], question_id: QuestionId) -> _QuestionSpan | None:
    hard_stop_index = _find_next_question_boundary(cells, 0, question_id)
    scored_indices: list[int] = []
    for idx, cell in enumerate(cells):
        if idx >= hard_stop_index:
            break
        score = _fallback_content_score(cell, question_id)
        if score >= 2:
            scored_indices.append(idx)

    if not scored_indices:
        return None

    groups: list[list[int]] = []
    current_group = [scored_indices[0]]
    for idx in scored_indices[1:]:
        if idx - current_group[-1] <= 2:
            current_group.append(idx)
            continue
        groups.append(current_group)
        current_group = [idx]
    groups.append(current_group)

    best_group = max(
        groups,
        key=lambda group: (
            len(group),
            sum(_fallback_content_score(cells[idx], question_id) for idx in group),
        ),
    )
    start_index = max(0, best_group[0] - 1)
    start_index = max(
        start_index,
        _find_previous_question_boundary(cells, best_group[0], question_id),
    )
    end_index = min(len(cells), best_group[-1] + 2)
    end_index = min(
        end_index,
        _find_next_question_boundary(cells, best_group[0] + 1, question_id),
    )
    if end_index <= start_index:
        return None
    return _QuestionSpan(
        start_index=start_index,
        end_index=end_index,
        mode="fallback_content",
        subquestion_anchor_count=0,
    )


def _normalize_question_anchor_source(text: str) -> str:
    candidate = text.strip()
    while candidate:
        updated = LEADING_HTML_ANCHOR_PATTERN.sub("", candidate, count=1)
        updated = LEADING_SEPARATOR_PATTERN.sub("", updated, count=1)
        updated = updated.lstrip()
        if updated == candidate:
            return candidate
        candidate = updated
    return candidate


def _fallback_content_score(cell: NotebookCell, question_id: QuestionId) -> int:
    rules = Q2_FALLBACK_RULES if question_id == "Q2" else Q3_FALLBACK_RULES
    score = 0
    for patterns in rules.values():
        if any(pattern.search(cell.lower_combined) for pattern in patterns):
            score += 1
    if question_id == "Q2" and _detect_feature_engineering_signal(cell):
        score += 1
    return score


def _collect_markdown_snippets(cells: list[NotebookCell]) -> list[EvidenceSnippet]:
    snippets: list[EvidenceSnippet] = []
    seen: set[str] = set()
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        if not _is_substantive_markdown(cell):
            continue
        snippet = _make_snippet("md", cell, cell.normalized_source)
        if snippet.content in seen:
            continue
        snippets.append(snippet)
        seen.add(snippet.content)
    return snippets


def _collect_code_snippets(cells: list[NotebookCell], question_id: QuestionId) -> list[EvidenceSnippet]:
    snippets: list[EvidenceSnippet] = []
    seen: set[str] = set()
    for cell in cells:
        if cell.cell_type != "code":
            continue
        if not _is_relevant_code(cell, question_id):
            continue
        snippet = _make_snippet("code", cell, cell.normalized_source)
        if snippet.content in seen:
            continue
        snippets.append(snippet)
        seen.add(snippet.content)
    return snippets


def _collect_output_snippets(cells: list[NotebookCell], question_id: QuestionId) -> list[EvidenceSnippet]:
    snippets: list[EvidenceSnippet] = []
    seen: set[str] = set()
    for cell in cells:
        if cell.cell_type != "code":
            continue
        if not cell.normalized_output:
            continue
        if not (_is_relevant_code(cell, question_id) or _fallback_content_score(cell, question_id) > 0):
            continue
        snippet = _make_snippet("out", cell, cell.normalized_output)
        if snippet.content in seen:
            continue
        snippets.append(snippet)
        seen.add(snippet.content)
    return snippets


def _is_substantive_markdown(cell: NotebookCell) -> bool:
    if not cell.normalized_source:
        return False
    if QUESTION_ONLY_PATTERN.match(cell.normalized_source):
        return False
    word_count = len(cell.normalized_source.split())
    return word_count >= 4 or _fallback_content_score(cell, "Q2") >= 2 or _fallback_content_score(cell, "Q3") >= 2


def _is_relevant_code(cell: NotebookCell, question_id: QuestionId) -> bool:
    if not cell.normalized_source:
        return False
    return (
        _fallback_content_score(cell, question_id) > 0
        or bool(cell.normalized_output)
        or bool(FIT_OR_PREDICT_PATTERN.search(cell.source))
    )


def _collect_signals(cells: list[NotebookCell], question_id: QuestionId) -> dict[str, list]:
    detected_models: set[str] = set()
    detected_metrics: set[str] = set()
    preprocessing_signals: set[str] = set()
    tuning_signals: set[str] = set()
    comparison_signals: set[str] = set()
    business_signals: set[str] = set()
    extracted_signals: dict[tuple[str, str], set[str]] = {}

    for cell in cells:
        if cell.cell_type == "markdown" and not _is_substantive_markdown(cell):
            continue

        text = cell.lower_combined
        source_ref = cell.source_ref

        model_hits = _detect_models(cell.source, text)
        metric_hits = _detect_named_hits(text, METRIC_PATTERNS)
        preprocessing_hits = _detect_named_hits(text, PREPROCESSING_RULES)
        tuning_hits = _detect_named_hits(text, TUNING_RULES)
        comparison_hits = _detect_named_hits(text, COMPARISON_RULES)
        business_hits = _detect_named_hits(text, BUSINESS_JUSTIFICATION_RULES)
        feature_engineering_detected = _detect_feature_engineering_signal(cell)

        detected_models.update(model_hits)
        detected_metrics.update(metric_hits)
        preprocessing_signals.update(preprocessing_hits)
        tuning_signals.update(tuning_hits)
        comparison_signals.update(comparison_hits)
        business_signals.update(business_hits)

        if question_id == "Q2":
            if _matches_any(text, _compile([r"\bbinary classification\b", r"\bsupervised binary classification\b"])):
                _add_extracted_signal(extracted_signals, "ml_problem_type", "binary_classification", source_ref)
            if feature_engineering_detected:
                _add_extracted_signal(extracted_signals, "feature_engineering", "engineered_features", source_ref)
            for split_value in _detect_named_hits(text, DATA_SPLIT_RULES):
                _add_extracted_signal(extracted_signals, "data_split", split_value.replace(" ", "_"), source_ref)
            for tuning_value in tuning_hits:
                _add_extracted_signal(extracted_signals, "hyperparameter_tuning", tuning_value.replace(" ", "_"), source_ref)
        else:
            metric_rationale_present = bool(metric_hits) and bool(
                business_hits
                or _matches_any(
                    text,
                    _compile([r"\bappropriate\b", r"\bgiven\b", r"\bbecause\b", r"\bpreferable\b"]),
                )
            )
            if metric_rationale_present:
                for metric in metric_hits:
                    _add_extracted_signal(extracted_signals, "metric_choice", metric, source_ref)
            if comparison_hits or (len(model_hits) >= 2 and bool(metric_hits)):
                _add_extracted_signal(extracted_signals, "model_comparison", "comparative_performance", source_ref)
            deployment_recommendation = _detect_deployment_recommendation_value(cell)
            if deployment_recommendation is not None:
                _add_extracted_signal(
                    extracted_signals,
                    "deployment_recommendation",
                    deployment_recommendation,
                    source_ref,
                )

        if comparison_hits and business_hits:
            _add_extracted_signal(extracted_signals, "business_justification", "business_rationale_present", source_ref)

    return {
        "detected_models": sorted(detected_models),
        "detected_metrics": sorted(detected_metrics),
        "preprocessing_signals": sorted(preprocessing_signals),
        "tuning_signals": sorted(tuning_signals),
        "comparison_signals": sorted(comparison_signals),
        "business_justification_signals": sorted(business_signals),
        "extracted_signals": [
            ExtractedSignal(signal=signal_name, value=value, evidence_refs=sorted(evidence_refs))
            for (signal_name, value), evidence_refs in sorted(extracted_signals.items())
        ],
    }


def _coverage_warnings(
    packet: EvidencePacket,
    question_id: QuestionId,
    notebook_path: Path,
) -> list[ExtractionWarning]:
    warnings: list[ExtractionWarning] = []
    if question_id == "Q2":
        extracted_signal_names = {(signal.signal, signal.value) for signal in packet.extracted_signals}
        if len(packet.detected_models) < 2:
            warnings.append(
                ExtractionWarning(
                    code="fewer_than_two_models_detected",
                    message=f"Fewer than two models were detected in the {question_id} evidence from {notebook_path.name}.",
                )
            )
        if not packet.preprocessing_signals:
            warnings.append(
                ExtractionWarning(
                    code="missing_preprocessing_signal",
                    message=f"No clear preprocessing signals were detected for {question_id} in {notebook_path.name}.",
                )
            )
        if ("feature_engineering", "engineered_features") not in extracted_signal_names:
            warnings.append(
                ExtractionWarning(
                    code="missing_feature_engineering_signal",
                    message=f"No explicit feature engineering evidence was detected for {question_id} in {notebook_path.name}.",
                )
            )
        if not any(signal.signal == "data_split" for signal in packet.extracted_signals):
            warnings.append(
                ExtractionWarning(
                    code="missing_data_split_signal",
                    message=f"No train/test or validation split evidence was detected for {question_id} in {notebook_path.name}.",
                )
            )
        if not packet.tuning_signals:
            warnings.append(
                ExtractionWarning(
                    code="missing_tuning_signal",
                    message=f"No explicit hyperparameter tuning evidence was detected for {question_id} in {notebook_path.name}.",
                )
            )
    else:
        if not packet.detected_metrics:
            warnings.append(
                ExtractionWarning(
                    code="missing_metric_choice_signal",
                    message=f"No explicit evaluation metric evidence was detected for {question_id} in {notebook_path.name}.",
                )
            )
        if not any(signal.signal == "model_comparison" for signal in packet.extracted_signals):
            warnings.append(
                ExtractionWarning(
                    code="missing_model_comparison_signal",
                    message=f"No explicit model comparison evidence was detected for {question_id} in {notebook_path.name}.",
                )
            )
        if not any(signal.signal == "deployment_recommendation" for signal in packet.extracted_signals):
            warnings.append(
                ExtractionWarning(
                    code="missing_deployment_recommendation_signal",
                    message=f"No explicit deployment recommendation evidence was detected for {question_id} in {notebook_path.name}.",
                )
            )
    return warnings


def _fallback_content_packet_is_usable(
    packet: EvidencePacket,
    question_id: QuestionId,
) -> bool:
    if question_id == "Q2":
        return _q2_fallback_packet_is_usable(packet)
    return _q3_fallback_packet_is_usable(packet)


def _heading_based_packet_is_usable(
    packet: EvidencePacket,
    question_id: QuestionId,
) -> bool:
    if question_id == "Q2":
        return _q2_heading_packet_is_usable(packet)
    return _q3_heading_packet_is_usable(packet)


def _q2_heading_packet_is_usable(packet: EvidencePacket) -> bool:
    workflow_signal_count = _q2_workflow_signal_count(packet)
    return (_has_model_signal(packet) and workflow_signal_count >= 1) or workflow_signal_count >= 2


def _q3_heading_packet_is_usable(packet: EvidencePacket) -> bool:
    return _q3_signal_cluster_size(packet) >= 2 or (
        _has_deployment_signal(packet) and bool(packet.detected_models)
    )


def _q2_fallback_packet_is_usable(packet: EvidencePacket) -> bool:
    return _has_model_signal(packet) and _q2_workflow_signal_count(packet) >= 1


def _q3_fallback_packet_is_usable(packet: EvidencePacket) -> bool:
    return _has_metric_signal(packet) and _q3_signal_cluster_size(packet) >= 2


def _has_model_signal(packet: EvidencePacket) -> bool:
    return bool(packet.detected_models) or any(
        FIT_OR_PREDICT_PATTERN.search(snippet.content)
        for snippet in packet.code_snippets
    )


def _q2_workflow_signal_count(packet: EvidencePacket) -> int:
    return sum(
        (
            bool(packet.preprocessing_signals),
            bool(packet.tuning_signals),
            any(signal.signal == "data_split" for signal in packet.extracted_signals),
            any(signal.signal == "feature_engineering" for signal in packet.extracted_signals),
        )
    )


def _has_metric_signal(packet: EvidencePacket) -> bool:
    return bool(
        packet.detected_metrics
        or any(signal.signal == "metric_choice" for signal in packet.extracted_signals)
    )


def _q3_signal_cluster_size(packet: EvidencePacket) -> int:
    return sum(
        (
            _has_metric_signal(packet),
            bool(
                packet.comparison_signals
                or any(signal.signal == "model_comparison" for signal in packet.extracted_signals)
            ),
            _has_deployment_signal(packet),
        )
    )


def _has_deployment_signal(packet: EvidencePacket) -> bool:
    return any(
        signal.signal == "deployment_recommendation"
        for signal in packet.extracted_signals
    )


def _detect_models(source_text: str, lower_text: str) -> set[str]:
    hits = _detect_named_hits(lower_text, MODEL_PATTERNS)
    for class_name in re.findall(r"\b([A-Z][A-Za-z0-9]+Classifier)\b", source_text):
        hits.add(_to_snake_case(class_name.replace("Classifier", "")))
    return hits


def _detect_feature_engineering_signal(cell: NotebookCell) -> bool:
    if _detect_named_hits(cell.lower_combined, FEATURE_ENGINEERING_RULES):
        return True

    source_text = cell.source
    lower_source = cell.lower_source
    if re.search(r"\b(?:feature_engineering|functiontransformer)\b", lower_source):
        return True

    column_names = FEATURE_ASSIGNMENT_PATTERN.findall(source_text)
    engineered_columns = [
        column_name
        for column_name in column_names
        if _looks_like_engineered_feature_name(column_name)
    ]
    if engineered_columns:
        if _has_engineering_operation(source_text, lower_source):
            return True
        if any(FEATURE_NAME_HINT_PATTERN.search(column_name) for column_name in engineered_columns):
            return True
        if len({column_name.lower() for column_name in engineered_columns}) >= 2:
            return True

    return bool(
        FEATURE_ASSIGN_CALL_PATTERN.search(source_text)
        and _has_engineering_operation(source_text, lower_source)
    )


def _has_engineering_operation(source_text: str, lower_source: str) -> bool:
    return any(
        (
            "pd.cut(" in lower_source,
            "np.where(" in lower_source,
            ".dt." in lower_source,
            ".map(" in lower_source,
            ".replace(" in lower_source,
            ".clip(" in lower_source,
            ".fillna(" in lower_source,
            ".astype(" in lower_source,
            ".groupby(" in lower_source,
            bool(re.search(r"\[[\"'][^\"']+[\"']\]\s*[-+*/]\s*[A-Za-z_\(]", source_text)),
            bool(re.search(r"[A-Za-z_\)]\s*[-+*/]\s*\[[\"'][^\"']+[\"']\]", source_text)),
            "interaction" in lower_source,
        )
    )


def _detect_deployment_recommendation_value(cell: NotebookCell) -> str | None:
    candidate_texts = [text for text in (cell.source, cell.output_text) if _normalize_text(text)]
    if cell.source and cell.output_text:
        candidate_texts.append(f"{cell.source}\n{cell.output_text}")

    matched_context = False
    grounded_models: set[str] = set()
    for text in candidate_texts:
        lower_text = text.lower()
        if not _matches_any(lower_text, DEPLOYMENT_PATTERNS):
            continue
        strong_context = _matches_any(lower_text, RECOMMENDATION_PATTERNS)
        if not strong_context and not _detect_models(text, lower_text):
            continue
        matched_context = True
        grounded_model = _ground_deployment_model(text)
        if grounded_model is not None:
            grounded_models.add(grounded_model)

    if len(grounded_models) == 1:
        return next(iter(grounded_models))
    if matched_context:
        return "deployment_recommendation_present"
    return None


def _ground_deployment_model(text: str) -> str | None:
    segments = [
        segment.strip()
        for segment in SENTENCE_SPLIT_PATTERN.split(text)
        if segment and segment.strip()
    ]
    if not segments:
        return None

    strong_segments = [
        segment for segment in segments if _matches_any(segment.lower(), RECOMMENDATION_PATTERNS)
    ]
    grounded_from_strong = _ground_models_from_segments(strong_segments)
    if len(grounded_from_strong) == 1:
        return next(iter(grounded_from_strong))
    if len(grounded_from_strong) > 1:
        return None

    deployment_segments = [
        segment for segment in segments if _matches_any(segment.lower(), DEPLOYMENT_PATTERNS)
    ]
    grounded_from_deployment = _ground_models_from_segments(deployment_segments)
    if len(grounded_from_deployment) == 1:
        return next(iter(grounded_from_deployment))
    return None


def _ground_models_from_segments(segments: list[str]) -> set[str]:
    grounded_models: set[str] = set()
    for segment in segments:
        segment_models = _detect_models(segment, segment.lower())
        if len(segment_models) == 1:
            grounded_models.update(segment_models)
            continue

        if len(segment_models) <= 1:
            continue

        keyword_windows = _ground_models_from_keyword_windows(segment)
        if len(keyword_windows) == 1:
            grounded_models.update(keyword_windows)
    return grounded_models


def _ground_models_from_keyword_windows(text: str) -> set[str]:
    grounded_models: set[str] = set()
    for match in re.finditer(
        r"\b(?:recommend(?:ed|ation)?|deploy(?:ed|ment|ing)?|best suited|selected|chosen|preferred)\b",
        text,
        flags=re.IGNORECASE,
    ):
        start = match.start()
        end = match.end()
        clause_boundary = max(
            text.rfind(".", 0, start),
            text.rfind(";", 0, start),
            text.rfind("\n", 0, start),
            text.rfind(",", 0, start),
            text.rfind(" but ", 0, start),
            text.rfind(" however ", 0, start),
            text.rfind(" while ", 0, start),
        )
        clause_window = text[max(0, clause_boundary + 1) : min(len(text), end + 100)]
        clause_models = _detect_models(clause_window, clause_window.lower())
        if len(clause_models) == 1:
            grounded_models.update(clause_models)
            continue
        if len(clause_models) > 1:
            continue

        before_window = text[max(0, start - 50) : end]
        after_window = text[start : min(len(text), end + 90)]

        before_models = _detect_models(before_window, before_window.lower())
        after_models = _detect_models(after_window, after_window.lower())
        if len(after_models) == 1:
            grounded_models.update(after_models)
            continue
        if len(before_models) == 1 and not after_models:
            grounded_models.update(before_models)
    return grounded_models


def _detect_named_hits(text: str, rules: dict[str, tuple[re.Pattern[str], ...]]) -> set[str]:
    hits: set[str] = set()
    for name, patterns in rules.items():
        if any(pattern.search(text) for pattern in patterns):
            hits.add(name)
    return hits


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _add_extracted_signal(
    signal_map: dict[tuple[str, str], set[str]],
    signal_name: str,
    value: str,
    source_ref: str,
) -> None:
    key = (signal_name, value)
    if key not in signal_map:
        signal_map[key] = set()
    signal_map[key].add(source_ref)


def _make_snippet(prefix: str, cell: NotebookCell, content: str) -> EvidenceSnippet:
    return EvidenceSnippet(
        snippet_id=f"{prefix}-{_slugify(cell.notebook_path.stem)}-{cell.index}",
        source_ref=cell.source_ref,
        content=_truncate(content),
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "notebook"


def _truncate(text: str, limit: int = SNIPPET_CHAR_LIMIT) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _collect_output_text(cell: nbformat.NotebookNode) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []):
        output_type = output.get("output_type", "")
        if output_type == "stream":
            parts.append(_coerce_text(output.get("text", "")))
            continue
        if output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if "text/plain" in data:
                parts.append(_coerce_text(data["text/plain"]))
            elif "text/markdown" in data:
                parts.append(_coerce_text(data["text/markdown"]))
            continue
        if output_type == "error":
            traceback = output.get("traceback", [])
            if traceback:
                parts.append(_coerce_text(traceback))
            else:
                parts.append(f"{output.get('ename', 'Error')}: {output.get('evalue', '')}")
    return "\n".join(part for part in parts if part)


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def _to_snake_case(value: str) -> str:
    transformed = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return transformed.strip("_")


def _looks_like_engineered_feature_name(column_name: str) -> bool:
    normalized = column_name.strip().lower()
    if not normalized or TARGET_COLUMN_HINT_PATTERN.search(normalized):
        return False
    return normalized not in {column.lower() for column in PUBLIC_DATASET_COLUMNS}
