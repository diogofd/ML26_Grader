from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import nbformat

from ..config import GradingConfig
from ..ingest.datasets import DatasetManifest, dataset_manifest_map
from ..ingest.submission import SubmissionArtifactPatterns, scan_submission_directory
from .deterministic import (
    compute_binary_f1,
    normalize_binary_label,
    validate_binary_predictions,
    validate_prediction_count,
)
from .execution import (
    DisabledQ4ExecutionBackend,
    Q4ExecutionBackend,
    Q4ExecutionRequest,
)
from .models import (
    FailureCategory,
    LeaderboardStatus,
    Q4ArtifactLayout,
    Q4ArtifactMode,
    Q4EvaluationResult,
    Q4ExecutionStatus,
)

LABEL_COLUMN = "Consumer disputed?"
FEATURE_ENGINEERING_PATTERNS = (
    re.compile(r"from\s+feature_engineering\s+import\s+feature_engineering", re.IGNORECASE),
    re.compile(r"\bfeature_engineering\s*\(", re.IGNORECASE),
    re.compile(r"\bfunctiontransformer\s*\(\s*feature_engineering", re.IGNORECASE),
    re.compile(r"%%writefile\s+feature_engineering\.py", re.IGNORECASE),
)


@dataclass(frozen=True)
class _DatasetInspection:
    manifest: DatasetManifest
    row_count: int
    labels: list[int] | None
    notes: list[str]


class Q4EvaluationPipeline:
    def __init__(
        self,
        patterns: SubmissionArtifactPatterns | None = None,
        *,
        dataset_manifests: Mapping[str, DatasetManifest] | None = None,
        timeout_seconds: int = 60,
        execution_backend: Q4ExecutionBackend | None = None,
    ) -> None:
        self._patterns = patterns or SubmissionArtifactPatterns()
        self._dataset_manifests = dict(dataset_manifests or _default_dataset_manifests())
        self._timeout_seconds = timeout_seconds
        self._execution_backend = execution_backend or DisabledQ4ExecutionBackend()

    @classmethod
    def from_config(
        cls,
        grading_config: GradingConfig,
        *,
        base_dir: Path = Path("."),
        execution_backend: Q4ExecutionBackend | None = None,
    ) -> "Q4EvaluationPipeline":
        return cls(
            patterns=grading_config.submission,
            dataset_manifests={
                "training": DatasetManifest(
                    name="training",
                    path=base_dir / grading_config.public_datasets.training,
                    includes_label=True,
                ),
                "test": DatasetManifest(
                    name="test",
                    path=base_dir / grading_config.public_datasets.test,
                    includes_label=True,
                ),
                "modeltesting": DatasetManifest(
                    name="modeltesting",
                    path=base_dir / grading_config.public_datasets.modeltesting,
                    includes_label=True,
                ),
            },
            timeout_seconds=grading_config.q4.timeout_seconds,
            execution_backend=execution_backend,
        )

    def inspect_artifacts(self, submission_root: Path) -> Q4ArtifactLayout:
        artifacts = scan_submission_directory(submission_root, self._patterns)

        notebook = artifacts.notebooks[0] if artifacts.notebooks else None
        requirements_file = artifacts.requirements_files[0] if artifacts.requirements_files else None
        feature_engineering_file = (
            artifacts.feature_engineering_files[0]
            if artifacts.feature_engineering_files
            else None
        )
        combined_pipeline = (
            artifacts.combined_pipeline_files[0]
            if artifacts.combined_pipeline_files
            else None
        )
        split_preprocessor = (
            artifacts.split_preprocessor_files[0]
            if artifacts.split_preprocessor_files
            else None
        )
        split_model = artifacts.split_model_files[0] if artifacts.split_model_files else None

        missing_artifacts: list[str] = []
        if notebook is None:
            missing_artifacts.append("notebook")
        if requirements_file is None:
            missing_artifacts.append("requirements_file")

        feature_engineering_required = (
            notebook is not None and _notebook_requires_feature_engineering(notebook)
        )
        if feature_engineering_required and feature_engineering_file is None:
            missing_artifacts.append("feature_engineering_file")

        if combined_pipeline is not None:
            artifact_mode = Q4ArtifactMode.COMBINED_PIPELINE
        elif split_preprocessor is not None and split_model is not None:
            artifact_mode = Q4ArtifactMode.SPLIT_PIPELINE
        else:
            artifact_mode = Q4ArtifactMode.MISSING
            if split_preprocessor is None and split_model is None:
                missing_artifacts.append("pipeline_artifact")
            else:
                if split_preprocessor is None:
                    missing_artifacts.append("split_preprocessor")
                if split_model is None:
                    missing_artifacts.append("split_model")

        return Q4ArtifactLayout(
            submission_root=artifacts.submission_root,
            notebook=notebook,
            requirements_file=requirements_file,
            feature_engineering_file=feature_engineering_file,
            combined_pipeline=combined_pipeline,
            split_preprocessor=split_preprocessor,
            split_model=split_model,
            artifact_mode=artifact_mode,
            feature_engineering_required=feature_engineering_required,
            missing_artifacts=sorted(set(missing_artifacts)),
        )

    def evaluate(
        self,
        submission_root: Path,
        *,
        dataset_name: str = "modeltesting",
    ) -> Q4EvaluationResult:
        try:
            artifact_layout = self.inspect_artifacts(submission_root)
        except (FileNotFoundError, NotADirectoryError) as exc:
            return self._failure_result(
                artifact_layout=Q4ArtifactLayout(
                    submission_root=submission_root.resolve(),
                    missing_artifacts=["submission_root"],
                ),
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                failure_category=FailureCategory.MISSING_ARTIFACTS,
                failure_reason=str(exc),
            )

        if artifact_layout.missing_artifacts:
            return self._failure_result(
                artifact_layout=artifact_layout,
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                failure_category=FailureCategory.MISSING_ARTIFACTS,
                failure_reason=(
                    "Missing required Q4 artifacts: "
                    + ", ".join(artifact_layout.missing_artifacts)
                ),
                execution_logs=[
                    f"Detected artifact mode: {artifact_layout.artifact_mode.value}.",
                    "Artifact validation failed before any execution step.",
                ],
            )

        try:
            dataset_inspection = self._inspect_dataset(dataset_name)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            return self._failure_result(
                artifact_layout=artifact_layout,
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                failure_category=FailureCategory.LOAD_FAILURE,
                failure_reason=f"Dataset inspection failed: {exc}",
            )

        execution_request = Q4ExecutionRequest(
            submission_root=artifact_layout.submission_root,
            artifact_layout=artifact_layout,
            dataset_name=dataset_inspection.manifest.name,
            dataset_path=dataset_inspection.manifest.path,
            timeout_seconds=self._timeout_seconds,
            input_row_count=dataset_inspection.row_count,
        )
        try:
            execution_response = self._execution_backend.run(execution_request)
        except Exception as exc:
            return self._failure_result(
                artifact_layout=artifact_layout,
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                dataset_name=dataset_inspection.manifest.name,
                dataset_path=dataset_inspection.manifest.path,
                input_row_count=dataset_inspection.row_count,
                labels_available=dataset_inspection.labels is not None,
                failure_category=FailureCategory.INFERENCE_FAILURE,
                failure_reason=f"Q4 execution backend failed: {exc}",
                execution_logs=list(dataset_inspection.notes),
            )
        execution_logs = [*dataset_inspection.notes, *execution_response.execution_logs]

        if execution_response.execution_status != Q4ExecutionStatus.SUCCEEDED:
            return Q4EvaluationResult(
                execution_status=execution_response.execution_status,
                leaderboard_status=(
                    LeaderboardStatus.FAILED
                    if execution_response.execution_status == Q4ExecutionStatus.FAILED
                    else LeaderboardStatus.NOT_RUN
                ),
                artifact_layout=artifact_layout,
                dataset_name=dataset_inspection.manifest.name,
                dataset_path=dataset_inspection.manifest.path,
                input_row_count=dataset_inspection.row_count,
                prediction_count=len(execution_response.predictions),
                predictions_valid=False,
                labels_available=dataset_inspection.labels is not None,
                failure_category=execution_response.failure_category,
                failure_reason=execution_response.failure_reason,
                execution_logs=execution_logs,
            )

        if not execution_response.predictions:
            return self._failure_result(
                artifact_layout=artifact_layout,
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                dataset_name=dataset_inspection.manifest.name,
                dataset_path=dataset_inspection.manifest.path,
                input_row_count=dataset_inspection.row_count,
                prediction_count=0,
                labels_available=dataset_inspection.labels is not None,
                failure_category=FailureCategory.EMPTY_PREDICTIONS,
                failure_reason="Prediction output is empty.",
                execution_logs=execution_logs,
            )

        try:
            normalized_predictions = validate_binary_predictions(execution_response.predictions)
        except ValueError as exc:
            return self._failure_result(
                artifact_layout=artifact_layout,
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                dataset_name=dataset_inspection.manifest.name,
                dataset_path=dataset_inspection.manifest.path,
                input_row_count=dataset_inspection.row_count,
                prediction_count=len(execution_response.predictions),
                labels_available=dataset_inspection.labels is not None,
                failure_category=FailureCategory.INVALID_PREDICTIONS,
                failure_reason=str(exc),
                execution_logs=execution_logs,
            )

        try:
            validate_prediction_count(normalized_predictions, dataset_inspection.row_count)
        except ValueError as exc:
            return self._failure_result(
                artifact_layout=artifact_layout,
                execution_status=Q4ExecutionStatus.FAILED,
                leaderboard_status=LeaderboardStatus.FAILED,
                dataset_name=dataset_inspection.manifest.name,
                dataset_path=dataset_inspection.manifest.path,
                input_row_count=dataset_inspection.row_count,
                prediction_count=len(normalized_predictions),
                labels_available=dataset_inspection.labels is not None,
                failure_category=FailureCategory.PREDICTION_COUNT_MISMATCH,
                failure_reason=str(exc),
                execution_logs=execution_logs,
            )

        f1_score: float | None = None
        leaderboard_status = LeaderboardStatus.NOT_RUN
        labels_available = dataset_inspection.labels is not None
        if labels_available:
            f1_score = compute_binary_f1(dataset_inspection.labels, normalized_predictions)
            leaderboard_status = LeaderboardStatus.VALID
            execution_logs.append(
                f"Computed F1 score from {len(dataset_inspection.labels)} labels."
            )
        else:
            execution_logs.append(
                "Dataset labels were unavailable or unusable, so no F1 score was computed."
            )

        execution_logs.append(
            f"Validated {len(normalized_predictions)} binary predictions against {dataset_inspection.row_count} input rows."
        )
        return Q4EvaluationResult(
            execution_status=Q4ExecutionStatus.SUCCEEDED,
            leaderboard_status=leaderboard_status,
            artifact_layout=artifact_layout,
            dataset_name=dataset_inspection.manifest.name,
            dataset_path=dataset_inspection.manifest.path,
            input_row_count=dataset_inspection.row_count,
            prediction_count=len(normalized_predictions),
            predictions_valid=True,
            labels_available=labels_available,
            f1_score=f1_score,
            execution_logs=execution_logs,
        )

    def evaluate_placeholder(
        self,
        submission_root: Path,
        *,
        dataset_name: str = "modeltesting",
    ) -> Q4EvaluationResult:
        return self.evaluate(submission_root, dataset_name=dataset_name)

    def _inspect_dataset(self, dataset_name: str) -> _DatasetInspection:
        try:
            manifest = self._dataset_manifests[dataset_name]
        except KeyError as exc:
            raise KeyError(f"Unsupported Q4 dataset: {dataset_name}") from exc

        row_count = 0
        labels: list[int] | None = [] if manifest.includes_label else None
        invalid_label_found = False

        with manifest.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{manifest.path} does not contain a valid CSV header.")
            label_column_present = LABEL_COLUMN in reader.fieldnames
            if labels is not None and not label_column_present:
                labels = None
            for row in reader:
                row_count += 1
                if labels is None or not label_column_present or invalid_label_found:
                    continue
                try:
                    labels.append(normalize_binary_label(row.get(LABEL_COLUMN, "")))
                except ValueError:
                    invalid_label_found = True
                    labels = None

        if row_count < 1:
            raise ValueError(f"{manifest.path} contains no input rows.")

        notes = [
            f"Loaded dataset {manifest.name} from {manifest.path}.",
            f"Dataset row count: {row_count}.",
        ]
        if manifest.includes_label:
            if labels is None:
                notes.append(
                    "Label column was unavailable or not fully binary, so F1 scoring is disabled for this run."
                )
            else:
                notes.append(f"Loaded {len(labels)} binary labels for F1 scoring.")

        return _DatasetInspection(
            manifest=manifest,
            row_count=row_count,
            labels=labels,
            notes=notes,
        )

    def _failure_result(
        self,
        *,
        artifact_layout: Q4ArtifactLayout,
        execution_status: Q4ExecutionStatus,
        leaderboard_status: LeaderboardStatus,
        failure_category: FailureCategory,
        failure_reason: str,
        dataset_name: str | None = None,
        dataset_path: Path | None = None,
        input_row_count: int | None = None,
        prediction_count: int | None = None,
        labels_available: bool = False,
        execution_logs: list[str] | None = None,
    ) -> Q4EvaluationResult:
        return Q4EvaluationResult(
            execution_status=execution_status,
            leaderboard_status=leaderboard_status,
            artifact_layout=artifact_layout,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            input_row_count=input_row_count,
            prediction_count=prediction_count,
            predictions_valid=False,
            labels_available=labels_available,
            failure_category=failure_category,
            failure_reason=failure_reason,
            execution_logs=list(execution_logs or []),
        )


def _default_dataset_manifests() -> dict[str, DatasetManifest]:
    manifests = dataset_manifest_map()
    return {
        "training": manifests["complaints_training"],
        "test": manifests["complaints_test"],
        "modeltesting": manifests["complaints_modeltesting100"],
    }


def _notebook_requires_feature_engineering(notebook_path: Path) -> bool:
    try:
        with notebook_path.open("r", encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        candidate_sources = [
            str(cell.get("source", ""))
            for cell in notebook.cells
            if cell.get("cell_type") == "code"
        ]
    except Exception:
        candidate_sources = [notebook_path.read_text(encoding="utf-8", errors="ignore")]

    combined_source = "\n".join(candidate_sources)
    return any(pattern.search(combined_source) for pattern in FEATURE_ENGINEERING_PATTERNS)
