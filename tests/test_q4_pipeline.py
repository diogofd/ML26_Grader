from __future__ import annotations

import csv
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from ml26_grader.ingest.datasets import DatasetManifest
from ml26_grader.q4.execution import Q4ExecutionRequest, Q4ExecutionResponse
from ml26_grader.q4.models import (
    FailureCategory,
    LeaderboardStatus,
    Q4ArtifactMode,
    Q4ExecutionStatus,
)
from ml26_grader.q4.pipeline import Q4EvaluationPipeline


class FakeExecutionBackend:
    def __init__(self, response: Q4ExecutionResponse) -> None:
        self.response = response
        self.requests: list[Q4ExecutionRequest] = []

    def run(self, request: Q4ExecutionRequest) -> Q4ExecutionResponse:
        self.requests.append(request)
        return self.response


def test_valid_combined_pipeline_submission_shape(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "combined_submission",
        notebook_code=["model.predict(X)"],
        combined_pipeline=True,
    )
    dataset_manifest = _write_dataset_manifest(tmp_path, labels=["Yes", "No", "Yes"])
    backend = FakeExecutionBackend(
        Q4ExecutionResponse(
            backend_name="fake",
            execution_status=Q4ExecutionStatus.SUCCEEDED,
            predictions=[1, 0, 1],
        )
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.artifact_layout.artifact_mode == Q4ArtifactMode.COMBINED_PIPELINE
    assert result.predictions_valid is True
    assert result.failure_reason is None
    assert backend.requests[0].artifact_layout.combined_pipeline is not None


def test_valid_split_model_submission_shape(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "split_submission",
        notebook_code=[
            "from feature_engineering import feature_engineering",
            "X = feature_engineering(X)",
            "pred = model.predict(X)",
        ],
        split_pipeline=True,
        include_feature_engineering=True,
    )
    dataset_manifest = _write_dataset_manifest(tmp_path, labels=["Yes", "No", "No"])
    backend = FakeExecutionBackend(
        Q4ExecutionResponse(
            backend_name="fake",
            execution_status=Q4ExecutionStatus.SUCCEEDED,
            predictions=[1, 0, 0],
        )
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.artifact_layout.artifact_mode == Q4ArtifactMode.SPLIT_PIPELINE
    assert result.artifact_layout.feature_engineering_required is True
    assert result.artifact_layout.feature_engineering_file is not None


def test_missing_artifact_failures(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "missing_feature_engineering",
        notebook_code=[
            "from feature_engineering import feature_engineering",
            "X = feature_engineering(X)",
        ],
        combined_pipeline=True,
        include_feature_engineering=False,
    )
    dataset_manifest = _write_dataset_manifest(tmp_path, labels=["Yes", "No"])
    pipeline = Q4EvaluationPipeline(dataset_manifests={"modeltesting": dataset_manifest})

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.MISSING_ARTIFACTS
    assert "feature_engineering_file" in result.artifact_layout.missing_artifacts


def test_invalid_prediction_count_fails(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "prediction_count_mismatch",
        notebook_code=["pred = model.predict(X)"],
        combined_pipeline=True,
    )
    dataset_manifest = _write_dataset_manifest(tmp_path, labels=["Yes", "No", "Yes"])
    backend = FakeExecutionBackend(
        Q4ExecutionResponse(
            backend_name="fake",
            execution_status=Q4ExecutionStatus.SUCCEEDED,
            predictions=[1, 0],
        )
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.PREDICTION_COUNT_MISMATCH
    assert result.predictions_valid is False


def test_non_binary_predictions_fail(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "invalid_predictions",
        notebook_code=["pred = model.predict(X)"],
        combined_pipeline=True,
    )
    dataset_manifest = _write_dataset_manifest(tmp_path, labels=["Yes", "No", "Yes"])
    backend = FakeExecutionBackend(
        Q4ExecutionResponse(
            backend_name="fake",
            execution_status=Q4ExecutionStatus.SUCCEEDED,
            predictions=[1, 2, 0],
        )
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.INVALID_PREDICTIONS
    assert result.predictions_valid is False


def test_empty_predictions_fail(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "empty_predictions",
        notebook_code=["pred = model.predict(X)"],
        combined_pipeline=True,
    )
    dataset_manifest = _write_dataset_manifest(tmp_path, labels=["Yes", "No"])
    backend = FakeExecutionBackend(
        Q4ExecutionResponse(
            backend_name="fake",
            execution_status=Q4ExecutionStatus.SUCCEEDED,
            predictions=[],
        )
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.EMPTY_PREDICTIONS
    assert result.prediction_count == 0


def test_f1_computation_path(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "f1_submission",
        notebook_code=["pred = model.predict(X)"],
        combined_pipeline=True,
    )
    dataset_manifest = _write_dataset_manifest(tmp_path, labels=["Yes", "No", "Yes", "Yes"])
    backend = FakeExecutionBackend(
        Q4ExecutionResponse(
            backend_name="fake",
            execution_status=Q4ExecutionStatus.SUCCEEDED,
            predictions=[1, 0, 0, 1],
        )
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.leaderboard_status == LeaderboardStatus.VALID
    assert result.f1_score == pytest.approx(0.8)
    assert result.predictions_valid is True


def _build_submission(
    submission_root: Path,
    *,
    notebook_code: list[str],
    combined_pipeline: bool = False,
    split_pipeline: bool = False,
    include_feature_engineering: bool = False,
) -> Path:
    submission_root.mkdir(parents=True, exist_ok=True)
    _write_notebook(
        submission_root / "12345_Complaints_Notebook.ipynb",
        notebook_code,
    )
    (submission_root / "12345_requirements.txt").write_text("scikit-learn==1.5.0\n", encoding="utf-8")
    if include_feature_engineering:
        (submission_root / "feature_engineering.py").write_text(
            "def feature_engineering(df):\n    return df\n",
            encoding="utf-8",
        )
    if combined_pipeline:
        (submission_root / "12345_Pipeline.pkl").write_bytes(b"pickle-placeholder")
    if split_pipeline:
        (submission_root / "12345_Preprocessor.pkl").write_bytes(b"preprocessor-placeholder")
        (submission_root / "12345_Model.pkl").write_bytes(b"model-placeholder")
    return submission_root


def _write_notebook(path: Path, code_cells: list[str]) -> None:
    notebook = new_notebook(cells=[new_code_cell(source=source) for source in code_cells])
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def _write_dataset_manifest(tmp_path: Path, *, labels: list[str]) -> DatasetManifest:
    dataset_path = tmp_path / "complaints_modeltesting100.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Complaint ID", "Consumer disputed?"])
        writer.writeheader()
        for index, label in enumerate(labels, start=1):
            writer.writerow(
                {
                    "Complaint ID": index,
                    "Consumer disputed?": label,
                }
            )
    return DatasetManifest(name="modeltesting", path=dataset_path, includes_label=True)
