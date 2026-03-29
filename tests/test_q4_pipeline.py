from __future__ import annotations

import csv
import importlib.util
import pickle
from pathlib import Path
import sys

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook

from ml26_grader.ingest.datasets import DatasetManifest
import ml26_grader.q4.execution as q4_execution_module
from ml26_grader.q4.execution import SubprocessQ4ExecutionBackend
from ml26_grader.q4.models import (
    FailureCategory,
    LeaderboardStatus,
    Q4ArtifactMode,
    Q4ExecutionStatus,
)
from ml26_grader.q4.pipeline import Q4EvaluationPipeline
from ml26_grader.q4.test_support import (
    FixedPredictionPipeline,
    PassthroughTransformer,
    RaisingPredictor,
    StringThresholdPredictor,
    ThresholdPredictor,
)


def test_combined_pipeline_execution_path_succeeds(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "combined_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path,
        rows=[
            {"Complaint ID": 1, "score": 0.8},
            {"Complaint ID": 2, "score": 0.1},
            {"Complaint ID": 3, "score": 0.9},
        ],
        includes_label=False,
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.leaderboard_status == LeaderboardStatus.NOT_RUN
    assert result.artifact_layout.artifact_mode == Q4ArtifactMode.COMBINED_PIPELINE
    assert result.predictions_valid is True
    assert result.labels_available is False
    assert result.prediction_count == 3
    assert result.requirements_env_used is False
    assert result.zero_grade_policy_applied is False
    assert result.zero_grade_policy_reason is None


def test_relative_dataset_manifest_is_absolutized_before_worker_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_root = _build_submission(
        tmp_path / "combined_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_path = _write_dataset_manifest(
        tmp_path / "data",
        rows=[
            {"Complaint ID": 1, "score": 0.8},
            {"Complaint ID": 2, "score": 0.1},
            {"Complaint ID": 3, "score": 0.9},
        ],
        includes_label=False,
    ).path
    monkeypatch.chdir(tmp_path)
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={
            "modeltesting": DatasetManifest(
                name="modeltesting",
                path=Path("data/complaints_modeltesting100.csv"),
                includes_label=False,
            )
        },
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.dataset_path == dataset_path.resolve()
    assert result.requirements_env_used is False


def test_split_preprocessor_and_model_execution_path_succeeds(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "split_submission",
        notebook_code=["predictions = model.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Preprocessor.pkl",
        PassthroughTransformer(),
    )
    _write_pickle(
        submission_root / "12345_Model.pkl",
        StringThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path,
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
            {"Complaint ID": 3, "score": 0.7, "Consumer disputed?": "Yes"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.artifact_layout.artifact_mode == Q4ArtifactMode.SPLIT_PIPELINE
    assert result.predictions_valid is True
    assert any("string-cast values" in log for log in result.execution_logs)


def test_feature_engineering_module_is_loaded_and_used(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "feature_engineering_submission",
        notebook_code=[
            "from feature_engineering import feature_engineering",
            "X = feature_engineering(X)",
            "predictions = model.predict(X)",
        ],
        feature_engineering_source=(
            "def feature_engineering(df):\n"
            "    engineered = df.copy()\n"
            "    engineered['engineered_flag'] = (engineered['days_to_response'] >= 3).astype(int)\n"
            "    return engineered\n"
        ),
    )
    feature_engineering_callable = _load_submission_feature_engineering_callable(
        submission_root / "feature_engineering.py"
    )
    _write_pickle(
        submission_root / "12345_Preprocessor.pkl",
        feature_engineering_callable,
    )
    _write_pickle(
        submission_root / "12345_Model.pkl",
        ThresholdPredictor("engineered_flag", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path,
        rows=[
            {"Complaint ID": 1, "days_to_response": 5, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "days_to_response": 1, "Consumer disputed?": "No"},
            {"Complaint ID": 3, "days_to_response": 4, "Consumer disputed?": "Yes"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.artifact_layout.feature_engineering_required is True
    assert result.predictions_valid is True
    assert any("Imported feature_engineering.py" in log for log in result.execution_logs)


def test_invalid_predictions_fail(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "invalid_predictions_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        FixedPredictionPipeline([1, 2, 0]),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path,
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
            {"Complaint ID": 3, "score": 0.7, "Consumer disputed?": "Yes"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.INVALID_PREDICTIONS
    assert result.zero_grade_policy_applied is True
    assert result.zero_grade_policy_reason == "invalid_predictions"


def test_missing_artifacts_fail_closed_before_execution(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "missing_feature_engineering_submission",
        notebook_code=[
            "from feature_engineering import feature_engineering",
            "X = feature_engineering(X)",
        ],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path,
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.MISSING_ARTIFACTS
    assert "feature_engineering_file" in result.artifact_layout.missing_artifacts
    assert result.zero_grade_policy_applied is True
    assert result.zero_grade_policy_reason == "missing_feature_engineering_file"


def test_missing_requirements_file_is_marked_as_zero_by_policy(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "missing_requirements_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    (submission_root / "12345_requirements.txt").unlink()
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "missing_requirements_dataset",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.MISSING_ARTIFACTS
    assert result.zero_grade_policy_applied is True
    assert result.zero_grade_policy_reason == "missing_requirements_file"
    assert "requirements_file" in result.artifact_layout.missing_artifacts
    assert "zero by policy" in (result.failure_reason or "")


def test_import_failure_is_marked_as_zero_by_policy(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "import_failure_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_unimportable_predictor_pickle(
        tmp_path / "non_importable_module.py",
        submission_root / "12345_Pipeline.pkl",
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "import_failure_dataset",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.IMPORT_FAILURE
    assert result.requirements_env_used is False
    assert result.zero_grade_policy_applied is True
    assert result.zero_grade_policy_reason == "import_failure"


def test_requirements_environment_execution_path_can_run_worker_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_root = _build_submission(
        tmp_path / "requirements_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "requirements_dataset",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    backend = SubprocessQ4ExecutionBackend(
        use_submission_requirements=True,
        requirements_env_root=tmp_path / "requirements_envs",
    )
    call_counts = _patch_requirements_backend_for_current_python(backend, monkeypatch)
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.requirements_env_used is True
    assert call_counts == {"create": 1, "install": 1}
    assert any("requirements-aware Q4 execution enabled" in log for log in result.execution_logs)


def test_requirements_environment_is_reused_for_identical_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_submission = _build_submission(
        tmp_path / "first_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    second_submission = _build_submission(
        tmp_path / "second_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        first_submission / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    _write_pickle(
        second_submission / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "reuse_dataset",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    backend = SubprocessQ4ExecutionBackend(
        use_submission_requirements=True,
        requirements_env_root=tmp_path / "requirements_envs",
        requirements_reuse_envs=True,
    )
    call_counts = _patch_requirements_backend_for_current_python(backend, monkeypatch)
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    first_result = pipeline.evaluate(first_submission)
    second_result = pipeline.evaluate(second_submission)

    assert first_result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert second_result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert first_result.requirements_env_used is True
    assert second_result.requirements_env_used is True
    assert call_counts == {"create": 1, "install": 1}


def test_requirements_environment_creation_failure_is_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_root = _build_submission(
        tmp_path / "requirements_creation_failure_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "requirements_creation_failure_dataset",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    backend = SubprocessQ4ExecutionBackend(
        use_submission_requirements=True,
        requirements_env_root=tmp_path / "requirements_envs",
    )

    def _raise_creation_failure(env_root: Path, logs: list[str]) -> None:
        raise q4_execution_module._RequirementsEnvFailure(
            FailureCategory.REQUIREMENTS_ENV_CREATION_FAILED,
            "Creating the submission-specific requirements environment failed with exit code 1.",
            logs=[*logs, "simulated env creation failure"],
        )

    monkeypatch.setattr(backend, "_create_virtual_environment", _raise_creation_failure)
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.REQUIREMENTS_ENV_CREATION_FAILED
    assert result.requirements_env_used is False
    assert result.zero_grade_policy_reason == "requirements_env_creation_failed"


def test_requirements_install_failure_is_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_root = _build_submission(
        tmp_path / "requirements_install_failure_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "requirements_install_failure_dataset",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    backend = SubprocessQ4ExecutionBackend(
        use_submission_requirements=True,
        requirements_env_root=tmp_path / "requirements_envs",
    )
    call_counts = _patch_requirements_backend_for_current_python(backend, monkeypatch)

    def _raise_install_failure(env_python: Path, requirements_file: Path, logs: list[str]) -> None:
        raise q4_execution_module._RequirementsEnvFailure(
            FailureCategory.REQUIREMENTS_INSTALL_FAILED,
            f"Installing requirements from {requirements_file} failed with exit code 1.",
            logs=[*logs, "simulated requirements install failure"],
        )

    monkeypatch.setattr(backend, "_install_submission_requirements", _raise_install_failure)
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.REQUIREMENTS_INSTALL_FAILED
    assert result.requirements_env_used is False
    assert result.zero_grade_policy_reason == "requirements_install_failed"
    assert call_counts["create"] == 1


def test_import_failure_after_requirements_install_is_distinguished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_root = _build_submission(
        tmp_path / "requirements_import_failure_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_unimportable_predictor_pickle(
        tmp_path / "non_importable_requirements_module.py",
        submission_root / "12345_Pipeline.pkl",
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "requirements_import_failure_dataset",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    backend = SubprocessQ4ExecutionBackend(
        use_submission_requirements=True,
        requirements_env_root=tmp_path / "requirements_envs",
    )
    _patch_requirements_backend_for_current_python(backend, monkeypatch)
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.IMPORT_FAILURE
    assert result.requirements_env_used is True
    assert result.zero_grade_policy_reason == "import_failure_after_requirements_install"


def test_shared_environment_mode_still_works_when_requirements_mode_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission_root = _build_submission(
        tmp_path / "shared_env_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "shared_env_dataset",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    backend = SubprocessQ4ExecutionBackend(use_submission_requirements=False)
    call_counts = _patch_requirements_backend_for_current_python(backend, monkeypatch)
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=backend,
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.requirements_env_used is False
    assert call_counts == {"create": 0, "install": 0}


def test_execution_failure_is_reported(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "raising_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        RaisingPredictor("boom"),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path,
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.FAILED
    assert result.failure_category == FailureCategory.INFERENCE_FAILURE
    assert "boom" in str(result.failure_reason)
    assert result.zero_grade_policy_applied is True
    assert result.zero_grade_policy_reason == "inference_failure"


def test_f1_computation_path_with_labels(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "f1_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        FixedPredictionPipeline([1, 0, 0, 1]),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path,
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
            {"Complaint ID": 3, "score": 0.7, "Consumer disputed?": "Yes"},
            {"Complaint ID": 4, "score": 0.8, "Consumer disputed?": "Yes"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.leaderboard_status == LeaderboardStatus.VALID
    assert result.f1_score == pytest.approx(0.8)
    assert result.predictions_valid is True


def test_f1_computation_path_with_lowercase_whitespace_labels(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "f1_whitespace_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        FixedPredictionPipeline([1, 0, 1, 0]),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "whitespace_labels",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": " yes "},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": " no "},
            {"Complaint ID": 3, "score": 0.7, "Consumer disputed?": "YES"},
            {"Complaint ID": 4, "score": 0.2, "Consumer disputed?": "No"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.labels_available is True
    assert result.leaderboard_status == LeaderboardStatus.VALID
    assert result.f1_score == pytest.approx(1.0)


def test_f1_computation_path_with_numeric_labels(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "f1_numeric_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        FixedPredictionPipeline([1, 0, 1, 0]),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "numeric_labels",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": 1},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": 0},
            {"Complaint ID": 3, "score": 0.7, "Consumer disputed?": 1},
            {"Complaint ID": 4, "score": 0.2, "Consumer disputed?": 0},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.labels_available is True
    assert result.leaderboard_status == LeaderboardStatus.VALID
    assert result.f1_score == pytest.approx(1.0)


def test_mixed_invalid_labels_disable_f1_scoring(tmp_path: Path) -> None:
    submission_root = _build_submission(
        tmp_path / "f1_invalid_labels_submission",
        notebook_code=["predictions = pipeline.predict(X)"],
    )
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        FixedPredictionPipeline([1, 0, 1, 0]),
    )
    dataset_manifest = _write_dataset_manifest(
        tmp_path / "invalid_labels",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
            {"Complaint ID": 3, "score": 0.7, "Consumer disputed?": "Maybe"},
            {"Complaint ID": 4, "score": 0.2, "Consumer disputed?": "No"},
        ],
    )
    pipeline = Q4EvaluationPipeline(
        dataset_manifests={"modeltesting": dataset_manifest},
        execution_backend=SubprocessQ4ExecutionBackend(),
    )

    result = pipeline.evaluate(submission_root)

    assert result.execution_status == Q4ExecutionStatus.SUCCEEDED
    assert result.labels_available is False
    assert result.leaderboard_status == LeaderboardStatus.NOT_RUN
    assert result.f1_score is None
    assert any(
        "Label column was unavailable or not fully binary" in log
        for log in result.execution_logs
    )


def _build_submission(
    submission_root: Path,
    *,
    notebook_code: list[str],
    feature_engineering_source: str | None = None,
) -> Path:
    submission_root.mkdir(parents=True, exist_ok=True)
    _write_notebook(
        submission_root / "12345_Complaints_Notebook.ipynb",
        notebook_code,
    )
    (submission_root / "12345_requirements.txt").write_text("pandas>=2.2\n", encoding="utf-8")
    if feature_engineering_source is not None:
        (submission_root / "feature_engineering.py").write_text(
            feature_engineering_source,
            encoding="utf-8",
        )
    return submission_root


def _write_notebook(path: Path, code_cells: list[str]) -> None:
    notebook = new_notebook(cells=[new_code_cell(source=source) for source in code_cells])
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def _write_pickle(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _write_unimportable_predictor_pickle(module_path: Path, pickle_path: Path) -> None:
    module_path.write_text(
        """
class NonImportablePredictor:
    def predict(self, frame):
        return [1 for _ in range(len(frame))]
""".strip(),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("non_importable_module", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import temporary module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["non_importable_module"] = module
    spec.loader.exec_module(module)
    try:
        _write_pickle(pickle_path, module.NonImportablePredictor())
    finally:
        sys.modules.pop("non_importable_module", None)


def _write_dataset_manifest(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]],
    includes_label: bool = True,
) -> DatasetManifest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_path = tmp_path / "complaints_modeltesting100.csv"
    fieldnames = list(rows[0].keys())
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return DatasetManifest(name="modeltesting", path=dataset_path, includes_label=includes_label)


def _patch_requirements_backend_for_current_python(
    backend: SubprocessQ4ExecutionBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    call_counts = {"create": 0, "install": 0}

    def _fake_create_virtual_environment(env_root: Path, logs: list[str]) -> None:
        call_counts["create"] += 1
        env_root.mkdir(parents=True, exist_ok=True)
        logs.append(f"simulated env creation at {env_root}")

    def _fake_install_requirements(env_python: Path, requirements_file: Path, logs: list[str]) -> None:
        call_counts["install"] += 1
        logs.append(f"simulated requirements install from {requirements_file}")

    monkeypatch.setattr(backend, "_create_virtual_environment", _fake_create_virtual_environment)
    monkeypatch.setattr(backend, "_install_submission_requirements", _fake_install_requirements)
    monkeypatch.setattr(backend, "_venv_python_executable", lambda env_root: Path(sys.executable))
    return call_counts


def _load_submission_feature_engineering_callable(path: Path):
    spec = importlib.util.spec_from_file_location("feature_engineering", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load feature_engineering.py from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["feature_engineering"] = module
    spec.loader.exec_module(module)
    return module.feature_engineering
