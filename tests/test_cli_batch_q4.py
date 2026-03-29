from __future__ import annotations

import csv
import json
import pickle
import sys
import zipfile
from pathlib import Path

import nbformat
import pytest

from ml26_grader.cli import main
import ml26_grader.q4.execution as q4_execution_module
from ml26_grader.q4.execution import SubprocessQ4ExecutionBackend
from ml26_grader.q4.test_support import FixedPredictionPipeline, ThresholdPredictor


def test_cli_batch_q4_success_path_writes_summary_and_leaderboard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _write_dataset(
        tmp_path / "modeltesting.csv",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.6, "Consumer disputed?": "Yes"},
            {"Complaint ID": 3, "score": 0.4, "Consumer disputed?": "No"},
            {"Complaint ID": 4, "score": 0.2, "Consumer disputed?": "No"},
        ],
    )
    config_path = _write_grading_config(tmp_path / "grading.toml", dataset_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    extract_root = tmp_path / "extracted"
    output_dir = tmp_path / "out"

    zipped_student_dir = batch_dir / "Student Zip"
    zipped_student_dir.mkdir()
    zipped_submission = _build_submission_tree(
        tmp_path / "zip_submission_source",
        predictor=ThresholdPredictor("score", threshold=0.5),
    )
    _write_submission_zip(zipped_student_dir / "submission.zip", zipped_submission)

    extracted_student_dir = batch_dir / "Student Extracted"
    _build_submission_tree(
        extracted_student_dir,
        predictor=ThresholdPredictor("score", threshold=0.75),
    )

    exit_code = main(
        [
            "grade-q4-batch",
            str(batch_dir),
            "--config",
            str(config_path),
            "--extract-root",
            str(extract_root),
            "--output-dir",
            str(output_dir),
            "--dataset",
            "modeltesting",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert len(payload["submissions"]) == 2
    assert output_dir.joinpath("q4_summary.json").exists()
    assert output_dir.joinpath("q4_summary.csv").exists()
    assert output_dir.joinpath("q4_leaderboard.csv").exists()

    summary_rows = list(csv.DictReader(output_dir.joinpath("q4_summary.csv").open("r", encoding="utf-8")))
    assert len(summary_rows) == 2
    summary_by_student = {row["student_folder_name"]: row for row in summary_rows}
    assert summary_by_student["Student Zip"]["execution_status"] == "succeeded"
    assert summary_by_student["Student Extracted"]["execution_status"] == "succeeded"
    assert summary_by_student["Student Zip"]["rank"] == "1"
    assert summary_by_student["Student Extracted"]["rank"] == "2"
    assert summary_by_student["Student Zip"]["requirements_env_used"] == "false"
    assert summary_by_student["Student Zip"]["zero_grade_policy_applied"] == "false"
    assert summary_by_student["Student Zip"]["zero_grade_policy_reason"] == ""
    assert summary_by_student["Student Zip"]["failure_category"] == ""
    assert summary_by_student["Student Extracted"]["zero_grade_policy_applied"] == "false"
    assert summary_by_student["Student Extracted"]["zero_grade_policy_reason"] == ""

    leaderboard_rows = list(
        csv.DictReader(output_dir.joinpath("q4_leaderboard.csv").open("r", encoding="utf-8"))
    )
    assert [row["student_folder_name"] for row in leaderboard_rows] == [
        "Student Zip",
        "Student Extracted",
    ]
    assert leaderboard_rows[0]["rank"] == "1"
    assert leaderboard_rows[1]["rank"] == "2"

    submission_json_paths = sorted(output_dir.joinpath("submissions").glob("*.json"))
    assert len(submission_json_paths) == 2
    first_payload = json.loads(submission_json_paths[0].read_text(encoding="utf-8"))
    assert "q4_result" in first_payload


def test_cli_batch_q4_mixed_batch_excludes_invalid_runs_from_leaderboard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _write_dataset(
        tmp_path / "modeltesting.csv",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.6, "Consumer disputed?": "Yes"},
            {"Complaint ID": 3, "score": 0.4, "Consumer disputed?": "No"},
            {"Complaint ID": 4, "score": 0.2, "Consumer disputed?": "No"},
        ],
    )
    config_path = _write_grading_config(tmp_path / "grading.toml", dataset_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    extract_root = tmp_path / "extracted"
    output_dir = tmp_path / "out"

    strong_student_dir = batch_dir / "Strong Student"
    strong_student_dir.mkdir()
    strong_submission = _build_submission_tree(
        tmp_path / "strong_submission_source",
        predictor=ThresholdPredictor("score", threshold=0.5),
    )
    _write_submission_zip(strong_student_dir / "submission.zip", strong_submission)

    weaker_student_dir = batch_dir / "Weaker Student"
    weaker_student_dir.mkdir()
    weaker_submission = _build_submission_tree(
        tmp_path / "weaker_submission_source",
        predictor=ThresholdPredictor("score", threshold=0.75),
    )
    _write_submission_zip(weaker_student_dir / "submission.zip", weaker_submission)

    invalid_student_dir = batch_dir / "Invalid Student"
    invalid_student_dir.mkdir()
    invalid_submission = _build_submission_tree(
        tmp_path / "invalid_submission_source",
        predictor=FixedPredictionPipeline([1, 2, 0, 0]),
    )
    _write_submission_zip(invalid_student_dir / "submission.zip", invalid_submission)

    missing_student_dir = batch_dir / "Missing Student"
    missing_student_dir.mkdir()
    missing_submission = _build_submission_tree(
        tmp_path / "missing_submission_source",
        predictor=None,
    )
    _write_submission_zip(missing_student_dir / "submission.zip", missing_submission)

    exit_code = main(
        [
            "grade-q4-batch",
            str(batch_dir),
            "--config",
            str(config_path),
            "--extract-root",
            str(extract_root),
            "--output-dir",
            str(output_dir),
            "--dataset",
            "modeltesting",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert len(payload["leaderboard"]) == 2
    assert [entry["submission_id"] for entry in payload["leaderboard"]] == [
        "Strong Student",
        "Weaker Student",
    ]

    summary_rows = list(csv.DictReader(output_dir.joinpath("q4_summary.csv").open("r", encoding="utf-8")))
    summary_by_student = {row["student_folder_name"]: row for row in summary_rows}
    assert summary_by_student["Strong Student"]["rank"] == "1"
    assert summary_by_student["Weaker Student"]["rank"] == "2"
    assert summary_by_student["Invalid Student"]["rank"] == ""
    assert summary_by_student["Missing Student"]["rank"] == ""
    assert summary_by_student["Invalid Student"]["execution_status"] == "failed"
    assert summary_by_student["Missing Student"]["execution_status"] == "failed"
    assert summary_by_student["Invalid Student"]["zero_grade_policy_applied"] == "true"
    assert summary_by_student["Invalid Student"]["zero_grade_policy_reason"] == "invalid_predictions"
    assert summary_by_student["Invalid Student"]["failure_category"] == "invalid_predictions"
    assert summary_by_student["Missing Student"]["zero_grade_policy_applied"] == "true"
    assert summary_by_student["Missing Student"]["zero_grade_policy_reason"] == "missing_pipeline_artifact"
    assert summary_by_student["Missing Student"]["failure_category"] == "missing_artifacts"

    leaderboard_rows = list(
        csv.DictReader(output_dir.joinpath("q4_leaderboard.csv").open("r", encoding="utf-8"))
    )
    assert [row["student_folder_name"] for row in leaderboard_rows] == [
        "Strong Student",
        "Weaker Student",
    ]


def test_cli_batch_q4_reports_missing_requirements_as_zero_by_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _write_dataset(
        tmp_path / "modeltesting.csv",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.2, "Consumer disputed?": "No"},
        ],
    )
    config_path = _write_grading_config(tmp_path / "grading.toml", dataset_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    extract_root = tmp_path / "extracted"
    output_dir = tmp_path / "out"

    student_dir = batch_dir / "Missing Requirements Student"
    student_dir.mkdir()
    submission_root = _build_submission_tree(
        tmp_path / "missing_requirements_source",
        predictor=ThresholdPredictor("score", threshold=0.5),
        include_requirements=False,
    )
    _write_submission_zip(student_dir / "submission.zip", submission_root)

    exit_code = main(
        [
            "grade-q4-batch",
            str(batch_dir),
            "--config",
            str(config_path),
            "--extract-root",
            str(extract_root),
            "--output-dir",
            str(output_dir),
            "--dataset",
            "modeltesting",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    q4_result = payload["submissions"][0]["q4_result"]
    assert q4_result["zero_grade_policy_applied"] is True
    assert q4_result["zero_grade_policy_reason"] == "missing_requirements_file"

    summary_rows = list(csv.DictReader(output_dir.joinpath("q4_summary.csv").open("r", encoding="utf-8")))
    assert summary_rows[0]["zero_grade_policy_applied"] == "true"
    assert summary_rows[0]["zero_grade_policy_reason"] == "missing_requirements_file"
    assert summary_rows[0]["failure_category"] == "missing_artifacts"


def test_cli_batch_q4_resolves_relative_dataset_path_from_grader_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "data"
    dataset_path = _write_dataset(
        dataset_dir / "complaints_modeltesting100.csv",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.2, "Consumer disputed?": "No"},
        ],
    )
    config_path = _write_grading_config(
        tmp_path / "grading.toml",
        dataset_path=Path("data/complaints_modeltesting100.csv"),
    )
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    extract_root = tmp_path / "extracted"
    output_dir = tmp_path / "out"

    zipped_student_dir = batch_dir / "Student Zip"
    zipped_student_dir.mkdir()
    zipped_submission = _build_submission_tree(
        tmp_path / "zip_submission_source",
        predictor=ThresholdPredictor("score", threshold=0.5),
    )
    _write_submission_zip(zipped_student_dir / "submission.zip", zipped_submission)

    monkeypatch.chdir(tmp_path)
    exit_code = main(
        [
            "grade-q4-batch",
            str(batch_dir),
            "--config",
            str(config_path),
            "--extract-root",
            str(extract_root),
            "--output-dir",
            str(output_dir),
            "--dataset",
            "modeltesting",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert len(payload["submissions"]) == 1
    q4_result = payload["submissions"][0]["q4_result"]
    assert q4_result["execution_status"] == "succeeded"
    assert q4_result["dataset_path"] == str(dataset_path.resolve())


def test_cli_batch_q4_surfaces_requirements_install_failure_in_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _write_dataset(
        tmp_path / "modeltesting.csv",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.2, "Consumer disputed?": "No"},
        ],
    )
    config_path = _write_grading_config(
        tmp_path / "grading.toml",
        dataset_path,
        use_submission_requirements=True,
        requirements_env_root=tmp_path / "requirements_envs",
    )
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    extract_root = tmp_path / "extracted"
    output_dir = tmp_path / "out"

    student_dir = batch_dir / "Student Zip"
    student_dir.mkdir()
    submission_root = _build_submission_tree(
        tmp_path / "requirements_install_source",
        predictor=ThresholdPredictor("score", threshold=0.5),
    )
    _write_submission_zip(student_dir / "submission.zip", submission_root)

    monkeypatch.setattr(
        SubprocessQ4ExecutionBackend,
        "_create_virtual_environment",
        lambda self, env_root, logs: env_root.mkdir(parents=True, exist_ok=True),
    )

    def _raise_install_failure(self, env_python: Path, requirements_file: Path, logs: list[str]) -> None:
        raise q4_execution_module._RequirementsEnvFailure(
            q4_execution_module.FailureCategory.REQUIREMENTS_INSTALL_FAILED,
            f"Installing requirements from {requirements_file} failed with exit code 1.",
            logs=[*logs, "simulated requirements install failure"],
        )

    monkeypatch.setattr(
        SubprocessQ4ExecutionBackend,
        "_install_submission_requirements",
        _raise_install_failure,
    )
    monkeypatch.setattr(
        SubprocessQ4ExecutionBackend,
        "_venv_python_executable",
        lambda self, env_root: Path(sys.executable),
    )

    exit_code = main(
        [
            "grade-q4-batch",
            str(batch_dir),
            "--config",
            str(config_path),
            "--extract-root",
            str(extract_root),
            "--output-dir",
            str(output_dir),
            "--dataset",
            "modeltesting",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    q4_result = payload["submissions"][0]["q4_result"]
    assert q4_result["execution_status"] == "failed"
    assert q4_result["failure_category"] == "requirements_install_failed"
    assert q4_result["zero_grade_policy_reason"] == "requirements_install_failed"
    assert q4_result["requirements_env_used"] is False

    summary_rows = list(csv.DictReader(output_dir.joinpath("q4_summary.csv").open("r", encoding="utf-8")))
    assert summary_rows[0]["failure_category"] == "requirements_install_failed"
    assert summary_rows[0]["zero_grade_policy_reason"] == "requirements_install_failed"
    assert summary_rows[0]["requirements_env_used"] == "false"


def _build_submission_tree(
    root: Path,
    *,
    predictor: object | None,
    include_requirements: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source="predictions = pipeline.predict(X)")]
    )
    with (root / "12345_Complaints_Notebook.ipynb").open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    if include_requirements:
        (root / "12345_requirements.txt").write_text("pandas>=2.2\n", encoding="utf-8")
    if predictor is not None:
        with (root / "12345_Pipeline.pkl").open("wb") as handle:
            pickle.dump(predictor, handle)
    return root


def _write_submission_zip(path: Path, submission_root: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for file_path in sorted(submission_root.rglob("*")):
            if file_path.is_dir():
                continue
            archive.write(
                file_path,
                arcname=f"assignsubmission_file/{file_path.relative_to(submission_root).as_posix()}",
            )
    return path


def _write_dataset(path: Path, *, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_grading_config(
    path: Path,
    dataset_path: Path,
    *,
    use_submission_requirements: bool = False,
    requirements_env_root: Path = Path("sandbox/q4_requirements_envs"),
) -> Path:
    path.write_text(
        f"""
Q2_1_MAX_POINTS = 1.0
Q2_2_MAX_POINTS = 3.0
Q3_1_MAX_POINTS = 2.0
Q3_2_MAX_POINTS = 2.0
Q4_MAX_POINTS = 4.0

[public_datasets]
training = "{dataset_path.as_posix()}"
test = "{dataset_path.as_posix()}"
modeltesting = "{dataset_path.as_posix()}"

[submission]
notebook_glob = "*Complaints*Notebook*.ipynb"
requirements_glob = "*_requirements.txt"
feature_engineering_glob = "feature_engineering.py"
combined_pipeline_glob = "*_Pipeline.pkl"
split_preprocessor_glob = "*_Preprocessor.pkl"
split_model_glob = "*_Model.pkl"

[q4]
timeout_seconds = 60
use_submission_requirements = {"true" if use_submission_requirements else "false"}
requirements_env_root = "{requirements_env_root.as_posix()}"
requirements_install_timeout_seconds = 600
requirements_reuse_envs = true

[llm]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    return path
