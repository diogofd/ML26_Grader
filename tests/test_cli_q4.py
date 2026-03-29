from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
import sys

import nbformat
import pytest

from ml26_grader.cli import main
from ml26_grader.q4.execution import SubprocessQ4ExecutionBackend
from ml26_grader.q4.test_support import RaisingPredictor, ThresholdPredictor


def test_inspect_q4_executes_single_submission_successfully(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission_root = _build_submission(tmp_path / "submission")
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_path = _write_dataset(
        tmp_path / "modeltesting.csv",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
            {"Complaint ID": 3, "score": 0.8, "Consumer disputed?": "Yes"},
        ],
    )
    config_path = _write_grading_config(tmp_path / "grading.toml", dataset_path)

    exit_code = main(
        [
            "inspect-q4",
            str(submission_root),
            "--config",
            str(config_path),
            "--dataset",
            "modeltesting",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_status"] == "succeeded"
    assert payload["leaderboard_status"] == "valid"
    assert payload["predictions_valid"] is True
    assert payload["failure_reason"] is None


def test_inspect_q4_returns_non_zero_on_execution_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission_root = _build_submission(tmp_path / "submission")
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        RaisingPredictor("boom"),
    )
    dataset_path = _write_dataset(
        tmp_path / "modeltesting.csv",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    config_path = _write_grading_config(tmp_path / "grading.toml", dataset_path)

    exit_code = main(
        [
            "inspect-q4",
            str(submission_root),
            "--config",
            str(config_path),
            "--dataset",
            "modeltesting",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_status"] == "failed"
    assert payload["failure_category"] == "inference_failure"
    assert "boom" in payload["failure_reason"]


def test_inspect_q4_can_use_submission_requirements_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission_root = _build_submission(tmp_path / "submission")
    _write_pickle(
        submission_root / "12345_Pipeline.pkl",
        ThresholdPredictor("score", threshold=0.5),
    )
    dataset_path = _write_dataset(
        tmp_path / "modeltesting.csv",
        rows=[
            {"Complaint ID": 1, "score": 0.9, "Consumer disputed?": "Yes"},
            {"Complaint ID": 2, "score": 0.1, "Consumer disputed?": "No"},
        ],
    )
    config_path = _write_grading_config(
        tmp_path / "grading.toml",
        dataset_path,
        use_submission_requirements=True,
    )

    monkeypatch.setattr(
        SubprocessQ4ExecutionBackend,
        "_create_virtual_environment",
        lambda self, env_root, logs: env_root.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        SubprocessQ4ExecutionBackend,
        "_install_submission_requirements",
        lambda self, env_python, requirements_file, logs: None,
    )
    monkeypatch.setattr(
        SubprocessQ4ExecutionBackend,
        "_venv_python_executable",
        lambda self, env_root: Path(sys.executable),
    )

    exit_code = main(
        [
            "inspect-q4",
            str(submission_root),
            "--config",
            str(config_path),
            "--dataset",
            "modeltesting",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_status"] == "succeeded"
    assert payload["requirements_env_used"] is True


def _build_submission(submission_root: Path) -> Path:
    submission_root.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source="predictions = pipeline.predict(X)")]
    )
    with (submission_root / "12345_Complaints_Notebook.ipynb").open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    (submission_root / "12345_requirements.txt").write_text("pandas>=2.2\n", encoding="utf-8")
    return submission_root


def _write_pickle(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _write_dataset(path: Path, *, rows: list[dict[str, object]]) -> Path:
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
requirements_env_root = "sandbox/q4_requirements_envs"
requirements_install_timeout_seconds = 600
requirements_reuse_envs = true

[llm]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    return path
