from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Callable

import nbformat
import pytest

from ml26_grader.cli import main
from ml26_grader.llm.interface import JudgeEvaluationAudit
from ml26_grader.llm.schemas import JudgeQuestionResult, JudgeRequest


class MockProviderJudge:
    def __init__(
        self,
        *,
        response_factory: Callable[[JudgeRequest], dict[str, Any]] | None = None,
    ) -> None:
        self._response_factory = response_factory
        self.calls: list[str] = []
        self.last_evaluation_audit: JudgeEvaluationAudit | None = None

    def evaluate(self, request: JudgeRequest) -> JudgeQuestionResult:
        self.calls.append(request.question_id)
        payload = self._response_factory(request) if self._response_factory is not None else {}
        self.last_evaluation_audit = JudgeEvaluationAudit(
            provider="openai",
            configured_model="gpt-4.1-mini",
            response_model="gpt-4.1-mini-2025-04-14",
            provider_request_id=f"mock-{request.question_id.lower()}",
            attempts=1,
            repair_attempted=False,
            usage={"total_tokens": 120},
            raw_output_text=json.dumps(payload, sort_keys=True),
        )
        return JudgeQuestionResult.model_validate(payload)


class RaisingProviderJudge:
    def __init__(self) -> None:
        self.last_evaluation_audit = None

    def evaluate(self, request: JudgeRequest) -> JudgeQuestionResult:
        raise RuntimeError(f"provider failure for {request.question_id}")


def _build_provider_payload(request: JudgeRequest) -> dict[str, Any]:
    subquestion_payloads: dict[str, dict[str, Any]] = {}
    total_score = 0.0
    for index, subquestion in enumerate(request.subquestions, start=1):
        score = subquestion.max_score if index == 1 else max(subquestion.max_score - 0.5, 0.0)
        total_score += score
        subquestion_payloads[subquestion.subquestion_id] = {
            "score": score,
            "max_score": subquestion.max_score,
            "confidence": 9.0,
            "student_feedback": f"{subquestion.subquestion_id} is supported by explicit notebook evidence.",
            "internal_notes": f"{subquestion.subquestion_id} evidence is explicit and traceable.",
            "evidence_used": ["explicit extracted evidence"],
            "missing_requirements": [],
        }

    return {
        "question_id": request.question_id,
        "score": total_score,
        "max_score": request.max_scores.overall,
        "confidence": 8.9,
        "student_feedback_overall": f"{request.question_id} is largely supported by explicit evidence.",
        "internal_notes_overall": f"{request.question_id} evidence is coherent across the extracted packet.",
        "review_recommended": False,
        "review_reasons": [],
        "subquestions": subquestion_payloads,
    }


def _write_grading_config(path: Path) -> None:
    path.write_text(
        """
Q2_1_MAX_POINTS = 1.0
Q2_2_MAX_POINTS = 3.0
Q3_1_MAX_POINTS = 2.0
Q3_2_MAX_POINTS = 2.0
Q4_MAX_POINTS = 4.0

[public_datasets]
training = "data/complaints_training.csv"
test = "data/complaints_test.csv"
modeltesting = "data/complaints_modeltesting100.csv"

[submission]
notebook_glob = "*Complaints*Notebook*.ipynb"
requirements_glob = "*_requirements.txt"
feature_engineering_glob = "feature_engineering.py"
combined_pipeline_glob = "*_Pipeline.pkl"
split_preprocessor_glob = "*_Preprocessor.pkl"
split_model_glob = "*_Model.pkl"

[q4]
timeout_seconds = 60

[llm]
enabled = true
fail_closed = true
provider = "openai"
model = "gpt-4.1-mini"
api_base_url = "https://api.openai.com/v1"
api_key_env_var = "TEST_OPENAI_API_KEY"
timeout_seconds = 60
temperature = 0.0
max_repair_attempts = 1
auto_accept_confidence = 8.5
""".strip(),
        encoding="utf-8",
    )


def _write_valid_rubrics(path: Path) -> None:
    path.write_text(
        """
[Q2]
rubric_version = "v1"
prompt_version = "p1"

[Q2.blocks."Q2.1"]
required_evidence = ["Explicit task framing"]
partial_credit_guidance = ["Partial if the task type is incomplete"]
common_failure_modes = ["Wrong learning task"]
score_band_guidance = ["Full marks require explicit binary classification framing"]
feedback_guidance = ["Explain whether the framing is explicit"]

[Q2.blocks."Q2.2"]
required_evidence = ["Two models", "Preprocessing", "Tuning"]
partial_credit_guidance = ["Partial if only one model is supported"]
common_failure_modes = ["No tuning evidence"]
score_band_guidance = ["Full marks require explicit workflow support"]
feedback_guidance = ["Tie feedback to concrete workflow evidence"]

[Q3]
rubric_version = "v1"
prompt_version = "p1"

[Q3.blocks."Q3.1"]
required_evidence = ["Metric justification"]
partial_credit_guidance = ["Partial if business risk is weakly connected"]
common_failure_modes = ["Metric choice not justified"]
score_band_guidance = ["Full marks require clear business and risk linkage"]
feedback_guidance = ["Comment on metric choice and rationale"]

[Q3.blocks."Q3.2"]
required_evidence = ["Model comparison", "Deployment recommendation"]
partial_credit_guidance = ["Partial if recommendation is not operationalized"]
common_failure_modes = ["Comparison is descriptive only"]
score_band_guidance = ["Full marks require a coherent deployment choice"]
feedback_guidance = ["Explain the main deployment gap or strength"]
""".strip(),
        encoding="utf-8",
    )


def _write_valid_submission_notebook(path: Path) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "## Q2.1\nThis is a supervised binary classification problem that predicts whether a complaint escalates into a dispute."
            ),
            nbformat.v4.new_markdown_cell("## Q2.2"),
            nbformat.v4.new_code_cell(
                "from sklearn.model_selection import train_test_split\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "X_train['delay_days'] = 3\n"
                "preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols), ('num', StandardScaler(), num_cols)])"
            ),
            nbformat.v4.new_code_cell(
                "grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid={'C': [0.1, 1, 10]}, scoring='f1')\n"
                "grid.fit(X_train, y_train)\n"
                "rf_model = RandomForestClassifier(n_estimators=300, random_state=42)\n"
                "rf_model.fit(X_train, y_train)"
            ),
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "## Q3.1\nF1 score is the primary metric because the classes are imbalanced and false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "## Q3.2\nRandom Forest is recommended for deployment because it improves recall while keeping the overall F1 score slightly higher."
            ),
            nbformat.v4.new_code_cell(
                "results = pd.DataFrame({'Model': ['Logistic Regression', 'Random Forest'], 'F1_score': [0.41, 0.45]})"
            ),
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def _write_helper_notebook(path: Path) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Packaging helper"),
            nbformat.v4.new_markdown_cell(
                "This helper notebook only reloads serialized artifacts and tests that inference still runs."
            ),
            nbformat.v4.new_code_cell(
                "with open('Student_Pipeline.pkl', 'rb') as handle:\n"
                "    pipeline = pickle.load(handle)\n"
                "predictions = pipeline.predict(X_external)\n"
                "print(predictions[:5])"
            ),
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def test_cli_grades_batch_and_writes_summary_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    extract_root = tmp_path / "extracted"
    output_dir = tmp_path / "out"
    config_path = tmp_path / "grading.toml"
    rubrics_path = tmp_path / "rubrics.toml"
    _write_grading_config(config_path)
    _write_valid_rubrics(rubrics_path)

    valid_student_dir = batch_dir / "Student One"
    valid_student_dir.mkdir()
    notebook_path = _write_valid_submission_notebook(tmp_path / "StudentComplaintsNotebook.ipynb")
    with zipfile.ZipFile(valid_student_dir / "submission.zip", "w") as archive:
        archive.write(
            notebook_path,
            arcname="assignsubmission_file/StudentComplaintsNotebook.ipynb",
        )

    no_zip_student_dir = batch_dir / "Student Two"
    no_zip_student_dir.mkdir()
    (no_zip_student_dir / "notes.txt").write_text("missing zip", encoding="utf-8")

    judge = MockProviderJudge(response_factory=_build_provider_payload)
    monkeypatch.setattr(
        "ml26_grader.scoring.pipeline.build_llm_judge",
        lambda grading_config: judge,
    )

    exit_code = main(
        [
            "grade-q23-batch",
            str(batch_dir),
            "--config",
            str(config_path),
            "--questions",
            str(Path(__file__).resolve().parents[1] / "specs" / "questions.toml"),
            "--rubrics",
            str(rubrics_path),
            "--extract-root",
            str(extract_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert len(payload["submissions"]) == 2
    assert output_dir.joinpath("batch_summary.json").exists()
    assert output_dir.joinpath("batch_summary.csv").exists()
    assert output_dir.joinpath("submissions").is_dir()

    submission_json_paths = sorted(output_dir.joinpath("submissions").glob("*.json"))
    assert len(submission_json_paths) == 2

    rows = list(csv.DictReader(output_dir.joinpath("batch_summary.csv").open("r", encoding="utf-8")))
    assert len(rows) == 2
    statuses = {row["student_folder_name"]: row["grading_status"] for row in rows}
    assert statuses["Student One"] == "scored"
    assert statuses["Student Two"] == "failed"
    assert judge.calls == ["Q2", "Q3"]


def test_cli_batch_uses_fallback_analysis_notebook_when_configured_helper_is_unusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    extract_root = tmp_path / "extracted"
    output_dir = tmp_path / "out"
    config_path = tmp_path / "grading.toml"
    rubrics_path = tmp_path / "rubrics.toml"
    _write_grading_config(config_path)
    _write_valid_rubrics(rubrics_path)

    student_dir = batch_dir / "Student One"
    student_dir.mkdir()
    helper_notebook = _write_helper_notebook(tmp_path / "Student_Complaints_Notebook_Helper.ipynb")
    analysis_notebook = _write_valid_submission_notebook(tmp_path / "Student_Assignment_Analysis.ipynb")
    with zipfile.ZipFile(student_dir / "submission.zip", "w") as archive:
        archive.write(
            helper_notebook,
            arcname="assignsubmission_file/Student_Complaints_Notebook_Helper.ipynb",
        )
        archive.write(
            analysis_notebook,
            arcname="assignsubmission_file/Student_Assignment_Analysis.ipynb",
        )

    judge = MockProviderJudge(response_factory=_build_provider_payload)
    monkeypatch.setattr(
        "ml26_grader.scoring.pipeline.build_llm_judge",
        lambda grading_config: judge,
    )

    exit_code = main(
        [
            "grade-q23-batch",
            str(batch_dir),
            "--config",
            str(config_path),
            "--questions",
            str(Path(__file__).resolve().parents[1] / "specs" / "questions.toml"),
            "--rubrics",
            str(rubrics_path),
            "--extract-root",
            str(extract_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    submission = payload["submissions"][0]
    q2_result = submission["grading"]["results"]["Q2"]
    q3_result = submission["grading"]["results"]["Q3"]

    for result in (q2_result, q3_result):
        assert result["status"] == "scored"
        assert any(
            note == f"Selected notebook: {analysis_notebook.name}"
            for note in result["extraction_result"]["notes"]
        )
        assert any(
            note == f"Trying fallback notebook candidates for {result['question_id']} because configured candidates were unusable."
            for note in result["extraction_result"]["notes"]
        )
        warning_codes = {
            warning["code"]
            for warning in result["extraction_result"]["evidence_packet"]["extraction_warnings"]
        }
        assert "notebook_candidate_rejected_unusable" in warning_codes
        assert "fallback_notebook_selected" in warning_codes


def test_cli_batch_fallback_does_not_hide_provider_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    extract_root = tmp_path / "extracted"
    output_dir = tmp_path / "out"
    config_path = tmp_path / "grading.toml"
    rubrics_path = tmp_path / "rubrics.toml"
    _write_grading_config(config_path)
    _write_valid_rubrics(rubrics_path)

    student_dir = batch_dir / "Student One"
    student_dir.mkdir()
    helper_notebook = _write_helper_notebook(tmp_path / "Student_Complaints_Notebook_Helper.ipynb")
    analysis_notebook = _write_valid_submission_notebook(tmp_path / "Student_Assignment_Analysis.ipynb")
    with zipfile.ZipFile(student_dir / "submission.zip", "w") as archive:
        archive.write(
            helper_notebook,
            arcname="assignsubmission_file/Student_Complaints_Notebook_Helper.ipynb",
        )
        archive.write(
            analysis_notebook,
            arcname="assignsubmission_file/Student_Assignment_Analysis.ipynb",
        )

    monkeypatch.setattr(
        "ml26_grader.scoring.pipeline.build_llm_judge",
        lambda grading_config: RaisingProviderJudge(),
    )

    exit_code = main(
        [
            "grade-q23-batch",
            str(batch_dir),
            "--config",
            str(config_path),
            "--questions",
            str(Path(__file__).resolve().parents[1] / "specs" / "questions.toml"),
            "--rubrics",
            str(rubrics_path),
            "--extract-root",
            str(extract_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    submission = payload["submissions"][0]
    q2_result = submission["grading"]["results"]["Q2"]

    assert q2_result["status"] == "failed"
    assert q2_result["review_reasons"] == ["judge_evaluation_failed"]
    assert any(
        note == f"Selected notebook: {analysis_notebook.name}"
        for note in q2_result["extraction_result"]["notes"]
    )
    warning_codes = {
        warning["code"]
        for warning in q2_result["extraction_result"]["evidence_packet"]["extraction_warnings"]
    }
    assert "fallback_notebook_selected" in warning_codes
