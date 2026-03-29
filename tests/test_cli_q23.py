from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import nbformat
import pytest

from ml26_grader.cli import main
from ml26_grader.llm.interface import JudgeEvaluationAudit
from ml26_grader.llm.openai_adapter import LLMProviderError
from ml26_grader.llm.schemas import JudgeQuestionResult, JudgeRequest


REPO_ROOT = Path(__file__).resolve().parents[1]


class MockProviderJudge:
    def __init__(
        self,
        *,
        response_factory: Callable[[JudgeRequest], dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response_factory = response_factory
        self._error = error
        self.calls: list[str] = []
        self.last_evaluation_audit: JudgeEvaluationAudit | None = None

    def evaluate(self, request: JudgeRequest) -> JudgeQuestionResult:
        self.calls.append(request.question_id)
        raw_output_text: str | None = None
        if self._response_factory is not None:
            raw_output_text = json.dumps(self._response_factory(request), sort_keys=True)

        self.last_evaluation_audit = JudgeEvaluationAudit(
            provider="openai",
            configured_model="gpt-4.1-mini",
            response_model="gpt-4.1-mini-2025-04-14",
            provider_request_id=f"mock-{request.question_id.lower()}",
            attempts=1,
            repair_attempted=False,
            usage={"total_tokens": 150},
            raw_output_text=raw_output_text,
            error=str(self._error) if self._error is not None else None,
        )

        if self._error is not None:
            raise self._error
        if self._response_factory is None:
            raise AssertionError("MockProviderJudge requires a response factory when no error is set.")
        return JudgeQuestionResult.model_validate(self._response_factory(request))


def _build_provider_payload(
    request: JudgeRequest,
    *,
    confidence: float = 9.1,
) -> dict[str, Any]:
    subquestion_payloads: dict[str, dict[str, Any]] = {}
    total_score = 0.0
    for index, subquestion in enumerate(request.subquestions, start=1):
        score = subquestion.max_score if index == 1 else max(subquestion.max_score - 0.5, 0.0)
        total_score += score
        subquestion_payloads[subquestion.subquestion_id] = {
            "score": score,
            "max_score": subquestion.max_score,
            "confidence": min(confidence + 0.2, 10.0),
            "student_feedback": f"{subquestion.subquestion_id} is supported by explicit notebook evidence.",
            "internal_notes": f"{subquestion.subquestion_id} evidence is explicit and traceable.",
            "evidence_used": ["explicit extracted evidence"],
            "missing_requirements": [],
        }

    return {
        "question_id": request.question_id,
        "score": total_score,
        "max_score": request.max_scores.overall,
        "confidence": confidence,
        "student_feedback_overall": f"{request.question_id} is largely supported by explicit evidence.",
        "internal_notes_overall": f"{request.question_id} evidence is coherent across the extracted packet.",
        "review_recommended": False,
        "review_reasons": [],
        "subquestions": subquestion_payloads,
    }


def _write_grading_config(path: Path, *, llm_enabled: bool = True) -> None:
    path.write_text(
        f"""
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
enabled = {"true" if llm_enabled else "false"}
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
                "results = pd.DataFrame({'Model': ['Logistic Regression', 'Random Forest'], 'F1_score': [0.41, 0.45], 'ROC_AUC': [0.62, 0.66]})\n"
                "print(results)",
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text="Model F1_score ROC_AUC\nLogistic Regression 0.41 0.62\nRandom Forest 0.45 0.66\n",
                    )
                ],
            ),
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def _write_unusable_submission_notebook(path: Path) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Exploratory Notes"),
            nbformat.v4.new_code_cell("df.head()"),
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def test_cli_grades_single_submission_successfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    _write_valid_submission_notebook(submission_dir / "StudentComplaintsNotebook.ipynb")
    config_path = tmp_path / "grading.toml"
    rubrics_path = tmp_path / "rubrics.toml"
    output_path = tmp_path / "out" / "result.json"
    _write_grading_config(config_path, llm_enabled=True)
    _write_valid_rubrics(rubrics_path)

    judge = MockProviderJudge(response_factory=_build_provider_payload)
    monkeypatch.setattr(
        "ml26_grader.scoring.pipeline.build_llm_judge",
        lambda grading_config: judge,
    )

    exit_code = main(
        [
            "grade-q23-submission",
            str(submission_dir),
            "--config",
            str(config_path),
            "--questions",
            str(REPO_ROOT / "specs" / "questions.toml"),
            "--rubrics",
            str(rubrics_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "scored"
    assert payload["review_required"] is False
    assert set(payload["results"]) == {"Q2", "Q3"}
    assert payload["results"]["Q2"]["status"] == "scored"
    assert payload["results"]["Q3"]["status"] == "scored"
    assert payload["results"]["Q2"]["score_summary"]["student_feedback_overall"]
    assert payload["results"]["Q3"]["score_summary"]["subquestions"]["Q3.2"]["student_feedback"]
    assert payload["results"]["Q2"]["judge_audit"]["provider"] == "openai"
    assert payload["results"]["Q2"]["judge_audit"]["provider_request_id"] == "mock-q2"
    assert judge.calls == ["Q2", "Q3"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_cli_fails_when_notebook_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    config_path = tmp_path / "grading.toml"
    rubrics_path = tmp_path / "rubrics.toml"
    _write_grading_config(config_path, llm_enabled=True)
    _write_valid_rubrics(rubrics_path)

    judge = MockProviderJudge(response_factory=_build_provider_payload)
    monkeypatch.setattr(
        "ml26_grader.scoring.pipeline.build_llm_judge",
        lambda grading_config: judge,
    )

    exit_code = main(
        [
            "grade-q23-submission",
            str(submission_dir),
            "--config",
            str(config_path),
            "--questions",
            str(REPO_ROOT / "specs" / "questions.toml"),
            "--rubrics",
            str(rubrics_path),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["review_required"] is True
    assert payload["results"]["Q2"]["status"] == "failed"
    assert payload["results"]["Q3"]["status"] == "failed"
    assert payload["results"]["Q2"]["failure_reason"] == "Evidence extraction did not produce a usable packet."
    assert payload["results"]["Q2"]["judge_request"] is None
    assert judge.calls == []


def test_cli_fails_closed_on_unusable_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    _write_unusable_submission_notebook(submission_dir / "StudentComplaintsNotebook.ipynb")
    config_path = tmp_path / "grading.toml"
    rubrics_path = tmp_path / "rubrics.toml"
    _write_grading_config(config_path, llm_enabled=True)
    _write_valid_rubrics(rubrics_path)

    judge = MockProviderJudge(response_factory=_build_provider_payload)
    monkeypatch.setattr(
        "ml26_grader.scoring.pipeline.build_llm_judge",
        lambda grading_config: judge,
    )

    exit_code = main(
        [
            "grade-q23-submission",
            str(submission_dir),
            "--config",
            str(config_path),
            "--questions",
            str(REPO_ROOT / "specs" / "questions.toml"),
            "--rubrics",
            str(rubrics_path),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["results"]["Q2"]["failure_reason"] == "Evidence extraction did not produce a usable packet."
    assert payload["results"]["Q2"]["review_reasons"]
    assert payload["results"]["Q2"]["extraction_result"]["status"] == "failed"
    assert judge.calls == []


def test_cli_fails_closed_on_provider_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    _write_valid_submission_notebook(submission_dir / "StudentComplaintsNotebook.ipynb")
    config_path = tmp_path / "grading.toml"
    rubrics_path = tmp_path / "rubrics.toml"
    _write_grading_config(config_path, llm_enabled=True)
    _write_valid_rubrics(rubrics_path)

    judge = MockProviderJudge(error=LLMProviderError("upstream provider unavailable"))
    monkeypatch.setattr(
        "ml26_grader.scoring.pipeline.build_llm_judge",
        lambda grading_config: judge,
    )

    exit_code = main(
        [
            "grade-q23-submission",
            str(submission_dir),
            "--config",
            str(config_path),
            "--questions",
            str(REPO_ROOT / "specs" / "questions.toml"),
            "--rubrics",
            str(rubrics_path),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["results"]["Q2"]["status"] == "failed"
    assert payload["results"]["Q2"]["failure_reason"] == "Judge evaluation failed: upstream provider unavailable"
    assert payload["results"]["Q2"]["judge_audit"]["error"] == "upstream provider unavailable"
    assert payload["results"]["Q3"]["judge_audit"]["provider_request_id"] == "mock-q3"
    assert judge.calls == ["Q2", "Q3"]
