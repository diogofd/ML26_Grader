from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import pytest

from ml26_grader.config import GradingConfig, LLMRuntimeConfig
from ml26_grader.llm.openai_adapter import LLMProviderError, OpenAIJudgeAdapter
from ml26_grader.llm.schemas import JudgeRequest
from ml26_grader.scoring import Q23GradingPipeline
from ml26_grader.scoring.models import QuestionGradingStatus


REPO_ROOT = Path(__file__).resolve().parents[1]


class StubTransport:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(
        self,
        *,
        api_key: str,
        base_url: str,
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "api_key": api_key,
                "base_url": base_url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self._responses:
            raise AssertionError("Stub transport ran out of responses.")
        next_response = self._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


def _build_request_payload(
    *,
    extraction_warning_codes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "question_id": "Q2",
        "question_text": "Full official wording for Q2.",
        "subquestions": [
            {
                "subquestion_id": "Q2.1",
                "question_text": "Identify the machine learning problem.",
                "max_score": 1.0,
            },
            {
                "subquestion_id": "Q2.2",
                "question_text": "Build two predictive models.",
                "max_score": 3.0,
            },
        ],
        "max_scores": {
            "overall": 4.0,
            "subquestions": {
                "Q2.1": 1.0,
                "Q2.2": 3.0,
            },
        },
        "rubric_blocks": [
            {
                "subquestion_id": "Q2.1",
                "required_evidence": ["Problem statement"],
                "partial_credit_guidance": ["Partial if task type is vague."],
                "common_failure_modes": ["Incorrect task framing"],
                "score_band_guidance": ["Full marks require explicit binary classification framing."],
                "feedback_guidance": ["Mention whether the task type is explicit and correct."],
            },
            {
                "subquestion_id": "Q2.2",
                "required_evidence": ["Two models", "Preprocessing", "Tuning"],
                "partial_credit_guidance": ["Partial if only one model is supported."],
                "common_failure_modes": ["No tuning evidence"],
                "score_band_guidance": ["Full marks require explicit technical workflow evidence."],
                "feedback_guidance": ["Tie comments to concrete evidence and missing workflow pieces."],
            },
        ],
        "evidence_packet": {
            "question_id": "Q2",
            "markdown_snippets": [
                {
                    "snippet_id": "md-1",
                    "source_ref": "notebook:12",
                    "content": "We solve a binary classification problem.",
                }
            ],
            "code_snippets": [],
            "output_snippets": [],
            "extracted_signals": [],
            "extraction_warnings": [
                {
                    "code": warning_code,
                    "message": f"Warning: {warning_code}",
                }
                for warning_code in (extraction_warning_codes or [])
            ],
            "detected_models": ["logistic_regression", "random_forest"],
            "detected_metrics": ["f1"],
            "preprocessing_signals": ["missing_value_imputation"],
            "tuning_signals": ["grid_search"],
            "comparison_signals": ["model_comparison_table"],
            "business_justification_signals": ["retention_risk"],
        },
        "metadata": {
            "submission_id": "S123",
            "rubric_version": "v1",
            "prompt_version": "p1",
            "extraction_version": "test",
        },
    }


def _build_result_payload() -> dict[str, Any]:
    return {
        "question_id": "Q2",
        "score": 3.0,
        "max_score": 4.0,
        "confidence": 8.9,
        "student_feedback_overall": "The response identifies the task and supports the models reasonably well.",
        "internal_notes_overall": "Overall evidence is mostly explicit.",
        "review_recommended": False,
        "review_reasons": [],
        "subquestions": {
            "Q2.1": {
                "score": 1.0,
                "max_score": 1.0,
                "confidence": 9.1,
                "student_feedback": "The task is clearly framed as binary classification.",
                "internal_notes": "Explicit task framing found in markdown.",
                "evidence_used": ["Binary classification statement"],
                "missing_requirements": [],
            },
            "Q2.2": {
                "score": 2.0,
                "max_score": 3.0,
                "confidence": 8.7,
                "student_feedback": "Two models are present, but tuning support is limited.",
                "internal_notes": "Need stronger hyperparameter evidence.",
                "evidence_used": ["Two models listed", "Preprocessing signal present"],
                "missing_requirements": ["hyperparameter_tuning_detail"],
            },
        },
    }


def _provider_response(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl_test_123",
        "model": "gpt-4.1-mini-2025-04-14",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ],
    }


def _build_runtime_config() -> LLMRuntimeConfig:
    return LLMRuntimeConfig(
        enabled=True,
        provider="openai",
        model="gpt-4.1-mini",
        api_key_env_var="TEST_OPENAI_API_KEY",
        temperature=0.0,
        max_repair_attempts=1,
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


def _write_notebook(path: Path) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "## Q2.1\nThis is a supervised binary classification problem."
            ),
            nbformat.v4.new_markdown_cell("## Q2.2"),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent'] - X_train['Date received']).dt.days.clip(lower=0)\n"
                "X_train['same_day_flag'] = np.where(X_train['response_delay_days'] == 0, 1, 0)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])\n"
                "grid = GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.1, 1]}, scoring='f1')\n"
                "rf = RandomForestClassifier(random_state=42)"
            ),
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def test_openai_adapter_parses_successful_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    request = JudgeRequest.model_validate(_build_request_payload())
    transport = StubTransport([_provider_response(json.dumps(_build_result_payload()))])
    adapter = OpenAIJudgeAdapter(_build_runtime_config(), transport=transport)

    result = adapter.evaluate(request)

    assert result.question_id == "Q2"
    assert result.subquestions["Q2.1"].score == 1.0
    assert adapter.last_evaluation_audit is not None
    assert adapter.last_evaluation_audit.provider == "openai"
    assert adapter.last_evaluation_audit.attempts == 1
    assert adapter.last_evaluation_audit.response_model == "gpt-4.1-mini-2025-04-14"
    assert transport.calls[0]["payload"]["response_format"]["json_schema"]["strict"] is True


def test_openai_adapter_prompt_encodes_stricter_evidence_quality_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    request = JudgeRequest.model_validate(
        _build_request_payload(
            extraction_warning_codes=[
                "limited_code_evidence",
                "limited_output_evidence",
            ]
        )
    )
    transport = StubTransport([_provider_response(json.dumps(_build_result_payload()))])
    adapter = OpenAIJudgeAdapter(_build_runtime_config(), transport=transport)

    adapter.evaluate(request)

    system_prompt = transport.calls[0]["payload"]["messages"][0]["content"]
    user_prompt = transport.calls[0]["payload"]["messages"][1]["content"]

    assert "Full marks require strong explicit evidence across the required rubric elements" in system_prompt
    assert "confidence must reflect the reliability of the grading decision" in system_prompt.lower()
    assert "missing explicit deployment recommendation evidence should matter" in system_prompt
    assert '"evidence_quality_summary"' in user_prompt
    assert '"material_warning_codes"' in user_prompt
    assert "limited_code_evidence" in user_prompt
    assert "narrative_only_packet" in user_prompt


def test_openai_adapter_retries_once_after_invalid_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    request = JudgeRequest.model_validate(_build_request_payload())
    transport = StubTransport(
        [
            _provider_response("{not valid json"),
            _provider_response(json.dumps(_build_result_payload())),
        ]
    )
    adapter = OpenAIJudgeAdapter(_build_runtime_config(), transport=transport)

    result = adapter.evaluate(request)

    assert result.question_id == "Q2"
    assert adapter.last_evaluation_audit is not None
    assert adapter.last_evaluation_audit.attempts == 2
    assert adapter.last_evaluation_audit.repair_attempted is True
    assert len(transport.calls) == 2
    assert "Repair the output" in transport.calls[1]["payload"]["messages"][1]["content"]


def test_openai_adapter_fails_closed_on_invalid_output_after_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    request = JudgeRequest.model_validate(_build_request_payload())
    transport = StubTransport(
        [
            _provider_response('{"question_id": "Q2"}'),
            _provider_response('{"question_id": "Q2"}'),
        ]
    )
    adapter = OpenAIJudgeAdapter(_build_runtime_config(), transport=transport)

    with pytest.raises(LLMProviderError, match="Structured output validation failed"):
        adapter.evaluate(request)

    assert adapter.last_evaluation_audit is not None
    assert adapter.last_evaluation_audit.attempts == 2
    assert adapter.last_evaluation_audit.error is not None


def test_pipeline_fails_closed_on_provider_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "test-key")
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _write_notebook(tmp_path / "ProviderFailureComplaintsNotebook.ipynb")
    transport = StubTransport([LLMProviderError("upstream provider unavailable")])
    judge = OpenAIJudgeAdapter(_build_runtime_config(), transport=transport)
    config = GradingConfig.from_toml(REPO_ROOT / "config" / "grading.example.toml")
    pipeline = Q23GradingPipeline.from_paths(
        config,
        REPO_ROOT / "specs" / "questions.toml",
        rubrics_path,
        judge=judge,
    )

    result = pipeline.grade_question(notebook_path, "Q2")

    assert result.status == QuestionGradingStatus.FAILED
    assert result.review_required is True
    assert "judge_evaluation_failed" in result.review_reasons
    assert result.judge_audit is not None
    assert result.judge_audit.error == "upstream provider unavailable"
