from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat

from ml26_grader.config import GradingConfig
from ml26_grader.llm.interface import JudgeEvaluationAudit
from ml26_grader.scoring import Q23GradingPipeline
from ml26_grader.scoring.models import QuestionGradingStatus, QuestionReviewTier


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeJudge:
    def __init__(self, response_factory, *, configured_model: str = "fake-first-pass") -> None:
        self._response_factory = response_factory
        self._configured_model = configured_model
        self.requests = []
        self.last_evaluation_audit = None

    def evaluate(self, request):
        self.requests.append(request)
        self.last_evaluation_audit = JudgeEvaluationAudit(
            provider="fake",
            configured_model=self._configured_model,
            attempts=1,
            repair_attempted=False,
        )
        return self._response_factory(request)


class FailingJudge:
    def __init__(self, message: str, *, configured_model: str = "fake-rescue") -> None:
        self._message = message
        self._configured_model = configured_model
        self.requests = []
        self.last_evaluation_audit = None

    def evaluate(self, request):
        self.requests.append(request)
        self.last_evaluation_audit = JudgeEvaluationAudit(
            provider="fake",
            configured_model=self._configured_model,
            attempts=1,
            repair_attempted=False,
            error=self._message,
        )
        raise RuntimeError(self._message)


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


def _write_incomplete_rubrics(path: Path) -> None:
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

[Q3]
rubric_version = "v1"
prompt_version = "p1"

[Q3.blocks."Q3.1"]
required_evidence = ["Metric justification"]
partial_credit_guidance = ["Partial if business risk is weakly connected"]
common_failure_modes = ["Metric choice not justified"]
score_band_guidance = ["Full marks require clear business and risk linkage"]
feedback_guidance = ["Comment on metric choice and rationale"]
""".strip(),
        encoding="utf-8",
    )


def _write_notebook(path: Path, cells: list) -> Path:
    notebook = nbformat.v4.new_notebook(cells=cells)
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def _build_q2_notebook(path: Path) -> Path:
    return _write_notebook(
        path,
        [
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
        ],
    )


def _build_unusable_notebook(path: Path) -> Path:
    return _write_notebook(
        path,
        [
            nbformat.v4.new_markdown_cell("# Exploratory Notes"),
            nbformat.v4.new_code_cell("df.head()"),
        ],
    )


def _build_light_warning_q3_notebook(path: Path) -> Path:
    return _write_notebook(
        path,
        [
            nbformat.v4.new_markdown_cell(
                "## Q3.1\nF1 score is the primary metric because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "## Q3.2\nLogistic Regression and Random Forest were compared, and Random Forest is recommended for deployment because it improves recall without materially reducing precision."
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
        ],
    )


def _build_warning_heavy_markdown_only_q2_notebook(path: Path) -> Path:
    return _write_notebook(
        path,
        [
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "## Q2.1\nThis is a supervised binary classification problem that predicts whether a complaint will escalate into a dispute."
            ),
            nbformat.v4.new_markdown_cell(
                "## Q2.2\nTwo predictive models were developed: Logistic Regression and Random Forest. "
                "The workflow included preprocessing with imputation, scaling, and encoding, feature engineering for response delay and calendar fields, "
                "a train/test split, and GridSearchCV tuning. Logistic Regression was retained as an interpretable baseline, while Random Forest captured nonlinear interactions. "
                "These choices were justified in technical terms and in business terms because missing disputed complaints increases escalation risk and customer harm."
            ),
        ],
    )


def _build_judge_payload(
    request,
    *,
    confidence: float = 9.3,
    overall_score_override: float | None = None,
    review_recommended: bool = False,
    review_reasons: list[str] | None = None,
    full_credit: bool = False,
) -> dict[str, Any]:
    subquestion_payloads: dict[str, dict[str, Any]] = {}
    total_score = 0.0
    for index, subquestion in enumerate(request.subquestions, start=1):
        score = (
            subquestion.max_score
            if full_credit or index == 1
            else max(subquestion.max_score - 0.5, 0.0)
        )
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
        "score": total_score if overall_score_override is None else overall_score_override,
        "max_score": request.max_scores.overall,
        "confidence": confidence,
        "student_feedback_overall": f"{request.question_id} is largely supported by explicit evidence.",
        "internal_notes_overall": f"{request.question_id} evidence is coherent across the extracted packet.",
        "review_recommended": review_recommended,
        "review_reasons": list(review_reasons or []),
        "subquestions": subquestion_payloads,
    }


def _build_pipeline(
    rubrics_path: Path,
    fake_judge: FakeJudge,
    review_rescue_judge=None,
    **llm_overrides: Any,
) -> Q23GradingPipeline:
    config = GradingConfig.from_toml(REPO_ROOT / "config" / "grading.example.toml")
    config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update=llm_overrides),
        }
    )
    return Q23GradingPipeline.from_paths(
        config,
        REPO_ROOT / "specs" / "questions.toml",
        rubrics_path,
        judge=fake_judge,
        review_rescue_judge=review_rescue_judge,
    )


def test_grades_q2_successfully_with_valid_mock_judge_response(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_q2_notebook(tmp_path / "SuccessComplaintsNotebook.ipynb")
    judge = FakeJudge(lambda request: _build_judge_payload(request))
    pipeline = _build_pipeline(rubrics_path, judge)

    result = pipeline.grade_question(notebook_path, "Q2")

    assert result.status == QuestionGradingStatus.SCORED
    assert result.review_tier == QuestionReviewTier.NONE
    assert result.review_required is False
    assert result.soft_auto_pass_applied is False
    assert result.hard_review_reasons == []
    assert result.score_summary is not None
    assert result.score_summary.student_feedback_overall.startswith("Q2")
    assert result.score_summary.subquestions["Q2.1"].student_feedback.startswith("Q2.1")
    assert len(judge.requests) == 1
    assert judge.requests[0].question_id == "Q2"
    assert judge.requests[0].evidence_packet.has_evidence() is True


def test_routes_low_confidence_results_to_review(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_q2_notebook(tmp_path / "LowConfidenceComplaintsNotebook.ipynb")
    judge = FakeJudge(lambda request: _build_judge_payload(request, confidence=7.0))
    pipeline = _build_pipeline(rubrics_path, judge)

    result = pipeline.grade_question(notebook_path, "Q2")

    assert result.status == QuestionGradingStatus.REVIEW
    assert result.review_tier == QuestionReviewTier.SOFT
    assert result.review_required is True
    assert "question_confidence_below_threshold" in result.review_reasons
    assert result.score_summary is not None
    assert result.score_summary.provisional is True


def test_soft_auto_pass_can_score_strong_light_warning_submission(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_light_warning_q3_notebook(tmp_path / "LightWarningQ3Notebook.ipynb")
    judge = FakeJudge(
        lambda request: _build_judge_payload(
            request,
            confidence=7.4,
            full_credit=True,
        )
    )
    pipeline = _build_pipeline(
        rubrics_path,
        judge,
        soft_auto_pass_enabled=True,
        auto_accept_confidence=7.8,
        soft_auto_pass_min_confidence=7.2,
        soft_auto_pass_min_score_ratio=0.75,
    )

    result = pipeline.grade_question(notebook_path, "Q3")

    assert result.status == QuestionGradingStatus.SCORED
    assert result.review_tier == QuestionReviewTier.NONE
    assert result.review_required is False
    assert result.soft_auto_pass_applied is True
    assert result.review_rescue_attempted is False
    assert "question_confidence_below_threshold" in result.soft_review_reasons
    assert "subquestion_anchor_missing" in result.soft_review_reasons
    assert result.review_reasons == []


def test_disabling_soft_auto_pass_restores_review_for_strong_below_threshold_case(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_light_warning_q3_notebook(tmp_path / "LightWarningQ3Notebook.ipynb")
    judge = FakeJudge(
        lambda request: _build_judge_payload(
            request,
            confidence=7.4,
            full_credit=True,
        )
    )
    pipeline = _build_pipeline(
        rubrics_path,
        judge,
        soft_auto_pass_enabled=False,
        auto_accept_confidence=7.8,
        soft_auto_pass_min_confidence=7.2,
        soft_auto_pass_min_score_ratio=0.75,
    )

    result = pipeline.grade_question(notebook_path, "Q3")

    assert result.status == QuestionGradingStatus.REVIEW
    assert result.review_tier == QuestionReviewTier.SOFT
    assert result.review_required is True
    assert result.soft_auto_pass_applied is False
    assert "question_confidence_below_threshold" in result.review_reasons


def test_soft_review_case_can_be_rescued_into_scored(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_light_warning_q3_notebook(tmp_path / "RescueQ3Notebook.ipynb")
    first_pass_judge = FakeJudge(
        lambda request: _build_judge_payload(
            request,
            confidence=7.4,
            full_credit=True,
            review_recommended=True,
            review_reasons=["light extraction ambiguity"],
        ),
        configured_model="fake-mini",
    )
    rescue_judge = FakeJudge(
        lambda request: _build_judge_payload(
            request,
            confidence=8.8,
            full_credit=True,
            review_recommended=False,
        ),
        configured_model="fake-strong",
    )
    pipeline = _build_pipeline(
        rubrics_path,
        first_pass_judge,
        review_rescue_judge=rescue_judge,
        soft_auto_pass_enabled=False,
        review_rescue_enabled=True,
        review_rescue_min_confidence=8.5,
    )

    result = pipeline.grade_question(notebook_path, "Q3")

    assert result.status == QuestionGradingStatus.SCORED
    assert result.review_required is False
    assert result.review_rescue_attempted is True
    assert result.review_rescue_changed_status is True
    assert result.review_rescue_initial_judge_audit is not None
    assert result.review_rescue_initial_judge_audit.configured_model == "fake-mini"
    assert result.review_rescue_judge_audit is not None
    assert result.review_rescue_judge_audit.configured_model == "fake-strong"
    assert result.judge_audit is not None
    assert result.judge_audit.configured_model == "fake-strong"
    assert len(first_pass_judge.requests) == 1
    assert len(rescue_judge.requests) == 1


def test_hard_failure_is_not_eligible_for_review_rescue(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_unusable_notebook(tmp_path / "UnusableComplaintsNotebook.ipynb")
    judge = FakeJudge(lambda request: _build_judge_payload(request))
    rescue_judge = FakeJudge(lambda request: _build_judge_payload(request, confidence=9.0))
    pipeline = _build_pipeline(
        rubrics_path,
        judge,
        review_rescue_judge=rescue_judge,
        review_rescue_enabled=True,
        review_rescue_min_confidence=8.5,
    )

    result = pipeline.grade_question(notebook_path, "Q2")

    assert result.status == QuestionGradingStatus.FAILED
    assert result.review_rescue_attempted is False
    assert len(judge.requests) == 0
    assert len(rescue_judge.requests) == 0


def test_rescue_provider_failure_leaves_soft_review_in_review(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_light_warning_q3_notebook(tmp_path / "RescueFailureQ3Notebook.ipynb")
    first_pass_judge = FakeJudge(
        lambda request: _build_judge_payload(
            request,
            confidence=7.4,
            full_credit=True,
            review_recommended=True,
        )
    )
    rescue_judge = FailingJudge("provider unavailable")
    pipeline = _build_pipeline(
        rubrics_path,
        first_pass_judge,
        review_rescue_judge=rescue_judge,
        soft_auto_pass_enabled=False,
        review_rescue_enabled=True,
        review_rescue_min_confidence=8.5,
    )

    result = pipeline.grade_question(notebook_path, "Q3")

    assert result.status == QuestionGradingStatus.REVIEW
    assert result.review_required is True
    assert result.review_rescue_attempted is True
    assert result.review_rescue_changed_status is False
    assert "Judge evaluation failed: provider unavailable" in (result.review_rescue_failure_reason or "")
    assert len(first_pass_judge.requests) == 1
    assert len(rescue_judge.requests) == 1


def test_disabling_review_rescue_restores_current_review_behavior(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_light_warning_q3_notebook(tmp_path / "NoRescueQ3Notebook.ipynb")
    first_pass_judge = FakeJudge(
        lambda request: _build_judge_payload(
            request,
            confidence=7.4,
            full_credit=True,
            review_recommended=True,
        )
    )
    rescue_judge = FakeJudge(lambda request: _build_judge_payload(request, confidence=9.0, full_credit=True))
    pipeline = _build_pipeline(
        rubrics_path,
        first_pass_judge,
        review_rescue_judge=rescue_judge,
        soft_auto_pass_enabled=False,
        review_rescue_enabled=False,
        review_rescue_min_confidence=8.5,
    )

    result = pipeline.grade_question(notebook_path, "Q3")

    assert result.status == QuestionGradingStatus.REVIEW
    assert result.review_rescue_attempted is False
    assert len(first_pass_judge.requests) == 1
    assert len(rescue_judge.requests) == 0


def test_routes_warning_heavy_markdown_only_packets_to_review(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_warning_heavy_markdown_only_q2_notebook(
        tmp_path / "WarningHeavyComplaintsNotebook.ipynb"
    )
    judge = FakeJudge(lambda request: _build_judge_payload(request, confidence=9.3))
    pipeline = _build_pipeline(
        rubrics_path,
        judge,
        soft_auto_pass_enabled=True,
        soft_auto_pass_min_confidence=8.0,
        soft_auto_pass_min_score_ratio=0.9,
    )

    result = pipeline.grade_question(notebook_path, "Q2")

    assert result.status == QuestionGradingStatus.REVIEW
    assert result.review_tier == QuestionReviewTier.SOFT
    assert result.review_required is True
    assert "warning_heavy_evidence_packet" in result.review_reasons
    assert "narrative_only_evidence_packet" in result.review_reasons
    assert result.soft_auto_pass_applied is False
    assert result.extraction_result is not None
    assert {
        warning.code
        for warning in result.extraction_result.evidence_packet.extraction_warnings
    } >= {"limited_code_evidence", "limited_output_evidence"}


def test_fails_closed_when_evidence_is_unusable(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_unusable_notebook(tmp_path / "UnusableComplaintsNotebook.ipynb")
    judge = FakeJudge(lambda request: _build_judge_payload(request))
    pipeline = _build_pipeline(rubrics_path, judge)

    result = pipeline.grade_question(notebook_path, "Q2")

    assert result.status == QuestionGradingStatus.FAILED
    assert result.review_tier == QuestionReviewTier.HARD
    assert result.review_required is True
    assert "evidence_extraction_failed" in result.review_reasons
    assert result.failure_reason == "Evidence extraction did not produce a usable packet."
    assert result.judge_request is None
    assert len(judge.requests) == 0


def test_score_consistency_issues_fail_closed_and_cannot_auto_pass(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    notebook_path = _build_q2_notebook(tmp_path / "ConsistencyComplaintsNotebook.ipynb")
    judge = FakeJudge(
        lambda request: _build_judge_payload(
            request,
            confidence=8.2,
            overall_score_override=0.5,
        )
    )
    pipeline = _build_pipeline(
        rubrics_path,
        judge,
        soft_auto_pass_enabled=True,
        soft_auto_pass_min_confidence=8.0,
        soft_auto_pass_min_score_ratio=0.9,
    )

    result = pipeline.grade_question(notebook_path, "Q2")

    assert result.status == QuestionGradingStatus.FAILED
    assert result.review_tier == QuestionReviewTier.HARD
    assert "score_consistency_issue" in result.review_reasons
    assert result.judge_result is not None
    assert result.failure_reason == "Judge output contained a score consistency issue and was failed closed."
    assert result.soft_auto_pass_applied is False


def test_incomplete_rubric_load_fails_closed_before_judging(tmp_path: Path) -> None:
    rubrics_path = tmp_path / "rubrics.toml"
    _write_incomplete_rubrics(rubrics_path)
    notebook_path = _build_q2_notebook(tmp_path / "IncompleteRubricComplaintsNotebook.ipynb")
    judge = FakeJudge(lambda request: _build_judge_payload(request))
    pipeline = _build_pipeline(rubrics_path, judge)

    result = pipeline.grade_question(notebook_path, "Q2")

    assert result.status == QuestionGradingStatus.FAILED
    assert result.review_tier == QuestionReviewTier.HARD
    assert result.review_required is True
    assert "invalid_rubric_spec" in result.review_reasons
    assert "Question specifications could not be loaded" in (result.failure_reason or "")
    assert len(judge.requests) == 0
