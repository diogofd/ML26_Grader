from __future__ import annotations

from pathlib import Path

import nbformat

from ml26_grader.extraction.service import ExtractionStatus, NotebookEvidenceExtractor


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_synthetic_notebook(path: Path) -> Path:
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


def _build_partial_q3_notebook(path: Path) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Q3"),
            nbformat.v4.new_markdown_cell(
                "F1 score is the most appropriate metric because the dataset is imbalanced and operational risk matters."
            ),
            nbformat.v4.new_code_cell(
                "print('Model F1_score ROC_AUC\\nLogistic Regression 0.40 0.61\\nGradient Boosting 0.43 0.64')",
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text="Model F1_score ROC_AUC\nLogistic Regression 0.40 0.61\nGradient Boosting 0.43 0.64\n",
                    )
                ],
            ),
        ]
    )
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def _write_notebook(path: Path, cells: list) -> Path:
    notebook = nbformat.v4.new_notebook(cells=cells)
    with path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return path


def _signal_pairs(result) -> set[tuple[str, str]]:
    return {
        (signal.signal, signal.value)
        for signal in result.evidence_packet.extracted_signals
    }


def _signal_values(result, signal_name: str) -> set[str]:
    return {
        signal.value
        for signal in result.evidence_packet.extracted_signals
        if signal.signal == signal_name
    }


def test_extracts_q2_and_q3_from_synthetic_notebook(tmp_path: Path) -> None:
    notebook_path = _build_synthetic_notebook(tmp_path / "analysis.ipynb")
    extractor = NotebookEvidenceExtractor()

    q2_result = extractor.extract(notebook_path, "Q2")
    q3_result = extractor.extract(notebook_path, "Q3")

    assert q2_result.status == ExtractionStatus.READY
    assert "logistic_regression" in q2_result.evidence_packet.detected_models
    assert "random_forest" in q2_result.evidence_packet.detected_models
    assert ("data_split", "train_test_split") in _signal_pairs(q2_result)
    assert ("feature_engineering", "engineered_features") in _signal_pairs(q2_result)
    assert q2_result.evidence_packet.preprocessing_signals
    assert q2_result.evidence_packet.tuning_signals

    assert q3_result.status == ExtractionStatus.READY
    assert "f1_score" in q3_result.evidence_packet.detected_metrics
    assert ("model_comparison", "comparative_performance") in _signal_pairs(q3_result)
    assert any(
        signal_name == "deployment_recommendation"
        for signal_name, _ in _signal_pairs(q3_result)
    )


def test_returns_partial_but_usable_q3_packet_with_warnings(tmp_path: Path) -> None:
    notebook_path = _build_partial_q3_notebook(tmp_path / "partial_q3.ipynb")
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert result.evidence_packet.has_evidence() is True
    assert "f1_score" in result.evidence_packet.detected_metrics
    assert ("model_comparison", "comparative_performance") in _signal_pairs(result)
    assert any(
        warning.code == "missing_deployment_recommendation_signal"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_heading_based_q2_partial_packet_can_still_be_ready_with_warnings(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "PartialHeadedQ2Notebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "This is a supervised binary classification task for dispute prediction."
            ),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent to company'] - X_train['Date received']).dt.days.clip(lower=0)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q2")

    assert result.status == ExtractionStatus.READY
    assert ("feature_engineering", "engineered_features") in _signal_pairs(result)
    assert ("data_split", "train_test_split") in _signal_pairs(result)
    assert any(
        warning.code == "fewer_than_two_models_detected"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_heading_based_q2_generic_analysis_fails_closed(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "HeadedGenericAnalysisNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "We summarize complaint timing patterns before deciding what to analyze next."
            ),
            nbformat.v4.new_code_cell(
                "df['response_delay_days'] = (df['Date sent to company'] - df['Date received']).dt.days.clip(lower=0)\n"
                "summary = df.groupby('Product')['response_delay_days'].mean()"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q2")

    assert result.status == ExtractionStatus.FAILED
    assert result.evidence_packet.has_evidence() is False
    assert any(
        warning.code in {"heading_content_not_corroborated", "empty_evidence_packet"}
        for warning in result.evidence_packet.extraction_warnings
    )


def test_prevents_q2_q3_signal_leakage_between_explicit_sections(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "LeakageComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "Binary classification is the correct framing for Q2.1."
            ),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent'] - X_train['Date received']).dt.days\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "grid = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, scoring='f1')\n"
                "rf = RandomForestClassifier(random_state=42)\n"
                "rf.fit(X_train, y_train)"
            ),
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "F1 score is preferred because false negatives carry business risk."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest were compared, and Random Forest is recommended for deployment."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    q2_result = extractor.extract(notebook_path, "Q2")
    q3_result = extractor.extract(notebook_path, "Q3")

    assert q2_result.status == ExtractionStatus.READY
    assert "deployment_recommendation" not in {
        signal.signal for signal in q2_result.evidence_packet.extracted_signals
    }
    assert "model_comparison" not in {
        signal.signal for signal in q2_result.evidence_packet.extracted_signals
    }

    assert q3_result.status == ExtractionStatus.READY
    assert "feature_engineering" not in {
        signal.signal for signal in q3_result.evidence_packet.extracted_signals
    }
    assert not q3_result.evidence_packet.tuning_signals


def test_q2_span_stops_before_q3_subquestion_only_section(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "SubquestionBoundaryComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "Binary classification is the correct framing for the dispute-prediction task."
            ),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent'] - X_train['Date received']).dt.days.clip(lower=0)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "grid = GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.1, 1.0]}, scoring='f1')\n"
                "rf = RandomForestClassifier(random_state=42)\n"
                "rf.fit(X_train, y_train)"
            ),
            nbformat.v4.new_markdown_cell(
                "## Q3.1\nF1 score is preferred because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "## Q3.2\nLogistic Regression and Random Forest were compared, and Logistic Regression is recommended for deployment because it remains interpretable."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    q2_result = extractor.extract(notebook_path, "Q2")
    q3_result = extractor.extract(notebook_path, "Q3")

    assert q2_result.status == ExtractionStatus.READY
    assert all(
        "Q3.1" not in snippet.content and "Q3.2" not in snippet.content
        for snippet in q2_result.evidence_packet.markdown_snippets
    )
    assert "deployment_recommendation" not in {
        signal.signal for signal in q2_result.evidence_packet.extracted_signals
    }

    assert q3_result.status == ExtractionStatus.READY
    assert ("model_comparison", "comparative_performance") in _signal_pairs(q3_result)
    assert any(
        signal_name == "deployment_recommendation"
        for signal_name, _ in _signal_pairs(q3_result)
    )


def test_headingless_q2_fallback_returns_ready_with_warning(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "HeadinglessComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "This is a supervised binary classification task that predicts whether a complaint will escalate."
            ),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent'] - X_train['Date received']).dt.days.clip(lower=0)\n"
                "X_train['same_day_flag'] = np.where(X_train['response_delay_days'] == 0, 1, 0)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n"
                "grid = GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.1, 1]}, scoring='f1')\n"
                "rf = RandomForestClassifier(random_state=42)"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q2")

    assert result.status == ExtractionStatus.READY
    assert result.evidence_packet.has_evidence() is True
    assert ("feature_engineering", "engineered_features") in _signal_pairs(result)
    assert any(
        warning.code == "question_section_inferred_from_content"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_headingless_q3_fallback_returns_ready_with_warning(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "HeadinglessQ3ComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "F1 score is the right primary metric because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest were compared, and Random Forest is recommended for deployment."
            ),
            nbformat.v4.new_code_cell(
                "print('Model F1_score ROC_AUC\\nLogistic Regression 0.41 0.62\\nRandom Forest 0.45 0.66')",
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
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert "f1_score" in result.evidence_packet.detected_metrics
    assert ("model_comparison", "comparative_performance") in _signal_pairs(result)
    assert any(
        warning.code == "question_section_inferred_from_content"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_headingless_q2_generic_data_manipulation_fails_closed(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "GenericAnalysisNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "We inspect complaint dates and derive response_delay_days for exploratory analysis."
            ),
            nbformat.v4.new_code_cell(
                "df['response_delay_days'] = (df['Date sent to company'] - df['Date received']).dt.days.clip(lower=0)\n"
                "df['same_day_flag'] = (df['response_delay_days'] == 0).astype(int)\n"
                "summary = df.groupby('Product')['response_delay_days'].mean()"
            ),
            nbformat.v4.new_code_cell(
                "print(summary.head())",
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text="Product\nCredit card 2.4\nMortgage 5.1\n",
                    )
                ],
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q2")

    assert result.status == ExtractionStatus.FAILED
    assert result.evidence_packet.has_evidence() is False
    assert any(
        warning.code in {"fallback_content_not_corroborated", "empty_evidence_packet"}
        for warning in result.evidence_packet.extraction_warnings
    )


def test_selects_strongest_notebook_when_multiple_relevant_notebooks_exist(tmp_path: Path) -> None:
    weak_path = _write_notebook(
        tmp_path / "WeakComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "F1 score matters because false negatives are costly."
            ),
        ],
    )
    strong_path = _write_notebook(
        tmp_path / "StrongComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "F1 score is the primary metric because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Gradient Boosting were compared, and Gradient Boosting is recommended for deployment because recall remains stronger."
            ),
            nbformat.v4.new_code_cell(
                "results = pd.DataFrame({'Model': ['Logistic Regression', 'Gradient Boosting'], 'F1_score': [0.41, 0.47]})"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q3")

    assert weak_path.exists() and strong_path.exists()
    assert result.status == ExtractionStatus.READY
    assert any(
        note == f"Selected notebook: {strong_path.name}"
        for note in result.notes
    )
    assert all(
        snippet.source_ref.startswith(f"{strong_path.name}#cell-")
        for snippet in result.evidence_packet.markdown_snippets
    )
    assert any(
        warning.code == "multiple_relevant_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )
    assert any(
        warning.code == "multiple_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_prefers_analysis_notebook_over_auxiliary_helper_notebook(tmp_path: Path) -> None:
    analysis_path = _write_notebook(
        tmp_path / "Student_Complaints_Notebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "## Q3.1\nF1 score is the best metric because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "## Q3.2\nRandom Forest is recommended for deployment because it improves recall without materially reducing precision."
            ),
            nbformat.v4.new_code_cell(
                "results = pd.DataFrame({'Model': ['Logistic Regression', 'Random Forest'], 'F1_score': [0.41, 0.45]})"
            ),
        ],
    )
    _write_notebook(
        tmp_path / "Student_Complaints_ModelTesting_Notebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Validation notebook"),
            nbformat.v4.new_markdown_cell(
                "F1 score remained acceptable after reload, and Random Forest can be deployed once the pickle file is loaded."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest both produced binary predictions during testing."
            ),
            nbformat.v4.new_code_cell(
                "with open('Student_Pipeline.pkl', 'rb') as handle:\n"
                "    pipeline = pickle.load(handle)\n"
                "predictions = pipeline.predict(X_external)\n"
                "print(predictions[:5])"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert any(
        note == f"Selected notebook: {analysis_path.name}"
        for note in result.notes
    )
    assert not any(
        warning.code == "multiple_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )
    assert not any(
        warning.code == "multiple_relevant_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_prefers_analysis_notebook_over_save_as_pickle_helper_notebook(tmp_path: Path) -> None:
    analysis_path = _write_notebook(
        tmp_path / "Assignment1_Complaints_Analysis.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "## Q2.1\nThis is a supervised binary classification problem for dispute prediction."
            ),
            nbformat.v4.new_markdown_cell(
                "## Q2.2\nI engineered response-delay features, used train_test_split, and compared Logistic Regression against Random Forest."
            ),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent'] - X_train['Date received']).dt.days.clip(lower=0)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "log_reg = LogisticRegression(max_iter=1000)\n"
                "rf = RandomForestClassifier(random_state=42)\n"
                "rf.fit(X_train, y_train)"
            ),
        ],
    )
    _write_notebook(
        tmp_path / "Assignment1_SaveAsPickle_v2.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Packaging helper"),
            nbformat.v4.new_markdown_cell(
                "This notebook saves the fitted pipeline and validates that the pickle can be reloaded for deployment."
            ),
            nbformat.v4.new_code_cell(
                "with open('Assignment1_Pipeline.pkl', 'rb') as handle:\n"
                "    pipeline = pickle.load(handle)\n"
                "predictions = pipeline.predict(X_external)\n"
                "print(predictions[:5])"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q2")

    assert result.status == ExtractionStatus.READY
    assert any(
        note == f"Selected notebook: {analysis_path.name}"
        for note in result.notes
    )
    assert not any(
        warning.code == "multiple_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_falls_back_to_nonmatching_analysis_notebook_when_configured_helper_is_unusable(
    tmp_path: Path,
) -> None:
    _write_notebook(
        tmp_path / "Student_Complaints_Notebook_Helper.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Packaging helper"),
            nbformat.v4.new_markdown_cell(
                "This helper notebook only reloads serialized artifacts and validates that inference still runs."
            ),
            nbformat.v4.new_code_cell(
                "with open('Student_Pipeline.pkl', 'rb') as handle:\n"
                "    pipeline = pickle.load(handle)\n"
                "predictions = pipeline.predict(X_external)\n"
                "print(predictions[:5])"
            ),
        ],
    )
    analysis_path = _write_notebook(
        tmp_path / "Student_Assignment_Analysis.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "## Q3.1\nF1 score is preferred because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "## Q3.2\nRandom Forest is recommended for deployment because it improves recall while keeping precision stable."
            ),
            nbformat.v4.new_code_cell(
                "results = pd.DataFrame({'Model': ['Logistic Regression', 'Random Forest'], 'F1_score': [0.41, 0.45]})"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert any(
        note == "Trying fallback notebook candidates for Q3 because configured candidates were unusable."
        for note in result.notes
    )
    assert any(
        note == f"Selected notebook: {analysis_path.name}"
        for note in result.notes
    )
    assert any(
        warning.code == "notebook_candidate_rejected_unusable"
        and "Student_Complaints_Notebook_Helper.ipynb" in warning.message
        for warning in result.evidence_packet.extraction_warnings
    )
    assert any(
        warning.code == "fallback_notebook_selected"
        for warning in result.evidence_packet.extraction_warnings
    )
    assert not any(
        warning.code == "configured_notebook_pattern_miss"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_fallback_candidates_still_fail_closed_when_all_notebooks_are_unusable(tmp_path: Path) -> None:
    _write_notebook(
        tmp_path / "Student_Complaints_Notebook_Helper.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Packaging helper"),
            nbformat.v4.new_code_cell("print('artifact reload only')"),
        ],
    )
    _write_notebook(
        tmp_path / "Student_Assignment_Analysis.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "This notebook discusses exploratory observations but does not contain Q3 answer structure."
            ),
            nbformat.v4.new_code_cell("summary = df.groupby('Product').size()"),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q3")

    assert result.status == ExtractionStatus.FAILED
    assert result.evidence_packet.has_evidence() is False
    assert any(
        note == "Trying fallback notebook candidates for Q3 because configured candidates were unusable."
        for note in result.notes
    )
    assert any(
        warning.code == "notebook_candidate_rejected_unusable"
        for warning in result.evidence_packet.extraction_warnings
    )
    assert any(
        warning.code == "empty_evidence_packet"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_single_notebook_extraction_does_not_emit_fallback_candidate_warnings(tmp_path: Path) -> None:
    notebook_path = _build_synthetic_notebook(tmp_path / "analysis.ipynb")
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q2")

    assert result.status == ExtractionStatus.READY
    assert all(
        warning.code not in {"notebook_candidate_rejected_unusable", "fallback_notebook_selected"}
        for warning in result.evidence_packet.extraction_warnings
    )


def test_configured_notebook_pattern_matches_realistic_submission_name(tmp_path: Path) -> None:
    real_notebook_path = _build_synthetic_notebook(
        tmp_path / "70142_Complaints_Notebook.ipynb.ipynb"
    )
    _write_notebook(
        tmp_path / "70142_Validation_Notebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Validation"),
            nbformat.v4.new_code_cell("print('validation notebook')"),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q2")

    assert result.status == ExtractionStatus.READY
    assert any(
        note == f"Selected notebook: {real_notebook_path.name}"
        for note in result.notes
    )
    assert not any(
        warning.code == "configured_notebook_pattern_miss"
        for warning in result.evidence_packet.extraction_warnings
    )
    assert not any(
        warning.code == "multiple_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_configured_notebook_pattern_matches_case_insensitive_realistic_name(tmp_path: Path) -> None:
    real_notebook_path = _build_synthetic_notebook(
        tmp_path / "71642_complaints_notebook.ipynb"
    )
    _write_notebook(
        tmp_path / "71642_Complaints_ModelTesting_Notebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Validation"),
            nbformat.v4.new_code_cell("print('validation notebook')"),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert any(
        note == f"Selected notebook: {real_notebook_path.name}"
        for note in result.notes
    )
    assert not any(
        warning.code == "configured_notebook_pattern_miss"
        for warning in result.evidence_packet.extraction_warnings
    )
    assert not any(
        warning.code == "multiple_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_checkpoint_notebooks_are_ignored_and_real_notebook_is_preferred(tmp_path: Path) -> None:
    real_notebook_path = _write_notebook(
        tmp_path / "StudentComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "F1 score is preferred because false negatives create business risk."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest were compared, and Random Forest is recommended for deployment."
            ),
        ],
    )
    checkpoint_dir = tmp_path / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    _write_notebook(
        checkpoint_dir / "StudentComplaintsNotebook.ipynb-checkpoint.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "F1 score is preferred because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Gradient Boosting were compared, and Gradient Boosting is recommended for deployment."
            ),
            nbformat.v4.new_code_cell(
                "results = pd.DataFrame({'Model': ['Logistic Regression', 'Gradient Boosting'], 'F1_score': [0.41, 0.47]})"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert any(
        note == f"Selected notebook: {real_notebook_path.name}"
        for note in result.notes
    )
    assert all(
        snippet.source_ref.startswith(f"{real_notebook_path.name}#cell-")
        for snippet in result.evidence_packet.markdown_snippets
    )
    assert not any(
        warning.code == "multiple_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_fallback_discovery_ignores_checkpoint_pollution(tmp_path: Path) -> None:
    real_notebook_path = _write_notebook(
        tmp_path / "analysis.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "F1 score is the right primary metric because false negatives create business risk."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest were compared, and Random Forest is recommended for deployment."
            ),
            nbformat.v4.new_code_cell(
                "print('Model F1_score\\nLogistic Regression 0.41\\nRandom Forest 0.45')",
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text="Model F1_score\nLogistic Regression 0.41\nRandom Forest 0.45\n",
                    )
                ],
            ),
        ],
    )
    checkpoint_dir = tmp_path / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    _write_notebook(
        checkpoint_dir / "analysis-checkpoint.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "This checkpoint copy should be ignored during fallback discovery."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(tmp_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert any(
        note == f"Selected notebook: {real_notebook_path.name}"
        for note in result.notes
    )
    assert any(
        warning.code == "configured_notebook_pattern_miss"
        for warning in result.evidence_packet.extraction_warnings
    )
    assert not any(
        warning.code == "multiple_notebooks_found"
        for warning in result.evidence_packet.extraction_warnings
    )


def test_grounded_deployment_recommendation_prefers_local_model_reference(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "DeploymentComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest were compared closely. "
                "After reviewing the trade-offs, Random Forest is recommended for deployment because it keeps recall higher."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert _signal_values(result, "deployment_recommendation") == {"random_forest"}


def test_ambiguous_deployment_recommendation_stays_generic_instead_of_guessing(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "AmbiguousDeploymentComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 3"),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest are both viable for deployment, "
                "and the final recommendation depends on operational budget and tolerance for false positives."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert _signal_values(result, "deployment_recommendation") == {
        "deployment_recommendation_present"
    }


def test_detects_feature_engineering_from_general_derived_column_patterns(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "FeatureEngineeringComplaintsNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell("# Question 2"),
            nbformat.v4.new_markdown_cell(
                "This is a binary classification problem."
            ),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent'] - X_train['Date received']).dt.days.clip(lower=0)\n"
                "X_train['same_day_flag'] = np.where(X_train['response_delay_days'] == 0, 1, 0)\n"
                "X_train['quality_x_timeliness'] = X_train['response_quality'] * X_train['Timely response?']\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "log_reg = LogisticRegression(max_iter=1000)\n"
                "gb = GradientBoostingClassifier(random_state=42)"
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q2")

    assert result.status == ExtractionStatus.READY
    assert ("feature_engineering", "engineered_features") in _signal_pairs(result)
    assert any(
        "response_delay_days" in snippet.content
        for snippet in result.evidence_packet.code_snippets
    )


def test_inferred_q2_span_stops_before_later_question_content(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "HeadinglessBoundaryNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "This is a supervised binary classification problem for dispute prediction."
            ),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent'] - X_train['Date received']).dt.days.clip(lower=0)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "grid = GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.1, 1.0]}, scoring='f1')\n"
                "rf = RandomForestClassifier(random_state=42)"
            ),
            nbformat.v4.new_markdown_cell(
                "# Question 4\nThe fitted pipeline is exported as a portable pickle artifact for deployment."
            ),
            nbformat.v4.new_markdown_cell(
                "# Question 5\nThe most important feature is company identity."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q2")

    assert result.status == ExtractionStatus.READY
    assert all(
        "Question 4" not in snippet.content and "Question 5" not in snippet.content
        for snippet in result.evidence_packet.markdown_snippets
    )


def test_inferred_q2_span_stops_before_html_prefixed_later_question_content(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "HeadinglessHtmlBoundaryNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "This is a supervised binary classification problem for dispute prediction."
            ),
            nbformat.v4.new_code_cell(
                "X_train['response_delay_days'] = (X_train['Date sent'] - X_train['Date received']).dt.days.clip(lower=0)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "grid = GridSearchCV(LogisticRegression(max_iter=1000), {'C': [0.1, 1.0]}, scoring='f1')\n"
                "rf = RandomForestClassifier(random_state=42)"
            ),
            nbformat.v4.new_markdown_cell(
                "<a id='s4'></a>\n---\n## Question 4\nThe fitted pipeline is exported as a portable pickle artifact for deployment."
            ),
            nbformat.v4.new_markdown_cell(
                "<a id='s5'></a>\n---\n## Question 5\nThe most important feature is company identity."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q2")

    assert result.status == ExtractionStatus.READY
    assert all(
        "Question 4" not in snippet.content and "Question 5" not in snippet.content
        for snippet in result.evidence_packet.markdown_snippets
    )


def test_inferred_q3_span_stops_before_later_question_content(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "HeadinglessQ3BoundaryNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "F1 score is the primary metric because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest were compared, and Random Forest is recommended for deployment."
            ),
            nbformat.v4.new_code_cell(
                "print('Model F1_score\\nLogistic Regression 0.41\\nRandom Forest 0.45')",
                outputs=[
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text="Model F1_score\nLogistic Regression 0.41\nRandom Forest 0.45\n",
                    )
                ],
            ),
            nbformat.v4.new_markdown_cell(
                "# Question 5\nThe strongest drivers of escalation are company identity and response delay."
            ),
            nbformat.v4.new_markdown_cell(
                "The insight section also compares XGBoost and Logistic Regression feature importance."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert all(
        "Question 5" not in snippet.content
        for snippet in result.evidence_packet.markdown_snippets
    )
    assert all(
        "feature importance" not in snippet.content.lower()
        for snippet in result.evidence_packet.markdown_snippets
    )


def test_inferred_q3_span_stops_before_html_prefixed_later_question_content(tmp_path: Path) -> None:
    notebook_path = _write_notebook(
        tmp_path / "HeadinglessQ3HtmlBoundaryNotebook.ipynb",
        [
            nbformat.v4.new_markdown_cell(
                "F1 score is the primary metric because false negatives create business and regulatory risk."
            ),
            nbformat.v4.new_markdown_cell(
                "Logistic Regression and Random Forest were compared, and Random Forest is recommended for deployment."
            ),
            nbformat.v4.new_code_cell(
                "print('Model F1_score\\nLogistic Regression 0.41\\nRandom Forest 0.45')",
                outputs=[
                    nbformat.v4.new_output(
                        output_type='stream',
                        name='stdout',
                        text="Model F1_score\nLogistic Regression 0.41\nRandom Forest 0.45\n",
                    )
                ],
            ),
            nbformat.v4.new_markdown_cell(
                "<a id='s4'></a>\n---\n## Question 4\nThe exported pipeline is ready for deployment."
            ),
            nbformat.v4.new_markdown_cell(
                "<a id='s5'></a>\n---\n## Question 5\nFeature importance highlights response delay and company identity."
            ),
        ],
    )
    extractor = NotebookEvidenceExtractor()

    result = extractor.extract(notebook_path, "Q3")

    assert result.status == ExtractionStatus.READY
    assert all(
        "Question 4" not in snippet.content and "Question 5" not in snippet.content
        for snippet in result.evidence_packet.markdown_snippets
    )


def test_fails_closed_for_unusable_reference_notebook() -> None:
    extractor = NotebookEvidenceExtractor()
    result = extractor.extract(
        REPO_ROOT / "reference" / "sample_materials" / "Complains.ipynb",
        "Q2",
    )

    assert result.status == ExtractionStatus.FAILED
    assert result.evidence_packet.has_evidence() is False
    assert any(
        warning.code in {"question_section_not_found", "empty_evidence_packet"}
        for warning in result.evidence_packet.extraction_warnings
    )


def test_extracts_real_student_sample_q2_evidence() -> None:
    extractor = NotebookEvidenceExtractor()
    result = extractor.extract(
        REPO_ROOT / "reference" / "student_samples" / "student_sample_2.ipynb",
        "Q2",
    )

    assert result.status == ExtractionStatus.READY
    assert result.evidence_packet.has_evidence() is True
    assert len(result.evidence_packet.detected_models) >= 2
    assert ("data_split", "train_test_split") in _signal_pairs(result)
    assert ("feature_engineering", "engineered_features") in _signal_pairs(result)


def test_extracts_real_student_sample_q3_evidence() -> None:
    extractor = NotebookEvidenceExtractor()
    result = extractor.extract(
        REPO_ROOT / "reference" / "student_samples" / "student_sample_1.ipynb",
        "Q3",
    )

    assert result.status == ExtractionStatus.READY
    assert "f1_score" in result.evidence_packet.detected_metrics
    assert ("model_comparison", "comparative_performance") in _signal_pairs(result)
    assert any(
        signal_name == "deployment_recommendation"
        for signal_name, _ in _signal_pairs(result)
    )


def test_real_student_sample_q3_can_be_partial_but_usable() -> None:
    extractor = NotebookEvidenceExtractor()
    result = extractor.extract(
        REPO_ROOT / "reference" / "student_samples" / "student_sample_2.ipynb",
        "Q3",
    )

    assert result.status == ExtractionStatus.READY
    assert result.evidence_packet.has_evidence() is True
    assert "f1_score" in result.evidence_packet.detected_metrics
    assert any(
        warning.code == "missing_deployment_recommendation_signal"
        for warning in result.evidence_packet.extraction_warnings
    )
