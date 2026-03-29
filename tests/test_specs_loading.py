from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ml26_grader.config import GradingConfig
from ml26_grader.llm.schemas import EvidencePacket
from ml26_grader.specs import load_llm_question_specs


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_grading_config_loads_llm_runtime_settings() -> None:
    config = GradingConfig.from_toml(REPO_ROOT / "config" / "grading.example.toml")

    assert config.q2_max_points == 4.0
    assert config.q3_max_points == 4.0
    assert config.llm.enabled is False
    assert config.llm.fail_closed is True
    assert config.llm.auto_accept_confidence == pytest.approx(8.5)
    assert config.llm.soft_auto_pass_enabled is False
    assert config.llm.soft_auto_pass_min_confidence == pytest.approx(8.0)
    assert config.llm.soft_auto_pass_min_score_ratio == pytest.approx(0.9)
    assert config.llm.soft_auto_pass_requires_no_failures is True
    assert "warning_heavy_evidence_packet" in config.llm.soft_auto_pass_disallowed_reasons
    assert "evidence_extraction_failed" in config.llm.soft_auto_pass_disallowed_reasons


def test_load_llm_question_specs_combines_questions_rubrics_and_scores(tmp_path: Path) -> None:
    config = GradingConfig.from_toml(REPO_ROOT / "config" / "grading.example.toml")
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)

    specs = load_llm_question_specs(
        config,
        REPO_ROOT / "specs" / "questions.toml",
        rubrics_path,
    )

    assert specs["Q2"].max_scores.overall == 4.0
    assert specs["Q2"].max_scores.subquestions["Q2.2"] == 3.0
    assert specs["Q3"].rubric_version == "v1"


def test_load_llm_question_specs_rejects_incomplete_placeholder_rubrics() -> None:
    config = GradingConfig.from_toml(REPO_ROOT / "config" / "grading.example.toml")

    with pytest.raises((ValidationError, ValueError)):
        load_llm_question_specs(
            config,
            REPO_ROOT / "specs" / "questions.toml",
            REPO_ROOT / "specs" / "rubrics.example.toml",
        )


def test_runtime_rubrics_encode_stricter_full_mark_calibration() -> None:
    config = GradingConfig.from_toml(REPO_ROOT / "config" / "grading.toml")

    specs = load_llm_question_specs(
        config,
        REPO_ROOT / "specs" / "questions.toml",
        REPO_ROOT / "specs" / "rubrics.toml",
    )

    q2_blocks = {
        block.subquestion_id: block
        for block in specs["Q2"].rubric_blocks
    }
    q3_blocks = {
        block.subquestion_id: block
        for block in specs["Q3"].rubric_blocks
    }
    q2_2_guidance = "\n".join(q2_blocks["Q2.2"].score_band_guidance)
    q3_2_guidance = "\n".join(q3_blocks["Q3.2"].score_band_guidance)
    q2_2_feedback = "\n".join(q2_blocks["Q2.2"].feedback_guidance)

    assert "supported mainly by prose" in q2_2_guidance
    assert "directly evidenced in the packet" in q3_2_guidance
    assert "lacked direct code or output support" in q2_2_feedback


def test_load_llm_question_specs_rejects_empty_required_rubric_content(tmp_path: Path) -> None:
    config = GradingConfig.from_toml(REPO_ROOT / "config" / "grading.example.toml")
    rubrics_path = tmp_path / "invalid_rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    rubrics_path.write_text(
        rubrics_path.read_text(encoding="utf-8").replace(
            'required_evidence = ["Explicit task framing"]',
            'required_evidence = [""]',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises((ValidationError, ValueError)):
        load_llm_question_specs(
            config,
            REPO_ROOT / "specs" / "questions.toml",
            rubrics_path,
        )


def test_loaded_question_spec_fails_closed_when_building_request_from_empty_evidence(
    tmp_path: Path,
) -> None:
    config = GradingConfig.from_toml(REPO_ROOT / "config" / "grading.example.toml")
    rubrics_path = tmp_path / "rubrics.toml"
    _write_valid_rubrics(rubrics_path)
    specs = load_llm_question_specs(
        config,
        REPO_ROOT / "specs" / "questions.toml",
        rubrics_path,
    )

    with pytest.raises(ValidationError):
        specs["Q2"].build_request(EvidencePacket(question_id="Q2"))
