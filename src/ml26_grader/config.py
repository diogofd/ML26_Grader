from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import CONFIDENCE_REVIEW_THRESHOLD
from .ingest.submission import SubmissionArtifactPatterns

DEFAULT_SOFT_AUTO_PASS_DISALLOWED_REASONS = (
    "warning_heavy_evidence_packet",
    "narrative_only_evidence_packet",
    "score_consistency_issue",
    "evidence_extraction_failed",
    "empty_evidence_packet",
    "question_section_not_found",
    "invalid_rubric_spec",
    "question_spec_missing",
    "invalid_judge_request",
    "judge_unavailable",
    "judge_evaluation_failed",
    "invalid_judge_output",
)
DEFAULT_REVIEW_RESCUE_DISALLOWED_REASONS = (
    "warning_heavy_evidence_packet",
    "narrative_only_evidence_packet",
    "score_consistency_issue",
    "evidence_extraction_failed",
    "empty_evidence_packet",
    "question_section_not_found",
    "invalid_rubric_spec",
    "question_spec_missing",
    "invalid_judge_request",
    "judge_unavailable",
    "judge_evaluation_failed",
    "invalid_judge_output",
)


class PublicDatasetPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training: Path = Path("data/complaints_training.csv")
    test: Path = Path("data/complaints_test.csv")
    modeltesting: Path = Path("data/complaints_modeltesting100.csv")


class Q4RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = Field(default=60, ge=1)
    use_submission_requirements: bool = False
    requirements_env_root: Path = Path("sandbox/q4_requirements_envs")
    requirements_install_timeout_seconds: int = Field(default=600, ge=1)
    requirements_reuse_envs: bool = True


class LLMRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fail_closed: bool = True
    provider: Literal["openai"] = "openai"
    model: str | None = None
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env_var: str = "OPENAI_API_KEY"
    timeout_seconds: int = Field(default=60, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_repair_attempts: int = Field(default=1, ge=0, le=1)
    auto_accept_confidence: float = Field(
        default=CONFIDENCE_REVIEW_THRESHOLD,
        ge=0,
        le=10,
    )
    soft_auto_pass_enabled: bool = False
    soft_auto_pass_min_confidence: float = Field(default=8.0, ge=0, le=10)
    soft_auto_pass_min_score_ratio: float = Field(default=0.9, ge=0, le=1)
    soft_auto_pass_requires_no_failures: bool = True
    soft_auto_pass_disallowed_reasons: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SOFT_AUTO_PASS_DISALLOWED_REASONS)
    )
    review_rescue_enabled: bool = False
    review_rescue_provider: Literal["openai"] | None = None
    review_rescue_model: str | None = "gpt-5.4"
    review_rescue_min_confidence: float = Field(default=8.5, ge=0, le=10)
    review_rescue_disallowed_reasons: list[str] = Field(
        default_factory=lambda: list(DEFAULT_REVIEW_RESCUE_DISALLOWED_REASONS)
    )

    @model_validator(mode="after")
    def validate_soft_auto_pass_settings(self) -> Self:
        if self.soft_auto_pass_min_confidence > self.auto_accept_confidence:
            raise ValueError(
                "soft_auto_pass_min_confidence cannot exceed auto_accept_confidence."
            )
        self.soft_auto_pass_disallowed_reasons = list(
            dict.fromkeys(
                reason
                for reason in self.soft_auto_pass_disallowed_reasons
                if reason
            )
        )
        self.review_rescue_disallowed_reasons = list(
            dict.fromkeys(
                reason
                for reason in self.review_rescue_disallowed_reasons
                if reason
            )
        )
        return self


class GradingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    q2_1_max_points: float = Field(alias="Q2_1_MAX_POINTS", gt=0)
    q2_2_max_points: float = Field(alias="Q2_2_MAX_POINTS", gt=0)
    q3_1_max_points: float = Field(alias="Q3_1_MAX_POINTS", gt=0)
    q3_2_max_points: float = Field(alias="Q3_2_MAX_POINTS", gt=0)
    q4_max_points: float = Field(alias="Q4_MAX_POINTS", gt=0)
    public_datasets: PublicDatasetPaths = Field(default_factory=PublicDatasetPaths)
    submission: SubmissionArtifactPatterns = Field(default_factory=SubmissionArtifactPatterns)
    q4: Q4RuntimeConfig = Field(default_factory=Q4RuntimeConfig)
    llm: LLMRuntimeConfig = Field(default_factory=LLMRuntimeConfig)

    @property
    def q2_max_points(self) -> float:
        return self.q2_1_max_points + self.q2_2_max_points

    @property
    def q3_max_points(self) -> float:
        return self.q3_1_max_points + self.q3_2_max_points

    def score_for_key(self, key: str) -> float:
        score_map = {
            "Q2_1_MAX_POINTS": self.q2_1_max_points,
            "Q2_2_MAX_POINTS": self.q2_2_max_points,
            "Q3_1_MAX_POINTS": self.q3_1_max_points,
            "Q3_2_MAX_POINTS": self.q3_2_max_points,
            "Q4_MAX_POINTS": self.q4_max_points,
        }
        try:
            return score_map[key]
        except KeyError as exc:
            raise KeyError(f"Unsupported score key: {key}") from exc

    @classmethod
    def from_toml(cls, path: Path) -> "GradingConfig":
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        return cls.model_validate(payload)
