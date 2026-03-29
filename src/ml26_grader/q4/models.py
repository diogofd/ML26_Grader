from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Q4ExecutionStatus(StrEnum):
    NOT_RUN = "not_run"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class LeaderboardStatus(StrEnum):
    NOT_RUN = "not_run"
    VALID = "valid"
    FAILED = "failed"


class FailureCategory(StrEnum):
    MISSING_ARTIFACTS = "missing_artifacts"
    LOAD_FAILURE = "load_failure"
    IMPORT_FAILURE = "import_failure"
    INFERENCE_FAILURE = "inference_failure"
    PREDICTION_COUNT_MISMATCH = "prediction_count_mismatch"
    INVALID_PREDICTIONS = "invalid_predictions"
    EMPTY_PREDICTIONS = "empty_predictions"
    TIMEOUT = "timeout"
    SANDBOX_VIOLATION = "sandbox_violation"
    EXECUTION_DISABLED = "execution_disabled"


class Q4ArtifactMode(StrEnum):
    COMBINED_PIPELINE = "combined_pipeline"
    SPLIT_PIPELINE = "split_pipeline"
    MISSING = "missing"


class Q4ArtifactLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_root: Path
    notebook: Path | None = None
    requirements_file: Path | None = None
    feature_engineering_file: Path | None = None
    combined_pipeline: Path | None = None
    split_preprocessor: Path | None = None
    split_model: Path | None = None
    artifact_mode: Q4ArtifactMode = Q4ArtifactMode.MISSING
    feature_engineering_required: bool = False
    missing_artifacts: list[str] = Field(default_factory=list)


class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str = Field(min_length=1)
    f1_score: float = Field(ge=0, le=1)
    rank: int | None = Field(default=None, ge=1)


class Q4EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_status: Q4ExecutionStatus
    leaderboard_status: LeaderboardStatus
    artifact_layout: Q4ArtifactLayout
    dataset_name: str | None = None
    dataset_path: Path | None = None
    input_row_count: int | None = Field(default=None, ge=1)
    prediction_count: int | None = Field(default=None, ge=0)
    predictions_valid: bool = False
    labels_available: bool = False
    f1_score: float | None = Field(default=None, ge=0, le=1)
    rank: int | None = Field(default=None, ge=1)
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None
    execution_logs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.leaderboard_status == LeaderboardStatus.VALID:
            if self.f1_score is None or not self.predictions_valid:
                raise ValueError("Valid leaderboard entries require a valid prediction set and f1_score.")

        if self.execution_status == Q4ExecutionStatus.FAILED:
            if self.failure_category is None or not self.failure_reason:
                raise ValueError("Failed executions require failure details.")

        if self.leaderboard_status == LeaderboardStatus.FAILED:
            if self.failure_category is None or not self.failure_reason:
                raise ValueError("Failed leaderboard entries require failure details.")

        if self.failure_category is None and self.failure_reason is not None:
            raise ValueError("failure_reason requires failure_category.")
        if self.failure_category is not None and not self.failure_reason:
            raise ValueError("failure_category requires failure_reason.")
        if self.predictions_valid and self.prediction_count is None:
            raise ValueError("Valid predictions require prediction_count.")
        if self.execution_status == Q4ExecutionStatus.SUCCEEDED and self.failure_category is not None:
            raise ValueError("Successful executions cannot include failure details.")

        return self
