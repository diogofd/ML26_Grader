from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import FailureCategory, Q4ArtifactLayout, Q4ExecutionStatus


class Q4ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_root: Path
    artifact_layout: Q4ArtifactLayout
    dataset_name: str = Field(min_length=1)
    dataset_path: Path
    timeout_seconds: int = Field(ge=1)
    input_row_count: int = Field(ge=1)


class Q4ExecutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(default="disabled", min_length=1)
    execution_status: Q4ExecutionStatus
    predictions: list[int | bool | str | float] = Field(default_factory=list)
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None
    execution_logs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.execution_status == Q4ExecutionStatus.FAILED:
            if self.failure_category is None or not self.failure_reason:
                raise ValueError("Failed execution responses require failure details.")
        if self.execution_status == Q4ExecutionStatus.SUCCEEDED and self.failure_category is not None:
            raise ValueError("Successful execution responses cannot include failure details.")
        if self.failure_category is None and self.failure_reason is not None:
            raise ValueError("failure_reason requires failure_category.")
        if self.failure_category is not None and not self.failure_reason:
            raise ValueError("failure_category requires failure_reason.")
        return self


class Q4ExecutionBackend(Protocol):
    def run(self, request: Q4ExecutionRequest) -> Q4ExecutionResponse:
        ...


class DisabledQ4ExecutionBackend:
    # Actual student-code execution belongs behind a dedicated subprocess backend.
    def run(self, request: Q4ExecutionRequest) -> Q4ExecutionResponse:
        return Q4ExecutionResponse(
            backend_name="disabled",
            execution_status=Q4ExecutionStatus.NOT_RUN,
            failure_category=FailureCategory.EXECUTION_DISABLED,
            failure_reason=(
                "Student-code execution is disabled because no Q4 execution backend is configured."
            ),
            execution_logs=[
                f"Prepared Q4 execution request for {request.dataset_name} with {request.input_row_count} rows.",
                "No subprocess backend was configured, so student artifacts were not loaded or executed.",
            ],
        )
