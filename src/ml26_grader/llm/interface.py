from __future__ import annotations

from typing import Any
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .schemas import JudgeQuestionResult, JudgeRequest


class JudgeEvaluationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    configured_model: str = Field(min_length=1)
    response_model: str | None = None
    provider_request_id: str | None = None
    attempts: int = Field(ge=1)
    repair_attempted: bool = False
    usage: dict[str, Any] = Field(default_factory=dict)
    raw_output_text: str | None = None
    error: str | None = None


class LLMJudge(Protocol):
    last_evaluation_audit: JudgeEvaluationAudit | None

    def evaluate(self, request: JudgeRequest) -> JudgeQuestionResult:
        ...


class PlaceholderLLMJudge:
    last_evaluation_audit: JudgeEvaluationAudit | None = None

    def evaluate(self, request: JudgeRequest) -> JudgeQuestionResult:
        raise NotImplementedError(
            "LLM judging is intentionally not implemented in the scaffold."
        )
