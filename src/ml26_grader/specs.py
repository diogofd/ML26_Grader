from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import GradingConfig
from .constants import QUESTION_SUBQUESTION_IDS
from .llm.schemas import (
    EvidencePacket,
    JudgeRequest,
    MaxScores,
    QuestionId,
    QuestionSubquestion,
    RubricBlock,
    SubquestionId,
)

def _is_placeholder_text(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return True
    return normalized.upper() in {"TODO", "TBD", "PLACEHOLDER"}


def _validate_rubric_text_list(field_name: str, values: list[str]) -> None:
    if not values:
        raise ValueError(f"{field_name} must contain at least one rubric item.")
    for value in values:
        if _is_placeholder_text(value):
            raise ValueError(f"{field_name} contains placeholder or empty rubric content.")


class QuestionSubquestionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    max_score_key: str = Field(min_length=1)


class QuestionSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_text: str = Field(min_length=1)
    subquestions: dict[str, QuestionSubquestionSource] = Field(min_length=2)


class RubricBlockSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_evidence: list[str] = Field(min_length=1)
    partial_credit_guidance: list[str] = Field(min_length=1)
    common_failure_modes: list[str] = Field(min_length=1)
    score_band_guidance: list[str] = Field(min_length=1)
    feedback_guidance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_placeholder_content(self) -> Self:
        _validate_rubric_text_list("required_evidence", self.required_evidence)
        _validate_rubric_text_list("partial_credit_guidance", self.partial_credit_guidance)
        _validate_rubric_text_list("common_failure_modes", self.common_failure_modes)
        _validate_rubric_text_list("score_band_guidance", self.score_band_guidance)
        _validate_rubric_text_list("feedback_guidance", self.feedback_guidance)
        return self


class RubricSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rubric_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    blocks: dict[str, RubricBlockSource] = Field(min_length=2)

    @model_validator(mode="after")
    def reject_placeholder_versions(self) -> Self:
        if _is_placeholder_text(self.rubric_version):
            raise ValueError("rubric_version must be set before LLM judging is enabled.")
        if _is_placeholder_text(self.prompt_version):
            raise ValueError("prompt_version must be set before LLM judging is enabled.")
        return self


class LoadedQuestionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: QuestionId
    question_text: str = Field(min_length=1)
    subquestions: list[QuestionSubquestion] = Field(min_length=2, max_length=2)
    max_scores: MaxScores
    rubric_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    rubric_blocks: list[RubricBlock] = Field(min_length=2, max_length=2)

    def build_request(
        self,
        evidence_packet: EvidencePacket,
        metadata: dict[str, Any] | None = None,
    ) -> JudgeRequest:
        payload = dict(metadata or {})
        payload.setdefault("rubric_version", self.rubric_version)
        payload.setdefault("prompt_version", self.prompt_version)
        return JudgeRequest(
            question_id=self.question_id,
            question_text=self.question_text,
            subquestions=self.subquestions,
            max_scores=self.max_scores,
            rubric_blocks=self.rubric_blocks,
            evidence_packet=evidence_packet,
            metadata=payload,
        )


def load_llm_question_specs(
    grading_config: GradingConfig,
    questions_path: Path,
    rubrics_path: Path,
) -> dict[QuestionId, LoadedQuestionSpec]:
    question_payload = _load_toml(questions_path)
    rubric_payload = _load_toml(rubrics_path)

    loaded_specs: dict[QuestionId, LoadedQuestionSpec] = {}
    for question_id in QUESTION_SUBQUESTION_IDS:
        raw_question = question_payload.get(question_id)
        if raw_question is None:
            raise ValueError(f"Missing question spec for {question_id}.")
        raw_rubric = rubric_payload.get(question_id)
        if raw_rubric is None:
            raise ValueError(f"Missing rubric spec for {question_id}.")

        question_source = QuestionSource.model_validate(raw_question)
        rubric_source = RubricSource.model_validate(raw_rubric)
        expected_subquestions = QUESTION_SUBQUESTION_IDS[question_id]
        if set(question_source.subquestions) != set(expected_subquestions):
            raise ValueError(
                f"Question spec for {question_id} must define {list(expected_subquestions)}."
            )
        if set(rubric_source.blocks) != set(expected_subquestions):
            raise ValueError(
                f"Rubric spec for {question_id} must define {list(expected_subquestions)}."
            )

        subquestions: list[QuestionSubquestion] = []
        rubric_blocks: list[RubricBlock] = []
        score_map: dict[SubquestionId, float] = {}

        for subquestion_id in expected_subquestions:
            question_sub = question_source.subquestions[subquestion_id]
            max_score = grading_config.score_for_key(question_sub.max_score_key)
            subquestions.append(
                QuestionSubquestion(
                    subquestion_id=subquestion_id,
                    question_text=question_sub.text,
                    max_score=max_score,
                )
            )
            score_map[subquestion_id] = max_score

            rubric_block = rubric_source.blocks[subquestion_id]
            rubric_blocks.append(
                RubricBlock(
                    subquestion_id=subquestion_id,
                    required_evidence=rubric_block.required_evidence,
                    partial_credit_guidance=rubric_block.partial_credit_guidance,
                    common_failure_modes=rubric_block.common_failure_modes,
                    score_band_guidance=rubric_block.score_band_guidance,
                    feedback_guidance=rubric_block.feedback_guidance,
                )
            )

        loaded_specs[question_id] = LoadedQuestionSpec(
            question_id=question_id,
            question_text=question_source.question_text,
            subquestions=subquestions,
            max_scores=MaxScores(
                overall=sum(score_map.values()),
                subquestions=score_map,
            ),
            rubric_version=rubric_source.rubric_version,
            prompt_version=rubric_source.prompt_version,
            rubric_blocks=rubric_blocks,
        )

    return loaded_specs


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)
