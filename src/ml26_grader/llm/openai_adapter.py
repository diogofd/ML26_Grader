from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import ValidationError

from ..config import LLMRuntimeConfig
from .interface import JudgeEvaluationAudit
from .schemas import JudgeQuestionResult, JudgeRequest


class LLMProviderError(RuntimeError):
    pass


class OpenAIChatCompletionsTransport(Protocol):
    def create_chat_completion(
        self,
        *,
        api_key: str,
        base_url: str,
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        ...


class DefaultOpenAIChatCompletionsTransport:
    def create_chat_completion(
        self,
        *,
        api_key: str,
        base_url: str,
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        endpoint = base_url.rstrip("/") + "/chat/completions"
        request = urllib_request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMProviderError(
                f"OpenAI HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib_error.URLError as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc.reason}") from exc

        try:
            return json.loads(raw_body)
        except JSONDecodeError as exc:
            raise LLMProviderError("OpenAI response body was not valid JSON.") from exc


class OpenAIJudgeAdapter:
    def __init__(
        self,
        config: LLMRuntimeConfig,
        *,
        transport: OpenAIChatCompletionsTransport | None = None,
    ) -> None:
        if config.provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        if not config.model:
            raise ValueError("LLM model must be configured for the OpenAI judge adapter.")

        self._config = config
        self._transport = transport or DefaultOpenAIChatCompletionsTransport()
        self.last_evaluation_audit: JudgeEvaluationAudit | None = None

    def evaluate(self, request: JudgeRequest) -> JudgeQuestionResult:
        api_key = os.environ.get(self._config.api_key_env_var)
        if not api_key:
            self.last_evaluation_audit = JudgeEvaluationAudit(
                provider="openai",
                configured_model=self._config.model or "unknown",
                attempts=1,
                repair_attempted=False,
                error=f"Environment variable {self._config.api_key_env_var} is not set.",
            )
            raise LLMProviderError(
                f"Environment variable {self._config.api_key_env_var} is not set."
            )

        schema = _build_response_schema(request)
        system_prompt = _build_system_prompt(
            confidence_threshold=self._config.auto_accept_confidence,
        )
        user_prompt = _build_user_prompt(request)

        attempts = 0
        repair_attempted = False
        last_error: str | None = None
        raw_output_text: str | None = None
        response_payload: dict[str, Any] | None = None

        while attempts <= self._config.max_repair_attempts:
            attempts += 1
            if attempts == 1:
                current_user_prompt = user_prompt
            else:
                repair_attempted = True
                current_user_prompt = _build_repair_prompt(
                    original_user_prompt=user_prompt,
                    raw_output_text=raw_output_text or "",
                    validation_error=last_error or "Invalid structured output.",
                )

            try:
                response_payload = self._transport.create_chat_completion(
                    api_key=api_key,
                    base_url=self._config.api_base_url,
                    payload=_build_chat_completions_payload(
                        model=self._config.model,
                        temperature=self._config.temperature,
                        system_prompt=system_prompt,
                        user_prompt=current_user_prompt,
                        response_schema=schema,
                    ),
                    timeout_seconds=self._config.timeout_seconds,
                )
                raw_output_text = _extract_message_text(response_payload)
            except Exception as exc:
                provider_error = (
                    exc
                    if isinstance(exc, LLMProviderError)
                    else LLMProviderError(f"OpenAI request failed: {exc}")
                )
                self.last_evaluation_audit = _build_audit_record(
                    configured_model=self._config.model,
                    attempts=attempts,
                    repair_attempted=repair_attempted,
                    response_payload=response_payload,
                    raw_output_text=raw_output_text,
                    error=str(provider_error),
                )
                raise provider_error from exc
            try:
                parsed_payload = json.loads(raw_output_text)
                result = JudgeQuestionResult.model_validate(parsed_payload)
            except (JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                if attempts > self._config.max_repair_attempts:
                    self.last_evaluation_audit = _build_audit_record(
                        configured_model=self._config.model,
                        attempts=attempts,
                        repair_attempted=repair_attempted,
                        response_payload=response_payload,
                        raw_output_text=raw_output_text,
                        error=f"Structured output validation failed: {last_error}",
                    )
                    raise LLMProviderError(
                        f"Structured output validation failed: {last_error}"
                    ) from exc
                continue

            self.last_evaluation_audit = _build_audit_record(
                configured_model=self._config.model,
                attempts=attempts,
                repair_attempted=repair_attempted,
                response_payload=response_payload,
                raw_output_text=raw_output_text,
            )
            return result

        raise AssertionError("Unreachable OpenAI judge retry loop state.")


def _build_chat_completions_payload(
    *,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "assignment_question_grade",
                "strict": True,
                "schema": response_schema,
            },
        },
    }


def _build_system_prompt(*, confidence_threshold: float) -> str:
    return (
        "You are a strict rubric-based grader for Assignment 1 questions Q2 and Q3.\n"
        "Use only the supplied rubric and evidence packet.\n"
        "Do not infer missing work, do not use outside assumptions, and do not reward implied evidence.\n"
        "Full marks require strong explicit evidence across the required rubric elements, not just fluent prose.\n"
        "When required workflow elements are described only in markdown and code or output evidence is absent or limited, full marks should be rare.\n"
        "Treat extraction warnings as first-class evidence quality indicators. Warnings such as limited_code_evidence, limited_output_evidence, missing_deployment_recommendation_signal, and similar ambiguity or incompleteness indicators should reduce confidence and may justify partial credit.\n"
        "Confidence must reflect the reliability of the grading decision, not the apparent quality of the submission writing.\n"
        f"If the evidence packet is materially incomplete or warning-heavy, confidence should usually remain below {confidence_threshold:.1f}.\n"
        "For Q3.2, missing explicit deployment recommendation evidence should matter unless a recommendation is directly and clearly supported in the evidence packet.\n"
        "Return JSON only, matching the provided schema exactly.\n"
        f"Set review_recommended to true whenever question-level confidence is below {confidence_threshold:.1f} "
        "or meaningful ambiguity remains.\n"
        "The overall score must equal the sum of the subquestion scores."
    )


def _build_user_prompt(request: JudgeRequest) -> str:
    payload = {
        "question_id": request.question_id,
        "question_text": request.question_text,
        "subquestions": [
            {
                "subquestion_id": item.subquestion_id,
                "question_text": item.question_text,
                "max_score": item.max_score,
            }
            for item in request.subquestions
        ],
        "max_scores": request.max_scores.model_dump(mode="json"),
        "rubric_blocks": [
            rubric_block.model_dump(mode="json")
            for rubric_block in request.rubric_blocks
        ],
        "evidence_packet": request.evidence_packet.model_dump(mode="json"),
        "evidence_quality_summary": _build_evidence_quality_summary(request),
        "calibration_instructions": {
            "full_marks_rule": (
                "Reserve full marks for cases where the required rubric elements are strongly and explicitly evidenced."
            ),
            "prose_only_rule": (
                "If required workflow elements appear mostly in prose and code/output support is absent or limited, full marks should be rare."
            ),
            "warning_rule": (
                "Extraction warnings are evidence-quality constraints, not cosmetic notes. Warning-heavy packets should reduce confidence and may justify partial credit."
            ),
            "confidence_rule": (
                "Confidence measures grading reliability. Materially incomplete or warning-heavy evidence should usually stay below the auto-accept threshold."
            ),
            "q32_rule": (
                "For Q3.2, a missing or weak deployment recommendation should reduce the score unless the recommendation is directly and clearly supported in the packet."
            ),
        },
        "metadata": request.metadata,
    }
    return (
        "Grade the following question using only the supplied evidence packet.\n"
        "Return exactly one JSON object matching the schema.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True, default=str)
    )


def _build_repair_prompt(
    *,
    original_user_prompt: str,
    raw_output_text: str,
    validation_error: str,
) -> str:
    return (
        original_user_prompt
        + "\n\nThe previous assistant response was invalid."
        + "\nValidation issue:\n"
        + validation_error
        + "\n\nPrevious output:\n"
        + raw_output_text
        + "\n\nRepair the output and return only valid JSON matching the same schema."
        + "\nPreserve the strict evidence-quality calibration rules from the original prompt."
    )


def _build_evidence_quality_summary(request: JudgeRequest) -> dict[str, Any]:
    warning_codes = [
        warning.code
        for warning in request.evidence_packet.extraction_warnings
    ]
    material_warning_codes = [
        warning_code
        for warning_code in warning_codes
        if warning_code
        in {
            "limited_code_evidence",
            "limited_output_evidence",
            "missing_preprocessing_signal",
            "missing_feature_engineering_signal",
            "missing_data_split_signal",
            "missing_tuning_signal",
            "missing_metric_choice_signal",
            "missing_model_comparison_signal",
            "missing_deployment_recommendation_signal",
            "fewer_than_two_models_detected",
        }
    ]
    return {
        "markdown_snippet_count": len(request.evidence_packet.markdown_snippets),
        "code_snippet_count": len(request.evidence_packet.code_snippets),
        "output_snippet_count": len(request.evidence_packet.output_snippets),
        "extraction_warning_codes": warning_codes,
        "material_warning_codes": material_warning_codes,
        "warning_heavy_packet": len(set(material_warning_codes)) >= 2,
        "narrative_only_packet": {
            "limited_code_evidence",
            "limited_output_evidence",
        }.issubset(set(material_warning_codes)),
    }


def _extract_message_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProviderError("Provider response did not contain any choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LLMProviderError("Provider choice payload was invalid.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LLMProviderError("Provider response did not contain a message payload.")
    refusal = message.get("refusal")
    if refusal:
        raise LLMProviderError(f"Provider refusal: {refusal}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderError("Provider response did not contain JSON content.")
    return content


def _build_audit_record(
    *,
    configured_model: str,
    attempts: int,
    repair_attempted: bool,
    response_payload: dict[str, Any] | None,
    raw_output_text: str | None,
    error: str | None = None,
) -> JudgeEvaluationAudit:
    response_payload = response_payload or {}
    return JudgeEvaluationAudit(
        provider="openai",
        configured_model=configured_model,
        response_model=_as_optional_string(response_payload.get("model")),
        provider_request_id=_as_optional_string(response_payload.get("id")),
        attempts=attempts,
        repair_attempted=repair_attempted,
        usage=response_payload.get("usage", {}) if isinstance(response_payload.get("usage"), dict) else {},
        raw_output_text=raw_output_text,
        error=error,
    )


def _as_optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _build_response_schema(request: JudgeRequest) -> dict[str, Any]:
    subquestion_properties: dict[str, Any] = {}
    required_subquestion_ids: list[str] = []
    for subquestion in request.subquestions:
        required_subquestion_ids.append(subquestion.subquestion_id)
        subquestion_properties[subquestion.subquestion_id] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": subquestion.max_score,
                },
                "max_score": {
                    "type": "number",
                    "const": subquestion.max_score,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                },
                "student_feedback": {"type": "string", "minLength": 1},
                "internal_notes": {"type": "string", "minLength": 1},
                "evidence_used": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "missing_requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "score",
                "max_score",
                "confidence",
                "student_feedback",
                "internal_notes",
                "evidence_used",
                "missing_requirements",
            ],
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_id": {
                "type": "string",
                "const": request.question_id,
            },
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": request.max_scores.overall,
            },
            "max_score": {
                "type": "number",
                "const": request.max_scores.overall,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 10,
            },
            "student_feedback_overall": {"type": "string", "minLength": 1},
            "internal_notes_overall": {"type": "string", "minLength": 1},
            "review_recommended": {"type": "boolean"},
            "review_reasons": {
                "type": "array",
                "items": {"type": "string"},
            },
            "subquestions": {
                "type": "object",
                "additionalProperties": False,
                "properties": subquestion_properties,
                "required": required_subquestion_ids,
            },
        },
        "required": [
            "question_id",
            "score",
            "max_score",
            "confidence",
            "student_feedback_overall",
            "internal_notes_overall",
            "review_recommended",
            "review_reasons",
            "subquestions",
        ],
    }
