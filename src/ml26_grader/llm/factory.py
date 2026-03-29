from __future__ import annotations

from ..config import GradingConfig
from .interface import LLMJudge, PlaceholderLLMJudge
from .openai_adapter import OpenAIJudgeAdapter


def build_llm_judge(grading_config: GradingConfig) -> LLMJudge:
    if not grading_config.llm.enabled:
        return PlaceholderLLMJudge()
    if grading_config.llm.provider == "openai":
        return OpenAIJudgeAdapter(grading_config.llm)
    raise ValueError(f"Unsupported LLM provider: {grading_config.llm.provider}")
