from __future__ import annotations

from ..config import GradingConfig, LLMRuntimeConfig
from .interface import LLMJudge, PlaceholderLLMJudge
from .openai_adapter import OpenAIJudgeAdapter


def build_llm_judge(
    grading_config: GradingConfig,
    llm_config: LLMRuntimeConfig | None = None,
) -> LLMJudge:
    runtime_config = llm_config or grading_config.llm
    if not runtime_config.enabled:
        return PlaceholderLLMJudge()
    if runtime_config.provider == "openai":
        return OpenAIJudgeAdapter(runtime_config)
    raise ValueError(f"Unsupported LLM provider: {runtime_config.provider}")
