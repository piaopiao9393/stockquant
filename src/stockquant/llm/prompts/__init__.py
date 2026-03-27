"""LLM 提示词模块"""

from stockquant.llm.prompts.eval_prompts import (
    EVAL_AGENT_SYSTEM_PROMPT,
    EVALUATE_FACTOR_PROMPT,
)
from stockquant.llm.prompts.factor_prompts import (
    BUILD_FACTOR_PROMPT,
    FACTOR_AGENT_SYSTEM_PROMPT,
    FEEDBACK_TEMPLATE,
    FIX_FACTOR_PROMPT,
    PARENT_FACTOR_TEMPLATE,
)
from stockquant.llm.prompts.idea_prompts import (
    GENERATE_HYPOTHESIS_PROMPT,
    IDEA_AGENT_SYSTEM_PROMPT,
    IMPROVE_HYPOTHESIS_PROMPT,
)

__all__ = [
    # Idea Agent
    "IDEA_AGENT_SYSTEM_PROMPT",
    "GENERATE_HYPOTHESIS_PROMPT",
    "IMPROVE_HYPOTHESIS_PROMPT",
    # Factor Agent
    "FACTOR_AGENT_SYSTEM_PROMPT",
    "BUILD_FACTOR_PROMPT",
    "PARENT_FACTOR_TEMPLATE",
    "FEEDBACK_TEMPLATE",
    "FIX_FACTOR_PROMPT",
    # Eval Agent
    "EVAL_AGENT_SYSTEM_PROMPT",
    "EVALUATE_FACTOR_PROMPT",
]
