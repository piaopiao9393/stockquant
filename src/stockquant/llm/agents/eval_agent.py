"""Eval Agent - 评估因子质量"""

import json
from typing import Any, Dict, List, Optional

from stockquant.llm.agents.base import AgentResponse, BaseAgent
from stockquant.llm.core import FactorAST
from stockquant.llm.prompts import EVALUATE_FACTOR_PROMPT, EVAL_AGENT_SYSTEM_PROMPT


class EvalAgent(BaseAgent):
    """Eval Agent - 评估因子质量并提供改进建议"""

    def __init__(self, llm_client, temperature: float = 0.3):
        super().__init__(
            name="EvalAgent",
            llm_client=llm_client,
            system_prompt=EVAL_AGENT_SYSTEM_PROMPT,
            temperature=temperature,
        )

    async def execute(self, context: Dict[str, Any]) -> AgentResponse:
        """评估因子

        Args:
            context: {
                "factor": FactorAST,
                "backtest_result": Dict,
                "ic_analysis": Dict,
                "regularization_result": Dict or RegularizationResult,
                "similar_factors": List[Tuple[str, float]],
            }

        Returns:
            AgentResponse with evaluation results
        """
        factor: FactorAST = context["factor"]
        backtest = context.get("backtest_result", {})
        ic = context.get("ic_analysis", {})
        reg = context.get("regularization_result", {})
        similar = context.get("similar_factors", [])

        # 从 regularization_result 中提取分数
        def get_score(reg_result, key):
            """支持 RegularizationResult 对象或 dict 格式"""
            try:
                # Try dict access first
                if isinstance(reg_result, dict):
                    details = reg_result.get("details", {})
                    item = details.get(key, {})
                    if isinstance(item, dict):
                        return item.get("score", 0)
                    # item is RegularizationResult
                    return getattr(item, "score", 0)
                # Try object attribute access
                if hasattr(reg_result, "details"):
                    detail = reg_result.details.get(key)
                    if detail is not None:
                        return getattr(detail, "score", 1.0 if getattr(detail, "passed", False) else 0.0)
            except Exception:
                pass
            return 0.0

        # 构建提示词
        prompt = EVALUATE_FACTOR_PROMPT.format(
            factor_name=factor.name,
            factor_expression=factor.to_expression(),
            factor_description=factor.description,
            hypothesis=factor.hypothesis,
            complexity=factor.complexity,
            depth=factor.depth,
            ic_mean=ic.get("ic_mean", 0),
            ic_std=ic.get("ic_std", 0),
            icir=ic.get("ir", 0),
            positive_ic_ratio=ic.get("positive_ratio", 0),
            annual_return=backtest.get("annual_return", 0),
            sharpe_ratio=backtest.get("sharpe_ratio", 0),
            max_drawdown=backtest.get("max_drawdown", 0),
            win_rate=backtest.get("win_rate", 0),
            volatility=backtest.get("volatility", 0),
            originality_score=get_score(reg, "originality"),
            complexity_score=get_score(reg, "complexity"),
            alignment_score=get_score(reg, "alignment") or 1.0,
            similar_factors="\n".join(f"- {name}: {sim:.2%}" for name, sim in similar[:5]),
        )

        # 调用 LLM
        messages = self._build_messages(prompt)
        response = await self.llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        # 解析响应
        try:
            data = json.loads(response.content)
            evaluation = data.get("evaluation", {})

            return AgentResponse(
                success=True,
                data={
                    "is_valid": evaluation.get("is_valid", False),
                    "score": evaluation.get("overall_score", 0),
                    "grade": evaluation.get("grade", "F"),
                    "assessment": evaluation.get("assessment", {}),
                    "issues": evaluation.get("issues", []),
                    "suggestions": evaluation.get("improvement_suggestions", []),
                    "should_continue": evaluation.get("should_continue", False),
                    "next_steps": evaluation.get("recommended_next_steps", ""),
                },
                reasoning=evaluation.get("assessment", {}),
                tokens_used=response.tokens_used,
            )
        except json.JSONDecodeError as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Failed to parse evaluation: {e}",
                tokens_used=response.tokens_used,
            )

    async def compare_factors(
        self, factors: List[FactorAST], metrics: List[Dict]
    ) -> AgentResponse:
        """比较多个因子，选择最佳"""
        from stockquant.llm.prompts import COMPARE_FACTORS_PROMPT

        factors_list = []
        for f, m in zip(factors, metrics):
            factors_list.append({
                "name": f.name,
                "expression": f.to_expression(),
                "ic_mean": m.get("ic_mean", 0),
                "sharpe": m.get("sharpe_ratio", 0),
                "complexity": f.complexity,
            })

        prompt = COMPARE_FACTORS_PROMPT.format(
            factors_list=json.dumps(factors_list, indent=2)
        )

        messages = self._build_messages(prompt)
        response = await self.llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.content)
            comparison = data.get("comparison", {})

            return AgentResponse(
                success=True,
                data={
                    "rankings": comparison.get("rankings", []),
                    "recommended": comparison.get("recommended_factor", ""),
                    "reasoning": comparison.get("selection_reasoning", ""),
                },
                tokens_used=response.tokens_used,
            )
        except json.JSONDecodeError as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Failed to parse comparison: {e}",
                tokens_used=response.tokens_used,
            )
