"""Factor Agent - 构建因子表达式"""

import json
from typing import Any, Dict, List, Optional

from stockquant.llm.agents.base import AgentResponse, BaseAgent
from stockquant.llm.core import ASTNode, FactorAST, NodeType
from stockquant.llm.prompts import (
    BUILD_FACTOR_PROMPT,
    FACTOR_AGENT_SYSTEM_PROMPT,
    FEEDBACK_TEMPLATE,
    PARENT_FACTOR_TEMPLATE,
)


class FactorAgent(BaseAgent):
    """Factor Agent - 将假设转化为因子表达式"""

    def __init__(self, llm_client, temperature: float = 0.6):
        super().__init__(
            name="FactorAgent",
            llm_client=llm_client,
            system_prompt=FACTOR_AGENT_SYSTEM_PROMPT,
            temperature=temperature,
        )

    async def execute(self, context: Dict[str, Any]) -> AgentResponse:
        """将假设转化为因子

        Args:
            context: {
                "hypothesis": Dict,        # 市场假设
                "parent_factor": FactorAST, # optional, for variation
                "feedback": List[Dict],    # optional, evaluation feedback
                "round": int,              # generation round
            }

        Returns:
            AgentResponse with FactorAST
        """
        hypothesis = context["hypothesis"]
        parent_factor = context.get("parent_factor")
        feedback = context.get("feedback", [])

        # 构建父因子信息
        parent_info = ""
        if parent_factor:
            parent_info = PARENT_FACTOR_TEMPLATE.format(
                parent_name=parent_factor.name,
                parent_expression=parent_factor.to_expression(),
                variation_direction=hypothesis.get("variation_direction", "optimize"),
                issues="; ".join(f.get("issue", "") for f in feedback),
            )

        # 构建反馈信息
        feedback_info = ""
        if feedback:
            feedback_info = FEEDBACK_TEMPLATE.format(
                feedback_list="\n".join(
                    f"- {f.get('issue')}: {f.get('suggestion')}" for f in feedback[-3:]
                )
            )

        # 构建提示词
        prompt = BUILD_FACTOR_PROMPT.format(
            hypothesis_statement=hypothesis.get("statement", ""),
            rationale=hypothesis.get("reasoning", ""),
            expected_form=hypothesis.get("expected_factor_form", ""),
            max_complexity=context.get("max_complexity", 20),
            window_sizes=context.get("window_sizes", "5, 10, 20, 60"),
            parent_factor_info=parent_info,
            feedback_info=feedback_info,
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
            factor_data = data.get("factor", {})

            # 构建 AST
            ast = self._parse_ast(factor_data.get("ast", {}))

            factor = FactorAST(
                name=factor_data.get("name", "unnamed_factor"),
                ast=ast,
                description=factor_data.get("description", ""),
                hypothesis=hypothesis.get("statement", ""),
                direction=factor_data.get("direction", 1),
                source_agent="FactorAgent",
                generation_round=context.get("round", 0),
                parent_factor=parent_factor.name if parent_factor else None,
            )

            self._update_history(prompt, response.content)

            return AgentResponse(
                success=True,
                data={
                    "factor": factor,
                    "alternatives": data.get("alternative_expressions", []),
                },
                reasoning=factor_data.get("rationale", ""),
                tokens_used=response.tokens_used,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.log_action("parse_error", {"error": str(e)})
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Failed to build factor: {e}",
                tokens_used=response.tokens_used,
            )

    def _parse_ast(self, data: Dict) -> ASTNode:
        """解析 AST JSON"""
        node_type = NodeType[data.get("type", "DATA")]
        params = data.get("params", {})
        children = [self._parse_ast(c) for c in data.get("children", [])]
        return ASTNode(node_type=node_type, children=children, params=params)

    async def fix_factor(
        self,
        original_expression: str,
        error_message: str,
        factor_context: Dict,
    ) -> AgentResponse:
        """修复有错误的因子"""
        from stockquant.llm.prompts import FIX_FACTOR_PROMPT

        prompt = FIX_FACTOR_PROMPT.format(
            original_expression=original_expression,
            error_message=error_message,
            factor_context=json.dumps(factor_context, ensure_ascii=False, indent=2),
        )

        messages = self._build_messages(prompt)
        response = await self.llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.content)
            factor_data = data.get("factor", {})
            ast = self._parse_ast(factor_data.get("ast", {}))

            factor = FactorAST(
                name=factor_data.get("name", "fixed_factor"),
                ast=ast,
                description=factor_data.get("description", ""),
                hypothesis=factor_context.get("hypothesis", ""),
                direction=factor_data.get("direction", 1),
                source_agent="FactorAgent",
            )

            return AgentResponse(
                success=True,
                data={"factor": factor},
                tokens_used=response.tokens_used,
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Failed to fix factor: {e}",
                tokens_used=response.tokens_used,
            )
