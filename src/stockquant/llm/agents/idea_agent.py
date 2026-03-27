"""Idea Agent - 生成市场假设"""

import json
from typing import Any, Dict, List, Optional

from stockquant.llm.agents.base import AgentResponse, BaseAgent
from stockquant.llm.prompts import GENERATE_HYPOTHESIS_PROMPT, IDEA_AGENT_SYSTEM_PROMPT


class IdeaAgent(BaseAgent):
    """Idea Agent - 基于市场观察生成假设"""

    def __init__(self, llm_client, temperature: float = 0.8):
        super().__init__(
            name="IdeaAgent",
            llm_client=llm_client,
            system_prompt=IDEA_AGENT_SYSTEM_PROMPT,
            temperature=temperature,
        )

    async def execute(self, context: Dict[str, Any]) -> AgentResponse:
        """生成市场假设

        Args:
            context: {
                "market_regime": str,  # bull/bear/sideways
                "volatility": str,     # high/medium/low
                "trend": str,          # upward/downward/ranging
                "recent_performance": Dict,
                "existing_hypotheses": List[str],
                "user_direction": str,  # optional
                "num_hypotheses": int,  # default 3
            }

        Returns:
            AgentResponse with hypotheses list
        """
        num_hypotheses = context.get("num_hypotheses", 3)

        # 构建提示词
        prompt = GENERATE_HYPOTHESIS_PROMPT.format(
            num_hypotheses=num_hypotheses,
            market_regime=context.get("market_regime", "unknown"),
            volatility=context.get("volatility", "unknown"),
            trend=context.get("trend", "unknown"),
            recent_performance=context.get("recent_performance", {}),
            existing_hypotheses="\n".join(
                f"- {h}" for h in context.get("existing_hypotheses", [])[-10:]
            ),
            user_direction=context.get("user_direction", ""),
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
            hypotheses = data.get("hypotheses", [])

            self._update_history(prompt, response.content)

            return AgentResponse(
                success=True,
                data={"hypotheses": hypotheses},
                reasoning=data.get("reasoning", ""),
                tokens_used=response.tokens_used,
            )
        except json.JSONDecodeError as e:
            self.log_action("parse_error", {"error": str(e), "content": response.content})
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Failed to parse JSON: {e}",
                tokens_used=response.tokens_used,
            )

    async def improve_hypothesis(
        self, hypothesis: Dict, feedback: str, num_variations: int = 2
    ) -> AgentResponse:
        """基于反馈改进假设"""
        from stockquant.llm.prompts import IMPROVE_HYPOTHESIS_PROMPT

        prompt = IMPROVE_HYPOTHESIS_PROMPT.format(
            hypothesis=json.dumps(hypothesis, ensure_ascii=False, indent=2),
            feedback=feedback,
            num_variations=num_variations,
        )

        messages = self._build_messages(prompt)
        response = await self.llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.content)
            return AgentResponse(
                success=True,
                data={"improved_hypotheses": data.get("improved_hypotheses", [])},
                tokens_used=response.tokens_used,
            )
        except json.JSONDecodeError as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Failed to parse improvement: {e}",
                tokens_used=response.tokens_used,
            )
