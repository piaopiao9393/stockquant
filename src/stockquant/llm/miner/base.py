"""LLM 因子挖掘器 - 整合三智能体进行因子挖掘"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from stockquant.llm.agents import EvalAgent, FactorAgent, IdeaAgent
from stockquant.llm.core import FactorAST, RegularizationPipeline


@dataclass
class MiningConfig:
    """挖掘配置"""

    max_iterations: int = 10
    max_factors_per_hypothesis: int = 3
    max_total_factors: int = 10

    # 正则化阈值
    originality_threshold: float = 0.85
    min_ic_threshold: float = 0.02
    min_sharpe_threshold: float = 0.5

    # 复杂度限制
    max_nodes: int = 30
    max_depth: int = 8

    # 回测参数
    backtest_start: str = "2020-01-01"
    backtest_end: str = "2023-12-31"
    rebalance_freq: str = "weekly"

    # LLM 参数
    idea_temperature: float = 0.8
    factor_temperature: float = 0.6
    eval_temperature: float = 0.3


@dataclass
class MiningResult:
    """挖掘结果"""

    success: bool
    factors: List[FactorAST] = field(default_factory=list)
    hypotheses: List[Dict] = field(default_factory=list)
    iterations: int = 0
    total_attempts: int = 0
    rejected_factors: List[Dict] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class LLMFactorMiner:
    """LLM 因子挖掘器"""

    def __init__(
        self,
        idea_agent: IdeaAgent,
        factor_agent: FactorAgent,
        eval_agent: EvalAgent,
        config: MiningConfig = None,
    ):
        self.idea_agent = idea_agent
        self.factor_agent = factor_agent
        self.eval_agent = eval_agent
        self.config = config or MiningConfig()

        self.regularization = RegularizationPipeline(
            originality_threshold=self.config.originality_threshold,
            max_nodes=self.config.max_nodes,
            max_depth=self.config.max_depth,
        )

        self.mining_history: List[Dict] = []
        self.discovered_factors: List[FactorAST] = []

    async def mine(
        self,
        context: Dict[str, Any],
        num_factors: int = 5,
    ) -> MiningResult:
        """执行因子挖掘

        Args:
            context: {
                "market_regime": str,
                "volatility": str,
                "trend": str,
                "market_data": Dict[str, pd.DataFrame],
                "returns": Dict[str, List[float]],
            }
            num_factors: 目标因子数量

        Returns:
            MiningResult
        """
        logger.info(f"Starting LLM factor mining, target: {num_factors} factors")

        result = MiningResult(success=False)
        result.hypotheses = []

        iteration = 0
        total_attempts = 0

        while len(result.factors) < num_factors and iteration < self.config.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.config.max_iterations}")

            # 1. 生成市场假设
            idea_response = await self.idea_agent.execute(
                {
                    "market_regime": context.get("market_regime", "unknown"),
                    "volatility": context.get("volatility", "unknown"),
                    "trend": context.get("trend", "unknown"),
                    "recent_performance": self._get_recent_performance(),
                    "existing_hypotheses": [h.get("statement", "") for h in result.hypotheses],
                    "num_hypotheses": 2,
                }
            )

            if not idea_response.success:
                logger.warning(f"Idea generation failed: {idea_response.reasoning}")
                continue

            hypotheses = idea_response.data.get("hypotheses", [])
            result.hypotheses.extend(hypotheses)

            # 2. 对每个假设生成因子
            for hypo in hypotheses:
                if len(result.factors) >= num_factors:
                    break

                factors_from_hypo = await self._process_hypothesis(
                    hypo, context, result.rejected_factors
                )

                for factor, evaluation in factors_from_hypo:
                    total_attempts += 1

                    if evaluation.get("is_valid"):
                        result.factors.append(factor)
                        self.discovered_factors.append(factor)
                        logger.info(f"Discovered valid factor: {factor.name}")
                    else:
                        result.rejected_factors.append(
                            {
                                "factor": factor,
                                "reason": evaluation.get("suggestions", []),
                            }
                        )

        result.success = len(result.factors) > 0
        result.iterations = iteration
        result.total_attempts = total_attempts
        result.metrics = {
            "success_rate": len(result.factors) / max(total_attempts, 1),
            "avg_complexity": np.mean([f.complexity for f in result.factors]) if result.factors else 0,
            "avg_ic": np.mean([f.metrics.get("ic_mean", 0) for f in result.factors]) if result.factors else 0,
        }

        logger.info(f"Mining complete: {len(result.factors)} factors discovered in {iteration} iterations")
        return result

    async def _process_hypothesis(
        self,
        hypothesis: Dict,
        context: Dict,
        previous_rejections: List[Dict],
    ) -> List[tuple]:
        """处理单个假设，生成并评估因子"""
        results = []
        max_attempts = self.config.max_factors_per_hypothesis

        for attempt in range(max_attempts):
            # 生成因子
            factor_response = await self.factor_agent.execute(
                {
                    "hypothesis": hypothesis,
                    "round": attempt,
                    "max_complexity": self.config.max_nodes,
                    "feedback": [r["reason"] for r in previous_rejections[-3:]],
                }
            )

            if not factor_response.success:
                continue

            factor: FactorAST = factor_response.data["factor"]

            # 正则化检查
            reg_passed, reg_details = self.regularization.validate(
                factor, self.discovered_factors
            )

            # 原创性不通过，直接跳过
            if not reg_details["details"]["originality"].passed:
                continue

            # 评估因子
            evaluation = await self._evaluate_factor(factor, context, reg_details)

            if evaluation.get("is_valid") or attempt == max_attempts - 1:
                results.append((factor, evaluation))

            if evaluation.get("is_valid"):
                break

        return results

    async def _evaluate_factor(
        self,
        factor: FactorAST,
        context: Dict,
        reg_details: Dict,
    ) -> Dict:
        """评估单个因子"""
        market_data = context.get("market_data", {})

        # 计算因子值
        try:
            ic_analysis = self._calculate_ic(factor, market_data, context.get("returns", {}))
        except Exception as e:
            logger.warning(f"IC calculation failed for {factor.name}: {e}")
            ic_analysis = {"ic_mean": 0, "ir": 0}

        # 简化的回测结果
        backtest_result = {
            "annual_return": 0.1,  # Placeholder
            "sharpe_ratio": 1.0,
            "max_drawdown": -0.1,
            "win_rate": 0.55,
            "volatility": 0.2,
        }

        # LLM 评估
        eval_response = await self.eval_agent.execute(
            {
                "factor": factor,
                "backtest_result": backtest_result,
                "ic_analysis": ic_analysis,
                "regularization_result": reg_details,
            }
        )

        if eval_response.success:
            evaluation = eval_response.data
            # 更新因子指标
            factor.metrics = {
                "ic_mean": ic_analysis.get("ic_mean", 0),
                "ir": ic_analysis.get("ir", 0),
                **backtest_result,
                "overall_score": evaluation.get("score", 0),
            }
            return evaluation

        return {"is_valid": False, "score": 0}

    def _calculate_ic(
        self,
        factor: FactorAST,
        market_data: Dict[str, pd.DataFrame],
        returns: Dict[str, List[float]],
    ) -> Dict:
        """计算 IC 指标"""
        executor = factor.to_executable()
        ics = []

        # 简化实现：计算单期 IC
        for symbol, df in market_data.items():
            if len(df) < 20 or symbol not in returns:
                continue

            try:
                factor_values = executor(df)
                future_returns = returns[symbol]

                if len(factor_values) == len(future_returns) and len(factor_values) > 10:
                    # 计算相关系数
                    valid_idx = ~(np.isnan(factor_values) | np.isnan(future_returns))
                    if valid_idx.sum() > 10:
                        ic = np.corrcoef(factor_values[valid_idx], future_returns[valid_idx])[0, 1]
                        if not np.isnan(ic):
                            ics.append(ic)
            except Exception:
                continue

        if not ics:
            return {"ic_mean": 0, "ic_std": 0, "ir": 0, "positive_ratio": 0}

        return {
            "ic_mean": np.mean(ics),
            "ic_std": np.std(ics),
            "ir": np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0,
            "positive_ratio": sum(1 for ic in ics if ic > 0) / len(ics),
        }

    def _get_recent_performance(self) -> Dict:
        """获取最近的表现（简化）"""
        return {
            "last_ic": 0.02,
            "best_performing_factor": "unknown",
            "market_condition": "neutral",
        }
