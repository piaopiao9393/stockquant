"""LLM Agent System 测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import tempfile

import numpy as np
import pandas as pd
from loguru import logger

# Disable verbose logging during tests
logger.remove()

from stockquant.llm.agents import EvalAgent, FactorAgent, IdeaAgent
from stockquant.llm.core import FactorAST
from stockquant.llm.miner import LLMFactorMiner, MiningConfig
from stockquant.llm.storage import FactorRepository
from stockquant.llm.utils.llm_client import MockLLMClient, LLMResponse


class AgentSpecificMockClient(MockLLMClient):
    """为不同 Agent 返回特定响应的 Mock 客户端"""

    async def generate(self, messages, temperature=0.7, max_tokens=2000, response_format=None):
        """根据消息内容返回特定响应"""
        prompt = messages[-1]["content"] if messages else ""

        # Idea Agent: 包含 "generate" 和 "hypotheses"
        if "generate" in prompt.lower() and "hypotheses" in prompt.lower():
            content = """{
                "hypotheses": [
                    {
                        "statement": "Mean reversion after high volatility",
                        "reasoning": "Stocks with high volatility tend to revert to mean",
                        "expected_factor_form": "Normalized deviation from moving average",
                        "confidence": 0.8
                    }
                ],
                "reasoning": "Based on current high volatility environment"
            }"""
            return LLMResponse(content=content, tokens_used=100, model="mock")

        # Factor Agent: 包含 "convert" 和 "factor"
        elif "convert" in prompt.lower() and "hypothesis" in prompt.lower():
            content = """{
                "factor": {
                    "name": "volatility_reversion",
                    "description": "Mean reversion after high volatility",
                    "rationale": "Captures the tendency of volatile stocks to revert",
                    "direction": -1,
                    "ast": {
                        "type": "TS_ZSCORE",
                        "params": {"window": 20},
                        "children": [
                            {
                                "type": "TS_MEAN",
                                "params": {"window": 5},
                                "children": [
                                    {
                                        "type": "DATA",
                                        "params": {"field": "close"},
                                        "children": []
                                    }
                                ]
                            }
                        ]
                    }
                },
                "alternative_expressions": []
            }"""
            return LLMResponse(content=content, tokens_used=100, model="mock")

        # Eval Agent: 包含 "evaluate" 或性能指标
        elif "evaluate" in prompt.lower() or "ic_mean" in prompt.lower():
            content = """{
                "evaluation": {
                    "is_valid": true,
                    "overall_score": 0.75,
                    "grade": "B+",
                    "assessment": {
                        "performance": "Good IC and Sharpe ratio",
                        "economic_intuition": "Clear mean reversion mechanism",
                        "robustness": "Stable across different periods"
                    },
                    "issues": [],
                    "improvement_suggestions": ["Consider different window sizes"],
                    "should_continue": true,
                    "recommended_next_steps": "Test on out-of-sample data"
                }
            }"""
            return LLMResponse(content=content, tokens_used=100, model="mock")

        # 默认响应
        return LLMResponse(
            content='{"result": "mock"}',
            tokens_used=50,
            model="mock",
        )


def test_idea_agent():
    """测试 Idea Agent"""
    print("Testing IdeaAgent...")

    async def async_test():
        client = AgentSpecificMockClient()
        agent = IdeaAgent(llm_client=client, temperature=0.8)

        context = {
            "market_regime": "high_volatility",
            "volatility": "high",
            "trend": "sideways",
            "recent_performance": {"last_ic": 0.02},
            "existing_hypotheses": [],
            "num_hypotheses": 1,
        }

        response = await agent.execute(context)

        assert response.success is True
        assert "hypotheses" in response.data
        assert len(response.data["hypotheses"]) > 0
        return True

    return asyncio.run(async_test())


def test_factor_agent():
    """测试 Factor Agent"""
    print("Testing FactorAgent...")

    async def async_test():
        client = AgentSpecificMockClient()
        agent = FactorAgent(llm_client=client, temperature=0.6)

        hypothesis = {
            "statement": "Mean reversion after high volatility",
            "reasoning": "Stocks with high volatility tend to revert",
        }

        context = {
            "hypothesis": hypothesis,
            "round": 0,
            "max_complexity": 20,
            "feedback": [],
        }

        response = await agent.execute(context)

        assert response.success is True
        assert "factor" in response.data
        factor = response.data["factor"]
        assert isinstance(factor, FactorAST)
        assert factor.name == "volatility_reversion"
        return True

    return asyncio.run(async_test())


def test_eval_agent():
    """测试 Eval Agent"""
    print("Testing EvalAgent...")

    async def async_test():
        client = AgentSpecificMockClient()
        agent = EvalAgent(llm_client=client, temperature=0.3)

        # Create a proper AST for the factor
        from stockquant.llm.core import DataNode, TS_Mean
        ast = TS_Mean(DataNode(field="close"), window=20)

        factor = FactorAST(
            name="test_factor",
            ast=ast,
            description="Test factor",
            hypothesis="Test hypothesis",
        )

        context = {
            "factor": factor,
            "backtest_result": {
                "annual_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.1,
                "win_rate": 0.55,
                "volatility": 0.2,
            },
            "ic_analysis": {
                "ic_mean": 0.03,
                "ic_std": 0.02,
                "ir": 1.5,
                "positive_ratio": 0.6,
            },
            "regularization_result": {
                "passed": True,
                "total_score": 0.9,
            },
            "similar_factors": [],
        }

        response = await agent.execute(context)

        assert response.success is True
        assert "is_valid" in response.data
        assert response.data.get("is_valid") is True
        return True

    return asyncio.run(async_test())


def test_factor_repository():
    """测试 Factor Repository"""
    print("Testing FactorRepository...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_factors.db"
        repo = FactorRepository(str(db_path))

        # Create test factor
        from stockquant.llm.core import DataNode, TS_Mean

        ast = TS_Mean(DataNode(field="close"), window=20)
        factor = FactorAST(
            name="test_sma20",
            ast=ast,
            description="20-day SMA test",
            hypothesis="Trend following",
            direction=1,
        )
        factor.metrics = {"ic_mean": 0.03, "ir": 0.5}

        # Save factor
        factor_id = repo.save_factor(factor)
        assert factor_id > 0

        # Get factor by name
        stored = repo.get_factor("test_sma20")
        assert stored is not None
        assert stored.name == "test_sma20"
        assert stored.metrics["ic_mean"] == 0.03

        # List factors
        factors = repo.list_factors()
        assert len(factors) == 1

        # Deactivate
        repo.deactivate_factor("test_sma20")
        active_factors = repo.list_factors(active_only=True)
        assert len(active_factors) == 0

        # Statistics
        stats = repo.get_factor_statistics()
        assert stats["total_factors"] == 1
        assert stats["active_factors"] == 0

        # Export/Import
        export_path = Path(tmpdir) / "export.json"
        repo.export_factors(str(export_path), active_only=False)
        assert export_path.exists()

        # Delete and re-import
        repo.delete_factor("test_sma20")
        assert len(repo.list_factors(active_only=False)) == 0

        imported = repo.import_factors(str(export_path))
        assert imported == 1

    return True


def test_llm_factor_miner():
    """测试 LLM Factor Miner 完整流程"""
    print("Testing LLMFactorMiner...")

    async def async_test():
        client = AgentSpecificMockClient()

        idea_agent = IdeaAgent(llm_client=client, temperature=0.8)
        factor_agent = FactorAgent(llm_client=client, temperature=0.6)
        eval_agent = EvalAgent(llm_client=client, temperature=0.3)

        config = MiningConfig(
            max_iterations=1,
            max_factors_per_hypothesis=1,
            max_total_factors=1,
            originality_threshold=0.85,
        )

        miner = LLMFactorMiner(idea_agent, factor_agent, eval_agent, config)

        # Create test market data
        dates = pd.date_range("2024-01-01", periods=100)
        market_data = {
            "000001.SZ": pd.DataFrame(
                {
                    "open": np.random.randn(100).cumsum() + 100,
                    "high": np.random.randn(100).cumsum() + 101,
                    "low": np.random.randn(100).cumsum() + 99,
                    "close": np.random.randn(100).cumsum() + 100,
                    "vol": np.random.randint(1000, 10000, 100),
                },
                index=dates,
            ),
        }

        returns = {
            "000001.SZ": list(np.random.randn(100) * 0.02),
        }

        context = {
            "market_regime": "high_volatility",
            "volatility": "high",
            "trend": "sideways",
            "market_data": market_data,
            "returns": returns,
        }

        result = await miner.mine(context, num_factors=1)

        assert result is not None
        assert result.iterations >= 0
        assert result.total_attempts >= 0

        # Check if at least tried to mine (might not find valid factor with mock data)
        print(f"  Mining result: {len(result.factors)} factors in {result.iterations} iterations")

        return True

    return asyncio.run(async_test())


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("LLM Agent System Tests")
    print("=" * 60)

    tests = [
        test_idea_agent,
        test_factor_agent,
        test_eval_agent,
        test_factor_repository,
        test_llm_factor_miner,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                print(f"  [OK] {test.__name__}")
                passed += 1
            else:
                print(f"  X {test.__name__} returned False")
                failed += 1
        except Exception as e:
            print(f"  X {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
