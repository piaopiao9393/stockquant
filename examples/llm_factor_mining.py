"""
StockQuant LLM 因子挖掘 - 使用示例

本示例展示如何使用 LLM 因子挖掘功能自动发现 A 股市场的 Alpha 因子
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd


async def example_1_basic_factor_mining():
    """示例1: 基础因子挖掘（使用 Mock LLM）"""
    print("=" * 60)
    print("示例1: 基础因子挖掘")
    print("=" * 60)

    from stockquant.llm.agents import IdeaAgent, FactorAgent, EvalAgent
    from stockquant.llm.miner import LLMFactorMiner, MiningConfig
    from stockquant.llm.utils.llm_client import MockLLMClient, LLMResponse

    # 创建自定义 Mock 客户端
    class ExampleMockClient(MockLLMClient):
        async def generate(self, messages, temperature=0.7, max_tokens=2000, response_format=None):
            prompt = messages[-1]["content"] if messages else ""

            # Idea Agent
            if "generate" in prompt.lower():
                return LLMResponse(content='''{
                    "hypotheses": [{
                        "statement": "High volume breakout with momentum",
                        "reasoning": "Volume surge indicates institutional interest",
                        "expected_factor_form": "Volume-adjusted price momentum",
                        "confidence": 0.8
                    }]
                }''')

            # Factor Agent
            elif "convert" in prompt.lower():
                return LLMResponse(content='''{
                    "factor": {
                        "name": "volume_momentum",
                        "description": "Volume-weighted price momentum",
                        "rationale": "Captures price momentum with volume confirmation",
                        "direction": 1,
                        "ast": {
                            "type": "MUL",
                            "params": {},
                            "children": [
                                {"type": "TS_MEAN", "params": {"window": 5}, "children": [
                                    {"type": "DATA", "params": {"field": "close"}, "children": []}
                                ]},
                                {"type": "DATA", "params": {"field": "vol"}, "children": []}
                            ]
                        }
                    }
                }''')

            # Eval Agent
            else:
                return LLMResponse(content='''{
                    "evaluation": {
                        "is_valid": true,
                        "overall_score": 0.72,
                        "grade": "B",
                        "assessment": {"predictive_power": "good", "economic_rationale": "strong"},
                        "issues": [],
                        "improvement_suggestions": ["Try different window sizes"],
                        "should_continue": true
                    }
                }''')

    # 创建客户端和智能体
    client = ExampleMockClient()
    idea_agent = IdeaAgent(client, temperature=0.8)
    factor_agent = FactorAgent(client, temperature=0.6)
    eval_agent = EvalAgent(client, temperature=0.3)

    # 配置挖掘器
    config = MiningConfig(
        max_iterations=2,
        max_factors_per_hypothesis=1,
        max_total_factors=2,
    )
    miner = LLMFactorMiner(idea_agent, factor_agent, eval_agent, config)

    # 准备测试数据
    dates = pd.date_range("2024-01-01", periods=50)
    market_data = {
        "000001.SZ": pd.DataFrame({
            "open": np.random.randn(50).cumsum() + 100,
            "high": np.random.randn(50).cumsum() + 101,
            "low": np.random.randn(50).cumsum() + 99,
            "close": np.random.randn(50).cumsum() + 100,
            "vol": np.random.randint(1000, 10000, 50),
        }, index=dates),
    }
    returns = {"000001.SZ": list(np.random.randn(50) * 0.02)}

    context = {
        "market_regime": "bull",
        "volatility": "medium",
        "trend": "upward",
        "market_data": market_data,
        "returns": returns,
    }

    # 执行挖掘
    result = await miner.mine(context, num_factors=1)

    print(f"挖掘结果: {len(result.factors)} 个因子")
    print(f"迭代次数: {result.iterations}")
    print(f"总尝试: {result.total_attempts}")

    for factor in result.factors:
        print(f"\n因子名称: {factor.name}")
        print(f"描述: {factor.description}")
        print(f"表达式: {factor.to_expression()}")
        print(f"复杂度: {factor.complexity} 节点")

    return result


async def example_2_factor_repository():
    """示例2: 因子仓库操作"""
    print("\n" + "=" * 60)
    print("示例2: 因子仓库操作")
    print("=" * 60)

    import tempfile
    from stockquant.llm.storage import FactorRepository
    from stockquant.llm.core import DataNode, TS_Mean, FactorAST

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建仓库
        repo = FactorRepository(f"{tmpdir}/factors.db")
        print(f"创建仓库: {tmpdir}/factors.db")

        # 创建示例因子
        ast = TS_Mean(DataNode(field="close"), window=20)
        factor = FactorAST(
            name="sma20",
            ast=ast,
            description="20-day simple moving average",
            hypothesis="Trend following",
            direction=1,
        )
        factor.metrics = {"ic_mean": 0.035, "ir": 0.8}

        # 保存因子
        factor_id = repo.save_factor(factor)
        print(f"保存因子 ID: {factor_id}")

        # 读取因子
        stored = repo.get_factor("sma20")
        print(f"读取因子: {stored.name}, IC: {stored.metrics.get('ic_mean')}")

        # 获取统计
        stats = repo.get_factor_statistics()
        print(f"统计: {stats}")

        # 导出
        export_path = f"{tmpdir}/factors.json"
        repo.export_factors(export_path)
        print(f"导出到: {export_path}")


def example_3_custom_factor():
    """示例3: 创建自定义因子"""
    print("\n" + "=" * 60)
    print("示例3: 创建自定义因子")
    print("=" * 60)

    from stockquant.llm.core import (
        DataNode, ConstantNode, TS_Mean, TS_Std,
        ASTNode, NodeType, FactorAST
    )
    import pandas as pd
    import numpy as np

    # 创建 Z-Score 因子: (close - mean(close, 20)) / std(close, 20)
    close = DataNode(field="close")
    mean20 = TS_Mean(close, window=20)
    std20 = TS_Std(close, window=20)

    # 分子: close - mean
    numerator = ASTNode(NodeType.SUB, children=[close, mean20])
    # 分母: std
    denominator = std20

    # Z-Score
    zscore = ASTNode(NodeType.DIV, children=[numerator, denominator])

    factor = FactorAST(
        name="price_zscore",
        ast=zscore,
        description="Price Z-Score (mean reversion)",
        hypothesis="Mean reversion after price deviation",
        direction=-1,  # 负向因子（越小越好）
    )

    print(f"因子名称: {factor.name}")
    print(f"表达式: {factor.to_expression()}")
    print(f"复杂度: {factor.complexity} 节点, 深度: {factor.depth}")

    # 测试计算
    dates = pd.date_range("2024-01-01", periods=30)
    df = pd.DataFrame({
        "close": np.random.randn(30).cumsum() + 100,
    }, index=dates)

    executor = factor.to_executable()
    result = executor(df)
    print(f"计算结果: {len(result)} 个值")
    print(f"最新值: {result.iloc[-1]:.4f}")

    return factor


async def main():
    """运行所有示例"""
    print("StockQuant LLM 因子挖掘 - 示例脚本")
    print("=" * 60)

    try:
        # 运行示例
        await example_1_basic_factor_mining()
        await example_2_factor_repository()
        example_3_custom_factor()

        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
