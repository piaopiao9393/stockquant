"""LLM 模块启动验证脚本 - 检查所有功能是否正常"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_imports():
    """检查所有模块导入"""
    print("=" * 60)
    print("1. 检查模块导入")
    print("=" * 60)

    checks = [
        ("stockquant.llm", "FactorAST", "Core AST"),
        ("stockquant.llm", "RegularizationPipeline", "Regularization"),
        ("stockquant.llm.agents", "IdeaAgent", "Idea Agent"),
        ("stockquant.llm.agents", "FactorAgent", "Factor Agent"),
        ("stockquant.llm.agents", "EvalAgent", "Eval Agent"),
        ("stockquant.llm.miner", "LLMFactorMiner", "Factor Miner"),
        ("stockquant.llm.miner", "MiningConfig", "Mining Config"),
        ("stockquant.llm.storage", "FactorRepository", "Factor Repository"),
        ("stockquant.llm.strategy", "LLMFactorStrategy", "LLM Strategy"),
        ("stockquant.llm.utils.llm_client", "create_llm_client", "LLM Client"),
    ]

    all_passed = True
    for module, name, desc in checks:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            print(f"  [OK] {desc}: {module}.{name}")
        except Exception as e:
            print(f"  [X] {desc}: {e}")
            all_passed = False

    return all_passed


def check_ast_functionality():
    """检查 AST 核心功能"""
    print("\n" + "=" * 60)
    print("2. 检查 AST 核心功能")
    print("=" * 60)

    try:
        from stockquant.llm.core import DataNode, TS_Mean, FactorAST
        import pandas as pd
        import numpy as np

        # 创建简单因子
        ast = TS_Mean(DataNode(field="close"), window=20)
        factor = FactorAST(
            name="test_factor",
            ast=ast,
            description="Test SMA factor",
            hypothesis="Trend following",
        )
        print(f"  [OK] 创建因子: {factor.name}")
        print(f"  [OK] 因子表达式: {factor.to_expression()}")
        print(f"  [OK] 复杂度: {factor.complexity} 节点")

        # 测试可执行性
        df = pd.DataFrame({
            "close": np.random.randn(50).cumsum() + 100,
        })
        executor = factor.to_executable()
        result = executor(df)
        print(f"  [OK] 因子计算: {len(result)} 个值")

        return True
    except Exception as e:
        print(f"  [X] AST 功能错误: {e}")
        return False


def check_regularization():
    """检查正则化功能"""
    print("\n" + "=" * 60)
    print("3. 检查正则化功能")
    print("=" * 60)

    try:
        from stockquant.llm.core import (
            DataNode, TS_Mean, FactorAST,
            RegularizationPipeline
        )

        # 创建两个相似因子
        ast1 = TS_Mean(DataNode(field="close"), window=20)
        factor1 = FactorAST(name="sma20", ast=ast1, description="20-day SMA", hypothesis="Trend")

        ast2 = TS_Mean(DataNode(field="close"), window=20)
        factor2 = FactorAST(name="sma20_copy", ast=ast2, description="Copy", hypothesis="Trend")

        # 检查原创性
        pipeline = RegularizationPipeline()
        passed, details = pipeline.validate(factor2, [factor1])

        print(f"  [OK] 原创性检查: {'通过' if passed else '未通过'}")
        print(f"  [OK] 综合分数: {details['total_score']:.4f}")

        return True
    except Exception as e:
        print(f"  [X] 正则化错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_llm_clients():
    """检查 LLM 客户端"""
    print("\n" + "=" * 60)
    print("4. 检查 LLM 客户端")
    print("=" * 60)

    try:
        from stockquant.llm.utils.llm_client import create_llm_client, MockLLMClient

        # Mock 客户端（无需 API 密钥）
        mock = create_llm_client("mock")
        print(f"  [OK] Mock 客户端: {type(mock).__name__}")

        # 检查真实客户端类是否存在
        from stockquant.llm.utils.llm_client import (
            OpenAIClient, AnthropicClient, DeepSeekClient
        )
        print(f"  [OK] OpenAI 客户端类")
        print(f"  [OK] Anthropic 客户端类")
        print(f"  [OK] DeepSeek 客户端类")

        return True
    except Exception as e:
        print(f"  [X] LLM 客户端错误: {e}")
        return False


def check_repository():
    """检查存储仓库"""
    print("\n" + "=" * 60)
    print("5. 检查因子存储仓库")
    print("=" * 60)

    try:
        import tempfile
        from pathlib import Path
        from stockquant.llm.storage import FactorRepository
        from stockquant.llm.core import DataNode, TS_Mean, FactorAST

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            repo = FactorRepository(str(db_path))

            # 保存因子
            ast = TS_Mean(DataNode(field="close"), window=20)
            factor = FactorAST(
                name="repo_test",
                ast=ast,
                description="Test factor",
                hypothesis="Test",
            )
            factor_id = repo.save_factor(factor)
            print(f"  [OK] 保存因子 ID: {factor_id}")

            # 读取因子
            stored = repo.get_factor("repo_test")
            print(f"  [OK] 读取因子: {stored.name}")

            # 统计信息
            stats = repo.get_factor_statistics()
            print(f"  [OK] 统计信息: {stats}")

        return True
    except Exception as e:
        print(f"  [X] 存储仓库错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主验证函数"""
    print("\n" + "=" * 60)
    print("StockQuant LLM 模块启动验证")
    print("=" * 60)

    results = {
        "导入检查": check_imports(),
        "AST 功能": check_ast_functionality(),
        "正则化功能": check_regularization(),
        "LLM 客户端": check_llm_clients(),
        "存储仓库": check_repository(),
    }

    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    for name, passed in results.items():
        status = "[OK]" if passed else "[X]"
        print(f"  {status} {name}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("所有检查通过！LLM 模块可以正常启动。")
        print("\n使用示例:")
        print("  from stockquant.llm import FactorAST")
        print("  from stockquant.llm.agents import IdeaAgent, FactorAgent, EvalAgent")
        print("  from stockquant.llm.miner import LLMFactorMiner")
        print("  from stockquant.llm.utils.llm_client import create_llm_client")
    else:
        print("部分检查失败，请修复上述问题。")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
