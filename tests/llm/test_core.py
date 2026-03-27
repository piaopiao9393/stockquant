"""LLM 核心模块测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from stockquant.llm.core import (
    ASTNode,
    ASTSimilarity,
    ComplexityConstraint,
    ConstantNode,
    DataNode,
    FactorAST,
    NodeType,
    OriginalityConstraint,
    RegularizationPipeline,
    TS_Mean,
    TS_Std,
)


class TestASTNode:
    """测试 AST 节点"""

    def test_data_node_creation(self):
        """测试数据节点创建"""
        node = DataNode(field="close")
        assert node.node_type == NodeType.DATA
        assert node.params["field"] == "close"
        assert node.depth() == 1
        assert node.node_count() == 1

    def test_constant_node_creation(self):
        """测试常数节点创建"""
        node = ConstantNode(value=10.5)
        assert node.node_type == NodeType.CONSTANT
        assert node.params["value"] == 10.5

    def test_ts_mean_creation(self):
        """测试时间序列均值节点"""
        data = DataNode(field="close")
        ts_mean = TS_Mean(data, window=20)
        assert ts_mean.node_type == NodeType.TS_MEAN
        assert ts_mean.params["window"] == 20
        assert ts_mean.depth() == 2
        assert ts_mean.node_count() == 2

    def test_binary_op(self):
        """测试二元运算"""
        left = DataNode(field="close")
        right = DataNode(field="open")
        add = ASTNode(NodeType.ADD, children=[left, right])
        assert add.depth() == 2
        assert add.node_count() == 3

    def test_canonical_string(self):
        """测试规范化字符串"""
        data = DataNode(field="close")
        ts_mean = TS_Mean(data, window=20)
        canonical = ts_mean._canonical_string()
        assert "TS_MEAN" in canonical
        assert "DATA(close)" in canonical

    def test_serialization(self):
        """测试序列化和反序列化"""
        data = DataNode(field="close")
        ts_mean = TS_Mean(data, window=20)
        ts_std = TS_Std(ts_mean, window=10)

        # 序列化
        data_dict = ts_std.to_dict()
        assert data_dict["type"] == "TS_STD"
        assert data_dict["params"]["window"] == 10

        # 反序列化
        restored = ASTNode.from_dict(data_dict)
        assert restored.node_type == NodeType.TS_STD
        assert restored.params["window"] == 10
        assert restored.depth() == ts_std.depth()


class TestFactorAST:
    """测试 FactorAST"""

    @pytest.fixture
    def simple_factor(self):
        """简单因子：20日均线"""
        data = DataNode(field="close")
        ast = TS_Mean(data, window=20)
        return FactorAST(
            name="sma20",
            ast=ast,
            description="20日均线",
            hypothesis="短期趋势跟踪",
        )

    @pytest.fixture
    def complex_factor(self):
        """复杂因子：价格波动率"""
        data = DataNode(field="close")
        delta = ASTNode(NodeType.DELTA, children=[data], params={"periods": 1})
        std = TS_Std(delta, window=20)
        return FactorAST(
            name="price_volatility",
            ast=std,
            description="价格变化率波动率",
            hypothesis="高波动后可能有反转",
        )

    def test_factor_id_computation(self, simple_factor):
        """测试因子 ID 计算"""
        assert len(simple_factor.factor_id) == 16
        assert simple_factor.factor_id == simple_factor._compute_id()

    def test_complexity_property(self, simple_factor, complex_factor):
        """测试复杂度属性"""
        assert simple_factor.complexity == 2
        assert complex_factor.complexity == 3

    def test_depth_property(self, simple_factor, complex_factor):
        """测试深度属性"""
        assert simple_factor.depth == 2
        assert complex_factor.depth == 3

    def test_to_expression(self, simple_factor, complex_factor):
        """测试表达式转换"""
        expr1 = simple_factor.to_expression()
        assert "ts_mean" in expr1
        assert "close" in expr1

        expr2 = complex_factor.to_expression()
        assert "ts_std" in expr2
        assert "delta" in expr2

    def test_to_dict(self, simple_factor):
        """测试字典序列化"""
        data = simple_factor.to_dict()
        assert data["name"] == "sma20"
        assert data["description"] == "20日均线"
        assert data["complexity"] == 2
        assert "factor_id" in data

    def test_from_dict(self, simple_factor):
        """测试字典反序列化"""
        data = simple_factor.to_dict()
        restored = FactorAST.from_dict(data)
        assert restored.name == simple_factor.name
        assert restored.to_expression() == simple_factor.to_expression()

    def test_get_required_fields(self, simple_factor, complex_factor):
        """测试获取所需字段"""
        fields1 = simple_factor.get_required_fields()
        assert fields1 == {"close"}

        fields2 = complex_factor.get_required_fields()
        assert fields2 == {"close"}

    def test_to_executable(self, simple_factor):
        """测试可执行函数生成"""
        # 创建测试数据
        dates = pd.date_range("2024-01-01", periods=30)
        df = pd.DataFrame(
            {
                "close": np.random.randn(30).cumsum() + 100,
                "open": np.random.randn(30).cumsum() + 100,
            },
            index=dates,
        )

        # 获取可执行函数
        executor = simple_factor.to_executable()

        # 执行计算
        result = executor(df)

        # 验证结果
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        # 前20个值应该是累积平均
        assert not result.isna().all()

    def test_to_executable_with_field_mapping(self):
        """测试带字段映射的可执行函数"""
        data = DataNode(field="close")
        ast = TS_Mean(data, window=5)
        factor = FactorAST(name="test", ast=ast)

        # 创建测试数据，使用不同字段名
        df = pd.DataFrame(
            {
                "close_adj": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

        # 使用字段映射
        executor = factor.to_executable(field_mapping={"close": "close_adj"})
        result = executor(df)

        assert len(result) == 10
        assert result.iloc[4] == 3.0  # 前5个平均


class TestASTSimilarity:
    """测试 AST 相似度计算"""

    @pytest.fixture
    def similarity(self):
        return ASTSimilarity()

    def test_identical_factors(self, similarity):
        """测试相同因子的相似度"""
        data = DataNode(field="close")
        ast1 = TS_Mean(data, window=20)
        ast2 = TS_Mean(DataNode(field="close"), window=20)

        sim = similarity.compute(ast1, ast2)
        assert sim == 1.0

    def test_different_factors(self, similarity):
        """测试不同因子的相似度"""
        ast1 = TS_Mean(DataNode(field="close"), window=20)
        ast2 = TS_Std(DataNode(field="volume"), window=10)

        sim = similarity.compute(ast1, ast2)
        assert 0 <= sim < 1.0

    def test_partial_similarity(self, similarity):
        """测试部分相似"""
        # 两个因子都使用 close 和 TS_MEAN
        ast1 = TS_Mean(DataNode(field="close"), window=20)
        ast2 = TS_Mean(DataNode(field="close"), window=30)

        sim = similarity.compute(ast1, ast2)
        assert 0.5 < sim < 1.0  # 应该比较相似但不是完全相同

    def test_is_original(self, similarity):
        """测试原创性检查"""
        # 创建现有因子库
        existing = [
            FactorAST(name="sma20", ast=TS_Mean(DataNode(field="close"), window=20)),
            FactorAST(name="sma30", ast=TS_Mean(DataNode(field="close"), window=30)),
        ]

        # 新因子 - 非常不同
        new_factor = FactorAST(
            name="volatility",
            ast=TS_Std(DataNode(field="volume"), window=20),
        )

        is_original, max_sim, most_similar = similarity.is_original(
            new_factor.ast, existing, threshold=0.85
        )

        assert is_original is True
        assert max_sim < 0.85

    def test_not_original(self, similarity):
        """测试非原创因子"""
        existing = [
            FactorAST(name="sma20", ast=TS_Mean(DataNode(field="close"), window=20)),
        ]

        # 几乎相同的因子
        new_factor = FactorAST(
            name="sma20_copy",
            ast=TS_Mean(DataNode(field="close"), window=20),
        )

        is_original, max_sim, most_similar = similarity.is_original(
            new_factor.ast, existing, threshold=0.85
        )

        assert is_original is False
        assert max_sim > 0.85
        assert most_similar == "sma20"

    def test_find_similar_factors(self, similarity):
        """测试查找相似因子"""
        existing = [
            FactorAST(name="f1", ast=TS_Mean(DataNode(field="close"), window=20)),
            FactorAST(name="f2", ast=TS_Std(DataNode(field="close"), window=20)),
            FactorAST(name="f3", ast=TS_Mean(DataNode(field="volume"), window=20)),
        ]

        query = TS_Mean(DataNode(field="close"), window=25)
        results = similarity.find_similar_factors(query, existing, top_k=2)

        assert len(results) == 2
        # f1 应该是最相似的
        assert results[0][0] == "f1"


class TestOriginalityConstraint:
    """测试原创性约束"""

    def test_original_factor(self):
        """测试原创因子通过"""
        constraint = OriginalityConstraint(threshold=0.85)

        existing = [
            FactorAST(name="sma20", ast=TS_Mean(DataNode(field="close"), window=20)),
        ]

        new_factor = FactorAST(
            name="volatility",
            ast=TS_Std(DataNode(field="volume"), window=20),
        )

        result = constraint.check(new_factor, existing)

        assert result.passed is True
        assert result.score > 0.15  # 1 - 相似度
        assert len(result.violations) == 0

    def test_duplicate_factor(self):
        """测试重复因子被拒绝"""
        constraint = OriginalityConstraint(threshold=0.85)

        existing = [
            FactorAST(name="sma20", ast=TS_Mean(DataNode(field="close"), window=20)),
        ]

        new_factor = FactorAST(
            name="sma20_v2",
            ast=TS_Mean(DataNode(field="close"), window=20),
        )

        result = constraint.check(new_factor, existing)

        assert result.passed is False
        assert len(result.violations) == 1
        assert "相似度过高" in result.violations[0]


class TestComplexityConstraint:
    """测试复杂度约束"""

    def test_normal_complexity(self):
        """测试正常复杂度通过"""
        constraint = ComplexityConstraint(max_nodes=50, max_depth=10)

        ast = TS_Mean(DataNode(field="close"), window=20)
        factor = FactorAST(name="test", ast=ast)

        result = constraint.check(factor)

        assert result.passed is True
        assert result.score > 0.5

    def test_too_complex(self):
        """测试过于复杂的因子被拒绝"""
        constraint = ComplexityConstraint(max_nodes=5, max_depth=3)

        # 构建一个复杂因子
        data = DataNode(field="close")
        ts_mean = TS_Mean(data, window=20)
        ts_std = TS_Std(ts_mean, window=10)
        delta = ASTNode(NodeType.DELTA, children=[ts_std], params={"periods": 1})

        factor = FactorAST(name="complex", ast=delta)

        result = constraint.check(factor)

        assert result.passed is False
        assert any("节点数" in v or "深度" in v for v in result.violations)

    def test_too_simple(self):
        """测试过于简单的因子被拒绝"""
        constraint = ComplexityConstraint(min_nodes=3)

        # 简单因子：只有数据节点
        factor = FactorAST(name="simple", ast=DataNode(field="close"))

        result = constraint.check(factor)

        assert result.passed is False
        assert any("节点数" in v for v in result.violations)


class TestRegularizationPipeline:
    """测试正则化管道"""

    def test_all_pass(self):
        """测试所有检查通过"""
        pipeline = RegularizationPipeline()

        existing = []
        factor = FactorAST(
            name="test",
            ast=TS_Mean(DataNode(field="close"), window=20),
            hypothesis="测试假设",
        )

        passed, details = pipeline.validate(factor, existing)

        assert passed is True
        assert details["passed"] is True
        assert len(details["violations"]) == 0

    def test_complexity_fail(self):
        """测试复杂度检查失败"""
        pipeline = RegularizationPipeline(max_nodes=2)

        # 复杂因子
        data = DataNode(field="close")
        ts_mean = TS_Mean(data, window=20)
        ts_std = TS_Std(ts_mean, window=10)

        factor = FactorAST(name="complex", ast=ts_std)

        passed, details = pipeline.validate(factor, [])

        assert passed is False
        assert any("[complexity]" in v for v in details["violations"])

    def test_originality_fail(self):
        """测试原创性检查失败"""
        pipeline = RegularizationPipeline(threshold=0.85)

        ast = TS_Mean(DataNode(field="close"), window=20)
        existing = [FactorAST(name="sma20", ast=ast)]
        new_factor = FactorAST(name="sma20_copy", ast=ast)

        passed, details = pipeline.validate(new_factor, existing)

        assert passed is False
        assert any("[originality]" in v for v in details["violations"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
