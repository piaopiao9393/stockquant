"""FactorAST - 因子的 AST 表示和可执行转换"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .ast_node import ASTNode, DataNode, NodeType


@dataclass
class FactorAST:
    """因子 AST 表示"""

    name: str
    ast: ASTNode
    description: str = ""
    hypothesis: str = ""  # 生成该因子的市场假设
    category: str = "llm"  # 因子类别
    direction: int = 1  # 1=正向, -1=负向

    # 元数据
    created_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    source_agent: str = ""  # 生成的智能体
    generation_round: int = 0  # 生成轮次
    parent_factor: Optional[str] = None  # 父因子（如果是变异）

    # 评估指标
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self._id = self._compute_id()

    def _compute_id(self) -> str:
        """计算因子唯一 ID（基于 AST 结构）"""
        canonical = self.ast._canonical_string()
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @property
    def factor_id(self) -> str:
        """获取因子 ID"""
        return self._id

    @property
    def complexity(self) -> int:
        """计算因子复杂度（节点数）"""
        return self.ast.node_count()

    @property
    def depth(self) -> int:
        """计算因子深度"""
        return self.ast.depth()

    def to_dict(self) -> Dict:
        """序列化"""
        return {
            "factor_id": self._id,
            "name": self.name,
            "ast": self.ast.to_dict(),
            "description": self.description,
            "hypothesis": self.hypothesis,
            "category": self.category,
            "direction": self.direction,
            "created_at": self.created_at,
            "source_agent": self.source_agent,
            "generation_round": self.generation_round,
            "parent_factor": self.parent_factor,
            "metrics": self.metrics,
            "complexity": self.complexity,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FactorAST":
        """反序列化"""
        return cls(
            name=data["name"],
            ast=ASTNode.from_dict(data["ast"]),
            description=data.get("description", ""),
            hypothesis=data.get("hypothesis", ""),
            category=data.get("category", "llm"),
            direction=data.get("direction", 1),
            created_at=data.get("created_at", ""),
            source_agent=data.get("source_agent", ""),
            generation_round=data.get("generation_round", 0),
            parent_factor=data.get("parent_factor"),
            metrics=data.get("metrics", {}),
        )

    def to_expression(self) -> str:
        """转换为可读表达式（用于展示）"""
        return self._node_to_expr(self.ast)

    def _node_to_expr(self, node: ASTNode) -> str:
        """递归转换节点为表达式"""
        if node.node_type == NodeType.DATA:
            return node.params.get("field", "unknown")

        if node.node_type == NodeType.CONSTANT:
            return str(node.params.get("value", 0))

        # 一元运算
        if len(node.children) == 1:
            child_expr = self._node_to_expr(node.children[0])
            op_map = {
                NodeType.TS_MEAN: f"ts_mean({child_expr}, {node.params.get('window', 20)})",
                NodeType.TS_STD: f"ts_std({child_expr}, {node.params.get('window', 20)})",
                NodeType.TS_RANK: f"ts_rank({child_expr}, {node.params.get('window', 20)})",
                NodeType.CS_RANK: f"cs_rank({child_expr})",
                NodeType.DELTA: f"delta({child_expr}, {node.params.get('periods', 1)})",
                NodeType.SHIFT: f"shift({child_expr}, {node.params.get('periods', 1)})",
            }
            return op_map.get(node.node_type, f"{node.node_type.name}({child_expr})")

        # 二元运算
        if len(node.children) == 2:
            left = self._node_to_expr(node.children[0])
            right = self._node_to_expr(node.children[1])
            op_map = {
                NodeType.ADD: f"({left} + {right})",
                NodeType.SUB: f"({left} - {right})",
                NodeType.MUL: f"({left} * {right})",
                NodeType.DIV: f"({left} / {right})",
                NodeType.POW: f"({left} ^ {right})",
            }
            return op_map.get(node.node_type, f"{node.node_type.name}({left}, {right})")

        # 多元运算
        children_exprs = [self._node_to_expr(c) for c in node.children]
        return f"{node.node_type.name}({', '.join(children_exprs)})"

    def to_executable(
        self, field_mapping: Optional[Dict[str, str]] = None
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """转换为可执行函数

        Args:
            field_mapping: 字段名映射，如 {"close": "close_adj"}

        Returns:
            可执行的因子计算函数
        """
        field_map = field_mapping or {}

        def executor(df: pd.DataFrame) -> pd.Series:
            """执行因子计算"""
            return self._execute_node(self.ast, df, field_map)

        return executor

    def _execute_node(
        self, node: ASTNode, df: pd.DataFrame, field_map: Dict[str, str]
    ) -> pd.Series:
        """递归执行节点"""

        # 数据节点
        if node.node_type == NodeType.DATA:
            field = node.params.get("field", "close")
            mapped_field = field_map.get(field, field)
            if mapped_field not in df.columns:
                raise ValueError(f"字段 {mapped_field} 不存在")
            return df[mapped_field]

        # 常数节点
        if node.node_type == NodeType.CONSTANT:
            value = node.params.get("value", 0)
            return pd.Series([value] * len(df), index=df.index)

        # 执行子节点
        children_results = [
            self._execute_node(child, df, field_map) for child in node.children
        ]

        # 算术运算
        if node.node_type == NodeType.ADD:
            return children_results[0] + children_results[1]
        if node.node_type == NodeType.SUB:
            return children_results[0] - children_results[1]
        if node.node_type == NodeType.MUL:
            return children_results[0] * children_results[1]
        if node.node_type == NodeType.DIV:
            # 避免除零
            denominator = children_results[1].replace(0, np.nan)
            return children_results[0] / denominator

        # 时间序列运算
        if node.node_type == NodeType.TS_MEAN:
            window = node.params.get("window", 20)
            return children_results[0].rolling(window=window, min_periods=1).mean()
        if node.node_type == NodeType.TS_STD:
            window = node.params.get("window", 20)
            return children_results[0].rolling(window=window, min_periods=1).std()
        if node.node_type == NodeType.TS_RANK:
            window = node.params.get("window", 20)
            return children_results[0].rolling(window=window).apply(
                lambda x: x.rank().iloc[-1] if len(x) > 0 else np.nan, raw=False
            )
        if node.node_type == NodeType.DELTA:
            periods = node.params.get("periods", 1)
            return children_results[0].diff(periods)
        if node.node_type == NodeType.SHIFT:
            periods = node.params.get("periods", 1)
            return children_results[0].shift(periods)

        # 截面运算（在单股票数据上无效，返回原值）
        if node.node_type in (NodeType.CS_RANK, NodeType.CS_ZSCORE):
            return children_results[0]

        raise NotImplementedError(f"未实现的节点类型: {node.node_type}")

    def get_required_fields(self) -> Set[str]:
        """获取所需的数据字段"""
        fields = set()
        self._collect_fields(self.ast, fields)
        return fields

    def _collect_fields(self, node: ASTNode, fields: Set[str]):
        """递归收集字段"""
        if node.node_type == NodeType.DATA:
            fields.add(node.params.get("field", "close"))
        for child in node.children:
            self._collect_fields(child, fields)
