"""AST 节点定义 - 基于 AlphaAgent 的因子表示"""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class NodeType(Enum):
    """节点类型"""

    # 数据节点
    DATA = auto()  # 原始数据字段
    CONSTANT = auto()  # 常数

    # 算术运算
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()

    # 统计运算
    MEAN = auto()
    STD = auto()
    MAX = auto()
    MIN = auto()
    SUM = auto()

    # 时间序列
    DELTA = auto()  # 差分
    SHIFT = auto()  # 滞后
    TS_MEAN = auto()  # 时序均值
    TS_STD = auto()  # 时序标准差
    TS_MAX = auto()
    TS_MIN = auto()
    TS_SUM = auto()
    TS_RANK = auto()  # 时序排名
    TS_ZSCORE = auto()  # 时序标准化

    # 横截面
    CS_RANK = auto()  # 截面排名
    CS_ZSCORE = auto()  # 截面标准化

    # 逻辑运算
    IF = auto()  # 条件
    AND = auto()
    OR = auto()
    NOT = auto()

    # 比较
    GT = auto()  # >
    LT = auto()  # <
    EQ = auto()  # ==
    GE = auto()  # >=
    LE = auto()  # <=


@dataclass
class ASTNode:
    """AST 节点"""

    node_type: NodeType
    children: List["ASTNode"] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "type": self.node_type.name,
            "params": self.params,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ASTNode":
        """从字典反序列化"""
        node_type = NodeType[data["type"]]
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            node_type=node_type,
            children=children,
            params=data.get("params", {}),
        )

    def depth(self) -> int:
        """计算节点深度"""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def node_count(self) -> int:
        """计算节点数量"""
        if not self.children:
            return 1
        return 1 + sum(child.node_count() for child in self.children)

    def __hash__(self) -> int:
        """用于原创性检测的哈希"""
        return hash(self._canonical_string())

    def _canonical_string(self) -> str:
        """规范化字符串表示（用于比较）"""
        children_str = ",".join(child._canonical_string() for child in self.children)
        params_str = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.node_type.name}({params_str})[{children_str}]"


# 工厂函数用于创建特定类型的节点

def DataNode(field: str) -> ASTNode:
    """创建数据字段节点"""
    return ASTNode(node_type=NodeType.DATA, params={"field": field})


def ConstantNode(value: float) -> ASTNode:
    """创建常数节点"""
    return ASTNode(node_type=NodeType.CONSTANT, params={"value": value})


def Add(left: ASTNode, right: ASTNode) -> ASTNode:
    """加法节点"""
    return ASTNode(node_type=NodeType.ADD, children=[left, right])


def Sub(left: ASTNode, right: ASTNode) -> ASTNode:
    """减法节点"""
    return ASTNode(node_type=NodeType.SUB, children=[left, right])


def Mul(left: ASTNode, right: ASTNode) -> ASTNode:
    """乘法节点"""
    return ASTNode(node_type=NodeType.MUL, children=[left, right])


def Div(left: ASTNode, right: ASTNode) -> ASTNode:
    """除法节点"""
    return ASTNode(node_type=NodeType.DIV, children=[left, right])


def TS_Mean(operand: ASTNode, window: int) -> ASTNode:
    """时间序列均值"""
    return ASTNode(
        node_type=NodeType.TS_MEAN, children=[operand], params={"window": window}
    )


def TS_Std(operand: ASTNode, window: int) -> ASTNode:
    """时间序列标准差"""
    return ASTNode(
        node_type=NodeType.TS_STD, children=[operand], params={"window": window}
    )


def TS_Rank(operand: ASTNode, window: int) -> ASTNode:
    """时间序列排名"""
    return ASTNode(
        node_type=NodeType.TS_RANK, children=[operand], params={"window": window}
    )


def CS_Rank(operand: ASTNode) -> ASTNode:
    """截面排名"""
    return ASTNode(node_type=NodeType.CS_RANK, children=[operand])


def Delta(operand: ASTNode, periods: int = 1) -> ASTNode:
    """差分"""
    return ASTNode(
        node_type=NodeType.DELTA, children=[operand], params={"periods": periods}
    )
