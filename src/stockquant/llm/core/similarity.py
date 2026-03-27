"""AST 相似度计算 - 用于原创性检测"""

from typing import List, Set, Tuple

import numpy as np

from .ast_node import ASTNode, NodeType
from .factor_ast import FactorAST


class ASTSimilarity:
    """AST 相似度计算器"""

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def compute(self, ast1: ASTNode, ast2: ASTNode) -> float:
        """计算两个 AST 的相似度 [0, 1]

        使用加权组合：
        - 结构相似度 (40%)
        - 节点类型相似度 (30%)
        - 子树重叠度 (30%)
        """
        structure_sim = self._structure_similarity(ast1, ast2)
        type_sim = self._node_type_similarity(ast1, ast2)
        subtree_sim = self._subtree_overlap(ast1, ast2)

        return 0.4 * structure_sim + 0.3 * type_sim + 0.3 * subtree_sim

    def _structure_similarity(self, ast1: ASTNode, ast2: ASTNode) -> float:
        """结构相似度 - 基于树编辑距离近似"""
        # 使用深度和节点数比较
        depth_diff = abs(ast1.depth() - ast2.depth())
        count_diff = abs(ast1.node_count() - ast2.node_count())

        depth_sim = max(0, 1 - depth_diff / max(ast1.depth(), ast2.depth(), 1))
        count_sim = max(0, 1 - count_diff / max(ast1.node_count(), ast2.node_count(), 1))

        return (depth_sim + count_sim) / 2

    def _node_type_similarity(self, ast1: ASTNode, ast2: ASTNode) -> float:
        """节点类型相似度 - 基于节点类型分布"""
        types1 = self._get_type_distribution(ast1)
        types2 = self._get_type_distribution(ast2)

        # 计算余弦相似度
        all_types = set(types1.keys()) | set(types2.keys())
        vec1 = np.array([types1.get(t, 0) for t in all_types])
        vec2 = np.array([types2.get(t, 0) for t in all_types])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _get_type_distribution(self, node: ASTNode) -> dict:
        """获取节点类型分布"""
        distribution = {}

        def count_types(n: ASTNode):
            distribution[n.node_type] = distribution.get(n.node_type, 0) + 1
            for child in n.children:
                count_types(child)

        count_types(node)
        return distribution

    def _subtree_overlap(self, ast1: ASTNode, ast2: ASTNode) -> float:
        """子树重叠度 - 基于子树哈希"""
        subtrees1 = self._get_subtree_hashes(ast1)
        subtrees2 = self._get_subtree_hashes(ast2)

        if not subtrees1 or not subtrees2:
            return 0.0

        intersection = len(subtrees1 & subtrees2)
        union = len(subtrees1 | subtrees2)

        return intersection / union if union > 0 else 0.0

    def _get_subtree_hashes(self, node: ASTNode) -> Set[str]:
        """获取所有子树的哈希"""
        hashes = set()

        def collect_hashes(n: ASTNode, depth: int = 0):
            if depth > self.max_depth:
                return

            # 使用规范化字符串作为哈希
            h = n._canonical_string()
            hashes.add(h)

            for child in n.children:
                collect_hashes(child, depth + 1)

        collect_hashes(node)
        return hashes

    def is_original(
        self,
        new_ast: ASTNode,
        existing_asts: List[FactorAST],
        threshold: float = 0.85,
    ) -> Tuple[bool, float, str]:
        """检查新因子是否原创

        Returns:
            (是否原创, 最高相似度, 最相似因子名)
        """
        if not existing_asts:
            return True, 0.0, ""

        max_sim = 0.0
        most_similar = ""

        for existing in existing_asts:
            sim = self.compute(new_ast, existing.ast)
            if sim > max_sim:
                max_sim = sim
                most_similar = existing.name

        is_original = max_sim < threshold
        return is_original, max_sim, most_similar

    def find_similar_factors(
        self, ast: ASTNode, existing_asts: List[FactorAST], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """查找最相似的因子"""
        similarities = []

        for existing in existing_asts:
            sim = self.compute(ast, existing.ast)
            similarities.append((existing.name, sim))

        # 排序并返回 top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
