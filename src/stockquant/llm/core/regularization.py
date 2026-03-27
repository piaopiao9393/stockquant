"""正则化约束 - 三大正则化：原创性、假设对齐、复杂度"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .factor_ast import FactorAST

from .ast_node import ASTNode, NodeType
from .similarity import ASTSimilarity


@dataclass
class RegularizationResult:
    """正则化检查结果"""

    passed: bool
    score: float  # 0-1，越高越好
    violations: List[str]
    details: dict


class OriginalityConstraint:
    """原创性约束"""

    def __init__(self, similarity_threshold: float = 0.85, max_depth: int = 5):
        self.similarity = ASTSimilarity(max_depth)
        self.threshold = similarity_threshold

    def check(
        self, new_factor: "FactorAST", existing_factors: List["FactorAST"]
    ) -> RegularizationResult:
        """检查原创性"""
        is_original, max_sim, most_similar = self.similarity.is_original(
            new_factor.ast, existing_factors, self.threshold
        )

        violations = []
        if not is_original:
            violations.append(
                f"与现有因子 '{most_similar}' 相似度过高 ({max_sim:.2%})"
            )

        # 分数：相似度越低分数越高
        score = 1.0 - max_sim

        return RegularizationResult(
            passed=is_original,
            score=score,
            violations=violations,
            details={
                "max_similarity": max_sim,
                "most_similar_factor": most_similar,
                "threshold": self.threshold,
            },
        )


class HypothesisAlignmentConstraint:
    """假设对齐约束"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def check(
        self, factor: "FactorAST", market_context: Optional[dict] = None
    ) -> RegularizationResult:
        """检查因子是否与生成假设对齐

        使用 LLM 评估因子表达式与市场假设的逻辑一致性
        """
        if not self.llm_client:
            # 如果没有 LLM 客户端，跳过检查
            return RegularizationResult(
                passed=True, score=1.0, violations=[], details={"skipped": True}
            )

        # 构造评估提示
        prompt = self._build_alignment_prompt(factor, market_context)

        try:
            response = self.llm_client.evaluate(prompt)
            alignment_score = response.get("alignment_score", 0.5)
            explanation = response.get("explanation", "")

            passed = alignment_score >= 0.6
            violations = []
            if not passed:
                violations.append(f"假设对齐度不足: {explanation}")

            return RegularizationResult(
                passed=passed,
                score=alignment_score,
                violations=violations,
                details={"explanation": explanation, "market_context": market_context},
            )
        except Exception as e:
            return RegularizationResult(
                passed=False, score=0.0, violations=[f"评估失败: {e}"], details={"error": str(e)}
            )

    def _build_alignment_prompt(
        self, factor: "FactorAST", market_context: Optional[dict]
    ) -> str:
        """构建对齐评估提示"""
        context_str = ""
        if market_context:
            context_str = f"""
市场背景:
- 当前市场阶段: {market_context.get('market_regime', 'unknown')}
- 近期波动率: {market_context.get('volatility', 'unknown')}
- 主要趋势: {market_context.get('trend', 'unknown')}
"""

        return f"""评估以下因子与市场假设的对齐程度。

市场假设: {factor.hypothesis}

因子表达式: {factor.to_expression()}
因子描述: {factor.description}
{context_str}

请评估该因子是否有效表达了市场假设，返回 JSON:
{{
    "alignment_score": 0.0-1.0,  // 对齐分数
    "explanation": "简要解释"
}}"""


class ComplexityConstraint:
    """复杂度约束"""

    def __init__(self, max_nodes: int = 50, max_depth: int = 10, min_nodes: int = 3):
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.min_nodes = min_nodes

    def check(self, factor: "FactorAST") -> RegularizationResult:
        """检查复杂度"""
        violations = []
        node_count = factor.ast.node_count()
        depth = factor.ast.depth()

        # 检查节点数
        if node_count > self.max_nodes:
            violations.append(f"节点数 {node_count} 超过上限 {self.max_nodes}")

        if node_count < self.min_nodes:
            violations.append(f"节点数 {node_count} 低于下限 {self.min_nodes}")

        # 检查深度
        if depth > self.max_depth:
            violations.append(f"深度 {depth} 超过上限 {self.max_depth}")

        # 计算复杂度分数（适中最好）
        optimal_nodes = (self.min_nodes + self.max_nodes) / 2
        node_score = 1.0 - abs(node_count - optimal_nodes) / optimal_nodes

        optimal_depth = self.max_depth / 2
        depth_score = 1.0 - abs(depth - optimal_depth) / self.max_depth

        score = (node_score + depth_score) / 2

        return RegularizationResult(
            passed=len(violations) == 0,
            score=max(0, score),
            violations=violations,
            details={
                "node_count": node_count,
                "depth": depth,
                "node_score": node_score,
                "depth_score": depth_score,
            },
        )


class RegularizationPipeline:
    """正则化管道 - 组合所有约束"""

    def __init__(
        self,
        originality_threshold: float = 0.85,
        max_nodes: int = 50,
        max_depth: int = 10,
        llm_client=None,
    ):
        self.originality = OriginalityConstraint(originality_threshold)
        self.alignment = HypothesisAlignmentConstraint(llm_client)
        self.complexity = ComplexityConstraint(max_nodes, max_depth)

    def validate(
        self,
        factor: "FactorAST",
        existing_factors: List["FactorAST"],
        market_context: Optional[dict] = None,
    ) -> Tuple[bool, dict]:
        """运行所有正则化检查

        Returns:
            (是否通过, 详细结果)
        """
        results = {
            "originality": self.originality.check(factor, existing_factors),
            "alignment": self.alignment.check(factor, market_context),
            "complexity": self.complexity.check(factor),
        }

        # 计算综合分数
        total_score = sum(r.score for r in results.values()) / len(results)

        # 检查是否通过
        all_passed = all(r.passed for r in results.values())

        # 收集所有违规
        all_violations = []
        for name, result in results.items():
            for v in result.violations:
                all_violations.append(f"[{name}] {v}")

        return all_passed, {
            "passed": all_passed,
            "total_score": total_score,
            "violations": all_violations,
            "details": results,
        }
